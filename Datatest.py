import sys
import numpy as np
import cv2
import os
import time
import torch
import pandas as pd
import threading
from datetime import datetime

import gelsight.gsdevice as gsdevice
import gelsight.gs3drecon as gs3drecon

# ROS相关导入
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

class SessionManager:
    def __init__(self, output_root):
        self.output_root = output_root
        self.session_timestamp = None
        self.current_session_path = None
        self.data_dir = None
        self.dm_dir = None
        self.photo_dir = None
        self.active_session = False
        self.current_episode_index = 0
        self.current_dm_dir = None
        self.current_photo_dir = None
        self.current_parquet_path = None
        self.current_frame_index = 0
        self.cached_markers = []

    def create_session(self):
        if self.active_session:
            self.close_episode()

        if self.session_timestamp is None:
            self.session_timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            self.current_session_path = os.path.join(self.output_root, self.session_timestamp)
            self.data_dir = os.path.join(self.current_session_path, 'data')
            self.dm_dir = os.path.join(self.current_session_path, 'dm')
            self.photo_dir = os.path.join(self.current_session_path, 'photo')
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.dm_dir, exist_ok=True)
            os.makedirs(self.photo_dir, exist_ok=True)
            self.current_episode_index = 0
        else:
            existing_parquets = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
            max_index = -1
            for f in existing_parquets:
                try:
                    index = int(f.split('.')[0])
                    max_index = max(max_index, index)
                except ValueError:
                    continue
            self.current_episode_index = max_index + 1

        episode_dm_dir = os.path.join(self.dm_dir, f'episode_{self.current_episode_index:03d}')
        os.makedirs(episode_dm_dir, exist_ok=True)
        
        episode_photo_dir = os.path.join(self.photo_dir, f'episode_{self.current_episode_index:03d}')
        os.makedirs(episode_photo_dir, exist_ok=True)
        
        parquet_path = os.path.join(self.data_dir, f'{self.current_episode_index:03d}.parquet')

        if os.path.exists(parquet_path):
            print(f"Episode {self.current_episode_index} already exists. Aborting.")
            return False

        self.active_session = True
        self.current_dm_dir = episode_dm_dir
        self.current_photo_dir = episode_photo_dir
        self.current_parquet_path = parquet_path
        self.current_frame_index = 0
        self.cached_markers = []
        print(f"New episode: {self.current_session_path}, episode {self.current_episode_index}")
        return True

    def close_episode(self):
        if self.active_session:
            if self.cached_markers:
                self.cached_markers[-1]['next_done'] = True
                self.flush_marker_data()
            print(f"Closed episode: {self.current_episode_index}")
        self.active_session = False
        self.current_dm_dir = None
        self.current_photo_dir = None
        self.current_parquet_path = None
        self.current_frame_index = 0
        self.cached_markers = []

    def add_marker_data(self, marker_dict):
        self.cached_markers.append(marker_dict)
        if len(self.cached_markers) >= 100:
            self.flush_marker_data()

    def flush_marker_data(self):
        if not self.cached_markers or not self.active_session:
            return
        df = pd.DataFrame(self.cached_markers)
        try:
            if not os.path.exists(self.current_parquet_path):
                df.to_parquet(self.current_parquet_path, engine='pyarrow')
            else:
                existing_df = pd.read_parquet(self.current_parquet_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_parquet(self.current_parquet_path, engine='pyarrow')
            self.cached_markers = []
        except Exception as e:
            print(f"Failed to write Parquet: {str(e)}")

    def increment_frame_index(self):
        self.current_frame_index += 1

class OrbbecCamera:
    def __init__(self):
        self.latest_image = None
        self.latest_time = 0
        self.lock = threading.Lock()
        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.encoding_map = {
            "bgr8": (np.uint8, 3, cv2.COLOR_BGR2RGB),
            "rgb8": (np.uint8, 3, cv2.COLOR_RGB2BGR),
            "mono8": (np.uint8, 1, None),
            "bgra8": (np.uint8, 4, cv2.COLOR_BGRA2RGBA),
            "rgba8": (np.uint8, 4, cv2.COLOR_RGBA2BGRA),
        }
    
    def callback(self, msg):
        try:
            # 获取图像尺寸和通道类型
            height, width = msg.height, msg.width
            encoding = msg.encoding.lower()
            
            # 检查是否支持该编码
            if encoding not in self.encoding_map:
                print(f"Unsupported encoding: {encoding}")
                return
                
            dtype, channels, conversion = self.encoding_map[encoding]
            
            # 创建numpy数组
            img_data = np.frombuffer(msg.data, dtype=dtype)
            img_data = img_data.reshape((height, width, channels))
            
            # 如果需要转换颜色空间
            if conversion is not None:
                img_data = cv2.cvtColor(img_data, conversion)
            
            with self.lock:
                self.latest_image = img_data
                self.latest_time = time.time()
        except Exception as e:
            print(f"Error processing Orbbec image: {str(e)}")
    
    def get_latest_image(self):
        with self.lock:
            return self.latest_image.copy() if self.latest_image is not None else None, self.latest_time

def main(argv):
    GPU = True
    MASK_MARKERS_FLAG = True
    OUTPUT_DIR = '/home/lin/gelsight_data'
    DATA_SAVE_INTERVAL = 1

    rospy.init_node('gelsight_publisher')
    raw_image_pub = rospy.Publisher('gelsight/raw_image', Image, queue_size=10)
    depthmap_pub = rospy.Publisher('gelsight/depthmap', Image, queue_size=10)
    markers_pub = rospy.Publisher('gelsight/markers', Float32MultiArray, queue_size=10)

    # 初始化奥比中光相机
    orbbec_cam = OrbbecCamera()

    session_mgr = SessionManager(OUTPUT_DIR)
    dev = gsdevice.Camera("GelSight Mini")
    dev.connect()

    nn = gs3drecon.Reconstruction3D(dev)
    net_path = 'nnmini.pt'
    nn.load_nn(net_path, "cuda" if GPU else "cpu")

    last_key_time = 0
    key_delay = 0.3

    try:
        frame_count = 0
        start_time = time.time()
        last_report_time = start_time
        last_orbbec_warn_time = 0
        last_orbbec_image = None

        while dev.while_condition:
            frame = dev.get_image()
            if frame is None:
                continue

            # 发布原始图像到ROS
            ros_frame = Image()
            ros_frame.header.stamp = rospy.Time.now()
            ros_frame.height, ros_frame.width = frame.shape[:2]
            ros_frame.encoding = 'bgr8'
            ros_frame.is_bigendian = 0
            ros_frame.step = frame.strides[0]
            ros_frame.data = frame.tobytes()
            raw_image_pub.publish(ros_frame)

            dm = nn.get_depthmap(frame, MASK_MARKERS_FLAG)

            # 发布深度图到ROS
            if dm is not None:
                ros_dm = Image()
                ros_dm.header.stamp = rospy.Time.now()
                ros_dm.height, ros_dm.width = dm.shape[:2]
                ros_dm.encoding = '32FC1'
                ros_dm.is_bigendian = 0
                ros_dm.step = dm.strides[0]
                ros_dm.data = dm.astype(np.float32).tobytes()
                depthmap_pub.publish(ros_dm)

            current_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            if current_time - last_key_time > key_delay:
                if key == ord('s'):
                    session_mgr.create_session()
                    last_key_time = current_time
                elif key == ord('c'):
                    session_mgr.close_episode()
                    last_key_time = current_time
                elif key == ord('q'):
                    break

            # 保存数据逻辑
            if session_mgr.active_session and (frame_count % DATA_SAVE_INTERVAL == 0):
                # 保存深度图
                dm_filename = f"{session_mgr.current_episode_index:03d}_{session_mgr.current_frame_index:06d}.npy"
                dm_path = os.path.join(session_mgr.current_dm_dir, dm_filename)
                np.save(dm_path, dm)

                # 获取并保存奥比中光图像
                orbbec_frame, orbbec_time = orbbec_cam.get_latest_image()
                if orbbec_frame is not None:
                    last_orbbec_image = orbbec_frame  # 缓存最新图像
                
                # 使用缓存图像如果当前帧无效
                save_image = last_orbbec_image if orbbec_frame is None else orbbec_frame
                
                # 确保图像有效且不为空
                if save_image is not None and save_image.size > 0:
                    photo_filename = f"{session_mgr.current_episode_index:03d}_{session_mgr.current_frame_index:06d}.jpg"
                    photo_path = os.path.join(session_mgr.current_photo_dir, photo_filename)
                    
                    # 仅当图像有效且形状正确时保存
                    if len(save_image.shape) >= 2 and save_image.shape[0] > 0 and save_image.shape[1] > 0:
                        # 保存为BGR格式
                        if save_image.shape[2] == 3:
                            cv2.imwrite(photo_path, cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))
                        elif save_image.shape[2] == 4:
                            cv2.imwrite(photo_path, cv2.cvtColor(save_image, cv2.COLOR_RGBA2BGRA))
                        elif save_image.shape[2] == 1:
                            cv2.imwrite(photo_path, save_image)
                        else:
                            print(f"Invalid image channels: {save_image.shape[2]}")
                    else:
                        print(f"Invalid image shape: {save_image.shape}")
                else:
                    print(f"Skipping invalid Orbbec image at frame {session_mgr.current_frame_index}")

                # 保存marker数据
                if nn.current_marker_matrix is not None:
                    current_time_sec = time.time() - start_time
                    markers = {}
                    index = 0
                    for i in range(7):
                        for j in range(9):
                            x = nn.current_marker_matrix[i, j, 0]
                            y = nn.current_marker_matrix[i, j, 1]
                            markers[f'x_{index}'] = x
                            markers[f'y_{index}'] = y
                            index += 1
                    markers['episode_index'] = session_mgr.current_episode_index
                    markers['frame_index'] = session_mgr.current_frame_index
                    markers['timestamp'] = current_time_sec
                    markers['next_done'] = False
                    markers['index'] = session_mgr.current_frame_index
                    session_mgr.add_marker_data(markers)

                    # 发布marker矩阵到ROS
                    markers_msg = Float32MultiArray()
                    markers_msg.data = nn.current_marker_matrix.flatten().tolist()
                    markers_msg.layout.dim = [
                        MultiArrayDimension(label="dim0", size=7, stride=126),
                        MultiArrayDimension(label="dim1", size=9, stride=18),
                        MultiArrayDimension(label="dim2", size=2, stride=2)
                    ]
                    markers_pub.publish(markers_msg)

                session_mgr.increment_frame_index()

            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                total_elapsed = current_time - start_time
                average_fps = frame_count / total_elapsed
                recent_fps = 30 / (current_time - last_report_time)
                print(
                    f"[{total_elapsed:.1f}s] Frames: {frame_count} | FPS: {average_fps:.2f} (Avg) / {recent_fps:.2f} (Current)")
                last_report_time = current_time

            display_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            cv2.imshow('Control Window', display_frame)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # 修复方法名错误
        session_mgr.close_episode()
        total_elapsed = time.time() - start_time
        print(f"\nTotal frames: {frame_count}")
        print(f"Runtime: {total_elapsed:.2f}s")
        print(f"Average FPS: {frame_count / total_elapsed:.2f}")

        dev.stop_video()
        cv2.destroyAllWindows()
        print(f"Data root: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main(sys.argv[1:])