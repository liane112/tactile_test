import sys
import numpy as np
import cv2
import os
import time
import torch
import pandas as pd
import pyarrow.parquet as pq
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
        self.session_timestamp = None  # 新增：存储首次创建的session时间戳
        self.current_session_path = None
        self.data_dir = None  # 新增：session的data目录
        self.dm_dir = None  # 新增：session的dm目录
        self.active_session = False
        self.current_episode_index = 0
        self.current_dm_dir = None
        self.current_parquet_path = None
        self.current_frame_index = 0
        self.cached_markers = []

    def create_session(self):
        # 关闭当前活动的episode
        if self.active_session:
            self.close_episode()

        # 初始化session（如果未创建）
        if self.session_timestamp is None:
            self.session_timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            self.current_session_path = os.path.join(self.output_root, self.session_timestamp)
            self.data_dir = os.path.join(self.current_session_path, 'data')
            self.dm_dir = os.path.join(self.current_session_path, 'dm')
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.dm_dir, exist_ok=True)
            self.current_episode_index = 0
        else:
            # 查找最大的episode索引并递增
            existing_parquets = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
            max_index = -1
            for f in existing_parquets:
                try:
                    index = int(f.split('.')[0])
                    max_index = max(max_index, index)
                except ValueError:
                    continue
            self.current_episode_index = max_index + 1

        # 创建episode目录
        episode_dm_dir = os.path.join(self.dm_dir, f'episode_{self.current_episode_index:03d}')
        os.makedirs(episode_dm_dir, exist_ok=True)
        parquet_path = os.path.join(self.data_dir, f'{self.current_episode_index:03d}.parquet')

        # 避免覆盖现有文件
        if os.path.exists(parquet_path):
            print(f"Episode {self.current_episode_index} already exists. Aborting.")
            return False

        # 激活新episode
        self.active_session = True
        self.current_dm_dir = episode_dm_dir
        self.current_parquet_path = parquet_path
        self.current_frame_index = 0
        self.cached_markers = []
        print(f"New episode: {self.current_session_path}, episode {self.current_episode_index}")
        return True

    def close_episode(self):  # 重命名并修改原有close_session方法
        if self.active_session:
            if self.cached_markers:
                self.cached_markers[-1]['next_done'] = True
                self.flush_marker_data()
            print(f"Closed episode: {self.current_episode_index}")
        self.active_session = False
        self.current_dm_dir = None
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

def main(argv):
    # SAVE_VIDEO_FLAG = False
    # SAVE_DATA_FLAG = True
    # FIND_ROI = False
    GPU = True
    MASK_MARKERS_FLAG = True
    OUTPUT_DIR = '/home/lin/gelsight_data'
    DATA_SAVE_INTERVAL = 1
    # mmpp = 0.0634

    # 初始化ROS节点
    rospy.init_node('gelsight_publisher')
    # 创建ROS发布者
    raw_image_pub = rospy.Publisher('gelsight/raw_image', Image, queue_size=10)
    depthmap_pub = rospy.Publisher('gelsight/depthmap', Image, queue_size=10)
    markers_pub = rospy.Publisher('gelsight/markers', Float32MultiArray, queue_size=10)

    session_mgr = SessionManager(OUTPUT_DIR)
    dev = gsdevice.Camera("GelSight Mini")
    dev.connect()

    # script_path = os.path.abspath(__file__)  # /home/lin/.../examples/ros/DataNew_ros.py
    # # 计算模型文件绝对路径
    # examples_dir = os.path.dirname(os.path.dirname(script_path))  # 向上两级到examples目录
    # model_path = os.path.join(examples_dir, 'nnmini.pt')  # /home/lin/.../examples/nnmini.pt

    nn = gs3drecon.Reconstruction3D(dev)
    net_path = 'nnmini.pt'
    nn.load_nn(net_path, "cuda" if GPU else "cpu")

    last_key_time = 0
    key_delay = 0.3

    try:
        frame_count = 0
        start_time = time.time()
        last_report_time = start_time

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
            # 修改按键处理部分
            if current_time - last_key_time > key_delay:
                if key == ord('s'):
                    session_mgr.create_session()  # 创建新episode
                    last_key_time = current_time
                elif key == ord('c'):
                    session_mgr.close_episode()  # 关闭当前episode
                    last_key_time = current_time
                elif key == ord('q'):
                    break

            if session_mgr.active_session and (frame_count % DATA_SAVE_INTERVAL == 0):
                dm_filename = f"{session_mgr.current_episode_index:03d}_{session_mgr.current_frame_index:06d}.npy"
                dm_path = os.path.join(session_mgr.current_dm_dir, dm_filename)
                np.save(dm_path, dm)

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
                    # 设置矩阵维度信息
                    markers_msg.layout.dim = [
                        MultiArrayDimension(label="dim0", size=7, stride=126),  # 7行，每行有9x2=18元素，7x18=126
                        MultiArrayDimension(label="dim1", size=9, stride=18),   # 9列，每列有2元素
                        MultiArrayDimension(label="dim2", size=2, stride=2)     # XY坐标
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
        session_mgr.close_session()
        total_elapsed = time.time() - start_time
        print(f"\nTotal frames: {frame_count}")
        print(f"Runtime: {total_elapsed:.2f}s")
        print(f"Average FPS: {frame_count / total_elapsed:.2f}")

        dev.stop_video()
        cv2.destroyAllWindows()
        print(f"Data root: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main(sys.argv[1:])