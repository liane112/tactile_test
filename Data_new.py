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


class SessionManager:
    def __init__(self, output_root):
        self.output_root = output_root
        self.active_session = False
        self.current_session_path = None
        self.current_episode_index = 0
        self.current_dm_dir = None
        self.current_parquet_path = None
        self.current_frame_index = 0
        self.cached_markers = []

    def create_session(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        main_dir = os.path.join(self.output_root, timestamp)
        data_dir = os.path.join(main_dir, 'data')
        dm_dir = os.path.join(main_dir, 'dm')

        try:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(dm_dir, exist_ok=True)
            existing_parquets = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
            self.current_episode_index = len(existing_parquets)
            episode_dm_dir = os.path.join(dm_dir, f'episode_{self.current_episode_index:03d}')
            os.makedirs(episode_dm_dir, exist_ok=True)
            parquet_path = os.path.join(data_dir, f'{self.current_episode_index:03d}.parquet')

            self.active_session = True
            self.current_session_path = main_dir
            self.current_dm_dir = episode_dm_dir
            self.current_parquet_path = parquet_path
            self.current_frame_index = 0
            self.cached_markers = []
            print(f"New session: {main_dir}, episode {self.current_episode_index}")
            return True
        except Exception as e:
            print(f"Session creation failed: {str(e)}")
            self.active_session = False
            return False

    def close_session(self):
        if self.active_session:
            if self.cached_markers:
                self.cached_markers[-1]['next_done'] = True
                self.flush_marker_data()
            print(f"Closed session: {self.current_session_path}")
        self.active_session = False
        self.current_session_path = None
        self.current_episode_index = 0
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
    SAVE_VIDEO_FLAG = False
    SAVE_DATA_FLAG = True
    FIND_ROI = False
    GPU = True
    MASK_MARKERS_FLAG = True
    OUTPUT_DIR = '/home/lin/gelsight_data'
    DATA_SAVE_INTERVAL = 1
    mmpp = 0.0634

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

        while dev.while_condition:
            frame = dev.get_image()
            if frame is None:
                continue

            dm = nn.get_depthmap(frame, MASK_MARKERS_FLAG)

            current_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            if current_time - last_key_time > key_delay:
                if key == ord('s'):
                    session_mgr.create_session()
                    last_key_time = current_time
                elif key == ord('c'):
                    session_mgr.close_session()
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