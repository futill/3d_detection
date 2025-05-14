import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import threading
import cvui
import socket
import struct
from pyorbbecsdk import (Config, Pipeline, FrameSet, OBSensorType, OBFormat,
                        VideoStreamProfile, OBAlignMode, frame_to_bgr_image)

# ================== 全局配置 ==================
ESC_KEY = 27
MODEL_PATHS = {
    'desk': '/home/jetson/ultralytics-main/black_desk_jet.pt',
    'object': '/home/jetson/ultralytics-main/objects6.pt',
    'jet': '/home/jetson/ultralytics-main/jet.pt'
}
NAMES = ['CA001', 'CA002', 'CA003', 'CA004', 'CB001', 'CB002', 'CB003', 'CB004',
        'CC001', 'CC002', 'CC003', 'CC004', 'CD001', 'CD002', 'CD003', 'CD004',
        'W001','W002','W003','W004']
CAMERA_PARAMS = {
    'fx': 475.312,
    'fy': 475.312,
    'cx': 327.94,
    'cy': 240.372
}
SERVER_CONFIG = ('192.168.1.66', 6666)

# ================== 线程安全类 ==================
class ThreadSafeData:
    def __init__(self):
        self.lock = threading.Lock()
        self.desk_pos = (0, 0)
        self.depth_data = None
        self.object_counts = {}
        self.current_frame = None
        self.running = True

# ================== 核心功能类 ==================
class DepthProcessor:
    def __init__(self, alpha=0.5):
        self.temporal_filter = TemporalFilter(alpha)
    
    def process_depth(self, depth_frame):
        try:
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            return depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        except Exception as e:
            print(f"Depth processing error: {e}")
            return None

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.last_detection = time.time()
    
    def detect(self, frame):
        try:
            if time.time() - self.last_detection > 0.1:  # 控制检测频率
                results = self.model.predict(source=frame)
                self.last_detection = time.time()
                return results
            return None
        except Exception as e:
            print(f"Detection error: {e}")
            return None

# ================== 辅助函数 ==================
def convert_resolution(x, y, src_w=1940, src_h=1080, dst_w=640, dst_h=480):
    return int(x * dst_w / src_w), int(y * dst_h / src_h)

def create_ui_window():
    window = np.zeros((800, 1000, 3), np.uint8)
    window[:] = (49, 52, 49)
    cvui.init('CVUI Demo')
    return window

# ================== 网络通信模块 ==================
class RefereeCommunicator:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()
    
    def connect(self):
        try:
            self.socket.connect(SERVER_CONFIG)
            print("Connected to referee box")
        except Exception as e:
            print(f"Connection failed: {e}")
    
    def send_data(self, data_type, data):
        try:
            header = struct.pack(">ii", data_type, len(data))
            self.socket.sendall(header + data)
            print(f"Sent data type {data_type}")
        except Exception as e:
            print(f"Send error: {e}")
            self.reconnect()
    
    def reconnect(self):
        self.socket.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()

# ================== 主程序 ==================
class MainApplication:
    def __init__(self):
        self.shared_data = ThreadSafeData()
        self.communicator = RefereeCommunicator()
        self.ui_window = create_ui_window()
        self.depth_processor = DepthProcessor()
        self.detectors = {
            'desk': ObjectDetector(MODEL_PATHS['desk']),
            'object': ObjectDetector(MODEL_PATHS['object']),
            'jet': ObjectDetector(MODEL_PATHS['jet'])
        }
        self.pipeline = self.init_camera()
        self.running = True
    
    def init_camera(self):
        pipeline = Pipeline()
        config = Config()
        try:
            # 配置相机流参数
            color_profile = self.get_stream_profile(pipeline, OBSensorType.COLOR_SENSOR, 1920)
            depth_profile = self.get_stream_profile(pipeline, OBSensorType.DEPTH_SENSOR, 640)
            
            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)
            
            device = pipeline.get_device()
            if device.get_device_info().get_pid() == 0x066B:
                config.set_align_mode(OBAlignMode.SW_MODE)
            
            pipeline.start(config)
            return pipeline
        except Exception as e:
            print(f"Camera init failed: {e}")
            return None
    
    def get_stream_profile(self, pipeline, sensor_type, width):
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        return profile_list.get_video_stream_profile(width, 0, 
                                                   OBFormat.MJPG if sensor_type == OBSensorType.COLOR_SENSOR else OBFormat.Y16,
                                                   30)
    
    def process_frame(self, frames):
        try:
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                # 更新共享数据
                with self.shared_data.lock:
                    self.shared_data.current_frame = frame_to_bgr_image(color_frame)
                    self.shared_data.depth_data = self.depth_processor.process_depth(depth_frame)
                
                # 界面更新
                self.update_ui()
                
            return True
        except Exception as e:
            print(f"Frame processing error: {e}")
            return False
    
    def detection_thread(self):
        while self.shared_data.running:
            with self.shared_data.lock:
                frame = self.shared_data.current_frame
            
            if frame is not None:
                # 执行检测逻辑
                results = self.detectors['object'].detect(frame)
                if results:
                    # 处理检测结果...
                    pass
    
    def update_ui(self):
        # 界面更新逻辑
        if cvui.button(self.ui_window, 750, 600, 100, 50, "START"):
            # 处理按钮点击...
            pass
        
        cvui.imshow('CVUI Demo', self.ui_window)
        cvui.update()
    
    def run(self):
        try:
            # 启动检测线程
            threading.Thread(target=self.detection_thread, daemon=True).start()
            
            while self.running:
                frames = self.pipeline.wait_for_frames(100)
                if frames and self.process_frame(frames):
                    key = cv.waitKey(1)
                    if key == ord("q"):
                        break
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.shared_data.running = False
        if self.pipeline:
            self.pipeline.stop()
        self.communicator.socket.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    app = MainApplication()
    app.run()