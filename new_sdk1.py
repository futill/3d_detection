import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import sys
import cvui
import socket
import struct
import threading
from pyorbbecsdk import Config, Pipeline, FrameSet, OBSensorType, OBFormat, VideoStreamProfile, OBError, OBAlignMode
from utils import frame_to_bgr_image

ESC_KEY = 27

# 全局变量
desk_x, desk_y = 0, 0
button_clicked = False
flag = False
flag1 = True
flag2 = 0
flag3 = True
depth_values = []
all_frame_results = []
average_depth_all = 0
object_counts = {}
results = None
depth_data = None
color_image1 = None
lock = threading.Lock()  # 线程锁

class TemporalFilter:
    """时间滤波器类"""
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        """处理帧"""
        if self.previous_frame is None:
            result = frame
        else:
            result = cv.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

def yolo_detect_desk(color_image, model):
    """YOLO 检测桌子"""
    results = model.predict(source=color_image)
    desk_x, desk_y = 0, 0
    if results:
        for r in results:
            for bbox in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                desk_x = (x1 + x2) // 2
                desk_y = (y1 + y2) // 2
    return desk_x, desk_y

def yolo_detect_objects(color_image, model, names):
    """YOLO 检测目标物体"""
    results = model.predict(source=color_image)
    object_counts = {}
    for r in results:
        for bbox, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if conf > 0.6:
                class_index = int(c.item())
                name = names[class_index]
                label = f"{name}"
                x1, y1, x2, y2 = map(int, bbox)
                object_center_x = (x1 + x2) // 2
                object_center_y = (y1 + y2) // 2
                object_counts[label] = object_counts.get(label, 0) + 1
    return object_counts

def depth_processing_thread(depth_data, fx, fy, cx, cy):
    """深度数据处理线程"""
    global average_depth_all, depth_values
    while True:
        if depth_data is not None:
            with lock:
                depth_data_cropped = depth_data  # 这里可以根据需要裁剪深度数据
                depth_mask = depth_data_cropped != 0
                valid_depth_pixels = depth_data_cropped[depth_mask]
                average_depth = np.mean(valid_depth_pixels) if valid_depth_pixels.size > 0 else 0
                depth_values.append(average_depth)
                average_depth_all = np.mean(depth_values) if depth_values else 0
        time.sleep(0.1)  # 控制线程频率

def yolo_detect_thread(color_image, model, names):
    """YOLO 检测线程"""
    global results, object_counts
    while True:
        if color_image is not None:
            with lock:
                object_counts = yolo_detect_objects(color_image, model, names)
        time.sleep(0.1)  # 控制线程频率

def main():
    global desk_x, desk_y, button_clicked, flag, flag1, flag2, flag3, depth_values, all_frame_results, average_depth_all, object_counts, results, depth_data, color_image1

    # 初始化模型
    model = YOLO('/home/jetson/ultralytics-main/black_desk_jet.pt', task='detect')
    model1 = YOLO('/home/jetson/ultralytics-main/objects6.pt', task='detect')
    names = ['CA001', 'CA002', 'CA003', 'CA004', 'CB001', 'CB002', 'CB003', 'CB004', 'CC001', 'CC002', 'CC003', 'CC004', 'CD001', 'CD002', 'CD003', 'CD004','W001','W002','W003','W004']
    fx, fy, cx, cy = 475.312, 475.312, 327.94, 240.372

    # 初始化管道
    pipeline = Pipeline()
    config = Config()
    try:
        color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(1920, 0, OBFormat.MJPG, 30)
        depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(640, 0, OBFormat.Y16, 30)
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        pipeline.start(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    # 初始化套接字
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("192.168.1.66", 6666))

    # 初始化 UI
    window = np.zeros((800, 1000, 3), np.uint8)
    window[:] = (49, 52, 49)
    cvui.init('CVUI Demo')

    # 启动深度数据处理线程
    depth_thread = threading.Thread(target=depth_processing_thread, args=(depth_data, fx, fy, cx, cy))
    depth_thread.daemon = True
    depth_thread.start()

    # 启动 YOLO 检测线程
    yolo_thread = threading.Thread(target=yolo_detect_thread, args=(color_image1, model1, names))
    yolo_thread.daemon = True
    yolo_thread.start()

    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            color_image1 = frame_to_bgr_image(color_frame)
            if color_image1 is None:
                print("Failed to convert frame to image")
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_frame.get_height(), depth_frame.get_width()))
            depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        except Exception as e:
            print(f"An error occurred: {e}")

        # UI 更新
        if cvui.button(window, 750, 600, 100, 50, "START"):
            button_clicked = True
            print('Button clicked!')

        cvui.imshow('CVUI Demo', window)
        cvui.update()
        window.fill(0)
        cvui.rect(window, 680, 40, 300, 420, 0xff0000)
        cvui.text(window, 700, 20, "Identification result output area", 0.5)

        if button_clicked:
            if flag3:
                cropped_frame = cv.cvtColor(color_image1, cv.COLOR_BGR2GRAY)
                cropped_frame = cv.cvtColor(cropped_frame, cv.COLOR_GRAY2BGR)
                desk_x, desk_y = yolo_detect_desk(cropped_frame, model)
                if desk_x != 0:
                    print("检测到桌子，结束内部循环。")
                    flag3 = False
                    flag = True

            if flag1:
                send_data_type_0(s, b"flycar6")
                flag1 = False

            if flag:
                with lock:
                    for label, count in object_counts.items():
                        cvui.text(window, 700, 60 + 20 * list(object_counts.keys()).index(label), f"Goal_ID={label};Num={count}", 0.5)

        if cv.waitKey(1) == ord("q"):
            break

    s.close()
    pipeline.stop()

if __name__ == "__main__":
    main()