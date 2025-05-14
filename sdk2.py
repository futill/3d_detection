import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import sys
import cvui
import socket
import struct
from pyorbbecsdk import Config
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import VideoStreamProfile
from pyorbbecsdk import OBError
from pyorbbecsdk import *
from utils import frame_to_bgr_image
ESC_KEY = 27

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

def convert_to_grayscale(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # 将灰度图像转换为彩色图像
    color_frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)

    return color_frame

def yolo_detect_desk(color_image,model):
    desk_x=0
    desk_y=0
    results = model.predict(source=color_image)
    if len(results)  >0:
        for r in results:
            for bbox in r.boxes.xyxy:
                x1, y1, x2, y2 = bbox
                x_1, y_1=convert_resolution_1940x1080_to_640x480(x1, y1)
                x_2, y_2=convert_resolution_1940x1080_to_640x480(x2, y2)
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                desk_x =int((x1+x2)/2)
                desk_y =int((y1+y2)/2)
    return desk_x,desk_y

def create_socket():
    # 创建一个TCP套接字
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return s

def connect_to_referee_box(s):
    # 设置裁判盒软件的IP地址和端口号
    referee_box_ip = "192.168.1.66" 
    # 请替换为裁判盒软件的IP地址
    referee_box_port = 6666

    # 连接到裁判盒软件
    s.connect((referee_box_ip, referee_box_port))

def communicate_with_referee_box(s, message):
    try:
        # 发送消息
        s.sendall(message)
        
        print("成功发送")
        return 
    except Exception as e:
        print("发送失败:", str(e))
        return str(e)

def send_data_type_1(s,file_path):
    try:
        with open(file_path, "rb") as file:
            # 读取文件数据
            data = file.read()
            # 数据类型和数据长度
            data_type = 1  # 假设发送的是DataType为1的文件数据
            data_type_bytes = struct.pack(">i", data_type)
            data_length_bytes = struct.pack(">i", len(data))
            # 构造消息
            message = data_type_bytes + data_length_bytes + data
            # 发送数据
            communicate_with_referee_box(s, message)
            return 
    except Exception as e:
        return str(e)
    
def send_data_type_0(s,team_id):
    try:
        # 构造消息
        data_type = 0
        data_type_bytes = struct.pack(">i", data_type)
        data_length_bytes = struct.pack(">i", len(team_id))
        
        # 构造消息
        message = data_type_bytes + data_length_bytes + team_id
        
        # 发送数据
        communicate_with_referee_box(s, message)
        return
    except Exception as e:
        return str(e)
    
def send_data_type_3(s):
    try:
        # 构造消息
        team_id = b"0000" 
        data_type = 3
        data_type_bytes = struct.pack(">i", data_type)
        data_length_bytes = struct.pack(">i", len(team_id))
        
        # 构造消息
        message = data_type_bytes + data_length_bytes + team_id
        
        # 发送数据
        communicate_with_referee_box(s, message)
        return
    except Exception as e:
        return str(e)



def pixel_to_camera_coordinates(u, v, depth, fx, fy, cx, cy):
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return X, Y, Z

def read_boxes_from_txt(txt_file):
    boxes = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            coords = line.strip().split()  #根据空格分隔每行的坐标
            box = []
            for coord_pair in coords:
                x, y = map(int, coord_pair.strip().split(','))  #根据中文逗号分隔坐标对
                box.append((x, y))
            boxes.append(box)
    return boxes

def depth_stream(x, y,depth_data,fx, fy, cx, cy,average_depth_all):

    temporal_filter = TemporalFilter(alpha=0.5)

    while True:
        try:

            depth_data = temporal_filter.process(depth_data)

            
            depth = average_depth_all

            print(depth,x,y)
            x, y, z = pixel_to_camera_coordinates(x, y, depth, fx, fy, cx, cy)

            break
        except KeyboardInterrupt:
            break
    return x, y, z

def convert_resolution_1940x1080_to_640x480(x, y):
    # 计算水平方向上的缩放比例
    scale_x = 640 / 1940
    # 计算垂直方向上的缩放比例
    scale_y = 480 / 1080
    # 计算新坐标
    x_new = x * scale_x
    y_new = y * scale_y
    return x_new, y_new

def main():
    flag = False
    flag1 = True
    flag2 = 0
    y=0
    i=0
    flag3 = True
    depth_values = []
    all_frame_results = []
    average_depth_all = 0
    window = np.zeros((800, 1000, 3), np.uint8)
    window[:] = (49, 52, 49)  # 背景颜色
    start_image=cv.imread(r'/home/jetson/ultralytics-main/jt.jpg')
    button_clicked = False
    initial_delay = 5
    jpg=0
    desk_x=0
    desk_y=0
    desk_time=0
    total_item_counts = {}
    model0 = YOLO('/home/jetson/ultralytics-main/black_desk_jet.pt', task='detect')
    model2 = YOLO('/home/jetson/ultralytics-main/jet.pt', task='detect')
    model1 = YOLO('/home/jetson/ultralytics-main/objects6.pt', task='detect')
    names= ['CA001', 'CA002', 'CA003', 'CA004', 'CB001', 'CB002', 'CB003', 'CB004', 'CC001', 'CC002', 'CC003', 'CC004', 'CD001', 'CD002', 'CD003', 'CD004','W001','W002','W003','W004']
    fx = 475.312
    fy = 475.312
    cx = 327.94
    cy = 240.372

    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    config = Config()    

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1920,0 ,OBFormat.MJPG, 30)
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
        assert depth_profile is not None
        print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                   color_profile.get_height(),
                                                   color_profile.get_fps(),
                                                   color_profile.get_format()))
        print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                   depth_profile.get_height(),
                                                   depth_profile.get_fps(),
                                                   depth_profile.get_format()))
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return
    
    if device_pid == 0x066B:
        # Femto Mega不支持硬件D2C，将其更改为软件D2C
        config.set_align_mode(OBAlignMode.SW_MODE)
    else:
        config.set_align_mode(OBAlignMode.HW_MODE)

    try:
        # 启动管道
        pipeline.start(config)
    except Exception as e:
        print(e)
        return

    s = create_socket()
    connect_to_referee_box(s)

    cvui.init('CVUI Demo')


 

    time.sleep(initial_delay)
    
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)  
            frames = pipeline.wait_for_frames(100)  # 等待帧，超时时间为100ms
            if frames is None:
                continue 
            # 获取彩色帧
            color_frame = frames.get_color_frame() 
            if color_frame is None:
                continue
            color_image1 = frame_to_bgr_image(color_frame)
            if color_image1 is None:
                print("failed to convert frame to image")
                continue
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()  # 获取深度帧
            if depth_frame is None:
                continue
            width = depth_frame.get_width()  # 获取帧的宽度
            height = depth_frame.get_height()  # 获取帧的高度
            scale = depth_frame.get_depth_scale()  # 获取深度缩放因子

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_image = cv.normalize(depth_data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            depth_image = cv.applyColorMap(depth_image, cv.COLORMAP_JET)
        except Exception as e:
                print("An error occurred:", e)
  
        if cvui.button(window, 750, 600, 100, 50, "START"):
            button_clicked = True
            print('Button clicked!')

            
        cvui.imshow('CVUI Demo', window)
    
        if jpg==1:
            time.sleep(5)
        window.fill(0)
            
        cvui.update()
        cvui.rect(window, 680, 40, 300, 420, 0xff0000)  
        cvui.text(window, 700, 20, "Identification result output area", 0.5)
        if button_clicked:

            if flag3:
                jpg=0
                desk_time +=1
                cropped_frame = convert_to_grayscale(color_image1)
                if i !=2:
                    desk_x,desk_y= yolo_detect_desk(cropped_frame,model0)
                if i ==2:
                    desk_x,desk_y= yolo_detect_desk(color_image1,model2)
                if desk_time==7:
                    desk_x=960
                    desk_y=530                    
                if desk_x !=0 :
                    desk_time=0
                    print("检测到桌子，结束内部循环。")
                    flag3 =  False
                    flag = True


        
            if flag:
                #cropped_frame = cv.resize(cropped_frame, (640, 480))
                #cvui.image(window, 0, 0, cropped_frame)
                results = model1.predict(source=color_image1)
                flag2+=1
                cvui.update()
                cvui.text(window, 680, 480, "predictoring", 1)

            if flag1:            
                team_id = b"flycar6"  # 示例数据
                send_data_type_0(s, team_id)
                flag1 = False
            
            if flag:
                for r in results:
                    object_counts = {}
                    for bbox, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        if conf > 0.6:
                            class_index = int(c.item())
                            name = names[class_index]
                            label = f"{name}"

                            x1, y1, x2, y2 = bbox
                    #x1=x1+desk_x1
                    #x2=x2+desk_x1
                    #y1=y1+desk_y1
                    #y2=y2+desk_y1
                            a1 = int(x1)
                            a2 = int(x2)
                            b1 = int(y1)
                            b2 = int(y2) 
                            print(a1, b1, a2, b2)
                            x_1, y_1=convert_resolution_1940x1080_to_640x480(x1, y1)
                            x_2, y_2=convert_resolution_1940x1080_to_640x480(x2, y2)
                            x1 = int(x1)
                            x2 = int(x2)
                            y1 = int(y1)
                            y2 = int(y2)  

                            object_center_x =int((x1+x2)/2)
                            object_center_y =int((y1+y2)/2)

                            depth_data_cropped = depth_data[y1:y2, x1:x2]
                            depth_mask = depth_data_cropped != 0

                            valid_depth_pixels = depth_data_cropped[depth_mask]

                            if len(valid_depth_pixels) > 0:
                                average_depth = np.mean(valid_depth_pixels)
                            else:
                                average_depth = 0  # 如果没有有效像素，平均深度为0

                            depth_values.append(average_depth)

                            X, Y, Z = pixel_to_camera_coordinates(object_center_x, object_center_y, average_depth, fx, fy, cx, cy)
                            print("物品坐标:",X, Y, Z)

                            y_diff = y - Y
                            if y_diff > 0:
                                cv.rectangle(color_image1, (a1, b1), (a2, b2), (0, 0, 255), 2)

                                average_depth_all = np.mean(depth_values) if depth_values else 0

                                if label in object_counts:
                            
                                    object_counts[label] += 1
                                else:
                                   object_counts[label] = 1

                                all_frame_results.append(object_counts)

                                j = 0
                                for j, (object_id, count) in enumerate(object_counts.items()):
                                    cvui.text(window, 700, 60 + 20 * j, f"Goal_ID={object_id};Num={count}", 0.5)
                                    j+=1
                                
                                max_results = max(all_frame_results, key=lambda x: sum(x.values()))

                average_depth_all = np.mean(depth_values) if depth_values else 0
                print("桌子", i + 1, "所有物品的平均深度:", average_depth_all)

            print("所有桌子上的物品总数:")
            for object_id, count in total_item_counts.items():
                print(f"物品 {object_id}: {count}")
            
            if desk_x != 0:
                x, y, z = depth_stream(desk_x,desk_y,depth_data,fx, fy, cx, cy,average_depth_all)
                valueble_center = (x, y, z)  # 桌面中心坐标
                print("桌子中心坐标为:",valueble_center) 

            if i==2:# 将结果写入文件
                with open("result_r/detection_results1.txt", "w") as file:
                    file.write("START\n")
                    for object_id, count in total_item_counts.items():
                        result_line = f"Goal_ID={object_id};Num={count}\n"
                        file.write(result_line)
                    file.write("END\n")

        # 测试发送DataType1的数据包文件
        if flag2==10:
            if i!=2:
                for object_id, count in max_results.items():
                    if object_id in total_item_counts:
                        total_item_counts[object_id] += count
                    else:
                        total_item_counts[object_id] = count
                send_data_type_3(s)
                flag=0
                jpg=1
                window[600:600+start_image.shape[0], 500:500+start_image.shape[1]] = start_image
                print("识别暂停,请转动摄像头")
                flag2=0
                i+=1
                all_frame_results = []
            if flag2 ==0:
                flag3=True
        if flag2 ==14:
            for label, count in max_results.items():
                if label in total_item_counts:
                    total_item_counts[label] += count
                else:
                    total_item_counts[label] = count
        if flag2 ==15:
            if i==2:
                file_path = "result_r/detection_results1.txt"
                send_data_type_1(s,file_path)
                print("成功发送3")
                break
            
        color_image1 = cv.resize(color_image1, (640, 480))
        cvui.image(window, 0, 0, color_image1)

    # 检查按键，如果按下 q 键则退出循环
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    s.close()

if __name__ == "__main__":
    main()
