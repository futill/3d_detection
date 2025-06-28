#摄像头参数252行
#模型路径241行
#ip修改58行
#模型分辨率66行
#置信度356行
#队伍名称49行
#每轮识别时间47行


import os
import cv2
import numpy as np
from time import time, sleep
from datetime import datetime
from ais_bench.infer.interface import InferSession
from pyorbbecsdk import *
import tkinter as tk
from PIL import Image, ImageTk
import sys
import csv
import getpass
import socket
import struct
from collections import Counter
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLASSES = ['CA001', 'CA002','CB001', 'CB002', 'CC001', 'CC002', 'CD001', 'CD002','W001', 'W002']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

all_frame_counts = {cls: [] for cls in CLASSES}
model = None
pipeline = None
running = False
detecting = False
root = None
canvas = None
photo = None
status_label = None
count_label = None
cumulative_text = None
last_boxes = []
global_item_counts = {cls: 0 for cls in CLASSES}
start_time = 0
DETECTION_TIME_LIMIT = 15
ROUND = 1
TEAM_ID = "flycar"
MAX_FRAMES_MEMORY = 10
tcp_socket = None

detection_round = 0
max_detection_rounds = 3
round_results = []
final_results = {}

TCP_HOST = '192.168.137.100'
TCP_PORT = 6666

# Desktop path
DESKTOP_PATH = os.path.expanduser(f"/home/{getpass.getuser()}/Desktop")
RESULT_DIR = os.path.join(DESKTOP_PATH, "result_r")
RESULT_FILE = os.path.join(RESULT_DIR, f"{TEAM_ID}-R{ROUND}.txt")

imgz = 1280 #根据模型分辨率修改

# Initialize CSV file
def init_csv():
    with open('item_counts.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Class', 'Count'])

def end_detection_round():
    global detection_round, round_results, final_results, detecting
    detecting = False
    detection_round += 1
    logging.info(f"Ended detection round {detection_round}")
    status_label.config(text=f"Detection round {detection_round} completed")

    current_counts = {}
    for cls in CLASSES:
        if len(all_frame_counts[cls]) > 0:
            most_common = Counter(all_frame_counts[cls]).most_common(1)[0][0]
            current_counts[cls] = most_common
        else:
            current_counts[cls] = 0
    
    round_results.append(current_counts)
    logging.info(f"Round {detection_round} results: {current_counts}")

    if detection_round == 1:
        final_results.update(current_counts)
    else:
        for cls, count in current_counts.items():
            if count > 0:
                final_results[cls] = final_results.get(cls, 0) + count

    cumulative_text.delete(1.0, tk.END)
    cumulative_text.insert(tk.END, f"Round {detection_round} results:\n")
    for cls, count in current_counts.items():
        if count > 0:
            cumulative_text.insert(tk.END, f"{cls}: {count}\n")

    next_target()

    if detection_round < max_detection_rounds:
        status_label.config(text=f"Waiting 3 seconds before starting round {detection_round + 1}...")
        start_next_round()
    else:
        save_final_results(final_results)
        send_result_file()
        detecting = False
        status_label.config(text="Detection completed, results sent.")

def start_next_round():
    sleep(3)
    start_detection_round()

def reset_all_frame_counts():
    global all_frame_counts
    all_frame_counts = {cls: [] for cls in CLASSES}
    logging.info("Frame counts reset")

def start_detection_round():
    global detecting, start_time, detection_round
    reset_all_frame_counts()
    detecting = True
    start_time = time()
    logging.info(f"Starting detection round {detection_round + 1}")
    status_label.config(text=f"Starting detection round {detection_round + 1}...")

def save_final_results(results):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    with open(RESULT_FILE, 'w') as f:
        f.write("START\n")
        for cls, count in results.items():
            if count > 0:
                f.write(f"Goal_ID={cls};Num={count}\n")
        f.write("END\n")
    logging.info(f"Final results saved to {RESULT_FILE}")
    status_label.config(text=f"Final results saved to {RESULT_FILE}")

def save_to_csv(new_counts):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('item_counts.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for cls, count in new_counts.items():
            if count > 0:
                writer.writerow([timestamp, cls, count])

def send_tcp_data(data_type, data):
    global tcp_socket
    if tcp_socket is None:
        status_label.config(text="TCP connection not established")
        logging.error("TCP connection not established")
        return False
    try:
        tcp_socket.settimeout(5)  # Set 5-second timeout
        data_bytes = data.encode('utf-8') if isinstance(data, str) else data
        data_length = len(data_bytes)
        packet = struct.pack('>II', data_type, data_length) + data_bytes
        tcp_socket.sendall(packet)
        logging.info(f"Sent TCP data: type={data_type}, data={data}")
        return True
    except Exception as e:
        status_label.config(text=f"TCP send failed: {e}")
        logging.error(f"TCP send failed: {e}")
        return False

def send_team_id():
    return send_tcp_data(0, TEAM_ID)

def next_target():
    success = send_tcp_data(3, TEAM_ID)
    logging.info(f"Sent next_target signal: {'Success' if success else 'Failed'}")
    return success

def send_result_file():
    global tcp_socket
    try:
        with open(RESULT_FILE, 'r') as f:
            result_content = f.read()
        if send_tcp_data(1, result_content):
            status_label.config(text="Result file sent")
            logging.info("Result file sent")
        else:
            status_label.config(text="Failed to send result file")
            logging.error("Failed to send result file")
        if tcp_socket:
            tcp_socket.close()
            tcp_socket = None
    except Exception as e:
        status_label.config(text=f"Failed to send result file: {e}")
        logging.error(f"Failed to send result file: {e}")

def connect_to_server():
    global tcp_socket, detecting, start_time, detection_round
    try:
        if tcp_socket is not None:
            tcp_socket.close()
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(5)  # Set connection timeout
        tcp_socket.connect((TCP_HOST, TCP_PORT))
        status_label.config(text="Connected to referee server")
        logging.info("Connected to referee server")
        if send_team_id():
            detection_round = 0
            round_results.clear()
            final_results.clear()
            start_detection_round()
        else:
            status_label.config(text="Failed to send team ID")
            logging.error("Failed to send team ID")
            if tcp_socket:
                tcp_socket.close()
                tcp_socket = None
    except Exception as e:
        status_label.config(text=f"Failed to connect to referee server: {e}")
        logging.error(f"Failed to connect to referee server: {e}")
        if tcp_socket:
            tcp_socket.close()
            tcp_socket = None

def compute_iou(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def initialize():
    global model, pipeline
    try:
        model = InferSession(device_id=0, model_path="/home/HwHiAiUser/3d_detection/1280x1280.om")
        status_label.config(text="Model loaded successfully")
    except Exception as e:
        status_label.config(text=f"Model loading failed: {e}")
        return False

    config = Config()
    pipeline = Pipeline()
    device = pipeline.get_device()
    #device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)#自动白平
    #device.set_bool_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, True)#对比度
    #device.set_bool_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, True)#曝光
    #device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, True)#颜色矫正
    try:
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 10)
        config.enable_stream(color_profile)
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_video_stream_profile(1280, 1024, OBFormat.Y16, 10)
        config.enable_stream(depth_profile)
        status_label.config(text="Camera initialized successfully")
        logging.info("Camera initialized successfully")
    except Exception as e:
        print(e)
        return
    config.set_align_mode(OBAlignMode.HW_MODE)
    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print(e)
    try:
        pipeline.start(config)
        return True
    except Exception as e:
        print(e)
        return

def preprocess_frame(original_image):
    height, width, _ = original_image.shape
    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / imgz
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(imgz, imgz), swapRB=True)
    return blob, scale

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, depth=0):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if depth > 0:
        cv2.putText(img, f"Z: {depth:.1f}mm", (x, y_plus_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

def run_inference(blob):
    begin_time = time()
    outputs = model.infer(feeds=[blob], mode="static")
    end_time = time()
    #logging.info(f"Model inference time: {end_time - begin_time:.3f} seconds")
    return outputs

def compute_box_depth(depth_data, box):
    x1, y1, w, h = [int(v) for v in box]
    center_x = x1 + w // 2
    center_y = y1 + h // 2
    region_size = 5
    half_region = region_size // 2
    depth_values = []
    for dy in range(-half_region, half_region + 1):
        for dx in range(-half_region, half_region + 1):
            px = center_x + dx
            py = center_y + dy
            # 边界检查
            if 0 <= px < depth_data.shape[1] and 0 <= py < depth_data.shape[0]:
                depth = depth_data[py, px]
                if depth > 0 and np.isfinite(depth):
                    depth_values.append(depth)
    if depth_values:
        avg_depth = float(np.mean(depth_values))
        return avg_depth
    else:
        return 0

def pixel_to_camera_coordinates(u, v, depth):
    fx = 475.312
    fy = 475.312
    cx = 327.94
    cy = 240.372
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return X, Y, Z

# Postprocess function
def postprocess(original_image, outputs, scale, depth_image):
    global last_boxes, global_item_counts, all_frame_counts, table_depth
    if not detecting:
        return original_image

    outputs = np.array([cv2.transpose(outputs[0][0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    depths = []
    current_counts = {cls: 0 for cls in CLASSES}
    new_counts = {cls: 0 for cls in CLASSES}
    valid_depths = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        _, maxScore, _, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.5:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3]
            ]
            x1 = round(box[0] * scale)
            y1 = round(box[1] * scale)
            w = round(box[2] * scale)
            h = round(box[3] * scale)
            object_center_x = x1 + w // 2
            object_center_y = y1 + h // 2
            depth = compute_box_depth(depth_image, [x1, y1, w, h]) if depth_image is not None else 0
            X,Y,Z=pixel_to_camera_coordinates(object_center_x, object_center_y, depth)
            if depth > 0 and np.isfinite(depth):  # Only include valid depths
                valid_depths.append(Y)
            boxes.append([x1, y1, w, h])
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
            depths.append(Y)

    if valid_depths:
        table_depth = float(np.mean(valid_depths))  
    else:
        table_depth = None  

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.6)
    depth_threshold = 300
    current_boxes = []
    iou_threshold = 0.5
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        depth = depths[i]
        x1 = round(box[0] * scale)
        y1 = round(box[1] * scale)
        w = round(box[2] * scale)
        h = round(box[3] * scale)
        object_center_x = x1 + w // 2
        object_center_y = y1 + h // 2
        object_depth = compute_box_depth(depth_image, [x1, y1, w, h]) if depth_image is not None else 0
        X,Y,Z=pixel_to_camera_coordinates(object_center_x, object_center_y, object_depth)
        if int(Y)>int(table_depth+100):
            print(f"Object at ({x1}, {y1}) filtered out: depth {Y:.1f} mm, table_depth {table_depth:.1f} mm")
            continue
        if Y <=0:
            continue
        is_new = True
        for frame_boxes in last_boxes:
            for last_box, last_class_id, last_depth, last_score, counted in frame_boxes:
                iou = compute_iou(box, last_box)
                depth_diff = abs(depth - last_depth) if depth > 0 and last_depth > 0 else float('inf')
                if iou >= iou_threshold and depth_diff <= depth_threshold and not counted:
                    is_new = False
                    break
        if is_new:
            new_counts[CLASSES[class_id]] += 1
            global_item_counts[CLASSES[class_id]] += 1

        current_counts[CLASSES[class_id]] += 1
        current_boxes.append((box, class_id, depth, score, is_new))
        draw_bounding_box(original_image, class_id, score, box[0], box[1], box[0] + box[2], box[1] + box[3],Y)

    last_boxes.append(current_boxes)
    if len(last_boxes) > MAX_FRAMES_MEMORY:
        last_boxes.pop(0)

    for cls in CLASSES:
        all_frame_counts[cls].append(current_counts[cls])

    count_text = ", ".join(f"{cls}: {cnt}" for cls, cnt in current_counts.items() if cnt > 0)
    count_label.config(text=f"Current frame counts: {count_text if count_text else 'No items'}")

    cumulative_text.delete(1.0, tk.END)
    cumulative_text.insert(tk.END, f"Round {detection_round + 1} most frequent counts:\n")
    for cls in CLASSES:
        if len(all_frame_counts[cls]) > 0:
            most_common_count = Counter(all_frame_counts[cls]).most_common(1)[0][0]
            if most_common_count > 0:
                cumulative_text.insert(tk.END, f"{cls}: {most_common_count}\n")
    if all(not any(all_frame_counts[cls]) for cls in CLASSES):
        cumulative_text.insert(tk.END, "No items")

    if any(new_counts[cls] > 0 for cls in CLASSES):
        save_to_csv(new_counts)

    return original_image

def update_frame():
    global running, photo, start_time, detecting
    if not running:
        return
    if detecting:
        elapsed_time = time() - start_time
        if elapsed_time >= DETECTION_TIME_LIMIT:
            detecting = False
            end_detection_round()
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        status_label.config(text="Unable to read frame")
        logging.warning("Unable to read frame")
        root.after(100, update_frame)
        return
    color_frame = frames.get_color_frame()
    if color_frame is None:
        status_label.config(text="Unable to get color frame")
        logging.warning("Unable to get color frame")
        root.after(100, update_frame)
        return
    depth_frame = frames.get_depth_frame()
    frame_data = color_frame.get_data()
    frame = np.asanyarray(frame_data).reshape(
        (color_frame.get_height(), color_frame.get_width(), 3))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
    scale_depth = depth_frame.get_depth_scale()
    depth_data = depth_data.astype(np.float32) * scale_depth
    if detecting:
        blob, scale_factor = preprocess_frame(frame)
        outputs = run_inference(blob)
        frame = postprocess(frame, outputs, scale_factor, depth_data)
    else:
        status_label.config(text="Camera feed active, click Start to begin detection...")
    
    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo
    root.after(100, update_frame)

def exit_program():
    global running, pipeline, tcp_socket
    running = False
    if pipeline:
        pipeline.stop()
    if tcp_socket:
        tcp_socket.close()
        tcp_socket = None
    cv2.destroyAllWindows()
    root.destroy()
    sys.exit()

def create_gui():
    global root, canvas, status_label, count_label, cumulative_text, photo, running
    root = tk.Tk()
    root.title("3d-detection")
    root.geometry("900x600")

    top_frame = tk.Frame(root)
    top_frame.pack(pady=5)

    status_label = tk.Label(top_frame, text="Initializing...", font=("Arial", 12))
    status_label.pack()

    count_label = tk.Label(top_frame, text="Detect: none", font=("Arial", 12))
    count_label.pack(pady=5)

    main_frame = tk.Frame(root)
    main_frame.pack()

    canvas = tk.Canvas(main_frame, width=640, height=480)
    canvas.pack(side=tk.LEFT)

    cumulative_text = tk.Text(main_frame, width=20, height=30, font=("Arial", 12))
    cumulative_text.pack(side=tk.LEFT, padx=10)
    cumulative_text.insert(tk.END, "Detect:\nnone")

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Start", command=connect_to_server, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Exit", command=exit_program, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

    root.protocol("WM_DELETE_WINDOW", exit_program)

    init_csv()
    if initialize():
        running = True
        update_frame()
    else:
        status_label.config(text="Initialization failed")

    root.mainloop()

if __name__ == "__main__":
    create_gui()
