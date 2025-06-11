import os
import cv2
import numpy as np
from time import time
from datetime import datetime
from ais_bench.infer.interface import InferSession
from pyorbbecsdk import Config, OBSensorType, OBFormat, Pipeline
import tkinter as tk
from PIL import Image, ImageTk
import sys
import csv
import getpass

# 类别定义
CLASSES = [
    'CA001', 'CA002', 'CA003', 'CA004', 'CB001', 'CB002', 'CB003', 'CB004',
    'CC001', 'CC002', 'CC003', 'CC004', 'CD001', 'CD002', 'CD003', 'CD004',
    'W001', 'W002', 'W003', 'W004'
]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 全局变量
model = None
pipeline = None
running = False
root = None
canvas = None
photo = None
status_label = None
count_label = None
cumulative_text = None
last_boxes = []  # 存储最近几帧的检测框和深度
global_item_counts = {cls: 0 for cls in CLASSES}  # 全局累计统计
start_time = None  # 检测开始时间
DETECTION_TIME_LIMIT = 15  # 检测时间限制（秒）
ROUND = 1  # 比赛轮次（1或2）
UNIT_ABBR = "UNIT"  # 报名单位英文缩写
TEAM_ABBR = "TEAM"  # 队伍名英文缩写
MAX_FRAMES_MEMORY = 10  # 检测框记忆帧数

# 桌面路径
DESKTOP_PATH = os.path.expanduser(f"/home/{getpass.getuser()}/Desktop")
RESULT_DIR = os.path.join(DESKTOP_PATH, "result_r")
RESULT_FILE = os.path.join(RESULT_DIR, f"{UNIT_ABBR}-{TEAM_ABBR}-R{ROUND}.txt")

# 初始化CSV文件
def init_csv():
    with open('item_counts.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Class', 'Count'])

# 保存统计数据到CSV
def save_to_csv(new_counts):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('item_counts.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for cls, count in new_counts.items():
            if count > 0:
                writer.writerow([timestamp, cls, count])

# 保存识别结果到比赛格式文件
def save_results():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    with open(RESULT_FILE, 'w') as f:
        f.write("START\n")
        for cls, count in global_item_counts.items():
            if count > 0:
                f.write(f"Goal_ID={cls};Num={count}\n")
        f.write("END\n")
    status_label.config(text=f"检测已结束，结果已保存到 {RESULT_FILE}")

# 计算IoU
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

# 初始化模型和摄像头
def initialize():
    global model, pipeline, running, start_time
    # 加载模型
    try:
        model = InferSession(device_id=0, model_path="/home/HwHiAiUser/3d_detection/yolov8x_24_0307_5381_1280_huawei.om")
        status_label.config(text="模型加载成功")
    except Exception as e:
        status_label.config(text=f"模型加载失败: {e}")
        return False

    # 初始化摄像头
    config = Config()
    pipeline = Pipeline()
    try:
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        pipeline.start(config)
        status_label.config(text="摄像头初始化成功")
    except Exception as e:
        status_label.config(text=f"摄像头配置失败: {e}")
        return False

    running = True
    start_time = time()  # 记录检测开始时间
    status_label.config(text="识别进行中...")
    return True

# 预处理帧
def preprocess_frame(original_image):
    height, width, _ = original_image.shape
    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(640, 640), swapRB=True)
    return blob, scale

# 绘制边界框
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, depth=0):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if depth > 0:
        cv2.putText(img, f"Z: {depth:.1f}mm", (x, y_plus_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# 运行推理
def run_inference(blob):
    begin_time = time()
    outputs = model.infer(feeds=[blob], mode="static")
    end_time = time()
    print("模型推理时间:", end_time - begin_time)
    return outputs

# 计算框内深度均值
def compute_box_depth(depth_image, box):
    x1, y1, w, h = [int(v) for v in box]
    x2, y2 = x1 + w, y1 + h
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(depth_image.shape[1], x2)
    y2 = min(depth_image.shape[0], y2)
    if x2 > x1 and y2 > y1:
        box_depth = depth_image[y1:y2, x1:x2]
        valid_depth = box_depth[box_depth > 0]
        return np.mean(valid_depth) if len(valid_depth) > 0 else 0
    return 0

# 后处理函数
def postprocess(original_image, outputs, scale, depth_image=None):
    global last_boxes, global_item_counts
    outputs = np.array([cv2.transpose(outputs[0][0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    depths = []
    current_counts = {cls: 0 for cls in CLASSES}  # 当前帧统计
    new_counts = {cls: 0 for cls in CLASSES}  # 新增物品统计

    # 提取检测结果
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        _, maxScore, _, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.4:
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
            depth = compute_box_depth(depth_image, [x1, y1, w, h]) if depth_image is not None else 0
            boxes.append([x1, y1, w, h])
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
            depths.append(depth)

    # 非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.6)

    # 去重和统计
    current_boxes = []
    iou_threshold = 0.5
    depth_threshold = 300  # 深度差异阈值（mm）
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        depth = depths[i]
        is_new = True

        # 与历史检测框比较
        for last_box, last_class_id, last_depth, last_score, counted in last_boxes:
            if class_id == last_class_id:
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
        draw_bounding_box(original_image, class_id, score, box[0], box[1], box[0] + box[2], box[1] + box[3], depth)

    # 更新历史检测框（限制记忆帧数）
    last_boxes.append(current_boxes)
    if len(last_boxes) > MAX_FRAMES_MEMORY:
        last_boxes.pop(0)

    # 更新GUI当前帧统计
    count_text = ", ".join(f"{cls}: {cnt}" for cls, cnt in current_counts.items() if cnt > 0)
    count_label.config(text=f"当前帧统计: {count_text if count_text else '无物品'}")
    print(f"检测框数量: {len(indices)}, 当前帧统计: {count_text if count_text else '无物品'}")

    # 更新GUI累计统计
    cumulative_text.delete(1.0, tk.END)
    cumulative_text.insert(tk.END, "累计统计:\n")
    cumulative_text.insert(tk.END, "\n".join(f"{cls}: {cnt}" for cls, cnt in global_item_counts.items() if cnt > 0) or "无物品")

    # 保存新增统计到CSV
    if any(new_counts[cls] > 0 for cls in CLASSES):
        save_to_csv(new_counts)

    return original_image

# 动态深度范围
def calculate_dynamic_depth_range(depth_image, percentile_low=10, percentile_high=90):
    depth_values = depth_image[depth_image > 0].flatten()
    if len(depth_values) == 0:
        return 500, 1500
    depth_min = np.percentile(depth_values, percentile_low)
    depth_max = np.percentile(depth_values, percentile_high)
    depth_min = max(100, depth_min - 100)
    depth_max = depth_max + 100
    return int(depth_min), int(depth_max)

# 更新视频帧
def update_frame():
    global running, photo, start_time
    if not running:
        return

    # 检查检测时间是否超过限制
    elapsed_time = time() - start_time
    if elapsed_time >= DETECTION_TIME_LIMIT:
        running = False
        save_results()
        return

    try:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            status_label.config(text="无法读取帧")
            root.after(100, update_frame)
            return

        color_frame = frames.get_color_frame()
        if color_frame is None:
            status_label.config(text="无法获取彩色帧")
            root.after(100, update_frame)
            return

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            status_label.config(text="无法获取深度帧")
            root.after(100, update_frame)
            return

        # 彩色帧转 BGR
        frame_data = color_frame.get_data()
        frame = np.asanyarray(frame_data).reshape(
            (color_frame.get_height(), color_frame.get_width(), 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 深度帧转 mm
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
        scale = depth_frame.get_depth_scale()
        depth_image = depth_image.astype(np.float32) * scale * 1000
        depth_image = np.where((depth_image > 20) & (depth_image < 10000), depth_image, 0).astype(np.uint16)

        # 动态深度范围
        depth_min, depth_max = calculate_dynamic_depth_range(depth_image)
        print(f"动态深度范围: {depth_min}mm - {depth_max}mm")

        # 预处理和推理
        blob, scale_factor = preprocess_frame(frame)
        outputs = run_inference(blob)

        # 后处理
        frame = postprocess(frame, outputs, scale_factor, depth_image=depth_image)

        # 调整帧大小以适应画布
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

    except Exception as e:
        status_label.config(text=f"处理错误: {e}")

    root.after(100, update_frame)

# 强制退出程序
def exit_program():
    global running, pipeline
    running = False
    if pipeline:
        pipeline.stop()
    cv2.destroyAllWindows()
    root.destroy()
    sys.exit()

# 创建GUI并自动启动
def create_gui():
    global root, canvas, status_label, count_label, cumulative_text, photo
    root = tk.Tk()
    root.title("目标检测系统")
    root.geometry("900x600")  # 调整窗口大小以容纳统计区域

    # 顶部框架：状态和当前帧统计
    top_frame = tk.Frame(root)
    top_frame.pack(pady=5)

    # 状态标签
    status_label = tk.Label(top_frame, text="初始化中...", font=("Arial", 12))
    status_label.pack()

    # 当前帧统计标签
    count_label = tk.Label(top_frame, text="当前帧统计: 无物品", font=("Arial", 12))
    count_label.pack(pady=5)

    # 主显示框架：画布和累计统计
    main_frame = tk.Frame(root)
    main_frame.pack()

    # 画布显示视频
    canvas = tk.Canvas(main_frame, width=640, height=480)
    canvas.pack(side=tk.LEFT)

    # 累计统计文本区域
    cumulative_text = tk.Text(main_frame, width=20, height=30, font=("Arial", 12))
    cumulative_text.pack(side=tk.LEFT, padx=10)
    cumulative_text.insert(tk.END, "累计统计:\n无物品")

    # 按钮框架
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    # 强制退出按钮
    tk.Button(button_frame, text="强制退出", command=exit_program, font=("Arial", 10)).pack(padx=5)

    # 窗口关闭事件
    root.protocol("WM_DELETE_WINDOW", exit_program)

    # 初始化CSV
    init_csv()

    # 自动初始化并开始识别
    if initialize():
        update_frame()

    root.mainloop()

if __name__ == "__main__":
    create_gui()