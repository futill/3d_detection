# <项目名称>
2025版先进视觉
<该项目是2025年中国机器人大赛先进视觉赛项存储库">  
---

## 📖 项目简介
- 部署平台：orangepiaipro 华为昇腾310芯片8T+16g版本、奥比中光astras深度相机
- 算法：yolov11，通过昇腾CANN软件栈的AI编程接口Ascend进行编写、调用npc加速。展

---

##  转模型
-参考https://blog.csdn.net/mao_hui_fei/article/details/139356518
-转换pt->onnx
yolo export model=/home/futill/...pt format=onnx dynamic=False opset=12
-转换onnx -> on
atc --framework=5 --model=yolov8x_24_0307_5381_1280.onnx  --input_format=NCHW  --input_shape="images:1,3,1280,1280" --output=yolov8x_24_0307_5381_1280_huawei --soc_version=Ascend310B4

---