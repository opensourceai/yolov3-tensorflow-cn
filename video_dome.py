import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
import cv2
import argparse

IMAGE_H, IMAGE_W = 416, 416
parser = argparse.ArgumentParser(description="gpu模式下不能设置score_thresh和iou_thresh")
parser.add_argument("--video_id", "-vi", default=0, help="传入相机的id,可以是图片,视频,网络摄像头(eg:http://admin:admin@ip:端口/")
parser.add_argument("--model", "-m", default="cpu", choices=["cpu", "gpu"], help="选择gpu中运行还是在cpu中运行")
parser.add_argument("--score_thresh", "-st", default=0.5, type=float, help="设置score_thresh值,越高所获得的box越少(仅在cpu模式下生效)")
parser.add_argument("--iou_thresh", "-it", default=0.5, type=float, help="设置score_thresh值,越高所获得的box越少(仅在cpu模式下生效)")
flags = parser.parse_args()

classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
graph = tf.Graph()
if flags.model == "cpu":
    input_tensor, output_tensors = utils.read_pb_return_tensors(graph, "data/checkpoint/yolov3_cpu_nms.pb",
                                                                ["Placeholder:0", "concat_9:0", "mul_6:0"])
else:
    input_tensor, output_tensors = utils.read_pb_return_tensors(graph, "data/checkpoint/yolov3_gpu_nms.pb",
                                                                ["Placeholder:0", "concat_10:0", "concat_11:0",
                                                                 "concat_12:0"])

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(flags.video_id)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image!")
        img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
        img_resized = img_resized / 255.
        prev_time = time.time()

        # 从模型中获取结果
        if flags.model == "cpu":
            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

        else:
            boxes, scores, labels = sess.run(output_tensors,
                                             feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})

        # 在图中进行标记
        image = utils.draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)
        image = utils.draw_Chinese(image, "按q退出", (0, 35))
        image = utils.draw_Chinese(image, "按k截图", (0, 55))
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" % (1000 * exec_time)
        cv2.putText(result, text=info, org=(0, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)

        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)

        keyboard = cv2.waitKey(10)
        # 按"k"进行截图
        if keyboard & 0xFF == ord('k'):
            now = int(round(time.time() * 1000))
            now02 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))
            filename = "screenshot/frames_%s.jpg" % now02
            cv2.imwrite(filename, result)
        # 按"q"退出
        if keyboard & 0xFF == ord('q'): break
