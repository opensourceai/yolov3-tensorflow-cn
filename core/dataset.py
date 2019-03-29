import cv2
import numpy as np
from core import utils
import tensorflow as tf


# 用于解析的类, 解析tfrecord文件,数据增强(对比度调整,翻转,剪裁....)
class Parser(object):
    def __init__(self, image_h, image_w, anchors, num_classes, debug=False):

        self.anchors = anchors
        self.num_classes = num_classes
        self.image_h = image_h
        self.image_w = image_w
        self.debug = debug

    def flip_left_right(self, image, gt_boxes):

        w = tf.cast(tf.shape(image)[1], tf.float32)  # 得到图像shape
        image = tf.image.flip_left_right(image)

        xmin, ymin, xmax, ymax, label = tf.unstack(gt_boxes, axis=1)
        xmin, ymin, xmax, ymax = w - xmax, ymin, w - xmin, ymax
        gt_boxes = tf.stack([xmin, ymin, xmax, ymax, label], axis=1)

        return image, gt_boxes

    def random_distort_color(self, image, gt_boxes):

        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        return image, gt_boxes

    def random_blur(self, image, gt_boxes):

        gaussian_blur = lambda image: cv2.GaussianBlur(image, (5, 5), 0)
        h, w = image.shape.as_list()[:2]
        image = tf.py_func(gaussian_blur, [image], tf.uint8)
        image.set_shape([h, w, 3])

        return image, gt_boxes

    def random_crop(self, image, gt_boxes, min_object_covered=0.8, aspect_ratio_range=[0.8, 1.2],
                    area_range=[0.5, 1.0]):

        #  h,w = tf.cast(tf.shape(image)[:2], tf.float32)
        h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
        xmin, ymin, xmax, ymax, label = tf.unstack(gt_boxes, axis=1)
        bboxes = tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=1)
        bboxes = tf.clip_by_value(bboxes, 0, 1)
        begin, size, dist_boxes = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, axis=0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range)
        # NOTE dist_boxes with shape: [ymin, xmin, ymax, xmax] and in values in range(0, 1)
        # Employ the bounding box to distort the image.
        croped_box = [dist_boxes[0, 0, 1] * w, dist_boxes[0, 0, 0] * h, dist_boxes[0, 0, 3] * w,
                      dist_boxes[0, 0, 2] * h]

        croped_xmin = tf.clip_by_value(xmin, croped_box[0], croped_box[2]) - croped_box[0]
        croped_ymin = tf.clip_by_value(ymin, croped_box[1], croped_box[3]) - croped_box[1]
        croped_xmax = tf.clip_by_value(xmax, croped_box[0], croped_box[2]) - croped_box[0]
        croped_ymax = tf.clip_by_value(ymax, croped_box[1], croped_box[3]) - croped_box[1]

        image = tf.slice(image, begin, size)
        gt_boxes = tf.stack([croped_xmin, croped_ymin, croped_xmax, croped_ymax, label], axis=1)

        return image, gt_boxes

    def preprocess(self, image, gt_boxes):
        image, gt_boxes = utils.resize_image_correct_bbox(image, gt_boxes, self.image_h, self.image_w)

        if self.debug: return image, gt_boxes

        y_true_13, y_true_26, y_true_52 = tf.py_func(self.preprocess_true_boxes, inp=[gt_boxes],
                                                     Tout=[tf.float32, tf.float32, tf.float32])
        image = image / 255.
        # image , cellxcellx3x(4+1+class_id)
        return image, y_true_13, y_true_26, y_true_52

    '''
    函数接受5维box(张量,左上角坐标,右下角坐标,class id). 一张图片中含有n个检测目标,则张量shape=(n,5)
    转化为中心坐标,
    计算该存在目标的box,与9个anchor的IOU(交并比)
    得到最大的IOU最大的anchor
    根据检测目标的5维信息,找到检测目标在那个feature_map, 属于那个格子, 那个anchor, class id
    '''

    def preprocess_true_boxes(self, gt_boxes):
        """
        将boxes处理成网络训练的格式
        Preprocess true boxes to training input format
        Parameters:
        -----------
        :param gt_boxes: numpy.ndarray of shape [T, 4]
                            T: the number of boxes in each image.每个图片含有的boxes数量
                            4: coordinate => x_min, y_min, x_max, y_max 坐标:左上角坐标,右下角坐标
        :param true_labels: class id
        :param input_shape: the shape of input image to the yolov3 network, [416, 416] 输入shape
        :param anchors: array, shape=[9,2], 9: the number of anchors, 2: width, height 数量x长与宽
        :param num_classes: integer, for coco dataset, it is 80 总共能够检测的class数量
        Returns:
        ----------
        y_true: list(3 array), shape like yolo_outputs, [13, 13, 3, 85]
                            13:cell szie, 3:number of anchors
                            85: box_centers, box_sizes, confidence, probability
                                中心坐标2,长宽2,存在目标的概率1,属于哪一类class的概率80
        """
        num_layers = len(self.anchors) // 3  # 每个特征图使用3个anchor, 得到feature_map的数量
        # anchor_mask：anchor box的索引数组，3个1组倒序排序，678对应13x13，345对应26x26，123对应52x52；
        # 即[[6, 7, 8], [3, 4, 5], [0, 1, 2]]； 如果只有2个feature_map
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

        # 最终的feature_map: 13x13，26x26，52x52；
        grid_sizes = [[self.image_h // x, self.image_w // x] for x in (32, 16, 8)]

        box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2  # the center of box boxes 获取box中心坐标
        box_sizes = gt_boxes[:, 2:4] - gt_boxes[:, 0:2]  # the height and width of box 得到box的长和宽

        # 将box坐标替换成中心点,长度和宽度
        gt_boxes[:, 0:2] = box_centers
        gt_boxes[:, 2:4] = box_sizes

        # 不同尺度下进行的标签的处理 cell x cell x anchors_number x (5(box_centers, box_sizes, confidence) + class_number)
        y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5 + self.num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5 + self.num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5 + self.num_classes], dtype=np.float32)

        y_true = [y_true_13, y_true_26, y_true_52]

        # 将anchors转换为中心坐标, 方便计算boxes和anchors的IOU
        anchors_max = self.anchors / 2.
        anchors_min = -anchors_max
        valid_mask = box_sizes[:, 0] > 0
        # 丢弃size为0boxes,即不存在的boxes
        gt_boxes = gt_boxes[valid_mask]
        wh = box_sizes[valid_mask]
        wh = np.expand_dims(wh, -2)  # 使得能够进行np.maximum(boxes_min, anchors_min)广播https://zhuanlan.zhihu.com/p/35010592
        boxes_max = wh / 2.
        boxes_min = -boxes_max
        # 计算交并比
        # https://zhuanlan.zhihu.com/p/51336725
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]

        anchor_area = self.anchors[:, 0] * self.anchors[:, 1]
        # 计算书交并比
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        iou = np.expand_dims(iou, -2)

        best_anchor = np.argmax(iou, axis=-1)  # 返回索引,(9个anchor中最大的) (n,9)

        # 最大 anchor在图中的坐标
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):  # 判断所存在feature_map中
                if n not in anchor_mask[l]: continue  # 不在该特征图的anchor

                # (box_x / image_w) * grid_sizes_x
                # i 和j 表示对应着feature map的元素来负责预测这个边框（这个边框的中心落在那）
                i = np.floor(gt_boxes[t, 0] / self.image_w * grid_sizes[l][1]).astype('int32')  # 向下取整
                j = np.floor(gt_boxes[t, 1] / self.image_h * grid_sizes[l][0]).astype('int32')  # 在格子中相对位置

                k = anchor_mask[l].index(n)  # 最佳anchor
                c = gt_boxes[t, 4].astype('int32')  # 得到class id

                y_true[l][j, i, k, 0:4] = gt_boxes[t, 0:4]  # box坐标,大小
                y_true[l][j, i, k, 4] = 1.  # 概率
                y_true[l][j, i, k, 5 + c] = 1.  # 类别

        return y_true_13, y_true_26, y_true_52

    def parser_example(self, serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], dtype=tf.string),
                'boxes': tf.FixedLenFeature([], dtype=tf.string),
            }
        )

        image = tf.image.decode_jpeg(features['image'], channels=3)  # 还原为图片
        image = tf.image.convert_image_dtype(image, tf.uint8)  # 还原为图片格式

        gt_boxes = tf.decode_raw(features['boxes'], tf.float32)
        gt_boxes = tf.reshape(gt_boxes, shape=[-1, 5])  # 5维 坐标, 长度, 属于3个anchor中的哪一个anchor

        return self.preprocess(image, gt_boxes)


class dataset(object):
    def __init__(self, parser, tfrecords_path, batch_size, shuffle=None, repeat=True):
        self.parser = parser
        self.filenames = tf.gfile.Glob(tfrecords_path)  # 张`正则路径下所有的文件
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self._buildup()

    def _buildup(self):
        try:
            self._TFRecordDataset = tf.data.TFRecordDataset(self.filenames)  # 读取tfrecord文件
        except:
            raise NotImplementedError("No tfrecords found!")

        self._TFRecordDataset = self._TFRecordDataset.map(map_func=self.parser.parser_example,  # 解析
                                                          num_parallel_calls=10)
        self._TFRecordDataset = self._TFRecordDataset.repeat() if self.repeat else self._TFRecordDataset

        if self.shuffle is not None:
            self._TFRecordDataset = self._TFRecordDataset.shuffle(self.shuffle)

        self._TFRecordDataset = self._TFRecordDataset.batch(self.batch_size).prefetch(self.batch_size)  # 用于缓存元素
        self._iterator = self._TFRecordDataset.make_one_shot_iterator()

    def get_next(self):
        return self._iterator.get_next()
