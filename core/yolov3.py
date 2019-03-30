import tensorflow as tf

from core import common

slim = tf.contrib.slim


class darknet53(object):
    """用于执行特征提取的网络
    https://images2018.cnblogs.com/blog/606386/201803/606386-20180327004340505-1572852891.png
    """

    def __init__(self, inputs):
        self.outputs = self.forward(inputs)

    def _darknet53_block(self, inputs, filters):
        """
        implement residuals block in darknet53
        """
        shortcut = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs

    def forward(self, inputs):

        inputs = common._conv2d_fixed_padding(inputs, 32, 3, strides=1)
        inputs = common._conv2d_fixed_padding(inputs, 64, 3, strides=2)  # 208
        inputs = self._darknet53_block(inputs, 32)  #
        inputs = common._conv2d_fixed_padding(inputs, 128, 3, strides=2)  # 104

        for i in range(2):
            inputs = self._darknet53_block(inputs, 64)

        inputs = common._conv2d_fixed_padding(inputs, 256, 3, strides=2)  # 52

        for i in range(8):
            inputs = self._darknet53_block(inputs, 128)

        route_1 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 512, 3, strides=2)  # 26

        for i in range(8):
            inputs = self._darknet53_block(inputs, 256)

        route_2 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 1024, 3, strides=2)  # 13

        for i in range(4):
            inputs = self._darknet53_block(inputs, 512)

        return route_1, route_2, inputs


class yolov3(object):

    def __init__(self, num_classes, anchors,
                 batch_norm_decay=0.9, leaky_relu=0.1):
        '''
        :param num_classes: class
        :param anchors: number of anchors 列表
        :param batch_norm_decay:
        :param leaky_relu:
        '''
        # self._ANCHORS =
        #               [[10 ,13], [16 , 30], [33 , 23],
        #               [30 ,61], [62 , 45], [59 ,119],
        #               [116,90], [156,198], [373,326]]
        self._ANCHORS = anchors
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._LEAKY_RELU = leaky_relu
        self._NUM_CLASSES = num_classes
        self.feature_maps = []  # [[None, 13, 13, 255], [None, 26, 26, 255], [None, 52, 52, 255]]

    def _yolo_block(self, inputs, filters):
        # if stride > 1 , padding
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        route = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs

    # 目标识别的层, 转换到合适的深度,以满足不同class_num数据的分类
    def _detection_layer(self, inputs, anchors):
        num_anchors = len(anchors)
        feature_map = slim.conv2d(inputs, num_anchors * (5 + self._NUM_CLASSES), 1,
                                  stride=1, normalizer_fn=None,
                                  activation_fn=None,
                                  biases_initializer=tf.zeros_initializer())
        return feature_map

    # 讲网络计算的的缩放量和偏移量与anchors,网格位置结合,得到在原图中的绝对位置与大小
    def _reorg_layer(self, feature_map, anchors):
        # 将张量转换为适合的格式
        num_anchors = len(anchors)  # num_anchors=3
        grid_size = feature_map.shape.as_list()[1:3]  # 网格数
        # the downscale image in height and weight
        stride = tf.cast(self.img_size // grid_size, tf.float32)  # [h,w] -> [y,x] 平均每个网络多少个像素值
        # 讲anchors 与 目标信息拆开  (batch_size, cell, cell , anchor_num * (5 + class_num)) -->
        #                         (batch_size, cell, cell , anchor_num ,5 + class_num)
        feature_map = tf.reshape(feature_map,
                                 [-1, grid_size[0], grid_size[1], num_anchors, 5 + self._NUM_CLASSES])  # 特征图

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)  # 分离各个值,在最后一个维度进行

        box_centers = tf.nn.sigmoid(box_centers)  # 使得偏移量变为非负,且在0~1之间, 超过1之后,中心点就偏移到了其他的单元中

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)

        a, b = tf.meshgrid(grid_x, grid_y)  # 构建网格 https://blog.csdn.net/MOU_IT/article/details/82083984
        '''
        a=[0,5,10]
        b=[0,5,15,20,25]
        A,B=tf.meshgrid(a,b)
        with tf.Session() as sess:
          print (A.eval())
          print (B.eval())
         
        结果：
        [[ 0  5 10]
         [ 0  5 10]
         [ 0  5 10]
         [ 0  5 10]
         [ 0  5 10]]
        [[ 0  0  0]
         [ 5  5  5]
         [15 15 15]
         [20 20 20]
         [25 25 25]]
         '''
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)  # 组合生产每个单元格左上角的坐标, 排列组合
        '''
        [0,0]
        [0,1]
        [0,2]
        .....
        [1,0]
        [1,1]
        .....
        [12,12]
        '''
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])  # 回复成5x5x1x2 的张量

        x_y_offset = tf.cast(x_y_offset, tf.float32)

        box_centers = box_centers + x_y_offset  # 物体的中心坐标
        box_centers = box_centers * stride[::-1]  # 在原图的坐标位置,反归一化 [h,w] -> [y,x]

        # tf.exp(box_sizes) 避免缩放出现负数, box_size[13,13,3,2], anchor[3,2]
        box_sizes = tf.exp(box_sizes) * anchors  # anchors -> [w, h] 使用网络计算出的缩放量对anchors进行缩放
        boxes = tf.concat([box_centers, box_sizes], axis=-1)  # 计算除所有的方框在原图中的位置
        return x_y_offset, boxes, conf_logits, prob_logits

    @staticmethod  # 静态静态方法不睡和类和实例进行绑定
    def _upsample(inputs, out_shape):  # 上采样, 放大图片
        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))  # 使用最近邻改变图像大小
        inputs = tf.identity(inputs, name='upsampled')

        return inputs

    # @staticmethod
    # def _upsample(inputs, out_shape):
    # """
    # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT 5 optimization
    # """
    # new_height, new_width = out_shape[1], out_shape[2]
    # filters = 256 if (new_height == 26 and new_width==26) else 128
    # inputs = tf.layers.conv2d_transpose(inputs, filters, kernel_size=3, padding='same',
    # strides=(2,2), kernel_initializer=tf.ones_initializer())
    # return inputs

    # 前向传播,得到3个feature_map
    def forward(self, inputs, is_training=False, reuse=False):
        """
        Creates YOLO v3 model.

        :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
               Dimension batch_size may be undefined. The channel order is RGB.
        :param is_training: whether is training or not.
        :param reuse: whether or not the network and its variables should be reused.
        :return:
        """
        # it will be needed later on 他在稍后将被需要
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,  # https://www.cnblogs.com/hellcat/p/8058092.html
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # Set activation_fn and parameters for conv2d, batch_norm.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common._fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                # 给定list(slim.conv2d)中的值设置默认值(normlizer,biase.....)
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self._LEAKY_RELU)):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, inputs = darknet53(inputs).outputs  # 得到图片张量
                    # route_1 : 52x52x256
                    # route_2 : 26x26x512
                    # inputs  : 13x13x1024

                with tf.variable_scope('yolo-v3'):
                    # https://github.com/YunYang1994/tensorflow-yolov3/raw/master/docs/images/levio.jpeg
                    # https://images2018.cnblogs.com/blog/606386/201803/606386-20180327004340505-1572852891.png
                    # feature_map1 13x13x1024 --> 13x13x[3x(5+class_num)]
                    route, inputs = self._yolo_block(inputs, 512)
                    feature_map_1 = self._detection_layer(inputs, self._ANCHORS[6:9])
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    # feature_map2 26x26x512 --> 26x26x[3x(5+class_num)]
                    inputs = common._conv2d_fixed_padding(route, 256, 1)
                    upsample_size = route_2.get_shape().as_list()
                    #  52x52 --> 26x26
                    inputs = self._upsample(inputs, upsample_size)  # 通过直接放大进行上采样
                    inputs = tf.concat([inputs, route_2], axis=3)  # 在axis=3 进行连接,
                    route, inputs = self._yolo_block(inputs, 256)
                    feature_map_2 = self._detection_layer(inputs, self._ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    # feature_map3 52x52x256 --> 52x52x[3x(5+class_num)]
                    inputs = common._conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    # 26x26 --> 52x52
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_1], axis=3)
                    route, inputs = self._yolo_block(inputs, 128)
                    feature_map_3 = self._detection_layer(inputs, self._ANCHORS[0:3])
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def _reshape(self, x_y_offset, boxes, confs, probs):
        # 构成一个(batch_size, cell*cell*len(anchors) , boxes)
        grid_size = x_y_offset.shape.as_list()[:2]  # 网格数
        boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])  # 3个anchor
        confs = tf.reshape(confs, [-1, grid_size[0] * grid_size[1] * 3, 1])  # 3个anchor分别对应概率
        probs = tf.reshape(probs, [-1, grid_size[0] * grid_size[1] * 3, self._NUM_CLASSES])  # 类别概率
        # print(boxes, confs, probs)

        return boxes, confs, probs

    # 给出在原图的位置
    def predict(self, feature_maps):
        """
        Note: given by feature_maps, compute the receptive field
              由给出的feature map 计算
              and get boxes, confs and class_probs
        input_argument: feature_maps -> [None, 13, 13, 255],
                                        [None, 26, 26, 255],
                                        [None, 52, 52, 255],
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_map_anchors = [(feature_map_1, self._ANCHORS[6:9]),
                               (feature_map_2, self._ANCHORS[3:6]),
                               (feature_map_3, self._ANCHORS[0:3])]

        # boxe 的相对位置转换为绝对位置
        results = [self._reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]
        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            # *result =  x_y_offset, boxes, confs, probs
            boxes, conf_logits, prob_logits = self._reshape(*result)
            # --> (batch_size, cell*cell*anchor_num, boxes/conf/prob)

            confs = tf.sigmoid(conf_logits)  # 转化成概率
            probs = tf.sigmoid(prob_logits)  # 转化成概率,每种类和不在为0

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # 将3个feature_map中所有的信息,整合到一个张量
        # shape : [Batch_size,10647,4]  10647 = 13x13x3 + 26x26x3 + 52x52x3
        boxes = tf.concat(boxes_list, axis=1)  # [Batch_size,10647,4]
        confs = tf.concat(confs_list, axis=1)  # [Batch_size,10647,1]
        probs = tf.concat(probs_list, axis=1)  # [Batch_size,10647,class_num]

        # 坐标转化:中心坐标转化为 左上角作案表,右下角坐标 --> 方便计算矩形框
        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x0 = center_x - width / 2.
        y0 = center_y - height / 2.
        x1 = center_x + width / 2.
        y1 = center_y + height / 2.

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs

    def compute_loss(self, pred_feature_map, y_true, ignore_thresh=0.5, max_box_per_image=8):
        """
        :param pred_feature_map: list [feature_map_1,feature_map_2,feature_map3]
                feature_map_1[13,13,3,(5 + self._NUM_CLASSES)]
        :param y_true: list [y_true_13, y_true_26, y_true_52]
               y_true_13 [13,13,3,(5 + self._NUM_CLASSES)] 只有含有目标的网格中存在信息,其余均为0.
        :param ignore_thresh: 0.5
        :param max_box_per_image:
        :return:
        """
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        total_loss = 0.
        # total_loss, rec_50, rec_75,  avg_iou    = 0., 0., 0., 0.
        _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

        # 计算每个featurn_map的损失
        for i in range(len(pred_feature_map)):
            result = self.loss_layer(pred_feature_map[i], y_true[i], _ANCHORS[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def loss_layer(self, feature_map_i, y_true, anchors):
        # y_ture [13,13,3,5+class_id]
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]  # cellxcell
        grid_size_ = feature_map_i.shape.as_list()[1:3]

        # 本身具有[-1, grid_size_[0], grid_size_[1], 3, 5 + self._NUM_CLASSES]的shape,
        # 但在进过tf.py_func方法时丢失shape信息,使用reshape重新赋予shape
        y_true = tf.reshape(y_true, [-1, grid_size_[0], grid_size_[1], 3, 5 + self._NUM_CLASSES])

        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        # 进过self._reorg_layer后会boxe会被换成绝对位置, 会使用ratio进行换算到cellxcell上
        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self._reorg_layer(feature_map_i, anchors)

        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]  # 该feature_map下所有的目标,有目标的为1,无目标的为0

        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box, 该feature_map下所有检测目标的数量
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4],
                                           tf.cast(object_mask[..., 0], 'bool'))  # 获取有每个(3个)anchor的中心坐标,长宽

        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou 计算每个pre_boxe与所有true_boxe的交并比.
        # true:[V,2],[V,2]
        # pre : [13,13,3,2]
        # out_shape: [N, 13, 13, 3, V],
        iou = self._broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # iou_shape : [N,13,13,3,V] 每个单元下每个anchor与所有的true_boxes的交并比
        best_iou = tf.reduce_max(iou, axis=-1)  # 选择每个anchor中iou最大的那个.
        # out_shape : [N,13,13,3]

        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)  # 如果iou低于0.5将会丢弃此anchor\
        # out_shape : [N,13,13,3] 0,1张量

        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # out_shape: [N, 13, 13, 3, 1] 0,1张量

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]  # 坐标反归一化
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset  # 绝对(image_size * image_size)信息 转换为 单元(cellxcell)相对信息
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset  # 获取网络真实输出值

        # get_tw_th, numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2],
        true_tw_th = y_true[..., 2:4] / anchors  # 缩放量
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability 稳定训练, 为0时不对anchors进行缩放, 在模型输出值特别小是e^out_put为0
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        # 还原网络最原始的输出值(有正负的)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # 较小的面接的box有较大的权重
        # shape: [N, 13, 13, 3, 1]  2. - 面积   为1时表示保持原始权重
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        # shape: [N, 13, 13, 3, 1] 方框损失值, 中心坐标均方差损失 * mask[N, 13, 13, 3, 1]
        # 仅仅计算有目标单元的loss, 不计算那些错误预测的boxes, 在预测是首先会排除那些conf,iou底的单元
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N  # N:batch_size
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask  # 只要存在目标的boxe
        conf_neg_mask = (1 - object_mask) * ignore_mask  # 选择不存在目标,同时iou小于阈值(0.5),

        # 分离正样本和负样本
        # 正样本损失
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        # 处理后的负样本损失,只计算那些是单元格中没有目标,同时IOU小于0.5的单元,
        # 只惩罚IOU<0.5,而不惩罚IOU>0.5 的原因是可能该单元内是有目标的,仅仅只是目标中心点却没有落在该单元中.所以不计算该loss
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N  # 平均交叉熵,同时提高正确分类,压低错误分类

        # shape: [N, 13, 13, 3, 1], 分类loss
        # boject_mask 只看与anchors相匹配的anchors
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:],
                                                                           logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def _broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match 只关心大小的匹配
        '''
        # shape:
        # true_box_??: [V, 2] V:目标数量
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2] , 扩张维度方便进行维度广播
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2] V:该尺度下分feature_map 下所有的目标是目标数量
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] --> [N, 13, 13, 3, V, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2] 维度广播
        # 真boxe,左上角,右下角, 假boxe的左上角,右小角,
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / + 2.,  # 取最靠右的左上角
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,  # 取最靠左的右下角
                                    true_box_xy + true_box_wh / 2.)
        # tf.maximun 去除那些没有面积交叉的矩形框, 置0
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)  # 得到重合区域的长和宽

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # 重合部分面积
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]  # 预测区域面积
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]  # 真实区域面积
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

        return iou
