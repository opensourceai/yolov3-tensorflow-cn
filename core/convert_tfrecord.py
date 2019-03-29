import sys
import argparse
import numpy as np
import tensorflow as tf


# 将训练图片转换为tfrecord文件

def main(argv):
    parser = argparse.ArgumentParser()
    # 物体映射表 , 图片地址, boxe , class_id 文件
    parser.add_argument("--dataset_txt", default='../data/train_dome_data/new_test.txt')
    parser.add_argument("--tfrecord_path_prefix",
                        default='../data/train_dome_data/images')
    # default='./data/train_data/quick_train_data/tfrecords/quick_train_data')
    flags = parser.parse_args()

    dataset = {}
    with open(flags.dataset_txt, 'r') as f:
        for line in f.readlines():
            example = line.split(' ')
            image_path = example[0]
            boxes_num = len(example[1:]) // 5  # boxs数量
            boxes = np.zeros([boxes_num, 5], dtype=np.float32)
            for i in range(boxes_num):
                boxes[i] = example[1 + i * 5:6 + i * 5]
                # print(boxes[i])
            dataset[image_path] = boxes

    image_paths = list(dataset.keys())
    images_num = len(image_paths)
    print(">> Processing %d images" % images_num)

    tfrecord_file = flags.tfrecord_path_prefix + "_" + flags.dataset_txt.split("_")[-1].split(".")[0] + ".tfrecords"
    with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
        for i in range(images_num):
            image = tf.gfile.FastGFile(image_paths[i], 'rb').read()  # 读取除二进制文件
            boxes = dataset[image_paths[i]]  # 得到图片的boxes
            boxes = boxes.tostring()  # 转出string

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'boxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes])),
                }
            ))
            sys.stdout.write("\r>> %d / %d" % (i + 1, images_num))
            sys.stdout.flush()
            record_writer.write(example.SerializeToString())
        print(">> Saving %d images in %s" % (images_num, tfrecord_file))


if __name__ == "__main__": main(sys.argv[1:])
