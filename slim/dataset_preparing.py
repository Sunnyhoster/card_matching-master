import os
import random
import sys
sys.path.insert(0, '../models/research/slim/')
from datasets import dataset_utils
import math
import os
import tensorflow as tf
import cv2

total = 0

#得到不重复的标签
def get_label_dict(dataset_path):
    label_dict = {}
    idx = 0
    fd = open("data/label_list_3600.txt", 'w')
    for label in os.listdir(dataset_path):
        if label == "._.DS_Store" or os.path.isfile(os.path.join(dataset_path, label)):
            continue
        if label[:6] == "Vocab_":
            label = label[6:]
        if label not in label_dict:
#         if not label_dict.has_key(label):
            label_dict[label] = idx
            idx += 1
            fd.write(label + '\n')
            print(label)
    fd.close()
    return label_dict
    
def generate_label_name(data_dir, output_path, label_dict):
    #data_dir = 'flower_photos/'
    #output_path = 'list.txt'
    fd = open(output_path, 'w')
    cls_list = os.listdir(data_dir)
    for cls in cls_list:
        if cls == "._.DS_Store" or os.path.isfile(os.path.join(data_dir, cls)):
            continue
        image_path = os.path.join(data_dir, cls)
        for image_name in os.listdir(image_path):
            print(os.path.join(image_path, image_name))
            #try:
            cv2.imread(os.path.join(image_path, image_name))
            #except Exception as e:
            #    continue
            label = cls
            if cls[:6] == "Vocab_":
                label = cls[6:]
            fd.write('{} {}\n'.format(os.path.join(cls, image_name), label_dict[label]))
            global total
            total += 1

    fd.close()

def dataset_split(list_path, train_list_path, val_list_path, label_dict):
    _NUM_VALIDATION = len(label_dict)
    _NUM_VALIDATION = int(0.1 * total)
    print(_NUM_VALIDATION)
    _RANDOM_SEED = 1
    #list_path = 'list.txt'
    #train_list_path = 'list_train.txt'
    #val_list_path = 'list_val.txt'
    fd = open(list_path)
    lines = fd.readlines()
    fd.close()
    random.seed(_RANDOM_SEED)
    random.shuffle(lines)
    fd = open(train_list_path, 'w')
    for line in lines[_NUM_VALIDATION:]:
        fd.write(line)
    fd.close()
    fd = open(val_list_path, 'w')
    for line in lines[:_NUM_VALIDATION]:
        fd.write(line.split(" ")[0] + "\n")
    fd.close()

def convert_dataset_to_record(list_path, data_dir, output_dir, _NUM_SHARDS=5):
    fd = open(list_path)
    lines = [line.split() for line in fd]
    fd.close()
    num_per_shard = int(math.ceil(len(lines) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_path = os.path.join(output_dir,
                    'data_{:02}-of-{:02}.tfrecord'.format(shard_id, _NUM_SHARDS))
                tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(
                        i + 1, len(lines), shard_id))
                    sys.stdout.flush()
                    try:
                        image_data = tf.gfile.FastGFile(os.path.join(data_dir, lines[i][0]), 'rb').read()
                        image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                    except Exception as e:
                        print(e)
                        continue
                    height, width = image.shape[0], image.shape[1]
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, int(lines[i][1]))
                    tfrecord_writer.write(example.SerializeToString())
                tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

#data_dir = "/home/lifeng/card_matching/full_img"
data_dir = "/home/yy/PreResearch/photos"
record_dir = "data/tf_record_3600"
label_dict = get_label_dict(data_dir)
print(len(label_dict))
generate_label_name(data_dir, "data/list_3600.txt", label_dict)
dataset_split("data/list_3600.txt", "data/list_train_3600.txt", "data/list_val_3600.txt", label_dict)

train_record_dir = os.path.join(record_dir, 'train')
# val_record_dir = os.path.join(record_dir, 'val')
os.system('mkdir -p ' + train_record_dir)
# os.system('mkdir -p ' + val_record_dir)
convert_dataset_to_record('data/list_train_3600.txt', data_dir, train_record_dir)
# convert_dataset_to_record('data/list_val_3600.txt', data_dir, val_record_dir)
