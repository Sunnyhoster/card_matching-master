from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'test_list', '', 'Test image list.')
tf.app.flags.DEFINE_string(
    'test_dir', '.', 'Test image directory.')
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Batch size.')
tf.app.flags.DEFINE_integer(
    'num_classes', 5, 'Number of classes.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')
tf.app.flags.DEFINE_string(
    'label_txt', 'label_list.txt', 'label_file')
FLAGS = tf.app.flags.FLAGS

def get_labels(label_txt):
    label = [line.rstrip('\n') for line in open(label_txt)]
    return label

def save_graph_to_file(sess, graph_file_name, output_tensor_name):
  output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph_def, [output_tensor_name])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return
    
def main(_):
    count = 0
    correct = 0
    label_list = get_labels(FLAGS.label_txt)
    if not FLAGS.test_list:
        raise ValueError('You must supply the test list with --test_list')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        # Select the model
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)
        # Select the preprocessing function
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        test_image_size = FLAGS.test_image_size or network_fn.default_image_size
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        batch_size = FLAGS.batch_size
        tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
        logits, _ = network_fn(tensor_input)
        logits = tf.nn.top_k(logits, 1, name="SpatialSqueeze")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        test_ids = [line.strip() for line in open(FLAGS.test_list)]
        tot = len(test_ids)
        results = list()
        with tf.Session(config=config) as sess:
            time_t = time.time()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
#             save_graph_to_file(sess, tf.Graph(), "mobilenet_v1_3600_imgs.pb", "MobilenetV1/Predictions/Reshape_1")
            #print("session init time: ", time.time() - time_t)
            for idx in range(0, tot, batch_size):
                images = list()
                idx_end = min(tot, idx + batch_size)
                #print(idx)
                labels = []
                img_path = []
                time_start = time.time()
                for i in range(idx, idx_end):
                    image_id = test_ids[i]
                    test_path = os.path.join(FLAGS.test_dir, image_id)
                    try:
#                         print(test_path)
                        image = open(test_path, 'rb').read()
                        image = tf.image.decode_jpeg(image, channels=3)
                        processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
                        processed_image = sess.run(processed_image)
                        images.append(processed_image)
                        gt = (image_id.split('/')[-2])
                        if gt[:6] == "Vocab_":
                            gt = gt[6:]
                        labels.append(gt)
                        img_path.append(image_id)
                    except Exception as e:
                        continue
                #print("batch prep time: ", time.time() - time_start)
                images = np.array(images)
                time_start = time.time()
                predictions = sess.run(logits, feed_dict = {tensor_input : images}).indices
                #print("batch running time: ", (time.time() - time_start))
                for i in range(len(predictions)):
                    try:
                        image = open(test_path, 'rb').read()
                        image = tf.image.decode_jpeg(image, channels=3)
                    except Exception as e:
                        continue
                    #print('{} {} {}'.format(img_path[i], label_list[predictions[i][0]], labels[i]))
                    count += 1
                    if label_list[predictions[i][0]] == labels[i]:
                        correct += 1
                    else:
                        print('{} {} {}'.format(img_path[i], label_list[predictions[i][0]], labels[i]))
            print("Accuracy: ", correct / count, " Corrct:", correct, " Total: ", count)
if __name__ == '__main__':
    tf.app.run()
