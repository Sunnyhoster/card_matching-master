from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
# from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import pickle
import os
import tensorflow as tf
import numpy as np

model_dir = './TFmodels/inception-v3'

graph_created = False
tf_session = None


def create_graph():
  global graph_created, tf_session
  if graph_created:
    return
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'tensorflow_inception_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
  graph_created = True
  tf_session = tf.Session()


def run_img_inference(img_dir):
  image_data = tf.gfile.FastGFile(img_dir, 'rb').read()
  create_graph()
  # sess = tf.Session()
  softmax_tensor = tf_session.graph.get_tensor_by_name('softmax:0')
  predictions = tf_session.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
  # (1,1008)->(1008,)
  predictions = np.squeeze(predictions)
  return predictions


def FeatureExtract(imagePath):
  """
  image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = np.sqrt(gray / float(np.max(gray)))
  inputImg = cv2.resize(gray, (640, 640))

  feat, hog_img = feature.hog(inputImg, orientations=9, pixels_per_cell=(50, 50),
         cells_per_block=(8, 8) , visualize=True)#, transform_sqrt=True, block_norm="L1")
  return feat
  """
  feat = run_img_inference(imagePath)
  return feat


# ==== main ====
ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="Path to the train image set")
ap.add_argument("--test", required=True, help="Path to the test image set")
args = vars(ap.parse_args())

data = []
labels = []

savedFile = "./model_cnn2.pkl"

if not os.path.isfile(savedFile):
  print "Calculate features and save to file"
  for imagePath in paths.list_images(args["train"]):
    label = (imagePath.split("/")[-1]).split("_")[0]
    try:
      feat = FeatureExtract(imagePath).reshape(1, -1)
    except Exception as e:
      print 'Failed to process %s' % imagePath
      print e
      continue
    data.append(feat.flatten())
    labels.append(label)
  npData = np.array(data)
  with open(savedFile, 'w') as f:
    pca = PCA(n_components=100)
    npData = pca.fit_transform(npData)
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(npData, labels)
    pickle.dump([model, pca], f)
else:
  with open(savedFile) as f:
    print "Read features from file"
    buf = pickle.load(f)
    model = buf[0]
    pca = buf[1]

count = 0
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
  try:
    feat = FeatureExtract(imagePath).reshape(1, -1)
  except Exception as e:
    print 'Failed to process %s' % imagePath
    print e
    continue
  feat = pca.transform(feat)
  pred = model.predict(feat)[0]
  gt = (imagePath.split("/")[-1]).split("_")[0]
  if pred == gt:
    count += 1
  else:
    print i, pred, gt

print "accuracy: ", float(count) / (i + 1)
if tf_session:
  tf_session.close()
"""
    # visualize the HOG image
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

    # draw the prediction on the test image and display it
    cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (0, 255, 0), 3)
    cv2.imshow("Test Image #{}".format(i + 1), image)
    cv2.waitKey(0)
"""
