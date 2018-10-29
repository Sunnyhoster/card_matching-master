from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
#from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import pickle
import os
import numpy as np

def FeatureExtract(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.sqrt(gray / float(np.max(gray)))
    """
    edged = imutils.auto_canny(gray)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if len(cnts) == 0:
        return
    c = max(cnts, key=cv2.contourArea)
 
    (x, y, w, h) = cv2.boundingRect(c)
    inputImg = gray[y:y + h, x:x + w]
    inputImg = cv2.resize(inputImg, (800, 800)) # (200, 100)
    """
    inputImg = cv2.resize(gray, (640, 640))

    feat, hog_img = feature.hog(inputImg, orientations=9, pixels_per_cell=(50, 50),
           cells_per_block=(8, 8) , visualize=True)#, transform_sqrt=True, block_norm="L1")
    return feat

# ==== main ====
ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="Path to the train image set")
ap.add_argument("--test", required=True, help="Path to the test image set")
args = vars(ap.parse_args())
 
data = []
labels = []

savedFile = "./model.pkl"

if not os.path.isfile(savedFile):
    print "Calculate features and save to file"
    for imagePath in paths.list_images(args["train"]):
        label = (imagePath.split("/")[-1]).split("_")[0]
        feat = FeatureExtract(imagePath)
        data.append(feat)
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
    feat = FeatureExtract(imagePath).reshape(1, -1)
    feat = pca.transform(feat)
    pred = model.predict(feat)[0]
    gt = (imagePath.split("/")[-1]).split("_")[0]
    if pred == gt:
        count += 1
    else:
        print i, pred, gt
print "accuracy: ", float(count) / (i + 1)
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
