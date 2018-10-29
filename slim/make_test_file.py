import os
import random
import sys
import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Make list file for test images')
parser.add_argument('-i', "--input", type=str, required=True, help="path to input dataset")
parser.add_argument('-o', "--output", type=str, nargs='?', default="list_val_new.txt", help="path of output file")
args = parser.parse_args()

dataset_path = args.input
output_path = args.output

def get_test_list(test_dataset_path, test_list):
    fd = open(test_list, 'w')    
    for img_name in os.listdir(test_dataset_path):
        img_path = os.path.join(test_dataset_path, img_name)
        try:
            image_data = cv2.imread(img_path)
            fd.write(img_name + '\n')
        except Exception as e:
            print e
            continue    
        
    fd.close()
    
get_test_list(dataset_path, output_path)