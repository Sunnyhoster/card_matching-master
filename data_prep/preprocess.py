import os
import random
import shutil

IMAGE_PATH = '/home/gaoxiao/Downloads/card_photos'
TRAIN_PATH = '/home/gaoxiao/code/card_matching/new_train_img'
TEST_PATH = '/home/gaoxiao/code/card_matching/new_test_img'

TEST_RATIO = 0.1


def list_files(dir):
  r = []
  subdirs = [x[0] for x in os.walk(dir)]
  print subdirs
  for subdir in subdirs:
    files = os.walk(subdir).next()[2]
    if (len(files) > 0):
      for file in files:
        r.append(subdir + "/" + file)
  return r


def main():
  all = list_files(IMAGE_PATH)
  print len(all)
  for f in all:
    file_name = os.path.basename(f)
    if file_name.endswith('DS_Store'):
      continue
    label = os.path.dirname(f)
    label = os.path.basename(label)
    new_file_name = '%s_%s' % (label, file_name)
    if random.random() > TEST_RATIO:
      dest = os.path.join(TRAIN_PATH, new_file_name)
    else:
      dest = os.path.join(TEST_PATH, new_file_name)
    shutil.copy(f, dest)


if __name__ == '__main__':
  main()
