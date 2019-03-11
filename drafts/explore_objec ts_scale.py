import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

datasets_root = '/home/xian/datasets'
dataset_name = 'VOC0712'

dataset_dir = os.path.join(datasets_root, dataset_name)
anns_dir = os.path.join(dataset_dir, 'annotations')
imgs_dir = os.path.join(dataset_dir, 'images')

train_names = os.path.join(dataset_dir, 'train_files.txt')
val_names = os.path.join(dataset_dir, 'val_files.txt')


def parse_annotations(imagefile, labelfile):
    image = cv2.imread(imagefile).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_height, img_width, _ = image.shape
    bboxes = []
    with open(labelfile, 'r') as fid:
        content = fid.read().splitlines()
        for line in content:
            line_split = line.split(' ')
            # Get coordinates from string:
            xmin = int(line_split[1])
            ymin = int(line_split[2])
            width = int(line_split[3])
            height = int(line_split[4])
            # Ensure coordinates fit in the image size:
            xmin = max(min(xmin, img_width - 2), 0)
            ymin = max(min(ymin, img_height - 2), 0)
            width = max(min(width, img_width - 1 - xmin), 1)
            height = max(min(height, img_height - 1 - ymin), 1)
            # Make relative coordinates:
            xmin = xmin / img_width
            ymin = ymin / img_height
            width = width / img_width
            height = height / img_height
            bboxes.append([xmin, ymin, width, height])
    return image, bboxes




bboxes = []
count = 0
with open(val_names, 'r') as fid:
    lines = fid.read().split('\n')
    lines = [line for line in lines if line != '']
    for line in lines:
        count += 1
        if count > 500:
            break
        imagefile = os.path.join(imgs_dir, line + '.jpg')
        labelfile = os.path.join(anns_dir, line + '.txt')
        _, img_bboxes = parse_annotations(imagefile, labelfile)
        bboxes.extend(img_bboxes)

nboxes = len(bboxes)
bboxes_array = np.zeros(shape=(nboxes, 4), dtype=np.float32)
for i in range(nboxes):
    bboxes_array[i, :] = bboxes[i]

bboxes_max_side = np.maximum(bboxes_array[:, 2], bboxes_array[:, 3])
bboxes_max_side.sort()

count = np.arange(nboxes) + 1
percent = count / float(nboxes)

plt.plot(bboxes_max_side, percent)
plt.xlabel('max obj side')
plt.ylabel('percent of boxes')
plt.show()


