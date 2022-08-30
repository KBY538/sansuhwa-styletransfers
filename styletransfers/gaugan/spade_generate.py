import cv2
import numpy as np
import os
from .test import evaluate
import glob

colors = {5: [250, 206, 135],  # 하늘
          2: [144, 238, 144],  # 앞산
          1: [170, 178, 32],  # 뒷산
          6: [0, 165, 255],  # 땅
          7: [153, 136, 119],  # 바위
          8: [128, 128, 240],  # 풀
          9: [225, 105, 65],  # 물
          4: [19, 69, 139],  # 가까운 나무
          3: [143, 143, 188],  # 먼 나무
          0: [0, 0, 0]  # None
          }

checkpoint_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

def distance(x1, x2):
    dist = 0
    for d in range(len(x1)):
        dist += (x1[d] - x2[d]) ** 2
    return dist

def label_maker(img):
    label = np.zeros((256, 256))
    img = cv2.resize(img, (256, 256), cv2.INTER_NEAREST)
    idx = 0

    values, _ = np.unique(img.reshape((-1, 3)), axis=0, return_counts=True)
    curr_colors = [c for c in values.tolist() if c in colors.values()]

    for i in range(256):
        for j in range(256):
            if list(img[i][j]) in curr_colors:
                idx = curr_colors.index(list(img[i][j]))
                val = [k for k, v in colors.items() if v == curr_colors[idx]][0]
            else:
                min_dist = float('inf')
                val = colors[0]
                for c in curr_colors:
                    dist = distance(img[i][j], c)
                    if min_dist > dist:
                        min_dist = dist
                        val = [k for k, v in colors.items() if v == curr_colors[idx]][0]
                if min_dist == float('inf'):
                    val = 0
            label[i][j] = val
    label = label.astype(int)
    return label

def generate_image(filepath, checkpoint_num, save_path="./result/", checkpoint_dir=checkpoint_dir):

    if type(filepath) == str:
        img = cv2.imread(filepath)
        name = os.path.basename(filepath).split('.')[0]+'_{}'.format(checkpoint_num)

    elif type(filepath) != np.ndarray:
        img = np.array(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        name = "templabel_{}".format(checkpoint_num)

    else:
        img = filepath
        name = "templabel_{}".format(checkpoint_num)

    label = label_maker(img)
    checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_num)
    return evaluate(label, name, save_path, checkpoint_dir)

def get_checkpoints(checkpoint_dir=checkpoint_dir):
    checkpoints  = glob.glob(checkpoint_dir+'/*')
    checkpoints.sort()
    return checkpoints