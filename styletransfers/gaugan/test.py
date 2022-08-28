"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np

from .model import Pix2PixModel
from .dataset import get_transform
from torchvision.transforms import ToPILImage
from PIL import Image
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def evaluate(labelmap, name, save_path, checkpoint_dir):
    opt = {
        'label_nc': 12, # num classes in coco model
        'crop_size': 256,
        'load_size': 256,
        'aspect_ratio': 1.0,
        'isTrain': False,
        'checkpoints_dir': checkpoint_dir,
        'which_epoch': 'latest',
        'use_gpu': False
    }
    model = Pix2PixModel(opt)
    model.eval()

    image = Image.fromarray(np.array(labelmap).astype(np.uint8))

    transform_label = get_transform(opt, method=Image.NEAREST, normalize=False)
    # transforms.ToTensor in transform_label rescales image from [0,255] to [0.0,1.0]
    # lets rescale it back to [0,255] to match our label ids
    label_tensor = transform_label(image) * 255.0
    label_tensor[label_tensor == 255] = opt['label_nc'] # 'unknown' is opt.label_nc
    #print("label_tensor:", label_tensor.shape)

    # not using encoder, so creating a blank image...
    transform_image = get_transform(opt)
    image_tensor = transform_image(Image.new('RGB', (500, 500)))

    data = {
        'label': label_tensor.unsqueeze(0),
        'instance': label_tensor.unsqueeze(0),
        'image': image_tensor.unsqueeze(0)
    }
    generated = model(data, mode='inference')
    #print("generated_image:", generated.shape)

    createFolder(save_path)
    generated_img = to_image(generated)
    generated_img.save(save_path+'/'+name+'.png')

    return generated_img, save_path+'/'+name+'.png'

def to_image(generated):
    to_img = ToPILImage()
    normalized_img = ((generated.reshape([3, 256, 256]) + 1) / 2.0) * 255.0
    return to_img(normalized_img.byte().cpu())