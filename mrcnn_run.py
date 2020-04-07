# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:36:59 2020

@author: jiayi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:38:21 2020

@author: jiayi
"""

import os
import sys
import cv2
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
import colorsys
import argparse
#import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
from os import listdir
import random


ROOT_DIR = os.path.abspath(".")

sys.path.append(ROOT_DIR)
sys.path.insert(0, '/content/Mask_RCNN')
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap


import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
from mrcnn import model as modellib, utils

DATA_ROOT_PATH = "../aridDataset/arid_40k_scene_dataset"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
utils.download_trained_weights("./mask_rcnn_coco.h5")

train_file = open('../train.txt', "r")
train_img_paths = train_file.readlines()
val_file = open('../val.txt', "r")
val_img_paths = val_file.readlines()
test_file = open('../test.txt', "r")
test_img_paths = test_file.readlines()
class_file = open('../arid.names')
classes = list()
for c in class_file.readlines():
  classes.append(c.split('\n')[0])

def get_paths(souce_paths):
  paths = list()
  for f in souce_paths:
    image_id = f.split('/')[-1].split('.')[0]
    root_path = f.split('/rgb')[0]
    anno_path = os.path.join(root_path, image_id) + '.json'
    path = {"image_path": f.split('\n')[0], "anno_path": anno_path}
    paths.append(path)
  return paths

def get_class_name(o):
  if not o.split('_')[1].isdigit():
    class_name = o.split('_')[0] + "_" + o.split('_')[1]
  else:
    class_name = o.split('_')[0]
  return class_name

#train_paths = get_paths(train_img_paths)
#val_paths = get_paths(val_img_paths)
#test_paths = get_paths(test_img_paths)

class AridDataSet(utils.Dataset):
    def load_dataset(self, is_train=True):
        #index = 1
        #add all classes 
        #classes = get_object_classes()
        for c in classes:
            self.add_class("arid", classes.index(c)+1, c)
            #index += 1
        #image_paths, json_paths = get_paths()
        img_id = 0
        if is_train:
          paths = get_paths(train_img_paths)
        if not is_train:
          paths = get_paths(val_img_paths)
        for i in range(0,len(paths)):
          ann_path = paths[i]["anno_path"]
          self.add_image('arid', image_id=img_id, path=paths[i]["image_path"], annotation=ann_path)
          img_id += 1

    
    def extract_boxes(self, filename):
        boxes=list()
        classes = list()
        annotations = json.load(open(filename))
        #annotations=list(annotations.values())
        #annotations = annotations[0]
        for anno in annotations['annotations']:
          if anno['id'] is None:
            continue
          xmin = int(anno['x'])
          xmax = int(anno['x'] + anno['width'])
          ymin = int(anno['y'])
          ymax = int(anno['y'] + anno['height'])
          coors = [xmin, ymin, xmax, ymax]
          boxes.append(coors)
          classes.append(get_class_name(anno['id']))
                    
        width = 640
        height = 480
        
        return boxes, width, height, classes
        
    def load_mask(self, index):
        info = self.image_info[index]        
        anno_path = info['annotation']
        boxes,w,h, classes = self.extract_boxes(anno_path)
        masks = zeros([h,w,len(boxes)], dtype='uint8')
        class_ids = list()  
        for i in range(len(boxes)):
            obj = classes[i]
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(obj))
        #info = self.image_info[1]
        return masks, asarray(class_ids, dtype='int32')
        
    def image_reference(self, index):
        info = self.image_info[index]        
        return info['path']


# define a configuration for the model
class AridConfig(Config):
  # Give the configuration a recognizable name
  NAME = "arid_cfg"
  # Number of classes (background + kangaroo)
  NUM_CLASSES = len(classes) + 1
  # Number of training steps per epoch
  STEPS_PER_EPOCH = 506
  GPU_COUNT = 2
  IMAGES_PER_GPU = 1
  LEARNING_RATE = 0.0005        

 
train_set = AridDataSet()    
train_set.load_dataset(is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

test_set = AridDataSet()
test_set.load_dataset(is_train=False)
test_set.prepare()
print('Validation: %d' % len(test_set.image_ids))
  
config = AridConfig()
config.display()
model = MaskRCNN(mode='training', model_dir='./backup', config=config)
model.load_weights('./mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=100, layers='heads')

































