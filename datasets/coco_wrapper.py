import pickle
from types import new_class
import torch
import numpy as np
import os
import json

from os.path import join, dirname, isdir, isfile, expanduser, realpath, basename
from random import shuffle, seed as set_seed
from PIL import Image

from itertools import combinations
from torchvision import transforms
from torchvision.transforms.transforms import Resize

from datasets.utils import blend_image_segmentation
from general_utils import get_from_repository

COCO_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
#😉 Apparently there are 79 coco classes.  

class COCOWrapper(object): #😉 This is a wrapper object to the original code. 

    def __init__(self, split, fold=0, image_size=400, aug=None, mask='separate', negative_prob=0,
                 with_class_label=False): #😉 split(😉train val split), fold(🙋‍♂️ k-fold?), image_size, augmentation, mask separate, w🙋‍♂️hat does this mean, does it mean the mask is given?, negative_prob .🙋‍♂️ Not sure.   
        super().__init__()

        self.mask = mask #😉 mask is here. 
        self.with_class_label = with_class_label
        self.negative_prob = negative_prob

        from third_party.hsnet.data.coco import DatasetCOCO #😉From hsnet github has to do with oneshot segmentation.

        get_from_repository('COCO-20i', ['COCO-20i.tar']) #😉Get from the repository this dataset.

        foldpath = join(dirname(__file__), '../third_party/hsnet/data/splits/coco/%s/fold%d.pkl') #🙋‍♂️ Not sure.

        def build_img_metadata_classwise(self):
            with open(foldpath % (self.split, self.fold), 'rb') as f:
                img_metadata_classwise = pickle.load(f)
            return img_metadata_classwise


        DatasetCOCO.build_img_metadata_classwise = build_img_metadata_classwise
        # DatasetCOCO.read_mask = read_mask
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]) #😉 Resizing. 

        self.coco = DatasetCOCO(expanduser('~/datasets/COCO-20i/'), fold, transform, split, 1, False)
    
        self.all_classes = [self.coco.class_ids]
        self.coco.base_path = join(expanduser('~/datasets/COCO-20i'))

    def __len__(self):
        return len(self.coco) #😉 length of coc. 

    def __getitem__(self, i):
        sample = self.coco[i] #😉the image sample. 

        label_name = COCO_CLASSES[int(sample['class_id'])] #😉 THe label name

        img_s, seg_s = sample['support_imgs'][0], sample['support_masks'][0] #😉 the next few things are support images. the image is sample['query_image']

        if self.negative_prob > 0 and torch.rand(1).item() < self.negative_prob: #😉 for a very few of the images,, 
            new_class_id = sample['class_id']
            while new_class_id == sample['class_id']:
                sample2 = self.coco[torch.randint(0, len(self), (1,)).item()]
                new_class_id = sample2['class_id'] #🙋‍♂️ Change the class id?
            img_s = sample2['support_imgs'][0]  #🙋‍♂️change the img?
            seg_s = torch.zeros_like(seg_s)  #🙋‍♂️no segmentation? 👌This is like a negative sample. where the label_name is something, and it is not in this image 

        mask = self.mask  #🙋‍♂️masks. possiblilities are separate, text_label, text, text_and, else(😉text_and_separate)
        if mask == 'separate':
            supp = (img_s, seg_s)  #😉 So we have the support image and its classification. not joinEd
        elif mask == 'text_label':   #😉 No text label is sued as support. 
            # DEPRECATED
            supp = [int(sample['class_id'])]
        elif mask == 'text':  #😉For text, we just use the label name. 
            supp = [label_name]      
        else:
            if mask.startswith('text_and_'):  #😉
                mask = mask[9:]  #😉Change mask to the word after text_and.
                label_add = [label_name]  #😉 Just use the label name
            else:
                label_add = []  #😉 Any other thing use nothing. 

            supp = label_add + blend_image_segmentation(img_s, seg_s, mode=mask)  #😉Support for not text, or separate is the [label_name, segmentation_blended]

        if self.with_class_label:
            label = (torch.zeros(0), sample['class_id'],) #😉 [[], class_id]
        else:
            label = (torch.zeros(0), ) #[[]]

        return (sample['query_img'],) + tuple(supp), (sample['query_mask'].unsqueeze(0),) + label #😉 img, support(img, segmentation), img_segmentation, label_id.