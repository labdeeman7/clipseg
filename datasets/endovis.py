
import torch
import numpy as np
import os
import random

from os.path import join, isdir, isfile, expanduser 
from PIL import Image

from torchvision import transforms
from torchvision.transforms.transforms import Resize

from torch.nn import functional as nnf
from general_utils import get_from_repository
import cv2

from skimage.draw import polygon2mask
import json



#😉 Assuming that I have a json file with all the information about the classes in each picture
# There is a reason why there is a find crop, and it is a different function, the authors are trying to ensure that the crop covers a lot of the wanted class. This is why segmentation is taken. I need to check the paper again. 
def random_crop_slices(origin_size, target_size):
    """Gets slices of a random crop. """
    assert origin_size[0] >= target_size[0] and origin_size[1] >= target_size[1], f'actual size: {origin_size}, target size: {target_size}'

    offset_y = torch.randint(0, origin_size[0] - target_size[0] + 1, (1,)).item()  # range: 0 <= value < high
    offset_x = torch.randint(0, origin_size[1] - target_size[1] + 1, (1,)).item()

    return slice(offset_y, offset_y + target_size[0]), slice(offset_x, offset_x + target_size[1])


def find_crop(seg, image_size, iterations=1000, min_frac=None, best_of=None):

    best_crops = []
    best_crop_not_ok = float('-inf'), None, None
    min_sum = 0

    seg = seg.astype('bool')
    
    if min_frac is not None:
        #min_sum = seg.sum() * min_frac
        min_sum = seg.shape[0] * seg.shape[1] * min_frac
    
    for iteration in range(iterations):
        sl_y, sl_x = random_crop_slices(seg.shape, image_size)
        seg_ = seg[sl_y, sl_x]
        sum_seg_ = seg_.sum()

        if sum_seg_ > min_sum:

            if best_of is None:
                return sl_y, sl_x, False
            else:
                best_crops += [(sum_seg_, sl_y, sl_x)]
                if len(best_crops) >= best_of:
                    best_crops.sort(key=lambda x:x[0], reverse=True)
                    sl_y, sl_x = best_crops[0][1:]
                    
                    return sl_y, sl_x, False

        else:
            if sum_seg_ > best_crop_not_ok[0]:
                best_crop_not_ok = sum_seg_, sl_y, sl_x
        
    else:
        # return best segmentation found
        return best_crop_not_ok[1:] + (best_crop_not_ok[0] <= min_sum,) 


class Endovis2017(object):

    def __init__(self, split, image_size=400, negative_prob=0, aug=None, aug_color=False, aug_crop=True,
                 min_size=0, remove_classes=None, with_visual=False, only_visual=False, mask=None, 
                 root_dir= "./datasets/Endovis2017/", mode="train"):
        super().__init__()

        self.info = json.load(open(join(root_dir, f'clipseg_{split}.json')))
        self.negative_prob = negative_prob
        self.image_size = image_size
        self.with_visual = with_visual #🙋‍♂️ I am guessing that with visual refers to using a visual prompt along with the text prompt. This is one-shot segmentation
        self.only_visual = only_visual #🙋‍♂️ only visual probably means, no text prompts?
        self.phrase_form = '{}'
        self.split = split
        self.mask = mask
        self.aug_crop = aug_crop
        self.root_dir = root_dir
        self.all_possible_instrument_classes = [
                    'background', 'bipolar_forceps', 'prograsp_forceps',
                    'large_needle_driver', 'vessel_sealer',
                    'grasping_retractor', 'monopolar_curved_scissors',
                    'other_medical_instruments'
        ]
        
        if aug_color:
            self.aug_color = transforms.Compose([
                transforms.ColorJitter(0.5, 0.5, 0.2, 0.05),
            ])
        else:
            self.aug_color = None

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std)

        self.sample_ids = [sample_ref["sample_id"] for sample_ref in self.info]

        self.seg_path = join(expanduser('~/datasets/PhraseCut/VGPhraseCut_v0/images/'))
        self.img_path = join(expanduser('~/datasets/PhraseCut/VGPhraseCut_v0/segmentation/'))
        self.mode = mode

    def __len__(self):
        return len(self.sample_ids)

    def load_sample(self, sample_i): #😉 load_samples.
                
        sample_ref_data = self.info[sample_i] #😉  loader get reference data?🙋‍♂️ Not sure what reference data is 

        phrases = sample_ref_data['phrases'] #😉 phrases.
        
        phrase = phrases[0]
        mask_path = sample_ref_data['mask_path']
        img_path = sample_ref_data['img_path']
        # img_path = img_path[1:]

        # print(self.root_dir)
        # print(img_path)
        # print(mask_path)
        
         
        seg =  np.array(cv2.imread(join(self.root_dir, mask_path), cv2.IMREAD_GRAYSCALE)) #😉 seg
        img = np.array(Image.open(join(self.root_dir, img_path))) #😉 img

        # print(img.shape)
        # print(seg.shape)

        min_shape = min(img.shape[:2]) #😉 min_shape. 

        if self.aug_crop: #😉 augment the crop, 
            sly, slx, exceed = find_crop(seg, (min_shape, min_shape), iterations=50, min_frac=0.05) #😉 get coordinates for the slicing of the crop
        else:
            sly, slx = slice(0, None), slice(0, None)
    
        seg = seg[sly, slx] #😉 segmentation
        img = img[sly, slx] #😉 image. 

        seg = seg.astype('uint8') 
        seg = torch.from_numpy(seg).view(1, 1, *seg.shape) #😉some resizing

        if img.ndim == 2:
            img = np.dstack([img] * 3)

        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() #😉 permute. 

        seg = nnf.interpolate(seg, (self.image_size, self.image_size), mode='nearest')[0,0] #😉 interpolate, segmentation to image size, ensure to use nearest. 
        img = nnf.interpolate(img, (self.image_size, self.image_size), mode='bilinear', align_corners=True)[0] #😉 img to bilinear,  

        # img = img.permute([2,0, 1])
        img = img / 255.0 #😉 divide image
        seg = seg / 255.0

        if self.aug_color is not None: #😉 augment if needed the color. 
            img = self.aug_color(img)

        # img = self.normalize(img) #😉 normlize #🛑 remeber to change this back

        return img, seg, phrase #😉 Image, segmentation, phrase, #😉 No support images done here or masking of the support images.   

    def __getitem__(self, i):
 
        sample_i= self.sample_ids[i]

        img, seg, phrase = self.load_sample(sample_i) #😉 Image, segmentation and phrase. 🙋‍♂️ I am not sure what sample_i and j stand for.
        # print(f"img.shape {img.shape}")
        # print(f"seg.shape {seg.shape}") 

        sample_ref_data = self.info[sample_i]
        sample_instrument_name = sample_ref_data["instrument_class_name"]
              

        if self.negative_prob > 0 and sample_instrument_name: #😉 Negative sampling enabled for instrument segmentation
            sample_instruments_in_image = sample_ref_data["instruments_in_image"] 
            instruments_not_in_image = [instrument for instrument in self.all_possible_instrument_classes if instrument not in sample_instruments_in_image ]       

            if torch.rand((1,)).item() < self.negative_prob: #😉 probability for negative probability.
                idx = torch.randint(0, len(instruments_not_in_image), (1,)).item()
                new_phrase = instruments_not_in_image[idx]
                phrase = new_phrase
                seg = torch.zeros_like(seg)

        if self.with_visual: 
            #😉 In the original paper, for every phrase, you can have visual prompts that corresponds with a phrase and the segmentation of the visual prompy
            #😉 Masks comes in here. "and" in the msk name acts as a delimiter. text for text only. text and separate, we have an input of text and a separate visual input.
            #😉 For the phrasecut.yaml, we have text_and_crop_blur_highlight352. So we have the text first. Text is always on its own. The bone of contention is whether we combine the mask of the visual prompt, with the visual prompt or not. If it is separate, we do not combine them if it is not separate we combine them. 
            #😉 These all affect the output. We do not mix text and visuals here. They can only be mixed in vector form. But we decide whether we apply segmentation in the beginning of the pipeline or inside the decoder here. if it is separate, it is inside the decoder, if it is not separate like the phrascut.yaml, then our dataset also produces a masked copy of the image. 
            #  text - only a text mask, we cannot do one-shot when mask is text. 
            #😉 separate - sepaerate the text and mask, as different support inputs. Do not blend. Useful for the original one-shot that was suggested. 
            #😉 there are others. crop, blur, highlight etc. 
            sample_ref_data = self.info[sample_i]
            # print(len(sample_ref_data["visual_prompt_img_path"]))
            idx = torch.randint(0, len(sample_ref_data["visual_prompt_img_path"]), (1,)).item()
            # print(idx)
            # print(sample_ref_data)
            # print(join(self.root_dir, sample_ref_data["visual_prompt_img_path"][idx]))

            img_s = np.array(cv2.imread(join(self.root_dir, sample_ref_data["visual_prompt_img_path"][idx]) ) ) #using 0 right now, because there is only one visual prompt.
            seg_s = np.array(cv2.imread(join(self.root_dir, sample_ref_data["visual_prompt_mask_path"][idx]), cv2.IMREAD_GRAYSCALE))

            img_s = torch.from_numpy(img_s).permute(2,0,1).float()
            seg_s = torch.from_numpy(seg_s)

            # print(f'img_s.shape{img_s.shape}')
            # print(f'seg_s.shape{seg_s.shape}')

            from datasets.utils import blend_image_segmentation #This is the one_shot segmentation. 

            if self.mask in {'separate', 'text_and_separate'}: #😉 return separated mask and shape. if only separate, and no text, do not return text. True for oneshot I gyess
                # assert img.shape[1:] == img_s.shape[1:] == seg_s.shape == seg.shape[1:]
                add_phrase = [phrase] if self.mask == 'text_and_separate' else []
                vis_s = add_phrase + [img_s, seg_s, True] #😉 if with_visual, phrase in samples_by_phrase and mask is either separate or text-and_separate,  vis_s is [phrase, img_s, seg_s, True] , img_s = support_img, seg_s = support_seg, 🙋‍♂️True not sure.   
            else:
                if self.mask.startswith('text_and_'):
                    mask_mode = self.mask[9:] #😉add phrase and remove the text from the mask.
                    label_add = [phrase]
                else:
                    mask_mode = self.mask
                    label_add = []

                masked_img_s = torch.from_numpy(blend_image_segmentation(img_s, seg_s, mode=mask_mode, image_size=self.image_size)[0])
                masked_img_s = masked_img_s/255
                # masked_img_s = self.normalize(masked_img_s)
                vis_s = label_add + [masked_img_s, True] #😉 if with_visual, phrase in samples_by_phrase and mask is not  separate or text-and_separate, vis_s is [phrase, masked_img_s, True], masked_img_s = a blend of the image and segmentation, 🙋‍♂️True not sure
                
           
        else:  #😉 No visual, just normal ref expression segmentation training. and not training on phrase cut+. We are interested in this.
            assert self.mask == 'text'  #😉 mask must be text. 
            vis_s = [phrase] #😉 if no visual prompt, just use the phrase, vis_s = [phrase]
        
        seg = seg.unsqueeze(0).float()

        data_x = (img,) + tuple(vis_s)

        return data_x, (seg, torch.zeros(0), i) #😉 So the output is data_x and (the segmentation, [], i)  data_x is data, and seg is segmentation. 
        #😉 data_x is made up of [img, vis_s]. vis_s changes depending on the condition.  
        #😉 if with_visual, phrase in samples_by_phrase and mask is either separate or text-and_separate,  vis_s is [phrase, img_s, seg_s, True] , img_s = Augmented_segmentation_image, seg_s = segmentation, 🙋‍♂️True not sure.   
        #😉 if with_visual, phrase in samples_by_phrase and mask is not  separate or text-and_separate, vis_s is [phrase, masked_img_s, True], masked_img_s = a blend of the image and segmentation, 🙋‍♂️True not sure
        #😉 if with_visual, unique_phrase, mask is separate or text-and_separate, vis-s = [phrase, zeros_img, zeros_w_h_img, False]
        #😉 if with_visual, unique_phrase, mask is text_and_, vis-s = [phrase, zeros_img, False]
        #😉 if with_visual, unique_phrase, mask is unknown, vis-s = [zeros_img, False]
        #😉 if no visual, vis_s = [phrase]    
