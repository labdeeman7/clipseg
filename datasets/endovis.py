
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

#😉 Assuming that I have a json file with all the information about the classes in each picture, 

class2sents = {
    'background': ['background', 'body tissues', 'organs'],
    'instrument': ['instrument', 'medical instrument', 'tool', 'medical tool'],
    'shaft': [
        'shaft', 'instrument shaft', 'tool shaft', 'instrument body',
        'tool body', 'instrument handle', 'tool handle'
    ],
    'wrist': [
        'wrist', 'instrument wrist', 'tool wrist', 'instrument neck',
        'tool neck', 'instrument hinge', 'tool hinge'
    ],
    'claspers': [
        'claspers', 'instrument claspers', 'tool claspers', 'instrument head',
        'tool head'
    ],
    'bipolar_forceps': ['bipolar forceps'],
    'prograsp_forceps': ['prograsp forceps'],
    'large_needle_driver': ['large needle driver', 'needle driver'],
    'vessel_sealer': ['vessel sealer'],
    'grasping_retractor': ['grasping retractor'],
    'monopolar_curved_scissors': ['monopolar curved scissors'],
    'other_medical_instruments': [
        'other instruments', 'other tools', 'other medical instruments',
        'other medical tools'
    ],
}

binary_factor = 255
parts_factor = 85
instruments_factor = 32

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
                 root_dir= "./datasets/endovis2017/cropped_train", segmentation="RES", method="oneshot"):
        super().__init__()

        self.image_size = image_size
        self.with_visual = with_visual
        self.only_visual = only_visual
        self.phrase_form = '{}'
        self.split = split
        self.mask = mask
        self.aug_crop = aug_crop
        self.root_dir = root_dir
        self.method = method  #segmentation can be refereing expression segmentation (RES), zero_shot(ZERO) and one_shot(ONE)
        self.segmentation  = segmentation
        self.img_paths = []
        self.seg_paths = [] #😉 I need to process the seg_paths such that all segmentation for an image can be used together.  
        self.ids = [] #😉 when shuffling, remember to shuffle with same seed for all img_paths, seg_paths and 
        
        if aug_color:
            self.aug_color = transforms.Compose([
                transforms.ColorJitter(0.5, 0.5, 0.2, 0.05),
            ])
        else:
            self.aug_color = None

        self.get_img_and_seg_paths()
        self.get_ids()

        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std)

        self.sample_ids = [(i, j) 
                           for i in self.img_paths 
                           for j in range(len(self.get_img_ref_data(i)['phrases']))]
        

        # self.all_phrases = list(set([p for i in self.refvg_loader.img_names for p in self.refvg_loader.get_img_ref_data(i)['phrases']]))

        from itertools import groupby
        samples_by_phrase = [(self.get_img_ref_data(i)['phrases'][j], (i, j)) 
                             for i, j in self.sample_ids]
        samples_by_phrase = sorted(samples_by_phrase)
        samples_by_phrase = groupby(samples_by_phrase, key=lambda x: x[0])
        
        self.samples_by_phrase = {prompt: [s[1] for s in prompt_sample_ids] for prompt, prompt_sample_ids in samples_by_phrase}

        self.all_phrases = list(set(self.samples_by_phrase.keys()))


        if self.only_visual:
            assert self.with_visual
            self.sample_ids = [(i, j) for i, j in self.sample_ids
                               if len(self.samples_by_phrase[self.refvg_loader.get_img_ref_data(i)['phrases'][j]]) > 1]

        # Filter by size (if min_size is set)
        sizes = [self.refvg_loader.get_img_ref_data(i)['gt_boxes'][j] for i, j in self.sample_ids]
        image_sizes = [self.refvg_loader.get_img_ref_data(i)['width'] * self.refvg_loader.get_img_ref_data(i)['height'] for i, j in self.sample_ids]
        #self.sizes = [sum([(s[2] - s[0]) * (s[3] - s[1]) for s in size]) for size in sizes]
        self.sizes = [sum([s[2] * s[3] for s in size]) / img_size for size, img_size in zip(sizes, image_sizes)]

        if min_size:
            print('filter by size')

        self.sample_ids = [self.sample_ids[i] for i in range(len(self.sample_ids)) if self.sizes[i] > min_size]

        self.seg_path = join(expanduser('~/datasets/PhraseCut/VGPhraseCut_v0/images/'))
        self.img_path = join(expanduser('~/datasets/PhraseCut/VGPhraseCut_v0/segmentation/'))

    def __len__(self):
        return len(self.sample_ids)

    def load_sample(self, sample_i, j): #😉 load_samples.
        print(sample_i)
        print(j)
        img_ref_data = self.refvg_loader.get_img_ref_data(sample_i) #😉  loader get reference data?🙋‍♂️ Not sure what reference data is 

        polys_phrase0 = img_ref_data['gt_Polygons'][j] #😉 polygon segmentation. 
        phrase = img_ref_data['phrases'][j] #😉 phrases. 
        phrase = self.phrase_form.format(phrase) #😉 Using a phrase form. 

        masks = [] #😉 masks, 
        for polys in polys_phrase0: 
            for poly in polys: #😉 for each polygon,
                poly = [p[::-1] for p in poly]  # swap x,y #😉 Reverse the direction. 
                masks += [polygon2mask((img_ref_data['height'], img_ref_data['width']), poly)] #😉 polygon to mask. 

        seg = np.stack(masks).max(0) #😉 seg, 
        img = np.array(Image.open(join(self.base_path, str(img_ref_data['image_id']) + '.jpg'))) #😉 img

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

        if self.aug_color is not None: #😉 augment if needed the color. 
            img = self.aug_color(img)

        img = self.normalize(img) #😉 normlize

        return img, seg, phrase #😉 Image, segmentation, phrase, #😉 No support images done here or masking of the support images.   

    def __getitem__(self, i):

        print(i)
 
        sample_i, j = self.sample_ids[i]

        img, seg, phrase = self.load_sample(sample_i, j) #😉 Image, segmentation and phrase. 🙋‍♂️ I am not sure what sample_i and j stand for. 

        if self.with_visual:
            # find a corresponding visual image
            if phrase in self.samples_by_phrase and len(self.samples_by_phrase[phrase]) > 1: #😉 samples_by_phrase gotten from the dataset.
                idx = torch.randint(0, len(self.samples_by_phrase[phrase]), (1,)).item()
                other_sample = self.samples_by_phrase[phrase][idx]
                #print(other_sample)
                img_s, seg_s, _ = self.load_sample(*other_sample)

                from datasets.utils import blend_image_segmentation

                if self.mask in {'separate', 'text_and_separate'}:
                    # assert img.shape[1:] == img_s.shape[1:] == seg_s.shape == seg.shape[1:]
                    add_phrase = [phrase] if self.mask == 'text_and_separate' else []
                    vis_s = add_phrase + [img_s, seg_s, True] #😉 if with_visual, phrase in samples_by_phrase and mask is either separate or text-and_separate,  vis_s is [phrase, img_s, seg_s, True] , img_s = Augmented_segmentation_image, seg_s = segmentation, 🙋‍♂️True not sure.   
                else:
                    if self.mask.startswith('text_and_'):
                        mask_mode = self.mask[9:]
                        label_add = [phrase]
                    else:
                        mask_mode = self.mask
                        label_add = []

                    masked_img_s = torch.from_numpy(blend_image_segmentation(img_s, seg_s, mode=mask_mode, image_size=self.image_size)[0])
                    vis_s = label_add + [masked_img_s, True] #😉 if with_visual, phrase in samples_by_phrase and mask is not  separate or text-and_separate, vis_s is [phrase, masked_img_s, True], masked_img_s = a blend of the image and segmentation, 🙋‍♂️True not sure
                
            else: #😉 Phrase is unique. it is not in the list of phrases. 
                # phrase is unique
                vis_s = torch.zeros_like(img)   

                if self.mask in {'separate', 'text_and_separate'}:
                    add_phrase = [phrase] if self.mask == 'text_and_separate' else []
                    vis_s = add_phrase + [vis_s, torch.zeros(*vis_s.shape[1:], dtype=torch.uint8), False] #😉 if with_visual, unique_phrase, mask is separate or text-and_separate, vis-s = [phrase, zeros_img, zeros_w_h_img, False]
                elif self.mask.startswith('text_and_'):
                    vis_s = [phrase, vis_s, False]   #😉 if with_visual, unique_phrase, mask is text_and_, vis-s = [phrase, zeros_img, False]
                else:
                    vis_s = [vis_s, False]  #😉 if with_visual, unique_phrase, mask is unknown, vis-s = [zeros_img, False]
        else:  #😉 No visual
            assert self.mask == 'text'  #😉 mask must be text. 
            vis_s = [phrase] #😉 if no visual, vis_s = [phrase]
        
        seg = seg.unsqueeze(0).float()

        data_x = (img,) + tuple(vis_s)

        return data_x, (seg, torch.zeros(0), i) #😉 So the output is data_x and (the segmentation, [], i) 
        #😉 data_x is made up of [img, vis_s]. vis_s changes depending on the condition.  
        #😉 if with_visual, phrase in samples_by_phrase and mask is either separate or text-and_separate,  vis_s is [phrase, img_s, seg_s, True] , img_s = Augmented_segmentation_image, seg_s = segmentation, 🙋‍♂️True not sure.   
        #😉 if with_visual, phrase in samples_by_phrase and mask is not  separate or text-and_separate, vis_s is [phrase, masked_img_s, True], masked_img_s = a blend of the image and segmentation, 🙋‍♂️True not sure
        #😉 if with_visual, unique_phrase, mask is separate or text-and_separate, vis-s = [phrase, zeros_img, zeros_w_h_img, False]
        #😉 if with_visual, unique_phrase, mask is text_and_, vis-s = [phrase, zeros_img, False]
        #😉 if with_visual, unique_phrase, mask is unknown, vis-s = [zeros_img, False]
        #😉 if no visual, vis_s = [phrase]    


    def get_img_ref_data(self, img_id): #😉 Similar to get everything about a single image. 
        """
        get a batch with one image and all refer data on that image
        """
        # Fetch feats according to the image_split_ix
        # wrapped = False #🙋‍♂️ Not sure what wrapped does.  But it does nto se
        max_index = len(self.ids) - 1

        # task_ids = []
        # p_structures = []
        phrases = [] #😉 phrases for each image are made. 
        seg_path = None

        # gt_Polygons = []
        # gt_boxes = []
        # img_ins_cats = []
        # img_ins_atts = []

        #😉 for each image, we have assigned phrases, task_ids, structures for the phrase(ategory, attributes, relation), instance_boxes(not used often), polygons(segmentation),    
        for task in self.ImgReferTasks[img_id]:
            phrases.append(task['phrase']) #😉 
            task_ids.append(task['task_id'])
            p_structures.append(task['phrase_structure'])
            if not self.input_anno_only:
                gt_boxes.append(task['instance_boxes'])
                gt_Polygons.append(task['Polygons'])
                img_ins_cats += [task['phrase_structure']['name']] * len(task['instance_boxes'])
                img_ins_atts += [task['phrase_structure']['attributes']] * len(task['instance_boxes'])

        # return data
        data = dict() #😉 we return data. So for endovis2018, we need to return all these values except
        #😉 add img_path and segmentation_path, remove everything that has to do with bounding boxes, categtories, and bounding boxes. not sure what p_structures is.
        data['image_id'] = img_id #😉 
        img_info = self.ImgInfo[img_id]
        data['width'] = img_info['width'] 
        data['height'] = img_info['height']
        data['split'] = img_info['split']

        data['task_ids'] = task_ids
        data['phrases'] = phrases #😉 phrases for an image are Stored here.
        data['p_structures'] = p_structures
        if not self.input_anno_only:
            data['img_ins_boxes'] = self.ImgInsBoxes[img_id]
            data['img_ins_Polygons'] = self.ImgInsPolygons[img_id]
            data['img_ins_cats'] = img_ins_cats
            data['img_ins_atts'] = img_ins_atts
            data['gt_Polygons'] = gt_Polygons
            data['gt_boxes'] = gt_boxes 

        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': max_index, 'wrapped': wrapped}

        return data 
        #😉 bounds, vg_obj_ids, vg_boxes, img_vg_boxes, gt_boxes, gt_Polygons, img_ins_atts
    def get_one_sample(self, image_file):
        if '.jpg' in image_file:
            suffix = '.jpg'
        elif '.png' in image_file:
            suffix = '.png'
        mask_path = os.path.join(
            self.save_dir,
            image_file.replace(suffix, '') + '_{}.png'.format(class_name))
        cv2.imwrite(mask_path, mask)
        cris_data = {
            'img_path': image_path.replace(root_dir, ''),
            'mask_path': mask_path.replace(root_dir, ''),
            'num_sents': len(class2sents[class_name]),
            'sents': class2sents[class_name],
        } 

    def get_img_information(self):
        cris_data_list = []
        if 'train' in root_dir:
            dataset_num = 8
        elif 'test' in root_dir:
            dataset_num = 10
        for i in range(1, dataset_num + 1):
            image_dir = os.path.join(root_dir, 'instrument_dataset_{}'.format(i),
                                    'images')
            print('process: {} ...'.format(image_dir))
            cris_masks_dir = os.path.join(root_dir,
                                        'instrument_dataset_{}'.format(i),
                                        'cris_masks')
            if not os.path.exists(cris_masks_dir):
                os.makedirs(cris_masks_dir)
            image_files = os.listdir(image_dir)
            image_files.sort()
            for image_file in image_files:
                print(image_file)
                image_path = os.path.join(image_dir, image_file)
                # binary
                binary_mask_file = image_path.replace('images',
                                                    'binary_masks').replace(
                                                        '.jpg', '.png')
                binary_mask = cv2.imread(binary_mask_file)
                binary_mask = (binary_mask / binary_factor).astype(np.uint8)
                for class_id, class_name in enumerate(['background',
                                                    'instrument']):
                    target_mask = (binary_mask == class_id) * 255
                    if target_mask.sum() != 0:
                        cris_data_list.append(
                            get_one_sample(root_dir, image_file, image_path,
                                        cris_masks_dir, target_mask,
                                        class_name))
                # parts
                parts_mask_file = image_path.replace('images',
                                                    'parts_masks').replace(
                                                        '.jpg', '.png')
                parts_mask = cv2.imread(parts_mask_file)
                parts_mask = (parts_mask / parts_factor).astype(np.uint8)
                for class_id, class_name in enumerate(
                    ['background', 'shaft', 'wrist', 'claspers']):
                    if class_id == 0:
                        continue
                    target_mask = (parts_mask == class_id) * 255
                    if target_mask.sum() != 0:
                        cris_data_list.append(
                            self.get_one_sample(root_dir, image_file, image_path,
                                        cris_masks_dir, target_mask,
                                        class_name))
                # instruments
                instruments_mask_file = image_path.replace(
                    'images', 'instruments_masks').replace('.jpg', '.png')
                instruments_mask = cv2.imread(instruments_mask_file)
                instruments_mask = (instruments_mask / instruments_factor).astype(
                    np.uint8)
                for class_id, class_name in enumerate([
                        'background', 'bipolar_forceps', 'prograsp_forceps',
                        'large_needle_driver', 'vessel_sealer',
                        'grasping_retractor', 'monopolar_curved_scissors',
                        'other_medical_instruments'
                ]):
                    if class_id == 0:
                        continue
                    target_mask = (instruments_mask == class_id) * 255
                    if target_mask.sum() != 0:
                        cris_data_list.append(
                            self.get_one_sample(root_dir, image_file, image_path,
                                        cris_masks_dir, target_mask,
                                        class_name))

        with open(os.path.join(root_dir, cris_data_file), 'w') as f:
            json.dump(cris_data_list, f)

    def get_phrase_information(self,):
        pass

    def get_img_and_seg_paths(self,):
        #😉 Currently I am only dealing with binary segmentation and multiclass segmentation, I would add in parts segmentation, later 
        self.img_paths = []
        self.mask_paths = []
        if self.split == 'train':
            dataset_num = 8
        elif self.split == 'test':
            dataset_num = 10

        for i in range(1, dataset_num + 1):
            # if self.segmentation == "full_segmentation":
            image_dir = os.path.join(self.root_dir, 'instrument_dataset_{}'.format(i),
                                    'images')
            print('process: {} ...'.format(image_dir))
            masks_dir = os.path.join(self.root_dir,
                                        'instrument_dataset_{}'.format(i),
                                        'instruments_masks')
            if not os.path.exists(masks_dir):
                os.makedirs(masks_dir)
            image_files = os.listdir(image_dir)
            image_files.sort()
            for image_file in image_files:
                print(image_file)
                image_path = sorted(os.path.join(image_dir, image_file))
                seg_path = sorted(os.path.join(masks_dir, image_file))
                self.img_paths.append(image_path)
                self.mask_paths.append(seg_path)
    
    def get_ids(self,):
        self.img_ids = [*range(len(self.img_paths) )]

    
