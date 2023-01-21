import os
import sys
import cv2
import json
import numpy as np
from os.path import join

#😉 A photo of is added to the sentence in the code. 
#😉 The only thing that we migth want to change is using two or more instruments as a class if this is available. 
#😉 Visual prompt img
class2ref = {
    'background': {
      'phrases': ['background', 'body tissues', 'organs'],
      'visual_prompt_img': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg','sample_6.jpg' ], 
      'visual_prompt_mask': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg','sample_6.jpg'], 
    },
    'instrument': {
      'phrases': ['instrument', 'medical instrument', 'tool', 'medical tool'],
      'visual_prompt_img': ['bipolar_forceps.jpg', 'prograsp_forceps.jpg',
                    'large_needle_driver.jpg', 'vessel_sealer.jpg',
                    'grasping_retractor.jpg', 'monopolar_curved_scissors.jpg'], 
      'visual_prompt_mask': ['bipolar_forceps.jpg', 'prograsp_forceps.jpg',
                    'large_needle_driver.jpg', 'vessel_sealer.jpg',
                    'grasping_retractor.jpg', 'monopolar_curved_scissors.jpg'], 
    },

    'shaft': {
      'phrases': ['shaft', 'instrument shaft', 'tool shaft', 'instrument body',
        'tool body', 'instrument handle', 'tool handle'],
      'visual_prompt_img': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg'], 
      'visual_prompt_mask': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg',], 
    },      
    'wrist': {
      'phrases': ['wrist', 'instrument wrist', 'tool wrist', 'instrument neck',
        'tool neck', 'instrument hinge', 'tool hinge'],
      'visual_prompt_img': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg'], 
      'visual_prompt_mask': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg'], 
    },
    'claspers': {
      'phrases': ['claspers', 'instrument claspers', 'tool claspers', 'instrument head',
        'tool head'],
      'visual_prompt_img': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg','sample_6.jpg'], 
      'visual_prompt_mask': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg', 'sample_4.jpg', 'sample_5.jpg','sample_6.jpg'], 
    },
        
    'bipolar_forceps': {
      'phrases': ['bipolar forceps'],
      'visual_prompt_img': ['bipolar_forceps.jpg'], 
      'visual_prompt_mask': ['bipolar_forceps.jpg'], 
    },
    'prograsp_forceps': {
      'phrases': ['prograsp forceps'],
      'visual_prompt_img': ['prograsp_forceps.jpg'], 
      'visual_prompt_mask': ['prograsp_forceps.jpg'], 
    },
    'large_needle_driver': {
      'phrases': ['large needle driver', 'needle driver'],
      'visual_prompt_img': ["needle_driver.jpg"], 
      'visual_prompt_mask': ["needle_driver.jpg"], 
    },
    'vessel_sealer': {
      'phrases': ['vessel sealer'],
      'visual_prompt_img': ["vessel_sealer.jpg"], 
      'visual_prompt_mask': ["vessel_sealer.jpg"], 
    },
    'grasping_retractor': {
      'phrases': ['grasping retractor'],
      'visual_prompt_img': ['grasping_retractor.jpg'], 
      'visual_prompt_mask': ['grasping_retractor.jpg'], 
    },
    'monopolar_curved_scissors': {
      'phrases': ['monopolar curved scissors'],
      'visual_prompt_img': ['monopolar_curved_scissors.jpg'], 
      'visual_prompt_mask': ['monopolar_curved_scissors.jpg'], 
    },
    'other_medical_instruments': {
      'phrases': ['other instruments', 'other tools', 'other medical instruments',
        'other medical tools'],
      'visual_prompt_img': ['other_medical_instruments.jpg'], #😉 There is no support image for this. Hence we need to rely on the text description, How do we describe other? 
      'visual_prompt_mask': ['other_medical_instruments.jpg'], 
    },
    # # mult_instruments: '{
    #     'phrases': [],
    #     'visual_prompt_img': [], 
    #   'visual_prompt_mask': [], 
    # # }'  
}


binary_factor = 255
parts_factor = 85
instruments_factor = 32
sample_id = 0


def get_one_sample(root_dir, image_file, image_path, save_dir, mask,
                   class_name, visual_prompt_img_dir, visual_prompt_mask_dir, instruments_in_image, instrument_class_name, problem_type):

    global sample_id

    if '.jpg' in image_file:
        suffix = '.jpg'
    elif '.png' in image_file:
        suffix = '.png'
    mask_path = os.path.join(
        save_dir,
        image_file.replace(suffix, '') + '_{}.png'.format(class_name))
    # cv2.imwrite(mask_path, mask)
    clipseg_data = {
        "sample_id": sample_id,
        "class_name": class_name,
        "problem_type": problem_type,
        "instrument_class_name": instrument_class_name, #have a value if it is instrument segmentation if not be null. 
        'instruments_in_image': instruments_in_image, #always have a value. 
        'img_path': image_path.replace(root_dir, ''),
        'mask_path': mask_path.replace(root_dir, ''),
        'num_phrases': len(class2ref[class_name]['phrases'] ),
        'phrases': class2ref[class_name]['phrases'], #😉 choose this randomly in other parts of the code. 
        "visual_prompt_img_path": [ join(visual_prompt_img_dir, img_path).replace(root_dir, '') 
                                  for img_path in class2ref[class_name]['visual_prompt_img'] ], #😉 choose this randomly in other parts of the code. 
        "visual_prompt_mask_path": [ join(visual_prompt_mask_dir, mask_path).replace(root_dir, '') 
                                    for mask_path in class2ref[class_name]['visual_prompt_mask'] ], #😉 This should be empty
    }
    sample_id += 1
    return clipseg_data


def process(root_dir, clipseg_data_file, split='train'):
    clipseg_data_list = []
    if split == 'train':
        dataset_num = 8
    elif split == 'test':
        dataset_num = 10
    else:
        raise ValueError("split can only be 'train' and 'test'")


    vis_prompt_dir = join(root_dir,  "visual_prompt", "no_combination")
    binary_seg_vis_prompt_dir = join(vis_prompt_dir, "binary_segmentation")
    instr_seg_vis_prompt_dir = join(vis_prompt_dir, "instrument_segmentation")
    part_seg_vis_prompt_dir = join(vis_prompt_dir, "part_segmentation")
    all_possible_instrument_classes = [
                    'background', 'bipolar_forceps', 'prograsp_forceps',
                    'large_needle_driver', 'vessel_sealer',
                    'grasping_retractor', 'monopolar_curved_scissors',
                    'other_medical_instruments'
      ]

    
    for i in range(1, dataset_num + 1):
        image_dir = os.path.join(root_dir, 'cropped_{}'.format(split) , 'instrument_dataset_{}'.format(i),
                                 'images')
        print('process: {} ...'.format(image_dir))
        clipseg_masks_dir = os.path.join(root_dir, 'cropped_{}'.format(split),
                                      'instrument_dataset_{}'.format(i),
                                      'clipseg_masks')
        if not os.path.exists(clipseg_masks_dir):
            os.makedirs(clipseg_masks_dir)
        image_files = os.listdir(image_dir)
        image_files.sort()

        
        for image_file in image_files:
            print(image_file)
            image_path = os.path.join(image_dir, image_file)

            # instruments
            problem_type = "instrument"
            instruments_mask_file = image_path.replace(
                'images', 'instruments_masks').replace('.jpg', '.png')
            instruments_mask = cv2.imread(instruments_mask_file)
            instruments_mask = (instruments_mask / instruments_factor).astype(
                np.uint8)
            instrument_ids_in_images = np.unique(instruments_mask)    
            instrument_ids_in_images = instrument_ids_in_images.tolist()
            instrument_ids_in_images.remove(0) #remove background.
            instruments_in_image = [all_possible_instrument_classes[instrument_id] for instrument_id in instrument_ids_in_images]
                      
            for class_id, class_name in enumerate([
                    'background', 'bipolar_forceps', 'prograsp_forceps',
                    'large_needle_driver', 'vessel_sealer',
                    'grasping_retractor', 'monopolar_curved_scissors',
                    'other_medical_instruments'
            ]):
                if class_id == 0:
                    continue
                target_mask = (instruments_mask == class_id) * 255
                class_vis_prompt_img_dir = join(instr_seg_vis_prompt_dir, "img")
                class_vis_prompt_mask_dir = join(instr_seg_vis_prompt_dir, "mask")

                instrument_class_name = class_name
                if target_mask.sum() != 0:
                    clipseg_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       clipseg_masks_dir, target_mask,
                                       class_name, class_vis_prompt_img_dir, 
                                       class_vis_prompt_mask_dir, instruments_in_image, instrument_class_name, problem_type))

            # binary
            problem_type = "binary"
            binary_mask_file = image_path.replace('images',
                                                  'binary_masks').replace(
                                                      '.jpg', '.png')
            binary_mask = cv2.imread(binary_mask_file)
            binary_mask = (binary_mask / binary_factor).astype(np.uint8)
            for class_id, class_name in enumerate(['background',
                                                   'instrument']):
                target_mask = (binary_mask == class_id) * 255

                class_vis_prompt_img_dir = join(binary_seg_vis_prompt_dir, class_name, "img")
                class_vis_prompt_mask_dir = join(binary_seg_vis_prompt_dir, class_name, "mask")
                instrument_class_name = None 

                if target_mask.sum() != 0:
                    clipseg_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       clipseg_masks_dir, target_mask,
                                       class_name, class_vis_prompt_img_dir, 
                                       class_vis_prompt_mask_dir, instruments_in_image, instrument_class_name, problem_type))

            # parts
            problem_type = "parts"
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

                class_vis_prompt_img_dir = join(part_seg_vis_prompt_dir, class_name, "img")
                class_vis_prompt_mask_dir = join(part_seg_vis_prompt_dir, class_name, "mask")
                instrument_class_name = None 

                if target_mask.sum() != 0:
                    clipseg_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       clipseg_masks_dir, target_mask,
                                       class_name, class_vis_prompt_img_dir, 
                                       class_vis_prompt_mask_dir, instruments_in_image, instrument_class_name, problem_type))

            

    with open(os.path.join(root_dir, clipseg_data_file), 'w') as f:
        json.dump(clipseg_data_list, f)


if __name__ == '__main__':
    # must add last "/"
    # /jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_test/
    root_dir = sys.argv[1]
    # clipseg_test.json
    clipseg_data_file = sys.argv[2]
    split = sys.argv[3]
    process(root_dir, clipseg_data_file, split)