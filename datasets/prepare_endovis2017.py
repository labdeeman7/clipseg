import os
import sys
import cv2
import json
import numpy as np

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


def get_one_sample(root_dir, image_file, image_path, save_dir, mask,
                   class_name):
    if '.jpg' in image_file:
        suffix = '.jpg'
    elif '.png' in image_file:
        suffix = '.png'
    mask_path = os.path.join(
        save_dir,
        image_file.replace(suffix, '') + '_{}.png'.format(class_name))
    cv2.imwrite(mask_path, mask)
    cris_data = {
        'img_path': image_path.replace(root_dir, ''),
        'mask_path': mask_path.replace(root_dir, ''),
        'num_sents': len(class2sents[class_name]),
        'sents': class2sents[class_name],
    }
    return cris_data


def process(root_dir, cris_data_file):
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
                        get_one_sample(root_dir, image_file, image_path,
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
                        get_one_sample(root_dir, image_file, image_path,
                                       cris_masks_dir, target_mask,
                                       class_name))

    with open(os.path.join(root_dir, cris_data_file), 'w') as f:
        json.dump(cris_data_list, f)


if __name__ == '__main__':
    # must add last "/"
    # /jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_test/
    root_dir = sys.argv[1]
    # cris_test.json
    cris_data_file = sys.argv[2]
    process(root_dir, cris_data_file)