import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import json
from os.path import join
from torch.nn import functional as nnf 
import torch

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

binary_factor = 255
parts_factor = 85
instruments_factor = 32


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() +
                                                    y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--dataset_path',
        type=str,
        default=
        './datasets/Endovis2017/',
        help='path where test images with ground truth are located')
    arg('--pred_path',
        type=str,
        default=
        './store_pred/Endovis2017/',
        help='path with predictions')
    arg('--problem_type',
        type=str,
        default='parts',
        choices=['binary', 'parts', 'instrument'])
    arg('--vis', action='store_true')
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []

    class_size = (352, 352) 

    mask_dir = join(args.dataset_path, "cropped_test")

    all_dict = json.load(open(join(args.dataset_path, "clipseg_all_data_dict.json"))) 
    instrument_dict = json.load(open(join(args.dataset_path, "clipseg_instrument_dict.json"))) 
    parts_dict = json.load(open(join(args.dataset_path,  "clipseg_parts_dict.json"))) 
    binary_dict  = json.load(open(join(args.dataset_path, "clipseg_binary_dict.json"))) 


    evaluate_dir = args.pred_path
    evaluate_gt = join(evaluate_dir, "gt", args.problem_type)
    evaluate_pred = join(evaluate_dir, "pred", args.problem_type)
    # print(f" evaluate_gt {evaluate_gt}")
    # print(f" evaluate_pred {evaluate_pred}")


    if args.problem_type == 'binary':
        class_name_list = ['background', 'instrument']
        factor = binary_factor
        evaluate_dict = binary_dict  
    elif args.problem_type == 'parts':
        class_name_list = ['background', 'shaft', 'wrist', 'claspers']
        factor = parts_factor
        evaluate_dict = parts_dict
    elif args.problem_type == 'instrument':
        class_name_list = [
            'background', 'bipolar_forceps', 'prograsp_forceps',
            'large_needle_driver', 'vessel_sealer', 'grasping_retractor',
            'monopolar_curved_scissors', 'other_medical_instruments'
        ]
        factor = instruments_factor
        evaluate_dict = instrument_dict

    # print(f"factor {factor}")  

  
    i = 0
    for seq_file_name, ref_data in evaluate_dict.items():
        if args.problem_type == "instrument":
            folder_name = "instruments" #made a mistake.
        else:
            folder_name =  args.problem_type   
        
        file_name = seq_file_name.split('_')[-1]
        seq_name = seq_file_name.split('_')[-2]
        seq_name = f"instrument_dataset_{seq_name}"
        mask_path = join(mask_dir, seq_name, f"{folder_name}_masks", f"{file_name}.png")
        y_true = cv2.imread(mask_path, 0).astype(np.uint8)
        y_true = np.expand_dims(y_true, (0,1))
        # print(f"{torch.from_numpy(y_true).shape} y_true.shape")
        y_true = nnf.interpolate(torch.from_numpy(y_true), class_size, mode='nearest').numpy()

        # print(f"mask_dir {mask_dir}")
        
        pred_image = np.zeros([len(class_name_list), class_size[0], class_size[1]]) 
        # print(pred_image.shape)
        # print(file_name)
        # print(seq_name)

        for data in ref_data:
            sample_id = data[0]
            class_name = data[1]
            idx = class_name_list.index(class_name)
            # print(f"{sample_id} sample_id")
            # print(f"{class_name} class_name")
            # print(f"{idx} idx")

            pred = np.load(join(evaluate_pred, f"{sample_id}.npy"))

            pred_image[idx, :, :] = pred

        y_pred = np.argmax(pred_image, axis=0)

        y_pred = y_pred * factor
        y_pred = y_pred.astype(np.uint8)

        result_jaccard += [general_jaccard(y_true, y_pred)]
        result_dice += [general_dice(y_true, y_pred)]

        # print(result_jaccard)
        # print(result_dice)
        # break

    print('Jaccard (IoU): mean={:.2f}, std={:.4f}'.format(
        np.mean(result_jaccard) * 100, np.std(result_jaccard)))
    print('Dice: mean={:.2f}, std={:.4f}'.format(
        np.mean(result_dice) * 100, np.std(result_dice)))