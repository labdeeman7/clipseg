
import torch
import numpy as np
import os

from os.path import join, isdir, isfile, expanduser 
from PIL import Image

from torchvision import transforms
from torchvision.transforms.transforms import Resize

from torch.nn import functional as nnf
from general_utils import get_from_repository

from skimage.draw import polygon2mask

#😉 polygon to mask important. phrasecut gives a json as the label. and the json is in the format of polygons.
#😉 Remeber that phrasecut was produced from the visual-genome dataset 

PASCAL_SYNSETS = ['person.n.01', 'bird.n.01', 'cat.n.01', 'cattle.n.01', 'dog.n.01', 'horse.n.01', 'sheep.n.01', 'aeroplane.n.01', 'bicycle.n.01', 
                    'boat.n.01', 'bus.n.01', 'car.n.01', 'motorcycle.n.01', 'train.n.01', 'bottle.n.01', 'chair.n.01', 'kitchen_table.n.01',  
                    'breakfast_table.n.01', 'trestle_table.n.01','pot_plant.n.01', 'sofa.n.01', 'television.n.03']  #😉 synset has to do with the removal of coco 20is dataset from the phrasecut dtaset. 

PASCAL_5I_SYNSETS_ORDERED = [
    'aeroplane.n.01', 'bicycle.n.01',  'bird.n.01', 'vessel.n.02', 'bottle.n.01', 'bus.n.01', 'car.n.01', 
     'cat.n.01',  'chair.n.01',  'cattle.n.01',   'table.n.02',  'dog.n.01', 'horse.n.01',  'motorcycle.n.01', 
      'person.n.01', 'pot_plant.n.01', 'sheep.n.01',  'sofa.n.01', 'train.n.01', 'television.n.03']  #😉 Synset is used to map classes from coco 20i to phrasecut to select which classes to remove. 

# Pascal-5i classes
PASCAL_5I_CLASS_IDS = {
    3: list(range(1, 16)),
    2: list(range(1, 11)) + list(range(16, 21)),
    1: list(range(1, 6)) + list(range(11, 21)),
    0: list(range(6, 21))
} #😉 pascal 5i IDs



def traverse_lemmas(synset): 
    """ list of all lemma names of hypernyms """

    out = synset.lemma_names()
    for h in synset.hypernyms():
        out += traverse_lemmas(h)

    return out #🙋‍♂️ Not sure. 


def traverse_lemmas_hypo(synset, max_depth=None, depth=0):
    """ list of all lemma names of hypernyms """

    if synset.name() == 'person.n.01':
        return ['person', 'human', 'man', 'woman', 'toddler', 'baby', 'body', 'child', 'infant'] 

    out = [l.name() for l in synset.lemmas()]

    # print(' '*depth, synset.name())

    if max_depth is not None and depth >= max_depth:
        return out

    # print(' '*(3*depth), out)
    for h in synset.hyponyms():
        out += traverse_lemmas_hypo(h, max_depth, depth+1)

    return out #😉Not sure, but I know it is not useful for me unless I want to search words and things like this.



def random_crop_slices(origin_size, target_size): #😉
    """Gets slices of a random crop. """
    assert origin_size[0] >= target_size[0] and origin_size[1] >= target_size[1], f'actual size: {origin_size}, target size: {target_size}' #😉 original size should be bigger than the target_size.    

    offset_y = torch.randint(0, origin_size[0] - target_size[0] + 1, (1,)).item()  # range: 0 <= value < high #😉 between the two values
    offset_x = torch.randint(0, origin_size[1] - target_size[1] + 1, (1,)).item()

    return slice(offset_y, offset_y + target_size[0]), slice(offset_x, offset_x + target_size[1])


def find_crop(seg, image_size, iterations=1000, min_frac=None, best_of=None): #😉 seg, image_size, iterations?, min_frac? best_of?

    best_crops = [] #😉 best_crops. 
    best_crop_not_ok = float('-inf'), None, None  #🙋‍♂️ [-inf, None, None]
    min_sum = 0 #🙋‍♂️ min sum?

    seg = seg.astype('bool') #😉 seg as bool? So two class semantic.
    
    if min_frac is not None: 
        #min_sum = seg.sum() * min_frac
        min_sum = seg.shape[0] * seg.shape[1] * min_frac #🙋‍♂️the min_ frac multiply. 
    
    for iteration in range(iterations): #🙋‍♂️ range iterations. 
        sl_y, sl_x = random_crop_slices(seg.shape, image_size) #😉 random crop slice is the difference between off and interval. 
        seg_ = seg[sl_y, sl_x] #😉 sl_y and sl_x. segmentation others. 
        sum_seg_ = seg_.sum() #😉 Segmentation extra is a sum 

        if sum_seg_ > min_sum: #😉 if sum of seg is greater than the min_sum, 

            if best_of is None:  #😉 best_of is not close. 
                return sl_y, sl_x, False #😉 return sl_y, sl_x, False. 
            else:
                best_crops += [(sum_seg_, sl_y, sl_x)]
                if len(best_crops) >= best_of:
                    best_crops.sort(key=lambda x:x[0], reverse=True)
                    sl_y, sl_x = best_crops[0][1:]
                    
                    return sl_y, sl_x, False #😉 we just send sl, sly I am not sure.  

        else:
            if sum_seg_ > best_crop_not_ok[0]:
                best_crop_not_ok = sum_seg_, sl_y, sl_x
        
    else:
        # return best segmentation found
        return best_crop_not_ok[1:] + (best_crop_not_ok[0] <= min_sum,)  #😉We have to select the best_crop_not_ok
    ##🙋‍♂️ Not still sure what this does.     


class PhraseCut(object):
    #😉 split is train vs val, image_size=400, negative_prob is for negative_sampling, augmentation, 
    #😉 aug_color, aug_crop, min_size of the image, remove_classes, visual, only_visual, mask.   
    def __init__(self, split, image_size=400, negative_prob=0, aug=None, aug_color=False, aug_crop=True,
                 min_size=0, remove_classes=None, with_visual=False, only_visual=False, mask=None):

        super().__init__()

        self.negative_prob = negative_prob #😉 Negative sampling
        self.image_size = image_size #😉 size.
        self.with_visual = with_visual #😉 with_visual, using visual and text
        self.only_visual = only_visual #😉 Only visual. 
        self.phrase_form = '{}' #😉 phrase_form. 
        self.mask = mask #😉 the text, seaparate etc.  
        self.aug_crop = aug_crop #😉 crop augmentation.
        
        if aug_color: #😉 aug_color, compose
            self.aug_color = transforms.Compose([
                transforms.ColorJitter(0.5, 0.5, 0.2, 0.05), #😉 color jitter.
            ])
        else:
            self.aug_color = None #😉 Augmented color. 

        # get_from_repository('PhraseCut', ['PhraseCut.tar'], integrity_check=lambda local_dir: all([
        #     isdir(join(local_dir, 'VGPhraseCut_v0')),
        #     isdir(join(local_dir, 'VGPhraseCut_v0', 'images')),
        #     isfile(join(local_dir, 'VGPhraseCut_v0', 'refer_train.json')),
        #     len(os.listdir(join(local_dir, 'VGPhraseCut_v0', 'images'))) in {108250, 108249}
        # ])) #😉 PhraseCut, PhraseCut.tar, VGPhraseCut_v0_images, 

        from third_party.PhraseCutDataset.utils.refvg_loader import RefVGLoader 
        self.refvg_loader = RefVGLoader(split=split) #😉 RefVGLoader, split, only split is passed in.   RefVGLoader get_img_ref_data takes id and gives a lot of information about image.  

        # img_ids where the size in the annotations does not match actual size
        invalid_img_ids = set([150417, 285665, 498246, 61564, 285743, 498269, 498010, 150516, 150344, 286093, 61530, 
                               150333, 286065, 285814, 498187, 285761, 498042])  #😉 invalid_img_id set, things that have a problem. 
        
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std) #😉 Image mean,std from ImageNet.   

        self.sample_ids = [(i, j) 
                           for i in self.refvg_loader.img_ids 
                           for j in range(len(self.refvg_loader.get_img_ref_data(i)['phrases']))
                           if i not in invalid_img_ids] #😉 sample_ids, is givven by: for every img_id, for every orresponding phrase list for this id, as long as the id is not an invalid img id, return the image id and the phrase list as a tuple. 
        

        # self.all_phrases = list(set([p for i in self.refvg_loader.img_ids for p in self.refvg_loader.get_img_ref_data(i)['phrases']]))

        from nltk.stem import WordNetLemmatizer #🙋‍♂️ why are we WorldNetLemmatizer.  
        wnl = WordNetLemmatizer()        

        # Filter by class (if remove_classes is set)
        if remove_classes is None: #😉 remove_classes.
            pass
        else:
            from nltk.corpus import wordnet #😉 worldnet  

            print('remove pascal classes...') #😉 pascal classes, are been removed from img_id and phrases. phrase contains all the values such as attribute, category, relationship  

            get_data = self.refvg_loader.get_img_ref_data  
            keep_sids = None  #😉 keep_sids 

            if remove_classes[0] == 'pas5i': #😉 pascal 5i, remove c;asses/ 
                subset_id = remove_classes[1]

                avoid = [PASCAL_5I_SYNSETS_ORDERED[i] for i in range(20) if i+1 not in PASCAL_5I_CLASS_IDS[subset_id]] #😉 Class names to be avoided. 
      

            elif remove_classes[0] == 'zs': #😉 remove classes for coco.
                stop = remove_classes[1]
                
                from datasets.pascal_zeroshot import PASCAL_VOC_CLASSES_ZS

                avoid = [c for class_set in PASCAL_VOC_CLASSES_ZS[:stop] for c in class_set]  #😉 Class names to be avoided.
                # print(avoid)

            elif remove_classes[0] == 'aff': #😉 Not sure was aff is. It maybe related to LaVIS. pa5i is pascal 5i, xeroshot is maybe coco or maybe still pascal, not sure. what is aff? 
                # avoid = ['drink.v.01', 'sit.v.01', 'ride.v.02']
                # all_lemmas = set(['drink', 'sit', 'ride'])
                avoid = ['drink', 'drinks', 'drinking', 'sit', 'sits', 'sitting', 
                         'ride', 'rides', 'riding',
                         'fly', 'flies', 'flying', 'drive', 'drives', 'driving', 'driven', 
                         'swim', 'swims', 'swimming',
                         'wheels', 'wheel', 'legs', 'leg', 'ear', 'ears']
                keep_sids = [(i, j) for i, j in self.sample_ids if 
                             all(x not in avoid for x in get_data(i)['phrases'][j].split(' '))]

            print('avoid classes:', avoid)


            if keep_sids is None: #😉 🙋‍♂️keep sids? lemma, all lemmas? Not sure
                all_lemmas = [s for ps in avoid for s in traverse_lemmas_hypo(wordnet.synset(ps), max_depth=None)]
                all_lemmas = list(set(all_lemmas))
                all_lemmas = [h.replace('_', ' ').lower() for h in all_lemmas]
                all_lemmas = set(all_lemmas)

                # divide into multi word and single word
                all_lemmas_s = set(l for l in all_lemmas if ' ' not in l)
                all_lemmas_m = set(l for l in all_lemmas if l not in all_lemmas_s)

                # new3
                phrases = [get_data(i)['phrases'][j] for i, j in self.sample_ids]
                remove_sids = set((i,j) for (i,j), phrase in zip(self.sample_ids, phrases)
                                  if any(l in phrase for l in all_lemmas_m) or 
                                  len(set(wnl.lemmatize(w) for w in phrase.split(' ')).intersection(all_lemmas_s)) > 0
                )
                keep_sids = [(i, j) for i, j in self.sample_ids if (i,j) not in remove_sids]

            print(f'Reduced to {len(keep_sids) / len(self.sample_ids):.3f}')
            removed_ids = set(self.sample_ids) - set(keep_sids)

            print('Examples of removed', len(removed_ids))
            for i, j in list(removed_ids)[:20]:
                print(i, get_data(i)['phrases'][j])

            self.sample_ids = keep_sids

        #😉 Okay, I reached this spot. 
        from itertools import groupby
        samples_by_phrase = [(self.refvg_loader.get_img_ref_data(i)['phrases'][j], (i, j)) 
                             for i, j in self.sample_ids]
        samples_by_phrase = sorted(samples_by_phrase)
        samples_by_phrase = groupby(samples_by_phrase, key=lambda x: x[0])
        
        self.samples_by_phrase = {prompt: [s[1] for s in prompt_sample_ids] for prompt, prompt_sample_ids in samples_by_phrase} #😉 prompt sample phrases,   

        self.all_phrases = list(set(self.samples_by_phrase.keys())) #😉 all phrases for all classes I guess not sure. 


        if self.only_visual: #😉 self.only_visual, when it is only visual,  
            assert self.with_visual
            self.sample_ids = [(i, j) for i, j in self.sample_ids
                               if len(self.samples_by_phrase[self.refvg_loader.get_img_ref_data(i)['phrases'][j]]) > 1] #😉 not sure what the sample_ids are.  

        # Filter by size (if min_size is set)
        sizes = [self.refvg_loader.get_img_ref_data(i)['gt_boxes'][j] for i, j in self.sample_ids] #😉 what do they mean by sizes, maybe sizes of gt_boxes
        image_sizes = [self.refvg_loader.get_img_ref_data(i)['width'] * self.refvg_loader.get_img_ref_data(i)['height'] for i, j in self.sample_ids] #😉 Image sizes,
        #self.sizes = [sum([(s[2] - s[0]) * (s[3] - s[1]) for s in size]) for size in sizes]
        self.sizes = [sum([s[2] * s[3] for s in size]) / img_size for size, img_size in zip(sizes, image_sizes)] #😉 all sizes I need to check everything. 

        if min_size:
            print('filter by size')

        self.sample_ids = [self.sample_ids[i] for i in range(len(self.sample_ids)) if self.sizes[i] > min_size] #😉  filter by size,🙋‍♂️ maybe a min size, 

        self.base_path = join(expanduser('datasets/PhraseCut/VGPhraseCut_v0/images')) #😉 wooow that is just init.

    def __len__(self):
        return len(self.sample_ids) #😉 length of all sample ids. 

    def load_sample(self, sample_i, j): #😉 load_samples.
        # print(sample_i)
        # print(j)
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

        # print(i)
 
        sample_i, j = self.sample_ids[i]

        img, seg, phrase = self.load_sample(sample_i, j) #😉 Image, segmentation and phrase. 🙋‍♂️ I am not sure what sample_i and j stand for. 

        if self.negative_prob > 0: #😉 Negative samples.  #😉 sampling for negative probability
            if torch.rand((1,)).item() < self.negative_prob: #😉 probability for negative probability.

                new_phrase = None 
                while new_phrase is None or new_phrase == phrase:
                    idx = torch.randint(0, len(self.all_phrases), (1,)).item()
                    new_phrase = self.all_phrases[idx]
                phrase = new_phrase
                seg = torch.zeros_like(seg)

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
                               

#😉 Phrasecutplus implementation is already in phrsecut, just with a different set of variables. 
class PhraseCutPlus(PhraseCut):

    def __init__(self, split, image_size=400, aug=None, aug_color=False, aug_crop=True, min_size=0, remove_classes=None, only_visual=False, mask=None):
        super().__init__(split, image_size=image_size, negative_prob=0.2, aug=aug, aug_color=aug_color, aug_crop=aug_crop, min_size=min_size, 
                         remove_classes=remove_classes, with_visual=True, only_visual=only_visual, mask=mask)
