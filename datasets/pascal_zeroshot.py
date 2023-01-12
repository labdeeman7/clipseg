from os.path import expanduser
import torch
import json
import torchvision
from general_utils import get_from_repository
from general_utils import log
from torchvision import transforms

#😉 THis is actually what I need to work with. It is the baseline they used for zero shot segmentation. pascal-10i. Not used anywhere so far
#😉 PASCAL_VOC_CLASSES_ZS Not use anywhere I can see, but seems to be used in calculating the score in pascal zeroshot, and in phrasecut. I believe for 10 classes, these are the classes selected. 
PASCAL_VOC_CLASSES_ZS = [['cattle.n.01', 'motorcycle.n.01'], ['aeroplane.n.01', 'sofa.n.01'], 
                         ['cat.n.01', 'television.n.03'], ['train.n.01', 'bottle.n.01'],
                          ['chair.n.01', 'pot_plant.n.01']] #😉 cattle, cycle, plane, sofa, cat, television, train, bottle, chair, potplant. 


class PascalZeroShot(object): #😉 Pascal zero shot

    def __init__(self, split, n_unseen, image_size=224) -> None: #😉 size 224, n_unseen(number of unseen classes.), split, self. 
        super().__init__()

        import sys #😉
        sys.path.append('third_party/JoEm') #😉Add to sys?
        from third_party.JoEm.data_loader.dataset import VOCSegmentation #😉 pascal segmentation.
        from third_party.JoEm.data_loader import get_seen_idx, get_unseen_idx, VOC #😉 data loader, get_seen_idx, get_unseen_idx.   

        self.pascal_classes = VOC #😉 pascal_classes from joem. It has more clsses. ['background','airplane', 'bicycle', 'bird', 'boat', 'bottle',
        #'bus', 'car', 'cat', 'chair', 'cow',
        #'diningtable', 'dog', 'horse', 'motorbike', 'person',
        #'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  [10, 14, 1, 18, 8, 20, 19, 5, 9, 16] cow, motobike, airplane, sofa.../ so the classes<zs are the class names for 10 seen. 
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), #😉 A resize on the class size, to the required sizes. 
        ])

        if split == 'train': #😉 Train split, get_unseen_idx is from pascal voc the author had a set of classes that are unseen, those that are seen, split,    
            self.voc = VOCSegmentation(get_unseen_idx(n_unseen), get_seen_idx(n_unseen), 
                                       split=split, transform=True, transform_args=dict(base_size=312, crop_size=312), 
                                       ignore_bg=False, ignore_unseen=False, remv_unseen_img=True)
        elif split == 'val':
            self.voc = VOCSegmentation(get_unseen_idx(n_unseen), get_seen_idx(n_unseen), 
                                       split=split, transform=False, 
                                       ignore_bg=False, ignore_unseen=False)

        #😉 The train and val dataloaders seem to have everything to be the same even split note split is a string, so they are not the same, it also has the labelling remove unseen.                                 

        self.unseen_idx = get_unseen_idx(n_unseen)

    def __len__(self):
        return len(self.voc) #😉 self.voc, is also a dataloader, so the len, gives length. 

    def __getitem__(self, i):

        sample = self.voc[i] #😉 Gets the sample.
        label = sample['label'].long()  #😉 The label is the segmentation. 
        all_labels = [l for l in torch.where(torch.bincount(label.flatten())>0)[0].numpy().tolist() if l != 255] #😉 All labels id as a list 🙋‍♂️ Not sure.  
        class_indices = [l for l in all_labels] #😉 indices. seems like same as all_labels.
        class_names = [self.pascal_classes[l] for l in all_labels] #😉 class names for the id.    

        image = self.transform(sample['image']) #😉 transform the image, into required shape.   

        label = transforms.Resize((self.image_size, self.image_size), 
            interpolation=torchvision.transforms.InterpolationMode.NEAREST)(label.unsqueeze(0))[0] #😉 Resize the label to same size. 

        return (image,), (label, ) #😉 Return image and label, no mask. I need to do this as my first test.


