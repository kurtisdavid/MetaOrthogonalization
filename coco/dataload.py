import torch
import os
import os.path
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
from io import StringIO
from glob import glob
from torchvision import transforms, datasets
import utils
import pickle
from copy import deepcopy
from itertools import cycle
import pandas as pd
import math
from copy import deepcopy
# ~~~~~~~~~~~~~~~~~~~~~~~ DATASETS ~~~~~~~~~~~~~~~~~~~~~~~

class IdenProf(Dataset):
    def __init__(self, root, train=True, transform=None, extra='labels.txt', exclusive=False):
        self.root = root
        self.train = train
        self.transform = transform
        if train:
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')

        self.class_names = [x for x in glob(os.path.join(self.img_path,'*')) if '.txt' not in x]
        self.idx_to_class = {
            i: x.split('/')[-1]
            for i,x in enumerate(self.class_names)
        }
        self.idx_to_class[10] = 'male'
        self.idx_to_class[11] = 'female'
        
        self.img_filename = []
        self.label = []
        self.gender = {}
        idx_map = {
            0: [0, 1],
            1: [1, 0],
        }
        g_map = {
            "F": [0, 1],
            "M": [1, 0],
            "B": [1, 1]
        }
        for i,class_name in enumerate(self.class_names):
            images = glob(os.path.join(class_name,'*.jpg'))
            self.img_filename.extend(images)
            self.label.extend([i]*len(images))
        count = np.array([0.0,0.0])
        with open(os.path.join(self.img_path, extra), 'r') as f:
            for line in f:
                fname, gender = line.strip().split()
                try:
                    self.gender[fname] = np.array(
                                            idx_map[int(gender)]
                                         )
                except:
                    if exclusive and gender == "B":
                        continue
                    self.gender[fname] = np.array(
                                            g_map[gender]
                                         )
                    if exclusive:
                        if gender == "F":
                            count[1] += 1
                        else:
                            count[0] += 1
        self.label = np.array(self.label, dtype=np.int64)
        
        if exclusive:
            N = count.sum()
            valids = []
            weights = []
            for i,img_name in enumerate(self.img_filename):
                if img_name in self.gender:
                    valids.append(i)
                    weights.append(
                        N / (self.gender[img_name] * count).sum()
                    )
            self.img_filename = [self.img_filename[x] for x in valids]
            self.weights = weights
            self.label = self.label[valids]
    
    def __getitem__(self, index):
        img = Image.open(self.img_filename[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(
                    np.array([self.label[index]])
                ).squeeze()
        gender = torch.from_numpy(
                    self.gender[self.img_filename[index]]
                 )
        return img, label, gender

    def __len__(self):
        return len(self.img_filename)

    def return_subset(self, idx):
        new = IdenProf(self.root, self.train, self.transform)
        
        new.img_filename = [new.img_filename[x] for x  in idx]
        new.label        = new.label[idx]
        return new

# based off of this https://github.com/jiangqy/Customized-DataLoader/blob/master/dataset_processing.py
# example use:
#    CelebA('./celeba','img_align_celeba/train','labels/train_labels.txt')
class CelebA(Dataset):
    def __init__(self, data_path, img_filepath, filename, transform=None):
        self.img_path = os.path.join(data_path, img_filepath)
        self.transform = transform
        # reading labels from file
        label_filepath = os.path.join(data_path, filename)
        # reading img file from file
        fp = open(label_filepath, 'r')
        self.img_filename = [x.split()[0].strip() for x in fp]
        fp.close()
        s = ""
        with open(label_filepath,'r') as fp:
            for line in fp:
                s += ' '.join(line.split()[1:]) + '\n'
        s = StringIO(s)
        labels = np.loadtxt(s, dtype=np.int64)
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).type(torch.FloatTensor)
        return img, label
    def __len__(self):
        return len(self.img_filename)

# based off of : https://github.com/JerryYLi/Dataset-REPAIR/blob/master/utils/datasets.py
class ColoredDataset(Dataset):
    def __init__(self, dataset, classes=None, colors=[0, 1], std=0.1):
        self.dataset = dataset
        self.colors = colors
        if classes is None:
            classes = max([y for _, y in dataset]) + 1

        if isinstance(colors, torch.Tensor):
            self.colors = colors
        elif isinstance(colors, list):
            self.colors = torch.Tensor(classes, 3, 1, 1).uniform_(colors[0], colors[1])
        else:
            raise ValueError('Unsupported colors!')
        self.perturb = std * torch.randn(len(self.dataset), 3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        color_img = (self.colors[label] + self.perturb[idx]).clamp(0, 1) * img
        return color_img, label, 0

# based off of : https://github.com/JerryYLi/Dataset-REPAIR/blob/master/utils/datasets.py
class BinaryColorDataset(Dataset):
    def __init__(self, dataset, classes=None, ratio=None):
        self.dataset = dataset
        self.colors = torch.zeros(
            (len(self.dataset), 3)
        )
        if classes is None:
            classes = max([y for _, y in dataset]) + 1
        if ratio is None:
            ratio = [0.5 for _ in range(classes)]
        class_to_idx = {y:{"idxs": []} for y in range(classes)}
        for i, data in enumerate(self.dataset):
            y = data[1]
            class_to_idx[y]["idxs"].append(i)
        for y in class_to_idx:
            idxs = np.array(class_to_idx[y]["idxs"])
            permuted_idxs = np.random.permutation(idxs)
            n = int(len(idxs) * ratio[y])
            pos_idxs = permuted_idxs[:n]
            neg_idxs = permuted_idxs[n:]
            self.colors[pos_idxs] += torch.Tensor([1,0,0]) # color them red
            self.colors[neg_idxs] += torch.Tensor([0,0,1]) # color them blue
        self.colors = self.colors.unsqueeze(-1).unsqueeze(-1)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        color_img = self.colors[idx] * img
        return color_img, label, self.colors[idx].squeeze()[[0,2]]

# this uses a subset of BAM dataset such that each scene class has a ratio of truck/zebra pasted
class SceneBAM(Dataset):
    # ratio -> ratio of truck / | scene |
    # specific -> do we only modify the given scene, if None, all get the given ratio (otherwise others get 0.5) 
    def __init__(self, 
                 root='bam_scenes/', 
                 train=True, 
                 transform=transforms.ToTensor(), 
                 ratio=0.5, 
                 specific=None, 
                 last=None,
                 next=None):
        assert specific is None or isinstance(specific, str) or isinstance(specific, tuple)
        self.root = root
        self.train = train
        self.transform = transform
        self.key = (train, specific, ratio)
        # this is used to make sure subsets of chosen images are valid
        if last is not None:
            assert isinstance(last, SceneBAM)
            assert last.key[0] == train
            assert last.key[1] == specific
            assert last.key[2] > ratio
        if train:
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'val')
        self.class_names = [x for x in glob(os.path.join(self.img_path,'*')) if '.txt' not in x]
        self.idx_to_class = {
            i: x.split('/')[-1]
            for i,x in enumerate(self.class_names)
        }
        self.idx_to_class[10] = 'truck'
        self.idx_to_class[11] = 'zebra' 
        self.img_filename = []
        self.label = []
        self.truck_zebra = []
        self.truck_idxs = []
        self.zebra_idxs = []
        for i,class_name in enumerate(self.class_names):
            # first extract all the names for truck and zebra
            truck_images = np.array(sorted(glob(os.path.join(class_name,'truck-*.jpg'))))
            zebra_images = np.array(sorted(glob(os.path.join(class_name,'zebra-*.jpg'))))
            assert truck_images.shape[0] == zebra_images.shape[0]
            N = truck_images.shape[0]
            if (
                (specific is None) 
                or self.idx_to_class[i] in specific
            ):
                n = int(N * ratio)
                if last is None:
                    assert ratio == 1.0
                    permuted_idxs = np.random.permutation(np.arange(0,N))
                    truck_idxs = permuted_idxs[:n]
                    zebra_idxs = permuted_idxs[n:]
                else:
                    # we want to sample from these idxs next...
                    curr_truck_idxs_length = last.truck_idxs[i].shape[0]
                    assert curr_truck_idxs_length > n
                    # if we don't have a restriction on which ones have already been chosen, just consider all
                    if next is None:
                        to_sample_from = last.truck_idxs[i]
                        # permuted_idxs = np.random.permutation(np.arange(0,curr_truck_idxs_length))
                    else:
                        # since next exists, collect the ones that you haven't chosen
                        to_sample_from = np.setdiff1d(last.truck_idxs[i], next.truck_idxs[i])
                        # you just need to get this many more for the current dataset
                        n = n - next.truck_idxs[i].shape[0]
                    permuted_idxs = np.random.permutation(np.arange(0,to_sample_from.shape[0]))
                    truck_idxs = to_sample_from[permuted_idxs[:n]]
                    if next is not None:
                        truck_idxs = np.concatenate(
                            [next.truck_idxs[i], truck_idxs],
                            axis = 0
                        )
                    # these get sent over to zebras, since they weren't chosen to stay as a truck
                    leftover_truck_idxs = np.setdiff1d(last.truck_idxs[i], truck_idxs)
                    zebra_idxs = np.concatenate(
                        [
                            last.zebra_idxs[i],
                            leftover_truck_idxs
                        ]
                    )
                    # guarantees that we've made subsets
                    assert np.setdiff1d(truck_idxs, last.truck_idxs[i]).shape[0] == 0
                    assert np.setdiff1d(last.zebra_idxs[i], zebra_idxs).shape[0] == 0
                    if next is not None:
                        assert np.setdiff1d(next.truck_idxs[i], truck_idxs).shape[0] == 0
                        assert np.setdiff1d(zebra_idxs, next.zebra_idxs[i]).shape[0] == 0
                # since we need this for later...
                n = int(N * ratio)
            else:
                n = int(N * 0.5)
                if last is None:
                    permuted_idxs = np.random.permutation(np.arange(0,N))
                    truck_idxs = permuted_idxs[:n]
                    zebra_idxs = permuted_idxs[n:]
                else:
                    # if coming from a previous experiment, make sure to copy the same idxs over...
                    truck_idxs = last.truck_idxs[i]
                    zebra_idxs = last.zebra_idxs[i]
            
            self.truck_idxs.append(truck_idxs)
            self.zebra_idxs.append(zebra_idxs)
            images = truck_images[truck_idxs].tolist() + zebra_images[zebra_idxs].tolist()
            assert len(images) == N 
            self.img_filename.extend(images)
            self.label.extend([i]*len(images))
            self.truck_zebra.extend([[1,0]]*n)
            self.truck_zebra.extend([[0,1]]*(N-n))

        self.label = np.array(self.label, dtype=np.int64)
        self.truck_zebra = np.array(self.truck_zebra, dtype=np.int64)
        self.verified = False 

    def verify(self, last=None, next=None):
        assert last is None or len(self.truck_idxs) == len(last.truck_idxs) 
        for i in range(len(self.truck_idxs)):
            # unit test to see if we have next <= curr <= last relation
            if last is not None:
                assert np.setdiff1d(self.truck_idxs[i], last.truck_idxs[i]).shape[0] == 0
                assert np.setdiff1d(last.zebra_idxs[i], self.zebra_idxs[i]).shape[0] == 0
            if next is not None:
                assert np.setdiff1d(next.truck_idxs[i], self.truck_idxs[i]).shape[0] == 0
                assert np.setdiff1d(self.zebra_idxs[i], next.zebra_idxs[i]).shape[0] == 0
            # unit test to make sure we have the desired ratio
            total = len(self.truck_idxs[i]) + len(self.zebra_idxs[i])
            curr  = self.idx_to_class[i]
            if self.key[1] is None or curr in self.key[1]:
                assert float(len(self.truck_idxs[i])) / total == self.key[2], str(i) + " | " + str(float(len(self.truck_idxs[i])) / total) + " | " + str(self.key[2])
            else:
                assert float(len(self.truck_idxs[i])) / total == 0.5
        # lastly double check that all images have a corresponding label
        assert len(self.img_filename) == self.label.shape[0] == self.truck_zebra.shape[0], (len(self.img_filename), self.label.shape[0] , self.truck_zebra.shape[0])

    def __getitem__(self, index):
        img = Image.open(self.img_filename[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(
                    np.array([self.label[index]])
                ).squeeze()
        truck_zebra = torch.from_numpy(
                    np.array([self.truck_zebra[index]])
                 ).squeeze()
        return img, label, truck_zebra

    def __len__(self):
        return len(self.img_filename)

class SubsetSplit(object):
    
    def __init__(self, dataset, ratio):
        self.subset = deepcopy(dataset)
        self.remaining = deepcopy(dataset)
        last = 0
        
        self.subset.img_filename = []
        self.subset.label = []
        self.subset.truck_zebra = []

        self.remaining.img_filename = []
        self.remaining.label = []
        self.remaining.truck_zebra = []


        for i in range(len(dataset.truck_idxs)):
            curr_truck_idxs = dataset.truck_idxs[i]
            curr_zebra_idxs = dataset.zebra_idxs[i]
            
            # this ensures we have stratified sampling subsets
            N1, N2 = int(ratio * curr_truck_idxs.shape[0]), int(ratio * curr_zebra_idxs.shape[0])

            N = curr_truck_idxs.shape[0] + curr_zebra_idxs.shape[0]


            # these are what will be used for the subset...
            subset_truck_idxs = np.random.choice(curr_truck_idxs.shape[0], N1, replace=False)
            subset_zebra_idxs = np.random.choice(curr_zebra_idxs.shape[0], N2, replace=False)

            remaining_truck_idxs = np.random.choice(curr_truck_idxs.shape[0], curr_truck_idxs.shape[0], replace=False)
            remaining_zebra_idxs = np.random.choice(curr_zebra_idxs.shape[0], curr_zebra_idxs.shape[0], replace=False)
            # these are the ones that are left for the main subset (i.e. split...)
            remaining_truck_idxs = np.setdiff1d(remaining_truck_idxs, subset_truck_idxs)
            remaining_zebra_idxs = np.setdiff1d(remaining_zebra_idxs, subset_zebra_idxs)

            class_images = np.array(dataset.img_filename[last:last+N])
            truck_images = class_images[:curr_truck_idxs.shape[0]]
            zebra_images = class_images[curr_truck_idxs.shape[0]:]

            subset_truck_images = truck_images[subset_truck_idxs]
            subset_zebra_images = zebra_images[subset_zebra_idxs]
            remaining_truck_images = truck_images[remaining_truck_idxs]
            remaining_zebra_images = zebra_images[remaining_zebra_idxs]

            subset_images    = subset_truck_images.tolist() + subset_zebra_images.tolist()
            remaining_images = remaining_truck_images.tolist() + remaining_zebra_images.tolist()

            self.subset.img_filename.extend(subset_images)
            self.remaining.img_filename.extend(remaining_images)

            self.subset.label.extend([i]*len(subset_images))
            self.remaining.label.extend([i]*len(remaining_images))
            
            self.subset.truck_zebra.extend([[1,0]]*subset_truck_images.shape[0])
            self.subset.truck_zebra.extend([[0,1]]*subset_zebra_images.shape[0])
            
            self.remaining.truck_zebra.extend([[1,0]]*remaining_truck_images.shape[0])
            self.remaining.truck_zebra.extend([[0,1]]*remaining_zebra_images.shape[0])

            last += N
    
    def return_split(self):
        return self.subset, self.remaining

# used to extract specific zebra/truck images from ImageNet
class SubsetImageFolder(datasets.ImageFolder):
    
    def __init__(self, root, class_to_idx, boxes = None, reference_area = None, **kwargs):
        self.class_to_idx = class_to_idx
        self.reference_area = reference_area
        super(SubsetImageFolder, self).__init__(root, **kwargs)
        # provide a way to remove certain images (we don't have good mask information for)
        # not needed for now, may need later
        if boxes is not None:
            assert isinstance(boxes, str) and 'csv' in boxes
            # to do: set up box reading...
            self.boxes = pd.read_csv(boxes, header=None)
            valid   = set(self.boxes[0].tolist())
            invalid = set(range(len(self.samples))).difference(valid)
            self.imgs = [self.samples[x] for x in range(len(self.samples)) if (x not in invalid)]
            self.samples = self.imgs
        else:
            self.boxes = None
    
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes      = self.class_to_idx.keys()
        class_to_idx = self.class_to_idx
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        new_transform = None
        if self.boxes is not None and self.reference_area is not None:
            bbox = self.boxes.iloc[index,1:].tolist()
            area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            # rescaling factor, to get the same scale as boxes found in BAM
            ratio = 1 / math.sqrt(area / self.reference_area)
            assert self.transform is not None
            new_transform = deepcopy(self.transform)
            new_transform.transforms[2].scale = [ratio, ratio] 
        if self.transform is not None and new_transform is None:
            sample = self.transform(sample)
        elif self.transform is not None:
            sample = new_transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target


# ~~~~~~~~~~~~~~~~~~~~~~~ DATALOADERS ~~~~~~~~~~~~~~~~~~~~~~~
def process_imagenet_loaders(loaders):
    loader = zip(
        cycle(
            zip(
                cycle(loaders['zebra']),
                loaders['truck']
            )
        ),
        loaders['negative']
    )
    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = zip(
                         cycle(
                             zip(
                                 cycle(loaders['zebra']),
                                 loaders['truck']
                             )
                         ),
                         loaders['negative']
                     ) 

# mini imagenet with only truck/zebra dataloaders + 
def get_imagenet_tz(datapath, train_batch_size=64, test_batch_size=64, workers=16, distributed=False, normalized=True, shuffled=True, resize=False, constant=True):
    # 0-3 are trucks, 4 zebra
    # TODO: how do we want to sample this? will need to update code
    valid_classes = {"truck": {"n03417042":0,
                               "n03930630":0,
                               "n04461696":0,
                               "n04467665":0},
                     "zebra": {"n02391049":1}
                    }
    all_classes = {}
    with open('./imagenet_labels.txt', 'r') as f:
        for line in f:
            label = line.rstrip()
            if label not in valid_classes['truck'] and label not in valid_classes['zebra']:
                all_classes[label] = 2
    assert len(all_classes.keys()) == 995
    # Data loading code
    traindir = os.path.join(datapath, 'train')
    valdir = os.path.join(datapath, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    def is_valid_file(x, subset, disjoint):
        if disjoint:
            for sub in valid_classes:
                for name in valid_classes[sub]:
                    if name in x:
                        return False
            return True
        else:
            for name in valid_classes[subset]:
                if name in x:
                    return True
            return False
    
    specific_train_datasets = {}
    specific_val_datasets   = {}
    # to do: we might have to make an eval transform
    #    at the moment, validation isn't used
    # MAKE THIS REGULAR RESIZE AND RETEST...
    transform = [transforms.Resize((224,224)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()]
    #TODO: add custom transform specific to individual images...
    if resize:
        transform = transform[:2]
        transform.append(
            transforms.RandomAffine(
                0., 
                translate=None, 
                scale=[1/2.5,1/2.5], # hyperparameter chosen so that bounding box areas are roughly the same now 
                shear=None, 
                resample=False, 
                fillcolor=(0,0,0)
            )
        )
        transform.append(
            transforms.RandomAffine(
                0., 
                translate=[.35,.35], 
                scale=None, 
                shear=None, 
                resample=False, 
                fillcolor=(0,0,0)
            ) 
        )
        transform.append(transforms.ToTensor())
    if normalized:
        transform.append(normalize)
    transform = transforms.Compose(transform)
    reference_areas = {
        'truck': 4127.1974,
        'zebra': 4784.1691
    }
    for key in valid_classes:
        curr_train_dataset = SubsetImageFolder(
                       traindir,
                       valid_classes[key],
                       transform=transform,
                       is_valid_file=lambda x: is_valid_file(x, key, False),
                       boxes = None if (constant or key=='negative') else './datasets/imagenet_' + key +'.csv',
                       reference_area = None if (constant or key=='negative') else reference_areas[key]
                       )
        curr_val_dataset = SubsetImageFolder(
                       valdir,
                       valid_classes[key],
                       transform=transform,
                       is_valid_file=lambda x: is_valid_file(x, key, False)
                       )
        specific_train_datasets[key] = curr_train_dataset
        specific_val_datasets[key]   = curr_val_dataset
    

    rest_train_dataset = SubsetImageFolder(
                       traindir,
                       all_classes,
                       transform=transform,
                       is_valid_file=lambda x: is_valid_file(x, "anything", True)
                    )
    rest_val_dataset = SubsetImageFolder(
                       valdir,
                       all_classes,
                       transform=transform,
                       is_valid_file=lambda x: is_valid_file(x, "anything", True)
                    )

   # rest_train_dataset.class_to_idx = DummyIdx()
   # rest_val_dataset.class_to_idx   = DummyIdx()

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loaders = {}
    val_loaders   = {}

    for key in specific_train_datasets:
        train_loaders[key] = torch.utils.data.DataLoader(
                                specific_train_datasets[key], batch_size=train_batch_size, shuffle=(train_sampler is None),
                                num_workers=workers, pin_memory=False, sampler=train_sampler)
        val_loaders[key] = torch.utils.data.DataLoader(
                                specific_val_datasets[key], batch_size=test_batch_size, shuffle=False,
                                num_workers=workers, pin_memory=False)
    train_loaders["negative"] = torch.utils.data.DataLoader(
                                rest_train_dataset, batch_size=train_batch_size, shuffle=(train_sampler is None),
                                num_workers=workers, pin_memory=False, sampler=train_sampler)
    val_loaders["negative"]   = torch.utils.data.DataLoader(
                                rest_val_dataset, batch_size=test_batch_size, shuffle=False,
                                num_workers=workers, pin_memory=False)

    return train_loaders, val_loaders

def get_split_dataloaders(trainloader, ratio):
    
    trainsplit = SubsetSplit(trainloader.dataset, ratio)
    shuffle = isinstance(trainloader.sampler, torch.utils.data.RandomSampler) 
    train_subset_loader = torch.utils.data.DataLoader(
        trainsplit.subset, batch_size=trainloader.batch_size, shuffle=shuffle,
        num_workers=trainloader.num_workers, pin_memory=True, sampler=None)

    train_remaining_loader = torch.utils.data.DataLoader(
        trainsplit.remaining, batch_size=int(trainloader.batch_size), shuffle=shuffle,
        num_workers=trainloader.num_workers, pin_memory=True, sampler=None)

    return train_subset_loader, train_remaining_loader


def get_data_loader_SceneBAM(seed=0,
                             root='bam_scenes/',
                             ratio=0.5,
                             specific=None,
                             train_batch_size=64, 
                             test_batch_size=64, 
                             workers=16, 
                             distributed=False, 
                             train_shuffle=True, 
                             val_shuffle=False,
                             training=True,
                             train_sampler=None,
                             test_sampler=None,
                             test_idx=None):
    # no dataset leaking
    if isinstance(specific, str):
        train_file = os.path.join('./datasets/SceneBAM/',
                                str(specific),
                                str(seed),
                                'train',
                                str(ratio)+'.pck')
        test_file = os.path.join('./datasets/SceneBAM/',
                                str(specific),
                                str(seed),
                                'test',
                                str(ratio)+'.pck')
    else:
        train_file = os.path.join('./datasets/SceneBAM/',
                                '.'.join(sorted(specific)),
                                str(seed),
                                'train',
                                str(ratio)+'.pck')
        test_file = os.path.join('./datasets/SceneBAM/',
                                '.'.join(sorted(specific)),
                                str(seed),
                                'test',
                                str(ratio)+'.pck')
    try:
        with open(train_file, 'rb') as f:
            train_dataset = pickle.load(f)
        assert train_dataset.verified
    except:
        raise Exception("Please create and verify datasets beforehand")  
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = train_sampler

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=train_shuffle and train_sampler is None,
        num_workers=workers, pin_memory=False, sampler=train_sampler)
    
    # no dataset leaking
    try:
        with open(test_file, 'rb') as f:
            test_dataset = pickle.load(f)
        assert train_dataset.verified
    except:
        raise Exception("Please create and verify datasets beforehand")  

    # figure out how to normalize, this will be done per dataset basis (i.e. given certain ratios...)
    if isinstance(train_dataset.transform,transforms.ToTensor):
        # compute average pixel per channel
        mu = 0
        for X,_,_ in train_loader:
            mu += X.sum(dim=-1).sum(dim=-1).sum(dim=0)
        mu /= (len(train_dataset)*X.shape[-1]*X.shape[-2])
        # compute std pixel per channel
        std = 0
        for X,_,_ in train_loader:
            std += (
                (X - mu.reshape(1,3,1,1))**2
            ).sum(dim=-1).sum(dim=-1).sum(dim=0)
        std /= (len(train_dataset)*X.shape[-1]*X.shape[-2])
        std = torch.sqrt(std)
        normalize = transforms.Normalize(mean=mu.data.numpy().tolist(),
                                         std=std.data.numpy().tolist())
        train_transform = transforms.Compose(
           [transforms.ToTensor(), normalize]
        )
        val_transform = transforms.Compose(
            [transforms.ToTensor(), normalize]
        )
        # just update the transform...
        train_dataset.transform = train_transform
        test_dataset.transform  = val_transform
        train_dataset.normal = [mu.data.numpy().tolist(),
                                std.data.numpy().tolist()]
        with open(train_file, 'wb') as f:
            pickle.dump(train_dataset,f)
        with open(test_file, 'wb') as f:
            pickle.dump(test_dataset,f)
    
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=val_shuffle,
        num_workers=workers, pin_memory=False,
        sampler=test_sampler)

    return train_loader, val_loader

def get_data_loader_coloredMNIST(root='./',
                                 binary_color=True,
                                 single_binary_ratio=0.5, 
                                 train_batch_size=64, 
                                 test_batch_size=64, 
                                 workers=16, 
                                 distributed=False, 
                                 train_shuffle=True, 
                                 val_shuffle=False,
                                 training=True,
                                 train_sampler=None,
                                 test_sampler=None,
                                 test_idx=None):

    if binary_color:
        dataset_type = lambda x: BinaryColorDataset(x, 
            ratio = [single_binary_ratio] + [0.5 for i in range(9)]
        )
    else:
        dataset_type = ColoredDataset
    # no dataset leaking
    try:
        with open('./datasets/colormnist/colormnist_train_'+str(single_binary_ratio)+'.pck', 'rb') as f:
            train_dataset = pickle.load(f)
    except:
        train_dataset = dataset_type(
            datasets.MNIST(
                root,
                train=True,
                transform=transforms.ToTensor(),
                download=True
            )
        )
        with open('./datasets/colormnist/colormnist_train_'+str(single_binary_ratio)+'.pck', 'wb') as f:
            pickle.dump(train_dataset,f)
    train_dataset.idx_to_class = {i: str(i) for i in range(10)}
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = train_sampler

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=train_shuffle and train_sampler is None,
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    # no dataset leaking
    try:
        with open('./datasets/colormnist/colormnist_test_'+str(single_binary_ratio)+'.pck', 'rb') as f:
            test_dataset = pickle.load(f)
    except:
        test_dataset = dataset_type(
            datasets.MNIST(
                root,
                train=False,
                transform=transforms.ToTensor(),
                download=True
            )
        )
        with open('./datasets/colormnist/colormnist_test_'+str(single_binary_ratio)+'.pck', 'wb') as f:
            pickle.dump(test_dataset,f)
    
    test_dataset.idx_to_class = {i: str(i) for i in range(10)}

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=val_shuffle,
        num_workers=workers, pin_memory=True,
        sampler=test_sampler)

    return train_loader, val_loader

def get_data_loader_idenProf(datapath, 
                             train_batch_size=64, 
                             test_batch_size=64, 
                             workers=16, 
                             distributed=False, 
                             train_shuffle=True, 
                             val_shuffle=False,
                             training=True,
                             train_sampler=None,
                             test_sampler=None,
                             test_idx=None,
                             exclusive=False,
                             gender_balance=False):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.5051, 0.4720, 0.4382],
                                     std=[0.3071, 0.3043, 0.3115])
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize])
    train_dataset = IdenProf(
        datapath,
        train=True,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]) if training else val_transform,
        exclusive=exclusive
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    elif gender_balance:
        assert exclusive
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            train_dataset.weights, len(train_dataset.weights)
        )
    else:
        train_sampler = train_sampler

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=train_shuffle and train_sampler is None,
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    if test_idx is not None:
        test_dataset = IdenProf(datapath,
                 train=False,
                 transform=val_transform,
                 exclusive=exclusive).return_subset(test_idx)
    else:
        test_dataset = IdenProf(datapath,
                 train=False,
                 transform=val_transform,
                 exclusive=exclusive)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=val_shuffle,
        num_workers=workers, pin_memory=True,
        sampler=test_sampler)

    return train_loader, val_loader

def get_data_loader_celebA(datapath, 
                           img_dir='img_align_celeba/', 
                           labels='labels/', 
                           train_batch_size=64, 
                           test_batch_size=64, 
                           workers=16, distributed=False, 
                           val_shuffle=False):
    normalize = transforms.Normalize(mean=[0.5063, 0.4258, 0.3832],
                                     std=[0.3107, 0.2904, 0.2897])
    train_dataset = CelebA(datapath,img_dir+'train',labels+'train.txt', 
                                 transform=transforms.Compose([
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,
                                  ])
                     )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        CelebA(datapath,
               img_dir+'eval',
               labels+'eval.txt',
               transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                 ])
                ),
        batch_size=test_batch_size, shuffle=val_shuffle,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader

