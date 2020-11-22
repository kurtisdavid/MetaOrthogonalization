import dataload
import torch
import torch.nn
import torchvision
import pickle
import os
import numpy as np
'''
directory structure:
-SceneBAM
    -Specific_Class
        -Seed0
            -Train
                -Ratio1
                -Ratio2
            -Test
                -Ratio1
                -Ratio2
        -Seed1
    -None
'''
def setupBAM(seed,
             train=True,
             specific=None,
             force=False,
             ratios=["1.0","0.9","0.8","0.75","0.7","0.6","0.5","0.4","0.3","0.25","0.2","0.1","0.0"]):
    if isinstance(specific, str):
        if not os.path.isdir('./datasets/SceneBAM/'+str(specific)):
            os.mkdir('./datasets/SceneBAM/'+str(specific))
        if not os.path.isdir('./datasets/SceneBAM/'+str(specific) + '/' + str(seed)):
            os.mkdir('./datasets/SceneBAM/'+str(specific) + '/' + str(seed))

        filename = './datasets/SceneBAM/' + str(specific) + '/' + str(seed)
    else:
        specific = tuple(sorted(specific))
        folder_name = '.'.join(specific)

        if not os.path.isdir('./datasets/SceneBAM/'+str(folder_name)):
            os.mkdir('./datasets/SceneBAM/'+str(folder_name))
        if not os.path.isdir('./datasets/SceneBAM/'+str(folder_name) + '/' + str(seed)):
            os.mkdir('./datasets/SceneBAM/'+str(folder_name) + '/' + str(seed))
        filename = './datasets/SceneBAM/' + str(folder_name) + '/' + str(seed)
    
    if train:
        filename += '/train/'
    else:
        filename += '/test/'
    if not os.path.isdir(filename[:-1]):
        os.mkdir(filename[:-1])
    last = None
    # so that we know when to pass last and next
    exists = [False for _ in range(len(ratios))]
    for idx, i in enumerate(ratios):
        curr_file = filename + str(i) + '.pck'
        if os.path.exists(curr_file):
            exists[idx] = True

    for idx,i in enumerate(ratios):
        curr_file = filename + str(i) + '.pck'
        next = None
        # check if the next file exists, since we want the following relation:
        # next <= curr <= last
        if (idx + 1) < len(exists) and exists[idx+1]:
            next_file = filename + str(ratios[idx+1]) + '.pck'
            with open(next_file,'rb') as f_next:
                next = pickle.load(f_next)
            if next.transform is None:
                next.transform = torchvision.transforms.ToTensor()
        if os.path.exists(curr_file) and not force:
            with open(curr_file,'rb') as f:
                curr = pickle.load(f)
            if curr.transform is None:
                curr.transform = torchvision.transforms.ToTensor()
        else:
            curr = dataload.SceneBAM(train=train, 
                                    ratio=float(i), 
                                    specific=specific,
                                    last=last,
                                    next=next)
        # should pass assertions...
        curr.verify(last, next)
        curr.verified = True # we will only use these datasets for dataloaders
        with open(curr_file, 'wb') as f:
            pickle.dump(curr, f)
        last = curr