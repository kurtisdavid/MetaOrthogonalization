import torch
import torch.nn as nn
import torchray
from torchray.attribution.common import get_module, Probe
import utils
import numpy as np
import torch.nn.functional as F

def extract_max(model, dataloader, module_name, device,
                k=10):
    layer = get_module(model, module_name)
    activation_probe = Probe(layer, 'output')
   
    activations = np.zeros(
        (len(dataloader.dataset), layer[0].weight.shape[0])
    )
    last = 0
    for X,y,gender in dataloader:
        _ = model(X.to(device))
        features = activation_probe.data[0]
        # global max pooling
        features = torch.max(
            torch.max(features, 
                      dim=-1, 
                      keepdim=True
            )[0].squeeze(),
            dim=-1,
            keepdim=True
        )[0].squeeze()
        features = features.data.cpu().numpy()
        activations[last:last+X.shape[0],:] = features
        last += X.shape[0]
   
    activation_probe.remove()
    return np.argsort(
        -activations, axis=0
    )[:k,:], activations.max(axis=0)


def receptive_field(model, dataloader, module_name, device,
                    max_values,
                    tau=0.2,
                    unit=0):
    layer = get_module(model, module_name)
    activation_probe = Probe(layer, 'output')
    
    collect = []
    for X,_,_ in dataloader:
        _ = model(X.to(device))
        # should only have spatial information now
        features = activation_probe.data[0][:, unit, :, :].squeeze()
        print(torch.max(
            torch.max(features, 
                      dim=-1, 
                      keepdim=True
            )[0].squeeze(),
            dim=-1,
            keepdim=True
        )[0].squeeze()
        )
        # run segmentation mask
        features[features/max_values[unit] < tau] = 0
        features[features/max_values[unit] >= tau] = 1
        features = F.interpolate(
            features.unsqueeze(1), (X.shape[-2],X.shape[-1]), mode="bilinear",align_corners=False)
        features = utils.unnormalize(X).to(device) * features
        collect.append(
            features.data.cpu().numpy()
        )
    return collect
