import alexnet
import lenet
import torchvision.models
import torch
import torch.nn as nn
import net2vec
import os.path
from torchray.attribution.common import Probe, get_module
import balanced_models

def update_probe(probed_model):
    module = get_module(probed_model.model, probed_model.module)
    probed_model.probe = Probe(module, target="output")

class ProbedModel(nn.Module):
    
    def __init__(self, model, net, module, switch_modes=True):
        super(ProbedModel,self).__init__()
        self.model  = model
        self.net    = net
        self.module = module
        self.switch_modes = switch_modes
        # recreate a new probe
        update_probe(self)
        '''
        def update_probe(new_self):
            new_module = get_module(new_self.model, new_self.module)
            new_self.probe = Probe(new_module, target="output")
        self.update_probe = update_probe # just so higher can use it
        self.update_probe(self)
        '''
    def forward(self, X):
        first_mode = self.model.training
        if self.switch_modes:
            self.model.eval()
            _ = self.model(X)
            if first_mode:
                self.model.train()
        else:
            _ = self.model(X)
        features = self.probe.data[0]
        if len(features.shape) == 4:
            features = torch.mean(features, (2,3), keepdim=True).squeeze()
        else:
            assert len(features.shape) == 2
        return self.net(features)

def load_models(device,
                model_loader,
                model_path=None,
                net2vec_pretrained=True,
                net2vec_path=None,
                module="conv5",
                num_attributes=12,
                model_init=False,
                n2v_init=False,
                loader=None,
                nonlinear=False,
                partial_projection=False,
                t=0,
                parallel=False,
                gpu_ids=None,
                ignore_net=False):
    print(model_loader)
    if model_loader is not None:
        model            = model_loader(model_path, device, model_init)
    else:
        class DummyArgs:
            num_object = 79
            finetune=False
            layer='generated_image'
            autoencoder_finetune=True
            finetune=True
        model = balanced_models.ObjectMultiLabelAdv(DummyArgs(), 79, 300, True, 1)
        ok    = torch.load('model_best.pth.tar', encoding='bytes')
        state_dict = {key.decode("utf-8"):ok[b'state_dict'][key] for key in ok[b'state_dict']}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        if module != 'fc':
            module = 'base_network.' + module
        else:
            module = 'finalLayer'
    if parallel:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    example_batch = None
    if loader is not None:
        example_batch = loader.dataset[0][0]
        if len(example_batch.shape) == 3:
            example_batch = example_batch.unsqueeze(0)
    if ignore_net:
        return model, None, None, None
    net, net_forward, activation_probe = net2vec.create_net2vec(
                                model,
                                module,
                                num_attributes,
                                device,
                                pretrained=net2vec_pretrained,
                                weights_path=net2vec_path,
                                initialize=n2v_init,
                                example_batch=example_batch,
                                nonlinear=nonlinear,
                                partial_projection=partial_projection,
                                t=t
                        )
    return model, net, net_forward, activation_probe

def mlp_(in_dim,
         out,
         hid_size=300):
    net = nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.Linear(in_dim,hid_size),
        nn.BatchNorm1d(hid_size),
        nn.LeakyReLU(),
        nn.Linear(hid_size,hid_size),
        nn.BatchNorm1d(hid_size),
        nn.LeakyReLU(),
        nn.Linear(hid_size,hid_size),
        nn.BatchNorm1d(hid_size),
        nn.LeakyReLU(),
        nn.Linear(hid_size,out)
    )
    return net

def resnet_(pretrained=True,
            custom_path=None,
            num_classes=10,
            device='cpu',
            initialize=False,
            size=50,
            linear_only=False):
    if size == 50:
        model = torchvision.models.resnet50(
                pretrained=pretrained
        )
    elif size == 34:
        model = torchvision.models.resnet34(
                pretrained=pretrained
        )
    model.fc = nn.Linear(model.fc.in_features, num_classes) # reshape last layer
    if linear_only:
        for param in model:
            param.requires_grad = False
        for param in model.fc:
            param.requires_grad = True

    model.to(device)
    if custom_path is not None:
        if os.path.exists(custom_path):
            print("found!")
            model.load_state_dict(
                torch.load(custom_path, map_location=device)
            )
        elif initialize:
            torch.save(model.state_dict(), custom_path)
        else:
            print(custom_path)
            raise Exception("If you want to create a new model, please denote initialize")
    return model

def alexnet_(pretrained=True,
             custom_path=None,
             num_classes=10,
             device='cpu',
             initialize=False):
    model = torchvision.models.alexnet(pretrained=pretrained and custom_path is None)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    model = alexnet.AlexNetFeatureExtractor(model)
    if custom_path is not None:
        if os.path.exists(custom_path):
            model.load_state_dict(
                torch.load(custom_path, map_location=device)
            )
        elif initialize:
            model = torchvision.models.alexnet(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(4096, num_classes)
            model = alexnet.AlexNetFeatureExtractor(model)
            torch.save(model.state_dict(), custom_path)
        else:
            raise Exception("If you want to create a new model, please denote initialize")
    return model.to(device)

def lenet_(pretrained=True,
           custom_path=None,
           num_classes=10,
           in_channels=3,
           device='cpu',
           initialize=False):
    model = lenet.LeNetFeatureExtractor(in_channels=3, out_dims=num_classes)
    if pretrained:
        assert custom_path is not None
        if os.path.exists(custom_path):
            model.load_state_dict(
                torch.load(custom_path, map_location=device)
            )
        elif initialize:
            torch.save(model.state_dict(), custom_path)
        else:
            raise Exception("If you want to create a new model, please denote initialize")
    return model.to(device)
