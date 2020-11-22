import numpy as np
import torch
import torch.nn as nn
import torchray
from torchray.attribution.common import Probe, get_module
import utils
import os.path
import torch.nn.functional as f
import models
import debias

# just a way to use f.normalize from functional


def normalize():
    return f.normalize


def zero_grad(m):
    if type(m) != nn.Linear or type(m) != nn.Conv2d:
        return
    m.weight.grad = None
    m.bias.grad = None


# recursively go through children from the latest layers to get the last linear transformation layer
def extract_last(module):
    children = list(module.children())
    if len(children) == 0:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            return module
        else:
            return None
    else:
        ans = None
        for i in range(len(children)-1, -1, -1):
            ans = extract_last(children[i])
            if ans is not None:
                break
        return ans


def create_net2vec(model,
                   module_name,
                   n_categories,
                   device,
                   pretrained=False,
                   weights_path=None,
                   initialize=False,
                   example_batch=None,
                   nonlinear=False,
                   partial_projection=False,
                   t=0):
    layer = get_module(model, module_name)
    activation_probe = Probe(layer, 'output')
    extracted = extract_last(layer)
    if extracted is None:
        # i.e. current layer doesn't have children and isn't a linear transform
        _ = model(example_batch.to(device))
        n_neurons = activation_probe.data[0].shape[1]
    else:
        n_neurons = extracted.weight.shape[0]
    if not nonlinear:
        net = nn.Linear(
            n_neurons, n_categories
        )
    else:
        net = models.mlp_(
            in_dim=n_neurons,
            out=n_categories
        )
    # regardless, this should work
    if pretrained:
        assert weights_path is not None
        if os.path.exists(weights_path):
            net.load_state_dict(
                torch.load(
                    weights_path, map_location=lambda storage, loc: storage)
            )
        elif initialize:
            torch.save(net.state_dict(), weights_path)
        else:
            raise Exception(
                "If you want to create a new model, please denote initialize for: " + str(weights_path))

    net.to(device)
    if partial_projection:
        embeddings = net.weight.data[:n_categories-2]
        vg = net.weight.data[-2] - net.weight.data[-1]
        embeddings = debias.partial_orthogonalization(
            embeddings,
            vg,
            t=1e-2
        )
        with torch.no_grad():
            net.weight.data = torch.cat(
                [embeddings,
                    net.weight.data[-2].unsqueeze(0),
                    net.weight.data[-1].unsqueeze(0)
                 ]
            )

    def net2vec(X, forward=False, switch_modes=True):
        # i.e. haven't updated the probe...
        if not forward:
            first_mode = model.training
            if switch_modes:
                model.eval()
                _ = model(X)
                if first_mode:
                    model.train()
            else:
                _ = model(X)
        features = activation_probe.data[0]
        if len(features.shape) == 4:
            # conv output
            features = torch.mean(features, (2, 3), keepdim=True).squeeze()
        elif len(features.shape) == 2:
            # linear output
            pass
        else:
            raise Exception("Neither Linear or Conv module given")
        return net(features)

    return net, net2vec, activation_probe


'''
Assumes the following output structure if multiple is True:

Class 1
Class 2
.
.
.
Class N
Gender Label 1
Gender Label 2
(Repeat Class Specific bias +9 times)
(Repeat Gender Label +9 times)
(Repeat other gender lable +9 times)
'''


def train_net2vec(model,
                  net,
                  net2vec,
                  epochs,
                  trainloader,
                  testloader,
                  device,
                  lr=0.01,
                  save_path='default.pt',
                  train_labels=None,
                  balanced=True,
                  p=0.5,
                  repeat=False,
                  n=None,
                  f=None,
                  multiple=False,
                  specific=None,
                  adam=False,
                  save_best=False,
                  criterion=None,
                  leakage=False):
    if adam:
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = None
    else:
        optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
    results = {
        'train_losses': [],
        'test_losses': [],
        'test_accs': []
    }
    train_losses = results['train_losses']
    test_losses = results['test_losses']
    test_accs = results['test_accs']
    best_acc = -1
    best_state = None
    model.eval()
    best_proj = 2
    best_proj_epoch = -1
    with open('projection_results.txt', 'a') as fp:
        print("Starting: " + str(epochs) + " " + str(lr), file=fp)
    for e in range(epochs):
        tmp_train_loss = []
        tmp_test_loss = []
        tmp_test_acc = []
        # model.train()
        net.train()
        k = 0
        for X, y, genders in trainloader:
            optim.zero_grad()
            if repeat:
                assert n is not None
                labels = utils.repeat_labels(genders[:, 0:1], n, device)
            else:
                labels = utils.merge_labels(y, genders, device)
            logits = net2vec(X.to(device), switch_modes=False)
            if train_labels is not None:
                if logits.shape[1] == labels.shape[1]:
                    logits = logits[:, train_labels]
                labels = labels[:, train_labels]
            if balanced:
                loss, _ = utils.balanced_loss(logits, labels, device, p=p)
            else:
                assert criterion is not None
                loss = criterion(logits, labels)
                if k % 10 == 0:
                    print(loss.item())
            loss.backward()
            tmp_train_loss.append(loss.item())
            optim.step()
            k += 1
        train_losses.append(np.mean(tmp_train_loss))
        model.eval()
        net.eval()
        with torch.no_grad():
            tmp_test_acc, (tmp_test_f1, tmp_test_mAP) = utils.net2vec_accuracy(
                testloader,
                net2vec,
                device,
                train_labels,
                repeat,
                n,
                leakage=leakage
            )
            if leakage:
                tmp_test_acc = 0.5 + abs(tmp_test_acc - 0.5)
        if np.max(tmp_test_acc) > best_acc:
            best_acc = np.max(tmp_test_acc)
            best_state = net.state_dict()
        # if scheduler is not None:
        #    scheduler.step(np.mean(tmp_test_acc))
        test_accs.append(tmp_test_acc)
        print("Epoch", e, " :", tmp_test_acc, "f1/mAP:",
              tmp_test_f1, "/", tmp_test_mAP, file=f)
        if isinstance(net, nn.Linear):
            W = net.weight.data
            vg = W[-2] - W[-1]
            vg = vg / vg.norm()
            normalized_W = normalize()(W, p=2, dim=1)
            mean_proj = (normalized_W @ vg.reshape(-1, 1)).mean().item()
            var_proj = (((normalized_W @ vg.reshape(-1, 1)) -
                         mean_proj) ** 2).sum().item() / (W.shape[0] - 1)
            proj = (
                mean_proj,
                ((W[0] / W[0].norm()).reshape(1, -1) @ vg.reshape(-1, 1)).item(),
                var_proj
            )
            print("projection:", proj, " |", save_path)
            with open('projection_results.txt', 'a') as fp:
                print("projection:", proj, " |", save_path, file=fp)
            if abs(proj[0]) < abs(best_proj):
                best_proj = proj[0]
                best_proj_epoch = e
        if e % 5 == 0 and e > 0:
            pass#torch.save(net.state_dict(), save_path.split('.pt')[
                       #0] + '_EPOCH_{}_'.format(e) + str(proj[0]) + '_' + str(lr) + '_var_{}.pt'.format(proj[2]))
    with open('projection_results.txt', 'a') as fp:
        print("", file=fp)
    if save_best:
        torch.save(best_state, save_path.split('.pt')[
                   0] + '_' + str(best_acc) + '_' + str(lr) + '.pt')
    else:
        torch.save(net.state_dict(), save_path.split('.pt')[0] + '{}_{}_'.format(
            best_proj, best_proj_epoch) + str(proj[0]) + '_' + str(lr) + '_var_{}.pt'.format(proj[2]))
        torch.save(net.state_dict(), save_path)
    return results

# compute directional derivatives for each output class given a concept vector
#   returns a (BS x N classes) tensor


def compute_directional_derivatives(X,
                                    concept_vector,
                                    model,
                                    activation_probe,
                                    device,
                                    softmax=False):
    if softmax:
        logits = nn.Softmax(dim=1)(model(X))
    else:
        logits = model(X)
    S = torch.zeros(
        X.shape[0],
        logits.shape[1]
    ).to(device)
    for i in range(logits.shape[1]):
        model.zero_grad()
        logits[:, i].backward(
            torch.ones(
                logits.shape[0]
            ).to(device), retain_graph=True
        )
        grads = activation_probe.data[0].grad
        if len(grads.shape) == 4:
            grads = torch.mean(
                grads,
                (2, 3),
                keepdim=True
            ).squeeze()
        elif len(grads.shape) == 2:
            pass
        else:
            raise Exception("Should be either a linear or conv layer")
        S[:, i] = f.normalize(grads, p=2, dim=1) @ f.normalize(
            concept_vector, p=2, dim=0)
    return S
