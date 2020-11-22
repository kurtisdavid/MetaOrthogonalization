import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import os
import gc
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
import torch.nn.functional as F

def view_tensors():
    print("VIEWING CURRENT TENSORS")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
def one_hot(labels, num_class = 10):
    one_hot = torch.FloatTensor(labels.shape[0], num_class)
    one_hot.zero_()
    one_hot.scatter_(1, labels.view(-1,1), 1)
    return one_hot

def repeat_labels(labels, n, device):
    return labels.repeat(1, n).to(device)

def merge_labels(y, extras, device):
    if len(y.shape) < 2:
        y = one_hot(y)
    extras = extras.type(torch.FloatTensor).reshape(extras.shape[0],-1)
    return torch.cat([y, extras], dim=1).to(device)

def logits_to_pred(logits, device):
    pred = torch.ones(logits.shape).to(device)
    pred[logits < 0] = 0.
    return pred

def balanced_loss(logits, labels, device, p=0.5, ids = None, stable=False, multiple=False, specific=None, shapes=[]):
    # loop through every binary classifier
    total_loss = 0
    n = 0
    losses = []
    if ids is None:
        if not multiple:  
            ids = [None for _ in range(logits.shape[1])]
        else:
            ids = [None for _ in range(labels.shape[1])]
    end = logits.shape[1]
    for i in range(end):
        curr_logits = None
        if not multiple or i < 12:
            curr_ids = ids[i]
            curr_logits, curr_labels, new_ids = random_logits_from_labels(
                logits[:, i].reshape(-1,1),
                labels[:, i].reshape(-1,1),
                device,
                p,
                curr_ids
            )
            # if curr_labels is not None:
            #     shapes.append(curr_labels.shape)
            # else:
            #     shapes.append(None)
            ids[i] = new_ids
        else:
            # class specific...
            if i >= 12 and i<21:
                curr_ids = ids[specific]
                curr_logits, curr_labels, new_ids = random_logits_from_labels(
                    logits[:,i].reshape(-1,1),
                    labels[:,specific].reshape(-1,1),
                    device,
                    p,
                    curr_ids
                )
                ids[specific] = new_ids
            elif i >=21 and i < 30:
                curr_ids = ids[-2]  
                curr_logits, curr_labels, new_ids = random_logits_from_labels(
                    logits[:,i].reshape(-1,1),
                    labels[:,-2].reshape(-1,1),
                    device,
                    p,
                    curr_ids
                )
                ids[-2] = new_ids
            elif i >= 30 and i < 39:
                curr_ids = ids[-1]  
                curr_logits, curr_labels, new_ids = random_logits_from_labels(
                    logits[:,i].reshape(-1,1),
                    labels[:,-1].reshape(-1,1),
                    device,
                    p,
                    curr_ids
                )
                ids[-1] = new_ids
        if curr_logits is not None:
            n += 1
            if not stable:
                losses.append( (i,nn.BCEWithLogitsLoss()(curr_logits, curr_labels.reshape(-1,1)), curr_logits, curr_labels) )
                shapes.append((i, losses[-1][1].item(), curr_labels.shape))
                total_loss += losses[-1][1] 
                '''
                if not multiple:
                    total_loss += nn.BCEWithLogitsLoss()(curr_logits, curr_labels.reshape(-1,1))
                else:
                    if i == specific or (i>=10):
                        total_loss += nn.BCEWithLogitsLoss()(curr_logits, curr_labels.reshape(-1,1)) / 10
                    else:
                        total_loss += nn.BCEWithLogitsLoss()(curr_logits, curr_labels.reshape(-1,1))
                '''
            else:
                sigmoid_out = nn.Sigmoid()(curr_logits)
                total_loss += nn.BCELoss()(sigmoid_out+1e-12,curr_labels.reshape(-1,1))
    return total_loss, ids

def random_logits_from_labels(logits,
                              labels,
                              device,
                              p=0.5,
                              ids=None):
    assert p > 0, "Non-zero probability must be passed in"
    with_label = logits[labels==1].reshape(-1,1)
    with_idx   = (
        (labels.squeeze())==1
    ).nonzero().squeeze()
    without_label = logits[labels==0].reshape(-1,1)
    without_idx = (
        (labels.squeeze())==0
    ).nonzero().squeeze()
    if ids is None: 
        if with_label.shape[0] == 0:
            # bad, so don't update
            return None, None, ids
        # if positives are less than desired
        elif float(with_label.shape[0]) / logits.shape[0] < p:
            num_needed = int(with_label.shape[0] / p) - with_label.shape[0]
            idx = list(random.sample(range(without_label.shape[0]), num_needed))
            final_logits = torch.zeros((with_label.shape[0] + num_needed, logits.shape[1])).to(device)
            final_logits[:with_label.shape[0], :] = with_label
            final_logits[with_label.shape[0]:, :] = without_label[idx]
            # create updated labels
            final_labels = torch.zeros(
                (with_label.shape[0] + num_needed)
            ).to(device)
            final_labels[:with_label.shape[0]] = 1
            # create new ids
            ids = torch.zeros(
                final_logits.shape[0]
            ).type(torch.LongTensor)
            ids[:with_label.shape[0]] = with_idx
            ids[with_label.shape[0]:] = without_idx[idx]
            return final_logits, final_labels, ids
        elif float(with_label.shape[0]) / logits.shape[0] > p:
            # when more positive examples exist, sample from those instead
            num_needed = int(without_label.shape[0] / (1-p)) - without_label.shape[0]
            idx = list(random.sample(range(with_label.shape[0]), num_needed))
            final_logits = torch.zeros((without_label.shape[0] + num_needed, logits.shape[1])).to(device)
            final_logits[:without_label.shape[0], :] = without_label
            final_logits[without_label.shape[0]:, :] = with_label[idx]
            # create updated labels
            final_labels = torch.zeros(
                (without_label.shape[0] + num_needed)
            ).to(device)
            final_labels[without_label.shape[0]:] = 1
            # create new ids
            ids = torch.zeros(
                final_logits.shape[0]
            ).type(torch.LongTensor)
            ids[:without_label.shape[0]] = without_idx
            ids[without_label.shape[0]:] = with_idx[idx]
            return final_logits, final_labels, ids
        else:
            # no need to modify
            # update ids just in case
            ids = list(range(labels.shape[0]))
            return logits, labels, ids
    else:
        # if we already know which ids to use, just use those
        final_logits = logits[ids].to(device)
        final_labels = labels[ids].to(device)
        return final_logits, final_labels, ids

def net2vec_accuracy(dataloader, inference_fn, device, train_labels=None,
                     repeat=False, n=None, balanced=True, multiple=False, specific=None,
                     leakage=False):
    test_acc = []
    #test_loss = []
    res = []
    for X,y,gender in dataloader:
        if repeat:
            assert n is not None
            labels = repeat_labels(gender[:,0:1],n,device)
        else:
            labels = merge_labels(y, gender, device)
        logits = inference_fn(X.to(device))
        if train_labels is not None:
            if logits.shape[1] == labels.shape[1]:
                logits = logits[:, train_labels]
            labels = labels[:, train_labels]
        logits = logits.reshape(X.shape[0],-1)
        prob   = torch.sigmoid(logits)
        softmax_pred = torch.max(
            logits, dim=1
        )[1]
        res.append((prob.data.cpu(), y.data.cpu(), gender.data.cpu(), softmax_pred.data.cpu()))
        pred   = logits_to_pred(logits, device)
        test_acc.append(
            (pred == labels).type(
                torch.FloatTensor
            ).sum(dim=0).data.cpu().numpy().reshape(1,-1)
        )
#        print(labels.shape)
        # test_loss.append(balanced_loss(logits, labels, device, p = 0.5, multiple=multiple,specific=specific)[0].item())
    test_acc = np.concatenate(test_acc).sum(axis=0) / len(dataloader.dataset)

    total_preds   = torch.cat([entry[0] for entry in res], 0)
    total_targets = torch.cat([entry[1] for entry in res], 0)
    total_genders = torch.cat([entry[2] for entry in res], 0)
    total_pred_genders = torch.cat([entry[3] for entry in res], 0)

    # sometimes dataloader will have 0 examples of an object, this filters that out
    valid = []
    max_  = total_targets.shape[1] if train_labels is None else len(train_labels)
    for i in range(max_):
        if (total_targets[:,i] > 0).sum().item() > 0:
            valid.append(i)
    
    total_preds   = total_preds[:, valid]
    total_targets = total_targets[:, valid]

    task_f1_score = f1_score(total_targets.long().numpy(), (total_preds >= 0.5).long().numpy(), average = 'macro')
    meanAP = average_precision_score(total_targets.long().numpy(), total_preds.numpy(), average='macro')

    if leakage:
        test_acc = accuracy_score(
            total_targets.long().numpy()[:,1], total_pred_genders.long().numpy()
        )

    return test_acc, (task_f1_score, meanAP)

def precision_recall(dataloader, inference_fn, device):
    tp = None
    fp = None
    fn = None
    for X,y,gender in dataloader:
        labels = merge_labels(y, gender, device)
        logits = inference_fn(X.to(device))
        pred = logits_to_pred(logits, device)
        for i in range(labels.shape[1]):
            if tp is None:
                tp = np.zeros(
                    (labels.shape[1])
                )
                fp = np.zeros(
                    (labels.shape[1])
                )
                fn = np.zeros(
                    (labels.shape[1])
                )
            
            with_label    = labels[:,i] == 1
            without_label = labels[:,i] == 0
            # true positives ( of the ones that are actually 1, which ones did we say 1 )
            pred_with_label = pred[with_label, i]
            tp[i] += pred_with_label[pred_with_label == 1].shape[0]

            # false positives ( of the ones that are actually 0, which ones did we say 1 )
            # extract predictions of network on examples without label
            pred_without_label = pred[without_label, i]
            fp[i] += pred_without_label[pred_without_label == 1].shape[0]
            
            # false negatives (of the ones that are actually 1, which ones did we sat 0) 
            # extract predictions of network on examples with label
            pred_with_label = pred[with_label, i]
            fn[i] += pred_with_label[pred_with_label == 0].shape[0]
    return tp, fp, fn, (tp / (tp + fp)), (tp / (tp + fn))

# modified from https://github.com/uvavision/Balanced-Datasets-Are-Not-Enough/blob/master/object_multilabel/train.py
def detection_results(dataloader, model, device):
    res = []
    model.eval()
    for images, targets, genders in dataloader:
        # Set mini-batch dataset
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images)
        preds = torch.sigmoid(preds)
        res.append((preds.data.cpu(), targets.data.cpu(), genders))
    
    total_preds   = torch.cat([entry[0] for entry in res], 0)
    total_targets = torch.cat([entry[1] for entry in res], 0)
    total_genders = torch.cat([entry[2] for entry in res], 0)

    # sometimes dataloader will have 0 examples of an object, this filters that out
    valid = []
    for i in range(total_targets.shape[1]):
        if (total_targets[:,i] > 0).sum().item() > 0:
            valid.append(i)
    
    total_preds   = total_preds[:, valid]
    total_targets = total_targets[:, valid]

    task_f1_score = f1_score(total_targets.long().numpy(), (total_preds >= 0.5).long().numpy(), average = 'macro')
    meanAP = average_precision_score(total_targets.long().numpy(), total_preds.numpy(), average='macro')
    return task_f1_score, meanAP

def detection_disparity(dataloader, model, device):
    test_pos_accs = [
        [] for _ in range(79)
    ]
    pos_cts = [
        0 for _ in range(79)
    ]
    test_neg_accs = [
        [] for _ in range(79)
    ]
    neg_cts = [
        0 for _ in range(79)
    ]
    for X,y_orig,gender_orig in dataloader:
        logits_orig = model(X.to(device))
        for specific in range(len(test_pos_accs)):
            logits = logits_orig[:, specific]
            y      = y_orig[:, specific]
            
            test_pos_acc = test_pos_accs[specific]
            test_neg_acc = test_neg_accs[specific]

            mask = (y==1)
            logits = logits[mask]
            gender = gender_orig[mask, ...]
            y      = y[mask] 
            if logits.shape[0] == 0:
                continue
            pred = torch.sigmoid(
                logits
            )
            pred[pred>=0.5] = 1
            pred[pred<0.5]  = 0
            # this is actually equivalent to checking if they are all 1, but kept it in original form
            test_pos_acc.append(
                (pred[gender[:,0] == 1] == y[gender[:,0] == 1].type(torch.FloatTensor).to(device)).sum().item() 
            )
            pos_cts[specific] += y[gender[:,0]==1].shape[0]

            test_neg_acc.append(
                (pred[gender[:,1] == 1] == y[gender[:,1] == 1].type(torch.FloatTensor).to(device)).sum().item() 
            )
            neg_cts[specific] += y[gender[:,1]==1].shape[0]
    total_ans = []
    for specific in range(len(test_pos_accs)):
        test_pos_acc = test_pos_accs[specific]
        test_neg_acc = test_neg_accs[specific]
        pos_ct = pos_cts[specific]
        neg_ct = neg_cts[specific]

        ans = (
            (np.sum(test_pos_acc) + np.sum(test_neg_acc)) / (pos_ct + neg_ct), 
            np.sum(test_pos_acc)/pos_ct if pos_ct > 0 else 0, # Pr(Zy = 1 | A = 0, Zy = 1) 
            np.sum(test_neg_acc)/neg_ct if neg_ct > 0 else 0, # Pr(Zy = 1 | A = 1, Zy = 1)
            pos_ct,
            neg_ct
        ) # disparity will be | ans[1] - ans[2] |
        total_ans.append(ans)
    true_res = detection_results(dataloader, model, device) 
    return total_ans, true_res
    

def classification_accuracy(dataloader, model, device):
    test_acc = []
    test_loss = []
    model.eval()
    for X,y,gender in dataloader:
        logits = model(X.to(device))
        pred = torch.max(
            logits,
            dim=1
        )[1]
        print("pred:", pred.shape)
        test_acc.append( 
            ((pred == y.type(torch.FloatTensor).to(device)).sum()).item()
        )
        test_loss.append(
            nn.CrossEntropyLoss()(
                logits,
                y.to(device)
            ).item()
        )
    return np.sum(test_acc)/len(dataloader.dataset), np.mean(test_loss)

def classification_specific_accuracy(dataloader, model, device):
    test_pos_accs = [
        [] for _ in range(10)
    ]
    pos_cts = [
        0 for _ in range(10)
    ]
    test_neg_accs = [
        [] for _ in range(10)
    ]
    neg_cts = [
        0 for _ in range(10)
    ]
    for X,y_orig,gender_orig in dataloader:
        logits_orig = model(X.to(device))
        for specific in range(len(test_pos_accs)):
            test_pos_acc = test_pos_accs[specific]
            test_neg_acc = test_neg_accs[specific]

            mask = (y_orig==specific)
            logits = logits_orig[mask]
            gender = gender_orig[mask]
            y      = y_orig[mask]
            
            if logits.shape[0] == 0:
                continue
            pred = torch.max(
                logits,
                dim=1
            )[1]

            test_pos_acc.append(
                (pred[gender[:,0] == 1] == y[gender[:,0] == 1].type(torch.FloatTensor).to(device)).sum().item() 
            )
            pos_cts[specific] += y[gender[:,0]==1].shape[0]

            test_neg_acc.append(
                (pred[gender[:,1] == 1] == y[gender[:,1] == 1].type(torch.FloatTensor).to(device)).sum().item() 
            )
            neg_cts[specific] += y[gender[:,1]==1].shape[0]
    total_ans = []
    overall_acc = 0
    ct = 0
    for specific in range(len(test_pos_accs)):
        test_pos_acc = test_pos_accs[specific]
        test_neg_acc = test_neg_accs[specific]
        pos_ct = pos_cts[specific]
        neg_ct = neg_cts[specific]

        ans = (
            (np.sum(test_pos_acc) + np.sum(test_neg_acc)) / (pos_ct + neg_ct), 
            np.sum(test_pos_acc)/pos_ct if pos_ct > 0 else 0, 
            np.sum(test_neg_acc)/neg_ct if neg_ct > 0 else 0,
            pos_ct,
            neg_ct
        )
        overall_acc += ans[0] * (pos_ct + neg_ct)
        ct += pos_ct + neg_ct
        total_ans.append(ans)
    return total_ans, overall_acc / ct 

# from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/4
def unnormalize(x, 
                mean = [0.5051, 0.4720, 0.4382],
                std = [0.3071, 0.3043, 0.3115]):
    x = x * torch.Tensor(std).reshape(1,3,1,1)
    x = x + torch.Tensor(mean).reshape(1,3,1,1)
    return x

def view_image(x, mean, std, idx=0):
    x = unnormalize(x, mean, std)
    plt.imshow(
        np.moveaxis(
            x[idx].data.cpu().numpy(),
            0,-1
        )
    )

def get_paths(base_folder,
              seed,
              specific,
              experiment='post_train',
              model_end="default.pt",
              n2v_end="default.pt",
              n2v_module=None,
              with_n2v=False):
    if not isinstance(specific, str) and specific is not None:
        specific = '.'.join(sorted(specific))
    model_path = get_model_path(base_folder, seed, specific, model_end, experiment, with_n2v, n2v_module)
    n2v_path   = get_net2vec_path(base_folder, seed, specific, n2v_module, n2v_end, experiment)
    return model_path, n2v_path

def get_model_path(base_folder,
                   seed,
                   specific,
                   end,
                   experiment='post_train',
                   with_n2v=False,
                   n2v_module=None):
    if not isinstance(specific, str) and specific is not None:
        specific = '.'.join(sorted(specific))
    if specific is not None:
        if not with_n2v:
            return os.path.join(base_folder,str(seed),str(specific),str(experiment),end)
        else:
            assert n2v_module is not None
            return os.path.join(base_folder,str(seed),str(specific),str(experiment),str(n2v_module),end)
    else:
        if not with_n2v:
            return os.path.join(base_folder,str(seed),str(experiment),end)
        else:
            assert n2v_module is not None
            return os.path.join(base_folder,str(seed),str(experiment),str(n2v_module),end)


def get_net2vec_path(base_folder,
                     seed,
                     specific,
                     module,
                     end,
                     experiment='post_train'):
    if not isinstance(specific, str) and specific is not None:
        specific = '.'.join(sorted(specific))
    if specific is not None:
        return os.path.join(base_folder,str(seed),str(specific),str(experiment),str(module),end)
    else:
        return os.path.join(base_folder,str(seed),str(experiment),str(module),end)

