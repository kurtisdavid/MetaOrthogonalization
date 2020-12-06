import torch
import torchray
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import net2vec
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import utils
import seaborn as sns
import models
import torch.nn.functional as f
#################### TCAV CODE ###########################

def compare_tcavs(dataloader,
                  model1,
                  net1,
                  probe1,
                  title1,
                  model2,
                  net2,
                  probe2,
                  title2,
                  device):
    collect1, labels1 = compute_tcavs(
        dataloader,
        model1,
        probe1,
        net1.weight[-2]-net1.weight[-1],
        device
    )
    collect2, labels2 = compute_tcavs(
        dataloader,
        model2,
        probe2,
        net2.weight[-2]-net2.weight[-1],
        device
    )

    
    for i in range(10):
        plt.figure(i, figsize=(10,10))
        plt.title(dataloader.dataset.idx_to_class[i])
        sns.distplot(collect1[labels1==i],
                 hist=False,
                 label=title1
        )
        sns.distplot(collect2[labels2==i],
             hist=False,
             label=title2,
             kde_kws={'linestyle':'--'}
        )



def compute_tcavs(dataloader,
                  model,
                  probe,
                  vg,
                  device,
                  softmax=False):
    model.eval()
    collect = np.zeros(
        len(dataloader.dataset)
    )
    labels = np.zeros(
        len(dataloader.dataset)
    ).astype(np.int)
    last = 0
    for X,y,gender in dataloader:
        S = net2vec.compute_directional_derivatives(
            X.to(device),
            vg,
            model,
            probe,
            device,
            softmax=softmax
        )
        S = torch.gather(
            S, 1, y.to(device).unsqueeze(-1)
        )
        collect[last:last+S.shape[0]] = S.data.cpu().numpy().squeeze()
        labels[last:last+y.shape[0]] = y.data.cpu().numpy().squeeze()
        last += S.shape[0]
    return collect, labels

################## PROJECTION CODE #######################

def compare_projections(net1, net2, idx_to_class):
    p1 = compute_projections(net1)
    p2 = compute_projections(net2)

    plt.figure(figsize=(8,8))
    plot_projections(p1, idx_to_class, marker='o')
    plot_projections(p2, idx_to_class, marker='x')
    plt.legend()
    plt.show()


def compute_projections(net):
    weights = net.weight.data.cpu()
    embeddings = f.normalize(weights[:-2],p=2,dim=1).data.cpu().numpy()
    weights = weights.numpy()
    v_g = weights[-2] - weights[-1]
    v_g = v_g / np.linalg.norm(v_g)
    return embeddings @ v_g

def plot_projections(projections, idx_to_class, marker='o'):
    for j in range(projections.shape[0]):
        plt.scatter(
            projections[j],
            np.linspace(-1,1,10)[j],
            marker=marker,
            label=idx_to_class[j]
        )
    plt.plot(
        np.zeros(10),
        np.linspace(-1,1,10),
        '--'
    )
#    plt.xlim(-2,2)

################## Net2Vec Classification ROC #####################

def net2vec_roc_data(dataloader,
                     net_forward,
                     device):
    all_proba  = None
    all_labels = None
    curr = 0
    for X,y,gender in dataloader:
        proba = nn.Sigmoid()(net_forward(X.to(device))).data.cpu().numpy()
        labels = utils.merge_labels(y, gender, device).data.cpu().numpy()
        if all_proba is None:
            all_proba = np.zeros(
                (len(dataloader.dataset),proba.shape[1])
            )
            all_labels = np.zeros(
                (len(dataloader.dataset),proba.shape[1])
            )
        all_proba[curr:curr+X.shape[0],:]  = proba
        all_labels[curr:curr+X.shape[0],:] = labels
        curr += X.shape[0]
    return all_proba, all_labels

def net2vec_roc_curves(dataloader,
                       net_forward,
                       device,
                       title="Default"):
    proba_, labels = net2vec_roc_data(dataloader, net_forward, device)
    idx_to_class = dataloader.dataset.idx_to_class
    fig, axarr = plt.subplots(4, 3, figsize=(20,30))
    fig.suptitle(title, fontsize=32)
    for i in range(proba_.shape[1]):
        r = i // 3
        c = i % 3
        fpr, tpr, _ = roc_curve(labels[:,i],proba_[:,i])
        auc_ = auc(fpr, tpr)
        axarr[r,c].plot(fpr, tpr, label='%s Eval ROC (AUC = %0.2f )' % (idx_to_class[i], auc_), 
                        color='steelblue')
        axarr[r,c].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                        label='Chance', alpha=.8)
        axarr[r,c].legend(fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show() 
         

################## Classification ROC CODE #######################
def compute_roc_data(dataloader, 
                     model,
                     device):
    female_y = []
    male_y   = []
    ys = (female_y,male_y)

    female_p = []
    male_p   = []
    ps = (female_p,male_p)
    
    model.eval()
    for X,y,gender in dataloader:
        pred = model(X.to(device))
        for i in range(2):
            filt = (gender[:,i] == 1)
            if y[filt].shape[0] > 0:
                ys[i].append(utils.one_hot(y[filt]).cpu().numpy())
                ps[i].append(pred[filt].data.cpu().numpy())
    female_ys = np.concatenate(ys[0])
    male_ys   = np.concatenate(ys[1])
    female_ps = np.concatenate(ps[0])
    male_ps   = np.concatenate(ps[1])

    return female_ys, male_ys, female_ps, male_ps

def compute_roc_curve(dataloader, 
                      model,
                      device):
    female_ys, male_ys, female_ps, male_ps = compute_roc_data(
        dataloader,
        model,
        device
    )
    # Compute ROC curve and ROC area for each class
    male_fpr = dict()
    male_tpr = dict()
    male_roc_auc = dict()

    female_fpr = dict()
    female_tpr = dict()
    female_roc_auc = dict()
    for i in range(10):
        male_fpr[i], male_tpr[i], _ = roc_curve(male_ys[:, i], male_ps[:, i])
        female_fpr[i], female_tpr[i], _ = roc_curve(female_ys[:, i], female_ps[:, i])
        male_roc_auc[i] = auc(male_fpr[i], male_tpr[i])
        female_roc_auc[i] = auc(female_fpr[i], female_tpr[i])

    # Compute micro-average ROC curve and ROC area
    male_fpr["micro"], male_tpr["micro"], _ = roc_curve(male_ys.ravel(), male_ps.ravel())
    female_fpr["micro"], female_tpr["micro"], _ = roc_curve(female_ys.ravel(), female_ps.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return male_fpr, male_tpr, male_roc_auc, female_fpr, female_tpr, female_roc_auc

def plot_roc_curve(dataloader, 
                   model,
                   device,
                   title="Default",
                   lw=2):
    idx_to_class = dataloader.dataset.idx_to_class
    male_fpr, male_tpr, male_roc_auc, female_fpr, female_tpr, female_roc_auc = compute_roc_curve(
        dataloader,
        model,
        device
    )
    fig, axarr = plt.subplots(4, 3, figsize=(20,30))
    fig.suptitle(title, fontsize=32)
    for i in range(10):
        if i < 9:
            r = i // 3
            c = i % 3
        else:
            r = i // 3
            c = i % 3 + 1
        axarr[r,c].plot(male_fpr[i], male_tpr[i],
             lw=lw, label='%s Male ROC (AUC = %0.2f)' % (idx_to_class[i], male_roc_auc[i]))
        axarr[r,c].plot(female_fpr[i], female_tpr[i],
             lw=lw, label='%s Female ROC (AUC = %0.2f)' % (idx_to_class[i], female_roc_auc[i]))
        axarr[r,c].legend(fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.delaxes(axarr[3,0])
    fig.delaxes(axarr[3,2])
    plt.show()
