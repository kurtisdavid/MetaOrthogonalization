import dataload
import torchvision
import torch
import torch.nn as nn
import importlib
import matplotlib.pyplot as plt
import numpy as np
import models
import utils
import analysis
import net2vec
import seaborn as sns
import pickle
import debias
import pandas as pd
import torch.nn.functional as f
import os
import post_train
import analysis
import seaborn as sns

## Repeated Code will be Here ##
def load_models(device,
                base_folder='./models/BAM/',
                specific="bowling_alley", 
                seed=0, 
                module="layer3",
                experiment="sgd_finetuned",
                ratio="0.5",
                adv=False,
                baseline=False,
                epoch=None,
                post=False,
                multiple=True,
                leakage=False,
                tcav=False,
                force=False,
                dataset='bam'):
    if leakage:
        assert post
    if epoch is not None:
        epoch = "_" + str(epoch)
    else:
        epoch = ""
    if baseline:
        model_end = "resnet_base_"+str(ratio)+epoch+'.pt'
        if not post:
            n2v_end   = "resnet_n2v_base_"+str(ratio)+epoch+'.pt'
        else:
            n2v_end   = "resnet_n2v_base_after_"+str(ratio)+epoch+'.pt'
    else:
        if not adv:
            model_end = "resnet_debias_"+str(ratio)+epoch+'.pt'
            if not post:
                n2v_end   = "resnet_n2v_debias_"+str(ratio)+epoch+'.pt'
            else:
                n2v_end   = "resnet_n2v_debias_after_"+str(ratio)+epoch+'.pt'
        else:
            model_end = "resnet_adv_"+str(ratio)+'.pt'
            if not post:
                n2v_end   = "resnet_n2v_adv_"+str(ratio)+'.pt'
            else:
                n2v_end   = "resnet_n2v_adv_after_"+str(ratio)+epoch+'.pt'
    if dataset != 'bam':
        model_end = model_end.replace('_'+str(ratio), '')
        n2v_end   = n2v_end.replace('_'+str(ratio), '')
    if dataset == 'bam':
        model_path, n2v_path = utils.get_paths(
                base_folder,
                seed,
                specific,
                model_end=model_end,
                n2v_end='leakage/' + n2v_end.replace('n2v','mlp') if leakage else n2v_end,
                n2v_module=module,
                experiment=experiment,
                with_n2v=True,
        )
    else:
        model_path = os.path.join(base_folder, str(seed), experiment, module, model_end)
        n2v_path = os.path.join(base_folder, str(seed), experiment, module, 'leakage/' + n2v_end.replace('n2v','mlp') if leakage else n2v_end)
    if dataset == 'bam':
        trainloader, _ = dataload.get_data_loader_SceneBAM(seed=seed,ratio=float(ratio), specific=specific)
        _, testloader = dataload.get_data_loader_SceneBAM(seed=seed,ratio=float(0.5), specific=specific)
    else:
        trainloader,testloader = dataload.get_data_loader_idenProf('idenprof',train_shuffle=True,
                                                                   train_batch_size=64,
                                                                   test_batch_size=64,
                                                                   exclusive=True)
    assert os.path.exists(model_path), model_path
    if post:
        # since we have to run a separate script, might not have finished...
        if not leakage:
            model_extra = '_adv' if adv else ('_base' if baseline else '_debias')
            n2v_extra   = model_extra + '_after'
            if tcav:
                pass
            elif force:
                post_train.train_net2vec(trainloader, 
                                        testloader, 
                                        device, 
                                        seed,
                                        specific=specific,
                                        p=ratio,
                                        n_epochs=20,
                                        module=module,
                                        lr=.01,
                                        out_file=None,
                                        base_folder=base_folder,
                                        experiment1=experiment,
                                        experiment2=experiment,
                                        model_extra=model_extra,
                                        n2v_extra=n2v_extra,
                                        with_n2v=True,
                                        nonlinear=False, # might want to change this later
                                        model_custom_end=epoch.replace('_',''),
                                        n2v_custom_end=epoch.replace('_',''),
                                        multiple=multiple,
                                        dataset=dataset
                )
            else:
                raise Exception('Run trial again')
        elif leakage:
            model_extra = '_adv' if adv else ('_base' if baseline else '_debias')
            n2v_extra   = model_extra + '_after'
            if force:
                post_train.train_leakage(trainloader, 
                                        testloader, 
                                        device, 
                                        seed,
                                        specific=specific,
                                        p=ratio,
                                        n_epochs=20,
                                        module=module,
                                        lr=5e-5, # leakage model uses adam
                                        out_file=None,
                                        base_folder=base_folder,
                                        experiment1=experiment,
                                        experiment2=experiment,
                                        model_extra=model_extra,
                                        n2v_extra=n2v_extra,
                                        with_n2v=True,
                                        nonlinear=True, # MLP leakage model
                                        model_custom_end='',
                                        n2v_custom_end='',
                                        dataset=dataset
                )
            else:
                raise Exception('Run trial again')
    else:
        # should've been saved during training
        assert os.path.exists(n2v_path)
    num_attributes = 10 + 9 + 20 if multiple else 12
    model, net, net_forward, activation_probe = models.load_models(
        device,
        lambda x,y,z: models.resnet_(pretrained=True, custom_path=x, device=y,initialize=z, size=50 if dataset == 'bam' else 34),
        model_path=model_path,
        net2vec_pretrained=True,
        net2vec_path=n2v_path,
        module='fc' if leakage else module,
        num_attributes=2 if leakage else num_attributes,
        model_init = False,
        n2v_init = False,
        nonlinear = leakage
    )
    return model, net, net_forward, activation_probe

# Metrics 1: Probe Projections on Bias
def collect_projection(device,
                       base_folder='./models/BAM/',
                       specific="bowling_alley", 
                       seed=0, 
                       module="layer3",
                       experiment="sgd_finetuned",
                       ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9", "1.0"],
                       adv=False,
                       baseline=False,
                       epoch=None,
                       post=False,
                       multiple=True,
                       force=False,
                       dataset='bam'):
    projs = {}
    for ratio in ratios:
        model, net, net_forward, activation_probe = load_models(
            device,
            base_folder=base_folder,
            specific=specific, 
            seed=seed, 
            module=module,
            experiment=experiment,
            ratio=ratio,
            adv=adv,
            baseline=baseline,
            epoch=epoch,
            post=post,
            multiple=multiple,
            leakage=False,
            force=force,
            dataset=dataset
        )
        num_attributes = 10 + 9 + 20 if multiple else 12
        W = net.weight.data
        # need to collect specific_idx projections for all copies
        proj_results = []
        truck_idx = 10
        zebra_idx = 11
        vg  = (W[truck_idx,:] - W[zebra_idx,:]).data.cpu().numpy()
        for s_idx in range(10):
            s   = W[s_idx,:].data.cpu().numpy()
            proj_results.append(
                np.dot(vg / (np.linalg.norm(vg)),
                    s  / (np.linalg.norm(s))
                )
            )
        projs[ratio] = proj_results
    
    return projs

# Metrics 2: TCAV Projections
def collect_tcav(
    device,
    base_folder='./models/BAM/',
    specific="bowling_alley", 
    seed=0, 
    module="layer3",
    experiment="sgd_finetuned",
    ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9", "1.0"],
    adv=False,
    baseline=False,
    epoch=None,
    post=False,
    multiple=True,
    force=False,
    dataset='bam'):
    if dataset == 'bam':
        _, testloader = dataload.get_data_loader_SceneBAM(seed=seed,ratio=float(0.5), specific=specific)
    else:
        _, testloader = dataload.get_data_loader_idenProf('idenprof',train_shuffle=True,
                                                                train_batch_size=64,
                                                                test_batch_size=64,
                                                                exclusive=True)
    model_type = "Baseline" if baseline else (
        "Adversarial" if adv else "Debias"
    )
    df = {}
    values = []
    r = []
    m = []
    s = []
    for ratio in ratios:
        print("loading model")
        model, net, net_forward, activation_probe = load_models(
            device,
            base_folder=base_folder,
            specific=specific, 
            seed=seed, 
            module=module,
            experiment=experiment,
            ratio=ratio,
            adv=adv,
            baseline=baseline,
            epoch=epoch,
            post=post,
            multiple=multiple,
            leakage=False,
            force=force,
            tcav=True,
            dataset=dataset
        )
        print("did load?")
        collect, labels = analysis.compute_tcavs(
            testloader,
            model,
            activation_probe,
            net.weight[-2]-net.weight[-1], #bias vector
            device
        )
        for s_idx in range(10):
            current = collect[labels==s_idx]
            values.append(current)
            r += [ratio] * current.shape[0]
            m += [model_type] * current.shape[0]
            s += [s_idx] * current.shape[0]
        
    values = np.concatenate(
        values
    )
    df['values'] = values
    df['ratio'] = r
    df['model'] = m
    df['class'] = s
    
    # return a Dataframe, then we can join together data from different experiments...
    return pd.DataFrame(df)

# Metric 3: Accuracy Disparity
def collect_accuracies(
    device,
    base_folder='./models/BAM/',
    specific="bowling_alley", 
    seeds=[0,1,2,3,4], 
    module="layer3",
    experiment="sgd_finetuned",
    ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9", "1.0"],
    adv=False,
    baseline=False,
    epoch=None,
    post=False,
    multiple=True,
    dataset='bam'):
    res = []
    for seed in seeds:
        curr = []
        if dataset == 'bam':
            _, testloader = dataload.get_data_loader_SceneBAM(seed=seed,ratio=float(0.5), specific=specific)
        else:
            _, testloader = dataload.get_data_loader_idenProf('idenprof',train_shuffle=True,
                                                                   train_batch_size=64,
                                                                   test_batch_size=32,
                                                                   exclusive=True)
        for ratio in ratios:
            model, _, _, _ = load_models(
                    device,
                    base_folder=base_folder,
                    specific=specific, 
                    seed=seed, 
                    module=module,
                    experiment=experiment,
                    ratio=ratio,
                    adv=adv,
                    baseline=baseline,
                    epoch=epoch,
                    post=False, # n2v doesnt matter, we should at least have the during n2v trained
                    multiple=multiple,
                    leakage=False,
                    dataset=dataset
                )
            model.eval()
            acc = utils.classification_specific_accuracy(
                testloader, 
                model, 
                device)
            curr.append(acc)
        res.append(curr)
    return res

# Metric 4: Leakage
def collect_leakage(device,
                    base_folder='./models/BAM/',
                    specific="bowling_alley", 
                    seed=0, 
                    module="layer3",
                    experiment="sgd_finetuned",
                    ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9", "1.0"],
                    adv=False,
                    baseline=False,
                    epoch=None,
                    multiple=True,
                    force=False,
                    dataset='bam'):
    results = {}
    if dataset == 'bam':
        _, testloader = dataload.get_data_loader_SceneBAM(seed=seed,ratio=float(0.5), specific=specific)
    else:
        _, testloader = dataload.get_data_loader_idenProf('idenprof',train_shuffle=True,
                                                                   train_batch_size=64,
                                                                   test_batch_size=64,
                                                                   exclusive=True)
    for ratio in ratios:
        model, net, net_forward, activation_probe = load_models(
            device,
            base_folder=base_folder,
            specific=specific, 
            seed=seed, 
            module=module,
            experiment=experiment,
            ratio=ratio,
            adv=adv,
            baseline=baseline,
            epoch=epoch,
            post=True,
            multiple=multiple,
            leakage=True,
            force=force,
            dataset=dataset
        )
        model.eval()
        net.eval()
        results[ratio],_ = utils.net2vec_accuracy(
            testloader, 
            net_forward, 
            device, 
            train_labels=[-2,-1]
        )
    return results
        
## ALL PLOTTING FUNCTIONS GO HERE ##
def plot_all_proj(results,
                  multiple=False, 
                  ylow=-.12,
                  yhigh=.15,
                  separate=False):
    # loop through all probes
    for epoch in results:
        res = results[epoch]
        plt.figure(figsize=(10,10))
        for i in range(10 if multiple else 1):
            if i == 0:
                linestyle = '-'
                extra = " (Primary)"
            else:
                linestyle = '-.'
                extra = ""
            l = []
            l1 = []
            l2 = []
            for ratio in res.keys():
                l.append(res[ratio][i][0])
                l1.append(res[ratio][i][1])
                l2.append(res[ratio][i][2])
            plt.plot(l, linestyle=linestyle, label="Probe " + str(i) + extra)
            if separate:
                plt.plot(l1, label="Truck Projection", linestyle='-.')
                plt.plot(l2, label="Zebra Projection", linestyle='-.')
            plt.xlabel("Bias Ratio")
            plt.ylabel("Bias Projection")
            plt.title("Epoch " + str(epoch), fontsize=16)
        plt.ylim(ylow,yhigh)
        plt.legend(fontsize=12)
        plt.savefig('pics/' + str(epoch) + "_multiple.png")
        plt.show()

def plot_tcav_comparison(df1, df2):
    df = df1.append(df2)
    # plot kdes
    pal = sns.cubehelix_palette(10,rot=-.75, light=0.7)
    grid = sns.FacetGrid(df, row="ratio", hue="model", aspect=5, height=2)
    grid = grid.map(sns.kdeplot, "values", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    grid = grid.map(plt.axhline, y=0, lw=2, clip_on=False).add_legend()
    grid = grid.set_titles("{row_name}")
    grid = grid.set(yticks=[])
    grid = grid.despine(bottom=True, left=True)

def process_subset_results_specific(results,
                                    specific=None,
                                    ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
                                    seeds=range(3),
                                    reduce=True):
    agg_results = []
    for i in seeds:
        r = []
        for j, ratio in enumerate(ratios):
            truck_acc = 0
            zebra_acc = 0
            truck_cts = 0
            zebra_cts = 0
            if specific is not None:
                truck_acc = results[i][j][0][specific][1]
                zebra_acc = results[i][j][0][specific][2]
            else:
                # get the overall accuracy of network on images with trucks and images with zebras
                for spc in range(len(results[i][j][0])):
                    truck_acc += results[i][j][0][spc][1] * results[i][j][0][spc][-2] 
                    zebra_acc += results[i][j][0][spc][2] * results[i][j][0][spc][-1] 
                    truck_cts += results[i][j][0][spc][-2] 
                    zebra_cts += results[i][j][0][spc][-1] 
                truck_acc /= truck_cts
                zebra_acc /= zebra_cts
            r.append([truck_acc, zebra_acc])
        agg_results.append(r)
    if reduce:
        means = np.mean(agg_results,axis=0)
        means = np.moveaxis(means,-1,0)
        stds  = np.std(agg_results,axis=0, ddof=1) / np.sqrt(len(seeds)-1)
        stds  = np.moveaxis(stds,-1,0)
        return means, stds
    else:
        return np.array(agg_results)

def plot_subset_accuracies_specific(base_results,
                    exp_results,
                    specific=0,
                    ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
                    base_label="Baseline",
                    exp_label="Experimental"):
    mean_baseline_subset_sp, std_baseline_subset_sp = process_subset_results_specific(base_results, specific)
    mean_exp_subset_sp, std_exp_subset_sp = process_subset_results_specific(exp_results,0)
    r = [float(ratio) for ratio in ratios]
    plt.figure(figsize=(12,8))
    plt.errorbar(r, mean_baseline_subset_sp[0,:], yerr=std_baseline_subset_sp[0,:], label = base_label, c='b')
    plt.errorbar(r, mean_exp_subset_sp[0,:], 
                 yerr=std_exp_subset_sp[0,:], 
                 label = exp_label, 
                 c='r',
                 linestyle='--')

    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Ratio", fontsize=14)
    plt.title("%s vs. %s | Truck Subset | %s" % (base_label, exp_label, str(idx_to_class[specific])), fontsize=18)
    plt.legend(fontsize=16,loc=8)
    plt.ylim(0.6,1.0)

    plt.figure(figsize=(12,8))
    plt.errorbar(r, mean_baseline_subset_sp[1,:], yerr=std_exp_subset_sp[1,:], label = base_label, c='b')
    plt.errorbar(r, mean_exp_subset_sp[1,:], yerr=std_exp_subset_sp[1,:], label = exp_label, c='r',
                 linestyle='--')

    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Ratio", fontsize=14)
    plt.title("%s vs. %s | Zebra Subset | %s" % (base_label, exp_label, str(idx_to_class[specific])), fontsize=18)
    plt.legend(fontsize=16,loc=8)
    plt.ylim(0.6,1.0)

def process_leakage_results(
    results,
    ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
    valid_ratios=[],
    seeds=range(3),
    reduce=True
):
    agg_results = []
    for i in seeds:
        r = []
        for ratio in ratios:
            if ratio in results[i]:
                leakage = np.max(results[i][ratio])
                r.append(leakage)
                if ratio not in valid_ratios:
                    valid_ratios.append(ratio)
        agg_results.append(r)
    if reduce:
        means = np.mean(agg_results,axis=0)
        means = np.moveaxis(means,-1,0)
        stds  = np.std(agg_results,axis=0, ddof=1) / np.sqrt(len(seeds)-1)
        stds  = np.moveaxis(stds,-1,0)
        return means, stds
    else:
        return np.array(agg_results)

def process_projection_results(
    results,
    specific,
    ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
    valid_ratios=[],
    seeds=range(3),
    reduce=True
):
    agg_results = []
    for i in seeds:
        r = []
        # if len(results[i].keys()) != len(ratios):
        #     continue
        for ratio in ratios:
            if ratio in results[i]:
                proj = results[i][ratio][specific]
                if isinstance(proj, list):
                    proj = proj[0]
                r.append(proj)
                if ratio not in valid_ratios:
                    valid_ratios.append(ratio)
        agg_results.append(r)
    if reduce:
        means = np.mean(agg_results,axis=0)
        means = np.moveaxis(means,-1,0)
        stds  = np.std(agg_results,axis=0, ddof=1) / np.sqrt(len(seeds)-1)
        stds  = np.moveaxis(stds,-1,0)
        return means, stds
    else:
        return np.array(agg_results)

def process_tcav_results(
    results,
    specific,
    ratios=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
    valid_ratios=[],
    seeds=range(3),
    reduced=True
):
    if reduced:
        agg_results = []
        for i in seeds:
            r = []
            df = results[i]
            df = df[df['class']==specific]
            all_ratios = df['ratio'].unique()
            for ratio in ratios:
                if ratio in all_ratios:
                    mean_tcav = df[df['ratio']==ratio]['values'].mean()
                    r.append(mean_tcav)
                    if ratio not in valid_ratios:
                        valid_ratios.append(ratio)
            agg_results.append(r)
        means = np.mean(agg_results,axis=0)
        means = np.moveaxis(means,-1,0)
        stds  = np.std(agg_results,axis=0, ddof=1) / np.sqrt(len(seeds)-1)
        stds  = np.moveaxis(stds,-1,0)
        return means, stds
    else:
        agg_results = []
        for i in seeds:
            r = []
            df = results[i]
            df = df[df['class']==specific]
            all_ratios = df['ratio'].unique()
            for ratio in ratios:
                if ratio in all_ratios:
                    tcavs = np.array(df[df['ratio']==ratio]['values']).reshape(-1)
                    r.append(tcavs)
                    if ratio not in valid_ratios:
                        valid_ratios.append(ratio)
            agg_results.append(r)
        agg_results = np.array(agg_results).mean(axis=0)
        return agg_results