import argparse
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
import random
import setup_datasets
import os
import time
import pickle
from itertools import cycle
import higher
import coco_dataload

torch.multiprocessing.set_sharing_strategy('file_system')
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_args():
    # General system running and configuration options
    parser = argparse.ArgumentParser(description='main.py')

    # true or false args
    parser.add_argument('-probe_eval_off', action='store_true', default=False, help='Default False --> we switch model to eval when running probe, otherwise True --> keep model in training mode (might help for training?')
    parser.add_argument('-no_log', action='store_true', default=False, help='logging experiments')
    parser.add_argument('-debias', action='store_true', default=False, help='use debias loss')
    parser.add_argument('-adv', action='store_true', default=False, help='use adversarial loss')
    parser.add_argument('-finetuned', action='store_true', default=False, help='if we want to start from pretrained...')
    parser.add_argument('-nonlinear', action='store_true', default=False, help='instead use resnet18 than a linear layer')
    parser.add_argument('-subset', action='store_true', default=False, help='split into subset/remaining')
    parser.add_argument('-save_every', action='store_true', default=False, help='save models during training')
    parser.add_argument('-experimental', action='store_true', default=False, help='if were applying lda')
    parser.add_argument('-multiple', action='store_true', default=False, help='have multiple probes')
    parser.add_argument('-debias_multiple', action='store_true', default=False, help='debias multiple probes')
    parser.add_argument('-reset', action='store_true', default=False, help='reinitialize probe every epoch')
    parser.add_argument('-n2v_start', action='store_true', default=False, help='start n2v fully trained')
    parser.add_argument('-adaptive_alpha', action='store_true', default=False, help='adaptively modify alpha s.t. loss * alpha \approx 1')
    parser.add_argument('-n2v_adam', action='store_true', default=False, help='use adam for n2v')
    parser.add_argument('-single', action='store_true', default=False, help='only use truck/zebra')
    parser.add_argument('-imagenet', action='store_true', default=False, help='training imagenet')
    parser.add_argument('-constant_resize', action='store_true', default=False, help='imagenet downsize the same for every image')
    parser.add_argument('-adaptive_resize', action='store_true', default=False, help='imagenet downsize every image')
    parser.add_argument('-no_class', action='store_true', default=False, help='no classification training')
    parser.add_argument('-partial_projection', action='store_true', default=False, help='reproject to match gamma')
    parser.add_argument('-constant_alpha', action='store_true', default=False, help='no alpha schedule')

    parser.add_argument('-jump_alpha', action='store_true', default=False, help='jump to 1000 after warm start')
    parser.add_argument('-linear_alpha', action='store_true', default=False, help='linear schedule after warm start')

    parser.add_argument('-mean_debias', action='store_true', default=False, help='use the sum or mean of bias (help with scaling?)')
    parser.add_argument('-no_limit', action='store_true', default=False, help='always have regularization on')
    parser.add_argument('-parallel', action='store_true', default=False, help='for big training')


    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='bam')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--main_lr', type=float, default=.01, help='downstream task learning rate')
    parser.add_argument('--n2v_lr', type=float, default=.001, help='n2v learning rate')
    parser.add_argument('--combined_n2v_lr', type=float, default=.1, help='meta learning rate for n2v parameters')
    parser.add_argument('--main_momentum', type=float, default=0, help='model momentum')
    parser.add_argument('--n2v_momentum', type=float, default=0, help='n2v momentum')

    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training')

    parser.add_argument('--train_bs', type=int, default=64, help='batch size')
    parser.add_argument('--test_bs', type=int, default=64, help='eval batch size')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha parameter...')
    parser.add_argument('--beta', type=float, default=1.0, help='beta parameter...')
    parser.add_argument('--gamma', type=float, default=0.0, help='extra bias parameter (i.e. 0 bias = 0 gamma)')
    parser.add_argument('--bias_norm', type=str, default='l2', help='whether to apply squared or absolute loss')
    # coco dataloader args
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--annotation_dir', type=str,
            default='./datasets/coco',
            help='annotation files path')
    parser.add_argument('--image_dir',
            default = './datasets/coco',
            help='image directory')
    
    parser.add_argument('-balanced', action='store_true', default=False, help='coco balanced by ratio')
    parser.add_argument('-gender_balanced', action='store_true', default=False, help='coco gender balanced in each object category')

    parser.add_argument('--data_dir', type=str, default='./bam_scenes', help='main data repository')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu')
    parser.add_argument('--ratio', type=str, default="0.5", help='bias ratio, can double up as coco ratio')
    parser.add_argument('--specific',nargs='+', default=None, help='which class(es) to control bias')
    parser.add_argument('--module',type=str, default='layer4', help='which module to extract from')
    parser.add_argument('--experiment1',type=str, default='sgd', help='type of learning for debias trials (where to get trained networks when finetuning)')
    parser.add_argument('--experiment2',type=str, default=None, help='type of learning for debias trials (where to save new models)')
    parser.add_argument('--verify_ratios',nargs='+', default=["1.0","0.9","0.8","0.75","0.7","0.6","0.5","0.4","0.3","0.25","0.2","0.1","0.0"], help='all the ratios we will test')
    parser.add_argument('--gpu_ids',nargs='+', default=None, help='which gpus to use dataparallel on')

    parser.add_argument('--subset_ratio', type=float, default=0.1, help='how much to subsample for ratio tests')
    parser.add_argument('--reset_counter', type=int, default=3, help='how often to reset')    
    # logger arguments
    parser.add_argument('--base_folder',type=str, default='./models/BAM/')
    parser.add_argument('--results_folder',type=str, default='./results/BAM/')
    parser.add_argument('--out_file', type=str, default=None) 
    args = parser.parse_args()
    return args

def train(trainloader, testloader, device, seed,
          debias_=True,
          specific=None,
          ratio = 0.5, # bias ratio in dataset
          n_epochs=5,
          model_lr=1e-3,
          n2v_lr=1e-3,
          combined_n2v_lr=1e-3, # metalearning rate for n2v
          alpha=100, # for debias,
          beta=0.1, # for adversarial loss
          out_file=None,
          base_folder="",
          results_folder="",
          experiment="sgd",
          momentum=0,
          module="layer4",
          finetuned=False,
          adversarial=False,
          nonlinear=False,
          subset=False,
          subset_ratio=0.1,
          save_every=False,
          model_momentum=0,
          n2v_momentum=0,
          experimental=False,
          multiple=False,
          debias_multiple=False,
          reset=False,
          reset_counter=1,
          n2v_start=False,
          experiment2=None,
          adaptive_alpha=False,
          n2v_adam=False,
          single=False,
          imagenet=False,
          train_batch_size=64,
          constant_resize=False,
          adaptive_resize=False,
          no_class=False,
          gamma=0,
          partial_projection=False,
          norm='l2',
          constant_alpha=False,
          jump_alpha=False,
          linear_alpha=False,
          mean_debias=False,
          no_limit=False,
          dataset='bam',
          parallel=False,
          gpu_ids=[],
          switch_modes=True):
    print("mu", momentum, "debias", debias_, "alpha", alpha, " | ratio:", ratio)
    
    def get_vg(W):
        if single:
            return W[-2,:]
        else:
            return W[-2,:] - W[-1,:]
    
    if dataset == 'bam' or dataset == 'coco':
        model_init_path, n2v_init_path = utils.get_paths(
                base_folder,
                seed,
                specific,
                model_end="resnet_init"+'.pt',
                n2v_end="resnet_n2v_init"+'.pt',
                n2v_module=module,
                experiment=experiment,
                with_n2v=False
        )
    else:
        model_init_path = os.path.join(base_folder, str(seed), experiment, 'resnet_init.pt')
        n2v_init_path = os.path.join(base_folder, str(seed), experiment, module, 'resnet_n2v_init.pt')
    if finetuned:
        if dataset == 'bam' or dataset == 'coco':
            model_init_path = utils.get_model_path(
                    base_folder,
                    seed,
                    specific,
                    "resnet_"+str(ratio)+".pt",
                    experiment='post_train' if not n2v_start else experiment.split('_finetuned')[0]
            )
        else:
            model_init_path = os.path.join(base_folder, str(seed), 'post_train' if not n2v_start else experiment.split('_finetuned')[0], 'resnet.pt')
        assert (debias_ and not adversarial) or (adversarial and not debias_) or (not adversarial and not debias_)
        if debias_ and n2v_start:
            ext = "_n2v_" if not nonlinear else "_mlp_"
            if dataset == 'bam' or dataset == 'coco':
                n2v_init_path = utils.get_net2vec_path(
                    base_folder,
                    seed,
                    specific,
                    module,
                    "resnet"+str(ext)+str(ratio)+".pt",
                    experiment=experiment.split('_finetuned')[0]
                )
            else:
                n2v_init_path = os.path.join(base_folder, str(seed), experiment.split('_finetuned')[0], module, 'resnet' + ext[:-1] + '.pt')
        # if we're also doing adversarial, make sure to load the matching n2v as init...
        if adversarial:
            ext = "_n2v_" if not nonlinear else "_mlp_"
            if dataset == 'bam' or dataset == 'coco':
                n2v_init_path = utils.get_net2vec_path(
                    base_folder,
                    seed,
                    specific,
                    module,
                    "resnet"+str(ext)+str(ratio)+".pt",
                    experiment='post_train'
                )
            else:
                n2v_init_path = os.path.join(base_folder, str(seed), 'post_train', module, 'resnet' + ext[:-1] + '.pt')
    num_classes = 10
    num_attributes = 12
    if nonlinear:
        num_attributes = 2
    if multiple:
        num_attributes = 10 + 9 + 2*10
    if dataset == 'coco':
        num_classes = 79
        num_attributes = 81
    model, net, net_forward, activation_probe = models.load_models(
        device,
        lambda x,y,z: models.resnet_(
            pretrained=True, 
            custom_path=x, 
            device=y,
            initialize=z,
            num_classes=num_classes,
            size=50 if (dataset=='bam' or dataset=='coco') else 34
        ),
        model_path=model_init_path,
        net2vec_pretrained=True,
        net2vec_path=n2v_init_path,
        module=module,
        num_attributes=num_attributes,
        # we want to make sure to save the inits if not finetuned...
        model_init=True if not finetuned else False, 
        n2v_init=True if not (finetuned and (adversarial or (debias_ and n2v_start))) else False,
        loader=trainloader,
        nonlinear=nonlinear,
        # parameters if we want to initially project probes to have a certain amount of bias
        partial_projection=partial_projection,
        t=gamma
    )
    print(model_init_path, n2v_init_path)
    model_n2v_combined = models.ProbedModel(
        model, net, module, switch_modes=switch_modes
    )
    if n2v_adam:
        combined_optim = torch.optim.Adam(
            [
                {'params': model_n2v_combined.model.parameters()},
                {'params': model_n2v_combined.net.parameters()}
            ],
            lr = n2v_lr
        )
        # TODO: allow for momentum training as well
        n2v_optim = torch.optim.Adam(net.parameters(), lr = n2v_lr)
    else:
        combined_optim = torch.optim.SGD(
            [
                {'params': model_n2v_combined.model.parameters()},
                {'params': model_n2v_combined.net.parameters(), 'lr': combined_n2v_lr, 'momentum': n2v_momentum}
            ],
            lr = model_lr, momentum = model_momentum
        )

        # TODO: allow for momentum training as well
        n2v_optim = torch.optim.SGD(net.parameters(), lr = n2v_lr, momentum = n2v_momentum)
    model_optim = torch.optim.SGD(model.parameters(), lr = model_lr, momentum = model_momentum)

    
    d_losses = []
    adv_losses = []
    n2v_train_losses = []
    n2v_accs = []
    n2v_val_losses = []
    class_train_losses = []
    class_accs = []
    class_val_losses = []
    alpha_log = []
    magnitudes = []
    magnitudes2 = []
    unreduced = []
    bias_grads = []
    loss_shapes = []
    loss_shapes2 = []

    results = {
        "debias_losses": d_losses,
        "n2v_train_losses": n2v_train_losses,
        "n2v_val_losses": n2v_val_losses,
        "n2v_accs": n2v_accs,
        "class_train_losses": class_train_losses,
        "class_val_losses": class_val_losses,
        "class_accs": class_accs,
        "adv_losses": adv_losses,
        "alphas": alpha_log,
        "magnitudes": magnitudes,
        "magnitudes2": magnitudes2,
        "unreduced": unreduced,
        "bias_grads": bias_grads,
        "loss_shapes": loss_shapes,
        "loss_shapes2": loss_shapes2
    }
    if debias_:
        results_end = str(ratio) + "_debias.pck"
    elif adversarial:
        results_end = str(ratio) + "_adv.pck"
        if nonlinear:
            results_end = str(ratio) + "_mlp_adv.pck"
    else:
        results_end = str(ratio) + "_base.pck"
    
    if dataset == 'bam' or dataset == 'coco':
        results_path = utils.get_net2vec_path(
            results_folder, 
            seed, 
            specific, 
            module, 
            results_end,
            experiment if experiment2 is None else experiment2
        )
    else:
        results_path = os.path.join(results_folder, str(seed), experiment if experiment2 is None else experiment2, module, results_end)
    if debias_:
        model_end = "resnet_debias_"+str(ratio)+'.pt'
        n2v_end   = "resnet_n2v_debias_"+str(ratio)+'.pt'
    elif adversarial:
        if not nonlinear:
            model_end = "resnet_adv_"+str(ratio)+'.pt'
        else:
            model_end = "resnet_adv_nonlinear_" + str(ratio) + '.pt'
        if not nonlinear:
            n2v_end   = "resnet_n2v_adv_"+str(ratio)+'.pt'
        else:
            n2v_end   = "resnet_mlp_adv_"+str(ratio)+'.pt'
    else:
        model_end = "resnet_base_"+str(ratio)+'.pt'
        n2v_end   = "resnet_n2v_base_"+str(ratio)+'.pt'
    
    if dataset != 'bam' and dataset != 'coco':
        model_end = model_end.replace('_'+str(ratio), '')
        n2v_end   = n2v_end.replace('_'+str(ratio), '')

    if dataset == 'bam' or dataset == 'coco':
        model_path, n2v_path = utils.get_paths(
                base_folder,
                seed,
                specific,
                model_end=model_end,
                n2v_end=n2v_end,
                n2v_module=module,
                experiment=experiment if experiment2 is None else experiment2,
                with_n2v=True,
        )
    else:
        model_path = os.path.join(base_folder, str(seed), experiment if experiment2 is None else experiment2, module, model_end)
        n2v_path = os.path.join(base_folder, str(seed), experiment if experiment2 is None else experiment2, module, n2v_end)
    if hasattr(trainloader.dataset, 'idx_to_class'):
        for key in trainloader.dataset.idx_to_class:
            if specific is not None and trainloader.dataset.idx_to_class[key] in specific:
                specific_idx = int(key)
            else:
                specific_idx = 0
    train_labels = None if not nonlinear else [-2,-1] 
    d_last = 0
    resize = constant_resize or adaptive_resize
    if imagenet:
        imagenet_trainloaders,_ = dataload.get_imagenet_tz('./datasets/imagenet', workers=8, train_batch_size=train_batch_size//8, resize=resize, constant=constant_resize)
        imagenet_trainloader    = dataload.process_imagenet_loaders(imagenet_trainloaders)

    params = list(model_n2v_combined.parameters())[:-2]
    init_alpha = alpha
    last_e = 0

    # setup training criteria
    if dataset == 'coco':
        object_weights = torch.FloatTensor(trainloader.dataset.getObjectWeights())
        gender_weights = torch.FloatTensor(trainloader.dataset.getGenderWeights())
        all_weights = torch.cat([object_weights, gender_weights])
        probe_criterion = nn.BCEWithLogitsLoss(weight=all_weights.to(device), reduction='elementwise_mean')
        downstream_criterion = nn.BCEWithLogitsLoss(weight=object_weights.to(device), reduction='elementwise_mean')
    else:
        probe_criterion = None
        downstream_criterion = nn.CrossEntropyLoss()

    for e in range(n_epochs):
        # save results every epoch...
        with open(results_path, 'wb') as f:
            print("saving results", e)
            print(results_path)
            pickle.dump(results, f)

        model.eval()
    
        with torch.no_grad():
            n2v_acc, n2v_val_loss = utils.net2vec_accuracy(testloader, net_forward, device, train_labels)
            n2v_accs.append(n2v_acc)
            n2v_val_losses.append(n2v_val_loss)

            if dataset != 'coco':
                class_acc, class_val_loss = utils.classification_accuracy(testloader, model, device)
                class_accs.append(class_acc)
                class_val_losses.append(class_val_loss)
            else:
                f1, mAP = utils.detection_results(testloader, model, device)
                print("Epoch", e, "| f1:", f1, "| mAP:", mAP )
                class_accs.append([f1, mAP])

        d_initial = 0
        if not adversarial:
            curr_W = net.weight.data.clone()
            if not multiple:
                vg = get_vg(curr_W).reshape(-1,1)
                d_initial = debias.debias_loss(curr_W[:-2], vg, t=0).item()
                print("Epoch", e, "bias", str(d_initial), " | debias: ", debias_)
            else:
                ds = np.zeros(10)
                for i in range(10):
                    if i == 0:
                        vg = (curr_W[10,:] - curr_W[11,:]).reshape(-1,1)
                    else:
                        vg = (curr_W[20+i,:] - curr_W[29+i,:]).reshape(-1,1)
                    ds[i] = debias.debias_loss(curr_W[:10], vg, t=0).item()
                print("Epoch", e, "bias", ds, " | debias: ", debias_)
                print("Accuracies:", n2v_acc)
                d_initial = ds[0]
        else:
            print("Epoch", e, "Adversarial", n2v_accs[-1])
        if adaptive_alpha and (e == 0 or ((d_last / d_initial) >= (5 / 2**(e - 1)) or (0.8 < (d_last / d_initial) < 1.2) )):
            #alpha = alpha
            old_alpha = alpha
            # we don't want to increase too much if it's already decreasing
            if (e == 0 or (d_last / d_initial) >= (5 / 2**(e-1))):
                alpha = min(alpha * 2, (15 / (2 ** e)) / (d_initial + 1e-10)) # numerical stability just in case d_initial gets really low
                #if e > 0 and old_alpha >= alpha:
                #    alpha = old_alpha # don't update if we're decreasing... 
                print("Option 1")
            if e>0 and alpha < old_alpha:
                # we want to increase if plateaud
                alpha = max(old_alpha * 1.5, alpha) # numerical stability just in case d_initial gets really low
                print("Option 2")
            # don't want to go over 1000...
            if alpha > 1000:
                alpha = 1000
            d_last = d_initial
        elif not adaptive_alpha and not constant_alpha:
            if jump_alpha and (e - last_e) > 2:
                if not mean_debias:
                    if alpha < 100:
                        alpha = min(alpha * 2, 100)
                        last_e = e
                    else:
                        # two jumps
                        # if (e-last_e) >= ((n_epochs - last_e) // 2):
                        #     alpha = 1000
                        # else:
                        alpha = init_alpha
                else:
                    if alpha < 1000:
                        alpha = min(alpha * 2, 1000)
                        last_e = e 
                    else:
                        alpha = 10000
            elif linear_alpha and (e - last_e) > 2:
                if alpha < 100:
                    alpha = min(alpha * 2, 100)
                    last_e = e
                else:
                    alpha += (1000 - 100) / (n_epochs - last_e)
            elif not jump_alpha and not linear_alpha:
                if (e+1) % 3 == 0:
                    # apply alpha schedule?
                    # alpha = min(alpha * 1.2, max(init_alpha,1000))
                    alpha = alpha * 1.5
        alpha_log.append(alpha)
        print("Current Alpha:,", alpha)
        if save_every and e % 10 == 0 and e > 0 and seed==0 and debias_:
            torch.save(net.state_dict(), n2v_path.split('.pt')[0] + '_' + str(e) + '.pt')
            torch.save(model.state_dict(), model_path.split('.pt')[0] + '_' + str(e) + '.pt')
        if reset and (e+1) % reset_counter == 0 and e > 0:
           print("resetting")
           net, net_forward, activation_probe = net2vec.create_net2vec(
                model,
                module,
                num_attributes,
                device,
                pretrained=False,
                initialize=True,
                nonlinear=nonlinear
           )
           n2v_optim = torch.optim.SGD(net.parameters(), lr = n2v_lr, momentum = n2v_momentum)
                
        model.train()
        ct = 0
        for X,y, genders in trainloader:
            ids = None
            ##### Part 1: Update the Embeddings #####
            model_optim.zero_grad()
            n2v_optim.zero_grad()
            labels = utils.merge_labels(y, genders ,device)
            logits = net_forward(X.to(device), switch_modes=switch_modes)
            # Now actually update net2vec embeddings, making sure to use the same batch
            if train_labels is not None:
                if logits.shape[1] == labels.shape[1]:
                    logits = logits[:, train_labels]
                labels = labels[:, train_labels]
            shapes  = []
            shapes2 = []
            if dataset=='coco':
                prelim_loss = probe_criterion(logits, labels)
            else:
                prelim_loss,ids = utils.balanced_loss(logits, labels, device, 0.5, ids=ids, multiple=multiple, specific=specific_idx, shapes=shapes)
            #print("prelim_loss:", prelim_loss.item())
            prelim_loss.backward()
            # we don't want to update these parameters, just in case
            model_optim.zero_grad()
            n2v_train_losses.append(prelim_loss.item())
            n2v_optim.step()
            try:
                magnitudes.append(
                    torch.norm(net.weight.data, dim=1).data.cpu().numpy()
                )
            except:
                pass

            ##### Part 2: Update Conv parameters for classification #####
            model_optim.zero_grad()
            n2v_optim.zero_grad()
            class_logits = model(X.to(device))
            class_loss = downstream_criterion(
                class_logits,
                y.to(device)
            )
            class_train_losses.append(class_loss.item())

            if debias_:
                W_curr = net.weight.data
                vg = get_vg(W_curr).reshape(-1,1)
                unreduced.append(
                    debias.debias_loss(
                        W_curr[:-2], 
                        vg, 
                        t=0, unreduced=True
                    ).data.cpu().numpy()
                )
            
            loss = class_loss
            #### Part 2a: Debias Loss
            if debias_:
                model_optim.zero_grad()
                n2v_optim.zero_grad()

                labels = utils.merge_labels(y, genders ,device)
                o = net.weight.clone() 
                combined_optim.zero_grad()
                with higher.innerloop_ctx(model_n2v_combined, combined_optim) as (fn2v, diffopt_n2v):
                    models.update_probe(fn2v)
                    logits = fn2v(X.to(device))
                    if dataset=='coco':
                        prelim_loss = probe_criterion(logits, labels)   
                    else:
                        prelim_loss, ids = utils.balanced_loss(logits, labels, device, 0.5, ids=ids, multiple=False, specific=specific_idx, shapes=shapes2)
                    diffopt_n2v.step(prelim_loss)
                    weights = list(fn2v.parameters())[-2]
                    vg = get_vg(weights).reshape(-1,1)
                    d_loss = debias.debias_loss(weights[:-2], vg, t=gamma, norm=norm, mean=mean_debias)
                    # only want to save the actual bias...
                    d_losses.append(d_loss.item())
                    grad_of_grads = torch.autograd.grad(
                        alpha * d_loss, 
                        list(fn2v.parameters(time=0))[:-2], 
                        allow_unused=True)
                    del prelim_loss
                    del logits
                    del vg
                    del fn2v
                    del diffopt_n2v
            #### Part 2b: Adversarial Loss
            if adversarial:
                logits = net_forward(None, forward=True)[:,-2:] # just use activation probe
                labels = genders.type(torch.FloatTensor).reshape(genders.shape[0],-1).to(device)
                adv_loss, _ = utils.balanced_loss(logits, labels, device, 0.5, ids=ids, stable=True)
                adv_losses.append(adv_loss.item())
                # getting too strong, let it retrain...
                if adv_loss < 2:
                    adv_loss = -beta * adv_loss
                    loss += adv_loss
            loss.backward()
            if debias_:
                # custom backward to include the bias regularization....
                max_norm_grad = -1
                param_idx = -1
                for ii in range(len(grad_of_grads)):
                    if (
                        grad_of_grads[ii] is not None
                    and params[ii].grad   is not None
                    and torch.isnan(grad_of_grads[ii]).long().sum() < grad_of_grads[ii].reshape(-1).shape[0]
                    ):
                        # just in case some or nan for some reason?
                        not_nan = ~torch.isnan(grad_of_grads[ii])
                        params[ii].grad[not_nan] += grad_of_grads[ii][not_nan]
                        if grad_of_grads[ii][not_nan].norm().item() > max_norm_grad:
                            max_norm_grad = grad_of_grads[ii][not_nan].norm().item()
                            param_idx = ii
                bias_grads.append((ii,max_norm_grad))
                # undo the last step and apply a smaller alpha to prevent stability issues
                if not no_limit and ((not mean_debias and max_norm_grad > 100) or (mean_debias and max_norm_grad > 100)):
                    for ii in range(len(grad_of_grads)):
                        if (
                            grad_of_grads[ii] is not None
                        and params[ii].grad   is not None
                        and torch.isnan(grad_of_grads[ii]).long().sum() < grad_of_grads[ii].reshape(-1).shape[0]
                        ):
                            # just in case some or nan for some reason?
                            not_nan = ~torch.isnan(grad_of_grads[ii])
                            params[ii].grad[not_nan] -= grad_of_grads[ii][not_nan]
                            # scale accordingly
                            # params[ii].grad[not_nan] += grad_of_grads[ii][not_nan] / max_norm_grad
            
            loss_shapes.append(shapes)
            loss_shapes2.append(shapes2)
            model_optim.step()
            #magnitudes2.append(
            #    torch.norm(net.weight.data, dim=1).data.cpu().numpy()
            #)
            ct += 1
    

    # save results every epoch...
    with open(results_path, 'wb') as f:
        print("saving results", e)
        print(results_path)
        pickle.dump(results, f)
    torch.save(net.state_dict(), n2v_path)
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    # setup our datasets...
    if args.dataset == 'bam':
        if args.specific is not None and len(args.specific) == 1:
            args.specific = args.specific[0]
        setup_datasets.setupBAM(args.seed,True,args.specific,ratios=args.verify_ratios)
        setup_datasets.setupBAM(args.seed,False,args.specific,ratios=args.verify_ratios)
        # now use the given ratio
        trainloader, _ = dataload.get_data_loader_SceneBAM(seed=args.seed,ratio=float(args.ratio), specific=args.specific, train_batch_size=args.train_bs)
        _, testloader = dataload.get_data_loader_SceneBAM(seed=args.seed,ratio=float("0.5"), specific=args.specific, test_batch_size=args.test_bs)
    elif args.dataset == 'idenprof':
        trainloader,testloader = dataload.get_data_loader_idenProf('idenprof',train_shuffle=True,
                                                                   train_batch_size=args.train_bs,
                                                                   test_batch_size=args.test_bs,
                                                                   exclusive=True)
    elif args.dataset == 'coco':
        trainloader, testloader = coco_dataload.get_data_loader_coco(
            args
        )
    # restart seed since we loop through dataloader to get mean/mu statistics
    set_seed(args.seed)
    device = torch.device('cuda:' + str(args.device))
    assert (args.nonlinear and args.adv) or not args.nonlinear
    args.experiment = args.experiment1
    if args.finetuned:
        args.experiment += "_finetuned"
  
    # setup model directories
    if args.dataset == 'bam':
        if not isinstance(args.specific, str):
            folder_name = '.'.join(
                sorted(args.specific)
            )
        else:
            folder_name = args.specific
        dir_parts = [args.base_folder, str(args.seed), folder_name, str(args.experiment), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))

        dir_parts = [args.base_folder, str(args.seed), folder_name, str(args.experiment2), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))
        dir_parts = [args.base_folder.replace('/models/','/results/'), str(args.seed), folder_name, str(args.experiment2), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))

    else:
        dir_parts = [args.base_folder, str(args.seed), str(args.experiment), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))

        dir_parts = [args.base_folder, str(args.seed), str(args.experiment2), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))

    # setup result directory
    result_parts = [args.results_folder, str(args.seed), str(args.experiment2), str(args.module)]
    print(result_parts)
    for i in range(len(result_parts)):
        if not os.path.isdir(os.path.join(*result_parts[:i+1])):
            os.mkdir(os.path.join(*result_parts[:i+1]))

    if args.imagenet:
        assert not args.single
    if args.parallel:
        args.gpu_ids = [int(args.device)] + [int(x) for x in list(args.gpu_ids)]
    #with torch.autograd.detect_anomaly():
    train(trainloader, testloader, device, args.seed,
          debias_=args.debias,
          specific=args.specific,
          ratio=args.ratio, # bias ratio in dataset
          n_epochs=args.epochs,
          model_lr=args.main_lr,
          n2v_lr=args.n2v_lr,
          alpha=args.alpha,
          beta=args.beta,
          out_file=args.out_file,
          base_folder=args.base_folder,
          results_folder=args.results_folder,
          experiment=args.experiment,
          momentum=args.momentum,
          module=args.module,
          finetuned=args.finetuned,
          adversarial=args.adv,
          nonlinear=args.nonlinear,
          subset=args.subset,
          subset_ratio=args.subset_ratio,
          save_every=args.save_every,
          model_momentum=args.main_momentum,
          n2v_momentum=args.n2v_momentum,
          experimental=args.experimental,
          multiple=args.multiple,
          debias_multiple=args.debias_multiple,
          reset=args.reset,
          reset_counter=args.reset_counter,
          n2v_start=args.n2v_start,
          experiment2=args.experiment2,
          adaptive_alpha=args.adaptive_alpha,
          n2v_adam=args.n2v_adam,
          single=args.single,
          imagenet=args.imagenet,
          train_batch_size=args.train_bs,
          constant_resize=args.constant_resize,
          adaptive_resize=args.adaptive_resize,
          no_class=args.no_class,
          gamma=args.gamma,
          partial_projection=args.partial_projection,
          combined_n2v_lr=args.combined_n2v_lr,
          norm=args.bias_norm,
          constant_alpha=args.constant_alpha,
          jump_alpha=args.jump_alpha,
          linear_alpha=args.linear_alpha,
          mean_debias=args.mean_debias,
          no_limit=args.no_limit,
          dataset=args.dataset,
          parallel=args.parallel,
          gpu_ids=args.gpu_ids,
          switch_modes=not args.probe_eval_off
          )
