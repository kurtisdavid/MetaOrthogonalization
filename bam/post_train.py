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
import coco_dataload
import torch.nn.functional as F

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
    parser.add_argument('-no_log', action='store_true', default=False, help='logging experiments')
    parser.add_argument('-main', action='store_true', default=False, help='train downstream task')
    parser.add_argument('-n2v', action='store_true', default=False, help='train net2vec')
    parser.add_argument('-with_n2v', action='store_true', default=False, help='where models are saved with n2v')
    parser.add_argument('-nonlinear', action='store_true', default=False, help='whether or not to use n2v linear or nonlinear resnet18')
    parser.add_argument('-multiple', action='store_true', default=False, help='do we want to store multiple probes...')
    parser.add_argument('-leakage', action='store_true', default=False, help='measure leakage from logits of trained models')
    parser.add_argument('-parallel', action='store_true', default=False, help='for big training')
    parser.add_argument('-linear_only', action='store_true', default=False, help='only train linear layers for main training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='bam')
    parser.add_argument('--device', type=int, default=0)
    
    parser.add_argument('--main_lr', type=float, default=.01, help='downstream task learning rate')
    parser.add_argument('--n2v_lr', type=float, default=.001, help='n2v learning rate')
    parser.add_argument('--main_epochs', type=int, default=30, help='downstream task epochs')
    parser.add_argument('--n2v_epochs', type=int, default=30, help='n2v epochs')

    parser.add_argument('--train_bs', type=int, default=64, help='batch size')
    parser.add_argument('--test_bs', type=int, default=64, help='eval batch size')
    
    parser.add_argument('--data_dir', type=str, default='./bam_scenes', help='main data repository')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu')
    parser.add_argument('--ratio', type=str, default="0.5", help='bias ratio')
    parser.add_argument('--specific',nargs='+', default=None, help='which class(es) to control bias')
    parser.add_argument('--module',type=str, default='layer3', help='which module to extract from')
    parser.add_argument('--experiment1',type=str, default='post_train', help='type of learning for debias trials')
    parser.add_argument('--experiment2',type=str, default='post_train', help='type of learning for debias trials')

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
    
    parser.add_argument('--model_custom_end', type=str, default='', help='when naming gets bad')
    parser.add_argument('--n2v_custom_end', type=str, default='', help='when naming gets bad')
    parser.add_argument('--model_extra',type=str, default='', help='extra to append to n2v name')
    parser.add_argument('--n2v_extra',type=str, default='', help='extra to append to n2v name')

    parser.add_argument('--verify_ratios',nargs='+', default=["1.0","0.9","0.8","0.75","0.7","0.6","0.5","0.4","0.3","0.25","0.2","0.1","0.0"], help='all the ratios we will test')
    parser.add_argument('--gpu_ids',nargs='+', default=None, help='which gpus to use dataparallel on')

    # logger arguments
    parser.add_argument('--base_folder',type=str, default='./models/BAM/')
    parser.add_argument('--out_file', type=str, default=None) 
    args = parser.parse_args()
    return args

def train_main(trainloader, testloader, device, seed,
               specific=None,
               p = 0.5,
               n_epochs=5,
               lr = 0.1,
               experiment="",
               out_file=None,
               base_folder="",
               dataset="bam",
               parallel=False,
               gpu_ids=[],
               linear_only=False):
    if out_file is not None:
        f = open(out_file,'a')
    else:
        f = None
    print("Downstream Training | Ratio: " + str(p) + " | lr = " + str(lr), file=f)
    num_classes = 10
    if dataset=='coco':
        num_classes = 79
    model = models.resnet_(pretrained=True, custom_path=os.path.join(base_folder,str(seed),"resnet_init.pt"), device=device, num_classes=num_classes, initialize=True,
                           size=50 if (dataset == 'bam' or dataset=='coco') else 34, linear_only=linear_only)
    if parallel:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    optim     = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    def scaler(epoch):
        return 0.75 ** (epoch // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=scaler if dataset == 'coco' else (lambda epoch: 0.95**epoch))
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    if dataset == 'coco':
        object_weights = torch.FloatTensor(trainloader.dataset.getObjectWeights())
        criterion = nn.BCEWithLogitsLoss(weight=object_weights.to(device), reduction='elementwise_mean')
    for e in range(n_epochs):
        if dataset != 'coco':
            with torch.no_grad():
                acc = utils.classification_accuracy(testloader, model, device)
            print("Epoch:", e, "| acc:", acc, file=f)
        else:
            with torch.no_grad():
                f1, mAP = utils.detection_results(testloader, model, device)
            print("Epoch:", e, "| f1:", f1, '| mAP:', mAP, '| lr:', scheduler.get_lr(), file=f)
            if f1 > best_f1:
                save_file = utils.get_model_path(base_folder, seed, specific, 'resnet_best' + str(p) + '_{}_{}.pt'.format(f1, mAP),
                                        experiment=experiment)
                best_f1 = f1
                torch.save(model.state_dict(), save_file)
        model.train()
        for X,y,color in trainloader:
            optim.zero_grad()
            loss = criterion(model(X.to(device)), y.to(device))
            loss.backward()
            optim.step()
        scheduler.step()
    end = time.time()
    print(start-end)
    if dataset == 'bam' or dataset == 'coco':
        if dataset == 'coco':
            with torch.no_grad():
                f1, mAP = utils.detection_results(testloader, model, device)
        # print("final", utils.classification_accuracy(testloader, model, device), file=f)
        save_file = utils.get_model_path(base_folder, seed, specific, 'resnet_' + str(p) + '_{}_{}.pt'.format(f1, mAP),
                                        experiment=experiment)
    else:
        save_file = os.path.join(base_folder, str(seed), experiment, 'resnet.pt')
    torch.save(model.state_dict(), save_file)
    if f is not None:
        f.close()

def train_net2vec(trainloader, testloader, device, seed,
                  specific=None,
                  p=0.5,
                  n_epochs=5,
                  module='layer4',
                  lr=0.5,
                  base_folder="",
                  out_file=None,
                  experiment1="",
                  experiment2="",
                  model_extra="",
                  n2v_extra="",
                  with_n2v=False,
                  nonlinear=False,
                  model_custom_end='',
                  n2v_custom_end='',
                  multiple=False,
                  dataset='bam',
                  parallel=False,
                  gpu_ids=[]):
    if out_file is not None:
        f = open(out_file,'a')
    else:
        f = None
    print("Training N2V | p =", p, file=f)
    if not nonlinear:
        n2v_extra = "n2v" + str(n2v_extra)
    else:
        n2v_extra = "mlp" + str(n2v_extra)
    if len(model_custom_end) > 0:
        model_custom_end = "_" + model_custom_end
    if len(n2v_custom_end) > 0:
        n2v_custom_end = "_" + n2v_custom_end
    if hasattr(trainloader.dataset, 'idx_to_class'):
        for key in trainloader.dataset.idx_to_class:
            if specific is not None and trainloader.dataset.idx_to_class[key] in specific:
                specific_idx = int(key)
    else:
        specific_idx = 0
    if dataset == 'bam' or dataset == 'coco':
        model_path = utils.get_model_path(
            base_folder,
            seed,
            specific,
            "resnet"+str(model_extra)+"_"+str(p)+model_custom_end+".pt",
            experiment=experiment1,
            with_n2v=with_n2v,
            n2v_module=module
        )
        n2v_path = utils.get_net2vec_path(
            base_folder,
            seed,
            specific,
            module,
            "resnet_"+str(n2v_extra)+"_"+str(p)+n2v_custom_end+".pt",
            experiment=experiment2,
        )
    else:
        if with_n2v:
            model_path = os.path.join(base_folder, str(seed), experiment1, module, "resnet" + str(model_extra) + model_custom_end + ".pt")
        else:
            model_path = os.path.join(base_folder, str(seed), experiment1, "resnet" + str(model_extra) + model_custom_end + ".pt")
        n2v_path   = os.path.join(base_folder, str(seed), experiment2, module, 'resnet_' + str(n2v_extra) + n2v_custom_end + ".pt")
    print(model_path, n2v_path)
    num_attributes = 12
    if nonlinear:
        num_attributes = 2
    if multiple:
        num_attributes = 10 + 9 + 2*10
    num_classes = 10
    if dataset == 'coco':
        num_classes = 79
        num_attributes = 81
    model, net, net_forward, activation_probe = models.load_models(
        device,
        lambda x, y, z: models.resnet_(
            pretrained=True, 
            custom_path=x, 
            device=y, 
            num_classes=num_classes, 
            initialize=z, 
            size=50 if (dataset=='bam' or dataset=='coco') else 34
        ),
        model_path=model_path,
        net2vec_pretrained=False,
        module=module,
        num_attributes=num_attributes,
        model_init=False, # don't need to initialize a new one
        n2v_init=True,
        loader=trainloader,
        nonlinear=nonlinear,
        parallel=parallel,
        gpu_ids=gpu_ids
    )
    if dataset == 'coco':
        object_weights = torch.FloatTensor(trainloader.dataset.getObjectWeights())
        gender_weights = torch.FloatTensor(trainloader.dataset.getGenderWeights())
        all_weights = torch.cat([object_weights, gender_weights])
        criterion = nn.BCEWithLogitsLoss(weight=all_weights.to(device), reduction='elementwise_mean')
        #criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = None
    net2vec.train_net2vec(
        model, net, net_forward,
        n_epochs,
        trainloader,
        testloader,
        device,
        lr=lr,
        save_path=n2v_path,
        f = f,
        train_labels=[-2,-1] if nonlinear else None,
        multiple = multiple,
        balanced = False if dataset == 'coco' else True,
        criterion = criterion,
        adam=True if dataset == 'coco' else False,
        leakage=True
    )
    if f is not None:
        f.close()

def train_leakage(trainloader, testloader, device, seed,
                  specific=None,
                  p=0.5,
                  n_epochs=5,
                  module='layer4',
                  lr=0.5,
                  base_folder="",
                  out_file=None,
                  experiment1="",
                  experiment2="",
                  model_extra="",
                  n2v_extra="",
                  with_n2v=False,
                  nonlinear=False,
                  model_custom_end='',
                  n2v_custom_end='',
                  multiple=False,
                  dataset='bam',
                  parallel=False,
                  gpu_ids=[]):
    if out_file is not None:
        f = open(out_file,'a')
    else:
        f = None
    print("Training Model Leakage | p =", p, file=f)
    if not nonlinear:
        n2v_extra = "n2v" + str(n2v_extra)
    else:
        n2v_extra = "mlp" + str(n2v_extra)
    if len(model_custom_end) > 0:
        model_custom_end = "_" + model_custom_end
    if len(n2v_custom_end) > 0:
        n2v_custom_end = "_" + n2v_custom_end
    if hasattr(trainloader.dataset, 'idx_to_class'):
        for key in trainloader.dataset.idx_to_class:
            if specific is not None and trainloader.dataset.idx_to_class[key] in specific:
                specific_idx = int(key)
            else:
                specific_idx = 0
    else:
        specific_idx = 0
    if dataset == 'bam' or dataset=='coco':
        model_path = utils.get_model_path(
            base_folder,
            seed,
            specific,
            "resnet"+str(model_extra)+"_"+str(p)+model_custom_end+".pt",
            experiment=experiment1,
            with_n2v=with_n2v,
            n2v_module=module
        )
        n2v_path = utils.get_net2vec_path(
            base_folder,
            seed,
            specific,
            module, 
            "leakage/resnet_"+str(n2v_extra)+"_"+str(p)+n2v_custom_end+".pt",
            experiment=experiment2,
        )
    else:
        if with_n2v:
            model_path = os.path.join(base_folder, str(seed), experiment1, module, "resnet" + str(model_extra) + model_custom_end + ".pt")
        else:
            model_path = os.path.join(base_folder, str(seed), experiment1, "resnet" + str(model_extra) + model_custom_end + ".pt")
        n2v_path   = os.path.join(base_folder, str(seed), experiment2, module, 'leakage/resnet_' + str(n2v_extra) + n2v_custom_end + ".pt")
     
    if dataset == 'bam':
        if specific is not None and not isinstance(specific, str):
            folder_name = '.'.join(
                sorted(specific)
            )
        else:
            folder_name = specific
        leakage_folder = os.path.join(
            str(base_folder),
            str(seed),
            folder_name,
            str(experiment2),
            str(module),
            'leakage'
        )
    else:
        leakage_folder = os.path.join(
            str(base_folder),
            str(seed),
            str(experiment2),
            str(module),
            'leakage'
        )
    if not os.path.isdir(leakage_folder):
        os.mkdir(leakage_folder)
    num_classes = 10
    if dataset == 'coco':
        num_classes = 79
    num_attributes = 2
    model, net, net_forward, activation_probe = models.load_models(
        device,
        lambda x, y, z: models.resnet_(
            pretrained=True, 
            custom_path=x, 
            device=y, 
            num_classes=num_classes, 
            initialize=z, 
            size=50 if (dataset=='bam' or dataset=='coco') else 34),
        model_path=model_path,
        net2vec_pretrained=False,
        module='fc', # leakage will come from the output logits...
        num_attributes=num_attributes,
        model_init=False, # don't need to initialize a new one
        n2v_init=True,
        loader=trainloader,
        nonlinear=nonlinear,
        parallel=parallel,
        gpu_ids=gpu_ids
    )
    def criterion(logits, genders):
        return F.cross_entropy(logits, genders[:, 1].long(), reduction='elementwise_mean')
    net2vec.train_net2vec(
        model, net, net_forward,
        n_epochs,
        trainloader,
        testloader,
        device,
        lr=lr,
        save_path=n2v_path,
        f = f,
        train_labels=[-2,-1],
        balanced = False,
        criterion = criterion,
        specific = specific_idx,
        adam=True,
        save_best=True,
        leakage=True
    )
    if f is not None:
        f.close()

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    # setup our datasets...
    if args.dataset == 'bam':
        if args.specific is not None and len(args.specific) == 1:
            args.specific = args.specific[0]
        if str(args.ratio) not in args.verify_ratios:
            args.verify_ratios = sorted(args.verify_ratios + [str(args.ratio)], reverse=True) 
        setup_datasets.setupBAM(args.seed,True,args.specific,ratios=args.verify_ratios)
        setup_datasets.setupBAM(args.seed,False,args.specific,ratios=args.verify_ratios)
        # now use the given ratio
        trainloader, _ = dataload.get_data_loader_SceneBAM(seed=args.seed,ratio=float(args.ratio), specific=args.specific)
        _, testloader = dataload.get_data_loader_SceneBAM(seed=args.seed,ratio=float("0.5"), specific=args.specific)
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
  
    
    # setup model directories
    if args.dataset == 'bam':
        if not isinstance(args.specific, str):
            folder_name = '.'.join(
                sorted(args.specific)
            )
        else:
            folder_name = args.specific
        dir_parts = [args.base_folder, str(args.seed), folder_name, str(args.experiment1), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))

        dir_parts = [args.base_folder, str(args.seed), folder_name, str(args.experiment2), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))
        
        dir_parts = [args.base_folder, str(args.seed), folder_name, str(args.experiment2), str(args.module), 'leakage']
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))
    else:
        dir_parts = [args.base_folder, str(args.seed), str(args.experiment1), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))

        dir_parts = [args.base_folder, str(args.seed), str(args.experiment2), str(args.module)]
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))
        
        dir_parts = [args.base_folder, str(args.seed), str(args.experiment2), str(args.module), 'leakage']
        for i in range(len(dir_parts)):
            if not os.path.isdir(os.path.join(*dir_parts[:i+1])):
                os.mkdir(os.path.join(*dir_parts[:i+1]))

    if args.parallel:
        args.gpu_ids = [int(args.device)] + [int(x) for x in list(args.gpu_ids)]
    if args.main:
        train_main(trainloader, testloader, device, args.seed,
                  specific=args.specific,
                  p=args.ratio,
                  n_epochs=args.main_epochs,
                  lr=args.main_lr,
                  out_file=args.out_file,
                  base_folder=args.base_folder,
                  experiment=args.experiment1,
                  dataset=args.dataset,
                  parallel=args.parallel,
                  gpu_ids=args.gpu_ids,
                  linear_only=args.linear_only)
    if args.n2v:
        train_net2vec(trainloader, testloader, device, args.seed,
                      specific=args.specific,
                      p=args.ratio,
                      n_epochs=args.n2v_epochs,
                      module=args.module,
                      lr=args.n2v_lr,
                      out_file=args.out_file,
                      base_folder=args.base_folder,
                      experiment1=args.experiment1,
                      experiment2=args.experiment2,
                      model_extra = args.model_extra,
                      n2v_extra = args.n2v_extra,
                      with_n2v=args.with_n2v,
                      nonlinear=args.nonlinear,
                      model_custom_end=args.model_custom_end,
                      n2v_custom_end=args.model_custom_end,
                      multiple=args.multiple,
                      dataset=args.dataset,
                      parallel=args.parallel,
                      gpu_ids=args.gpu_ids)
    if args.leakage:
        train_leakage(trainloader, testloader, device, args.seed,
                      specific=args.specific,
                      p=args.ratio,
                      n_epochs=args.n2v_epochs,
                      module=args.module,
                      lr=args.n2v_lr,
                      out_file=args.out_file,
                      base_folder=args.base_folder,
                      experiment1=args.experiment1,
                      experiment2=args.experiment2,
                      model_extra = args.model_extra,
                      n2v_extra = args.n2v_extra,
                      with_n2v=args.with_n2v,
                      nonlinear=args.nonlinear,
                      model_custom_end=args.model_custom_end,
                      n2v_custom_end=args.model_custom_end,
                      dataset=args.dataset,
                      parallel=args.parallel,
                      gpu_ids=args.gpu_ids)
