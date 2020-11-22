import argparse
import torch
import metrics
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import seaborn as sns
from cycler import cycler
import dataload
plt.style.use('ggplot')
ggplot_colors = [color for color in plt.rcParams['axes.prop_cycle']]
map_ = [1,2,0] + list(range(3,len(ggplot_colors)))
ggplot_colors = np.array(ggplot_colors)[map_].tolist()
plt.rcParams['axes.prop_cycle'] = cycler(color=[c['color'] for c in ggplot_colors])

class_to_idx = {
    'bowling_alley': 0,
    'runway': 1,
    'track': 2,
    'bamboo_forest': 3,
    'cockpit': 4,
    'ski_slope': 5,
    'bedroom': 6,
    'bus_interior': 7,
    'laundromat': 8,
    'corn_field': 9,
    'truck': 10,
    'zebra': 11
}
idx_to_class = {
    v:k for k,v in class_to_idx.items()
}
models = ['baseline','debias','adv']

def get_args():
    # General system running and configuration options
    parser = argparse.ArgumentParser(description='settings to compile all metric information, given provided model settings')

    # computing or producing final results
    parser.add_argument('-final', action='store_true', default=False, help='create final products (results/pdfs/etc)')
    parser.add_argument('-comparison', action='store_true', default=False, help='only valid if final, compares 1/3/5/7/10 class versions')
    parser.add_argument('-aggregated', action='store_true', default=False, help='counted maxes')

    # type of experiment
    parser.add_argument('-baseline', action='store_true', default=False, help='baseline experiment')
    parser.add_argument('-debias', action='store_true', default=False, help='debias experiment')
    parser.add_argument('-adv', action='store_true', default=False, help='adv experiment')

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

    # experiment settings
    # boolean settings
    parser.add_argument('-multiple', action='store_true', default=False, help='if using multiple probes')
    parser.add_argument('-post', action='store_true', default=False, help='if using probes not used during training')
    parser.add_argument('-fresh', action='store_true', default=False, help='if we want to start fresh and NOT use cache')
    # nonboolean settings
    parser.add_argument('--experiment',type=str, default='post_train', help='folder with our model')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--module',type=str, default=None, help='where to apply regularization')
    parser.add_argument('--debias_extra',type=str, default="", help="if we want more specific file naming")
    parser.add_argument('--adv_extra',type=str, default="", help="if we want more specific file naming")
    parser.add_argument('--class_',type=int, default=None, help='which class to graph truck/zebra results')
    parser.add_argument('--dataset', type=str, default='bam', help='bam or idenprof')
    parser.add_argument('--custom_end',type=str,default='',help='hparamappend')
    # list settings
    parser.add_argument('--specific',nargs='+', default=None, help='which class(es) to control bias')
    parser.add_argument('--ratios', nargs='+', default=["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"], help='Which ratios did you test experiment on')
    parser.add_argument('--epochs', nargs='+', default=[None], help='have custom epochs if desired, default will obtain model from the end')
    parser.add_argument('--seeds',  nargs='+', default=[0, 1, 2, 3, 4], help='which trials')
    parser.add_argument('--models',  nargs='+', type=int, default=[0, 1, 2], help='which models to collect from')
    parser.add_argument('--metrics', nargs='+', default=['accuracy','projection','tcav','leakage'])

    parser.add_argument('--specifics',nargs='+', default=None, help='which specifics to make comparison on')
    

    # logger arguments
    parser.add_argument('--base_folder',type=str, default='./models/BAM/')
    
    args = parser.parse_args()
    return args


def prepare_save(args,
                 file_end=None):
    specifics = args.specifics
    if args.specifics is None:
        if args.dataset == 'bam':
            specifics = ['.'.join(sorted(args.specific))]
        else:
            specifics = [None]
    probe_filenames = []
    general_filenames = []
    for specific in specifics:
        # ensure the folder to save directories ready
        if args.dataset == 'bam':
            probe_dir_parts = [args.base_folder.replace('models','final_results'),
                        specific, 
                        str(args.module),
                        'post' if args.post else 'during']
        else:
            probe_dir_parts = [args.base_folder.replace('models','final_results'),
                        str(args.module),
                        'post' if args.post else 'during']
        for i in range(len(probe_dir_parts)):
            if not os.path.isdir(os.path.join(*probe_dir_parts[:i+1])):
                os.mkdir(os.path.join(*probe_dir_parts[:i+1]))
        if file_end is None:
            file_end = 'baseline' if args.baseline else (
                'debias'+args.debias_extra if args.debias else 'adv' + args.adv_extra
            )
        if args.multiple:
            file_end += '_multiple'
        probe_filename = os.path.join(
            *probe_dir_parts,
            file_end + '.pck'
        )
        general_filename = os.path.join(
            *probe_dir_parts[:-1],
            file_end + '.pck'
        )
        probe_filenames.append(probe_filename)
        general_filenames.append(general_filename)
    if len(probe_filenames) == 1 and len(general_filenames) == 1:
        return probe_filenames[0], general_filenames[0]
    else:
        return probe_filenames, general_filenames

def update_projection_data(device,
                    args,
                    results={},
                    filename="",
                    force=False):
    print("Collecting Projection Data")
    if 'projection' not in results:
        res = {}
        results['projection'] = res
    res = results['projection']
    for epoch in args.epochs:
        if epoch not in res:
            res[epoch] = {}
        for seed in args.seeds:
            if seed not in res[epoch]:
                try:
                    res[epoch][seed] = metrics.collect_projection(
                        device,
                        base_folder=args.base_folder,
                        specific=args.specific, 
                        seed=seed, 
                        module=args.module,
                        experiment=args.experiment,
                        ratios=args.ratios,
                        adv=args.adv,
                        baseline=args.baseline,
                        epoch=epoch,
                        post=args.post,
                        multiple=args.multiple,
                        force=force,
                        dataset=args.dataset
                    )
                    # save after each just in case it crashes...
                    with open(filename, 'wb') as f:
                        pickle.dump(results, f)
                except:
                    assert not force


def update_tcav_data(device,
              args,
              results={},
              filename="",
              force=False):
    print("Collecting TCAV Data")
    if 'tcav' not in results:
        res = {}
        results['tcav'] = res
    res = results['tcav']
    for epoch in args.epochs:
        if epoch not in res:
            res[epoch] = {}
        for seed in args.seeds:
            if seed not in res[epoch]:
                try:
                    res[epoch][seed] = metrics.collect_tcav(
                        device,
                        base_folder=args.base_folder,
                        specific=args.specific, 
                        seed=seed, 
                        module=args.module,
                        experiment=args.experiment,
                        ratios=args.ratios,
                        adv=args.adv,
                        baseline=args.baseline,
                        epoch=epoch,
                        post=args.post,
                        multiple=args.multiple,
                        force=force,
                        dataset=args.dataset,
                        args=args
                    )
                    # save after each just in case it crashes...
                    with open(filename, 'wb') as f:
                        pickle.dump(results, f)
                except Exception as e:
                    print(e)
                    exit(-1)
                    assert not force
            

def update_accuracy_data(device,
              args,
              results={},
              filename=""):
    print("Collecting Accuracy Data")
    if 'accuracy' not in results:
        res = {}
        results['accuracy'] = res
    res = results['accuracy']
    for epoch in args.epochs:
        if epoch not in res:
            res[epoch] = metrics.collect_accuracies(
                device,
                base_folder=args.base_folder,
                specific=args.specific, 
                seeds=args.seeds, 
                module=args.module,
                experiment=args.experiment,
                ratios=args.ratios,
                adv=args.adv,
                baseline=args.baseline,
                post=args.post,
                multiple=args.multiple,
                epoch=epoch,
                dataset=args.dataset,
                args=args
            )
            # save after each just in case it crashes...
            print("saving" + filename + "...")
            print(results)
            with open(filename, 'wb') as f:
                pickle.dump(results, f)

def update_leakage_data(device,
              args,
              results={},
              filename="",
              force=False):
    print("Collecting Leakage Data")
    if 'leakage' not in results:
        res = {}
        results['leakage'] = res
    res = results['leakage']
    for epoch in args.epochs:
        if epoch not in res:
            res[epoch] = {}
        for seed in args.seeds:
            if seed not in res[epoch]:
                try:
                    res[epoch][seed] = metrics.collect_leakage(
                        device,
                        base_folder=args.base_folder,
                        specific=args.specific, 
                        seed=seed, 
                        module=args.module,
                        experiment=args.experiment,
                        ratios=args.ratios,
                        adv=args.adv,
                        baseline=args.baseline,
                        multiple=args.multiple,
                        epoch=epoch,
                        force=force,
                        dataset=args.dataset,
                        args=args
                    )
                    # save after each just in case it crashes...
                    with open(filename, 'wb') as f:
                        pickle.dump(results, f)
                except:
                    assert not force

def collect_results(device,
                    args,
                    probe_result={},
                    probe_filename="",
                    general_result={},
                    general_filename="",
                    # a way to control whether to run the post training scripts
                    force=False):
    # should only need to be run once, unlike the rest
    if not force:
        # Class Specific Data
        if 'accuracy' in args.metrics:
            update_accuracy_data(
                device,
                args,
                results=general_result,
                filename=general_filename
            )
    # Run projection data, this is one that can require a post trained n2v
    if 'projection' in args.metrics:
        update_projection_data(
            device, 
            args, 
            results=probe_result,
            filename=probe_filename,
            force=force
        )
    # Collect TCAV dfs, using same as above
    if 'tcav' in args.metrics:
        update_tcav_data(
            device, 
            args, 
            results=probe_result,
            filename=probe_filename,
            force=False
        )
    if 'leakage' in args.metrics:
        update_leakage_data(
            device,
            args,
            results=general_result,
            filename=general_filename,
            force=force
        )
    '''
    TODO:
        - Fairness metrics? <- save in directory shared with post/during
    '''

def obtain_results(probe_filename,
                   general_filename):
    probe_result = {}
    general_result = {}
    if os.path.exists(probe_filename):
        with open(probe_filename, 'rb') as f:
            probe_result = pickle.load(f)
    if os.path.exists(general_filename):
        with open(general_filename, 'rb') as f:
            general_result = pickle.load(f)
    return probe_result, general_result

def save_xls(list_dfs, xls_path, sheet_names):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,sheet_names[n])
        writer.save()

def process_accuracies(
    args,
    files,
    df_list=[]
):
    truck_df = pd.DataFrame(columns=['class','model'] + args.ratios)
    zebra_df = pd.DataFrame(columns=['class','model'] + args.ratios)
    if args.specific == None:
        specific_class = list(range(10))
    else:
        specific_class = [class_to_idx[specific] for specific in args.specific]
    for s_idx in specific_class:
        start = len(truck_df)
        for i,(filename,model) in enumerate(zip(files,models)):
            with open(filename, 'rb') as f:
                res = pickle.load(f)
            if 'accuracy' not in res or 'accuracy' not in args.metrics:
                df_list.append(truck_df)
                df_list.append(zebra_df)
                return
            accs = metrics.process_subset_results_specific(
                    res['accuracy'][None],
                    s_idx,
                    ratios=args.ratios,
                    seeds=args.seeds)
            for j in range(2):
                truck_df.loc[start+i*2 + j,args.ratios] = accs[j][0,:]
                zebra_df.loc[start+i*2 + j,args.ratios] = accs[j][1,:]
                truck_df.loc[start+i*2 + j,['model']] = model
                zebra_df.loc[start+i*2 + j,['model']] = model
                truck_df.loc[start+i*2 + j,['class']] = s_idx
                zebra_df.loc[start+i*2 + j,['class']] = s_idx
    
    df_list.append(truck_df)
    df_list.append(zebra_df)

def process_leakage(
    args,
    files,
    df_list=[]
):
    leakage_df = pd.DataFrame(columns=['model'] + args.ratios)
    for i,(filename,model) in enumerate(zip(files,models)):
        with open(filename, 'rb') as f:
            res = pickle.load(f)
        if 'leakage' not in res or 'leakage' not in args.metrics:
            df_list.append(leakage_df)
            return
        lkg = metrics.process_leakage_results(
                res['leakage'][None],
                ratios=args.ratios,
                seeds=args.seeds)
        for j in range(2):
            leakage_df.loc[i*2 + j,args.ratios] = lkg[j]
            leakage_df.loc[i*2 + j,['model']] = model
    
    df_list.append(leakage_df)

def process_projections(
    args,
    files,
    df_list=[]
):
    projections_df = pd.DataFrame(columns=['class','model','epoch'] + args.ratios)

    if args.specific == None:
        specific_class = list(range(10))
    else:
        specific_class = [class_to_idx[specific] for specific in args.specific]
    for s_idx in specific_class:
        start = len(projections_df)
        for i,(filename,model) in enumerate(zip(files,models)):
            with open(filename, 'rb') as f:
                res = pickle.load(f)
            if 'projection' not in res or 'projection' not in args.metrics:
                df_list.append(projections_df)
                return
            print(filename)
            for epoch in res['projection']:
                valid_ratios = []
                proj = metrics.process_projection_results(
                        res['projection'][epoch],
                        s_idx,
                        ratios=args.ratios,
                        valid_ratios=valid_ratios,
                        seeds=args.seeds)
                print(valid_ratios, proj)
                for j in range(2):
                    projections_df.loc[start + j,sorted(valid_ratios)] = proj[j]
                    projections_df.loc[start + j,['model']] = model
                    projections_df.loc[start + j,['class']] = s_idx
                    projections_df.loc[start + j,['epoch']] = epoch
                start += 2
    
    df_list.append(projections_df)

def process_tcav(
    args,
    files,
    df_list=[]
):
    tcav_df = pd.DataFrame(columns=['class', 'model'] + args.ratios)
    if args.specific == None:
        specific_class = list(range(10))
    else:
        specific_class = [class_to_idx[specific] for specific in args.specific]
    for s_idx in specific_class:
        start = len(tcav_df)
        for i,(filename,model) in enumerate(zip(files,models)):
            with open(filename, 'rb') as f:
                res = pickle.load(f)
            if 'tcav' not in res or 'tcav' not in args.metrics:
                df_list.append(tcav_df)
                return
            valid_ratios = []
            tcav = metrics.process_tcav_results(
                    res['tcav'][None],
                    s_idx,
                    ratios=args.ratios,
                    valid_ratios=valid_ratios,
                    seeds=args.seeds)
            for j in range(2):
                tcav_df.loc[start+i*2 + j,valid_ratios] = tcav[j]
                tcav_df.loc[start+i*2 + j,['model']] = model
                tcav_df.loc[start+i*2 + j,['class']] = s_idx
        
    df_list.append(tcav_df)

def process_results(
    args
):
    desired_models = [model for model in models]
    if len(args.debias_extra) > 0:
        desired_models[1] += args.debias_extra
    if len(args.adv_extra) > 0:
        desired_models[2] += args.adv_extra
    
    desired_models = [desired_models[i] for i in args.models]
    filenames = [prepare_save(args, file_end=model) for model in desired_models]
    probe_files   = [x[0] for x in filenames]
    general_files = [x[1] for x in filenames]

    df_list = []
    sheet_names = [
        'Truck Acc',
        'Zebra Acc',
        'Leakage',
        'Projection',
        'TCAV'
    ]
    # get truck/zebra specific accuracies
    process_accuracies(
        args,
        general_files, 
        df_list
    )
    # get maximum leakage of truck/zebra class
    process_leakage(
        args,
        general_files,
        df_list
    )
    # get projections on bias
    process_projections(
        args,
        probe_files,
        df_list
    )
    # get tcav projections on bias
    process_tcav(
        args,
        probe_files,
        df_list
    )
    # Now create an excel sheet
    if args.post:
        end = 'results_post.xlsx'
    else:
        end = 'results.xlsx'
    xls_filename = '/'.join(general_files[0].split('/')[:-1] + [end])
    print(xls_filename)
    save_xls(df_list, xls_filename, sheet_names)

def accuracy_curves(args,
                    desired_models,
                    general_files):
    
    markers = ['o','^','p','P','*']
    colors = ['k', 'g', 'b']
    colors = colors[:len(desired_models)]
    titles = ['Standard', 'Meta Orthogonalization', 'Adversarial']
    titles = titles[:len(desired_models)]
    labels = ['1 Class', '3 Class', '5 Class', '7 Class', '10 Class']
    fig, axes = plt.subplots(2,3,figsize=(30,20))
    opportunity_diff = {
        "model": [],
        "ratio": [],
        "diff": [],
        "specific": []
    }
    for j,specific in enumerate(args.specifics):
        for i,(filename,model) in enumerate(zip(general_files,desired_models)):
            with open(filename[j], 'rb') as f:
                res = pickle.load(f)
            accs, stds = metrics.process_subset_results_specific(
                    res['accuracy'][None],
                    args.class_,
                    ratios=args.ratios,
                    seeds=args.seeds)
            for k,ratio in enumerate(args.ratios):
                opportunity_diff["model"].append(model)
                opportunity_diff["ratio"].append(ratio)
                opportunity_diff["diff"].append(np.abs(accs[0][k] - accs[1][k]))
                opportunity_diff["specific"].append(specific)
            axes[0][i].plot(args.ratios, accs[0], marker=markers[j], label=labels[j])
            axes[1][i].plot(args.ratios, accs[1], marker=markers[j], label=labels[j])
    for i in range(axes.shape[1]):
        ax0 = axes[0][i]
        ax1 = axes[1][i]
        ax0.legend(fontsize=18)
        ax0.set_ylim([0.5,1.0])
        ax0.set_title(titles[i], fontsize=24)
        ax1.legend(fontsize=18)
        ax1.set_ylim([0.5,1.0])
        ax1.set_title(titles[i], fontsize=24)
    if args.class_ is None:
        plt.savefig('graphs/overall_accuracies.pdf', bbox_inches='tight')
    else:
        plt.savefig('graphs/%d_accuracies.pdf' % args.class_, bbox_inches='tight')
    
    df  = pd.DataFrame.from_dict(opportunity_diff)
    g = sns.catplot(x="ratio", y="diff",
                hue="model", row="specific",
                data=df, kind="bar",
                height=4, aspect=1.2)
    if args.class_ is None:
        plt.savefig('graphs/overall_opp.pdf', bbox_inches='tight')
    else:
        plt.savefig('graphs/%d_opp.pdf' % args.class_, bbox_inches='tight')

    # ~~~~~~~~~~~~~~~ #

    fig, axes = plt.subplots(2,1,figsize=(30,20))
    correlations = [[] for _ in range(len(desired_models))]
    for j,specific in enumerate(args.specifics):
        for i,(filename,model) in enumerate(zip(general_files,desired_models)):
            with open(filename[j], 'rb') as f:
                res = pickle.load(f)
            accs, stds = metrics.process_subset_results_specific(
                    res['accuracy'][None],
                    args.class_,
                    ratios=args.ratios,
                    seeds=args.seeds)
            _, _, truck_r_value, _, _ = scipy.stats.linregress([float(r) for r in args.ratios], accs[0])
            _, _, zebra_r_value, _, _ = scipy.stats.linregress([float(r) for r in args.ratios], accs[1])
            correlations[i].append([truck_r_value, zebra_r_value])
    # should have shape (3, 5, 2)
    correlations = np.array(correlations)
    x = np.arange(5)
    width = 0.2
    for i in range(2):
        ax = axes[i]
        corr = correlations[:,:,i]
        for j in range(3):
            ax.bar(x + (width * (j-1)), corr[j], width, label=titles[j])
            # for k, v in enumerate(corr[j]):
            #     ax.text(x[k] + (width * (j-1)), corr[j][k] + .25, str(corr[j][k]), color='blue', fontweight='bold')
        ax.legend()
        ax.set_ylabel('Ratio:Accuracy Correlation')
        ax.set_ylim(-1,1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('# Classes with Bias Ratio')
    
    axes[0].set_title('Truck Correlations')
    axes[1].set_title('Zebra Correlations')
    if args.class_ is None:
        plt.savefig('graphs/overall_correlations.pdf', bbox_inches='tight')
    else:
        plt.savefig('graphs/%d_correlations.pdf' % args.class_, bbox_inches='tight')

    # aggregated
    if not args.aggregated:
        return
    fig, axes = plt.subplots(2,1,figsize=(30,20))
    correlations = {
        specific: [
            [] for _ in range(len(desired_models))
        ] for specific in args.specifics
    }
    for i,(filename,model) in enumerate(zip(general_files,desired_models)):
        for j,specific in enumerate(args.specifics):
            with open(filename[j], 'rb') as f:
                res = pickle.load(f)
            for class_ in range(10):
                accs, stds = metrics.process_subset_results_specific(
                        res['accuracy'][None],
                        class_,
                        ratios=args.ratios,
                        seeds=args.seeds)
                _, _, truck_r_value, _, _ = scipy.stats.linregress([float(r) for r in args.ratios], accs[0])
                _, _, zebra_r_value, _, _ = scipy.stats.linregress([float(r) for r in args.ratios], accs[1])
                correlations[specific][i].append([truck_r_value, zebra_r_value])
    total_counts = [
        [
            [] for _ in range(len(desired_models))
        ] for _ in range(2)
    ]
    for specific in correlations:
        # should have shape (# models x # classes x 2 (truck/zebra))
        all_res = np.array(correlations[specific])
        # we want to find, for each class, which model has the lowest in absolute value
        for j in range(2):
            maxes = np.argmax(-np.abs(all_res[:,:,j]), axis=0)
            unique, counts = np.unique(maxes, return_counts=True)
            dict_cts = {
                unique[ii]: counts[ii] for ii in range(len(unique)) 
            }
            cts = [
                0 if ii not in dict_cts else dict_cts[ii] for ii in range(len(desired_models))
            ]
            for i,i_ct in enumerate(cts):
                total_counts[j][i].append(i_ct)
    
    for i in range(2):
        ax = axes[i]
        corr = total_counts[i]
        for j in range(3):
            ax.bar(x + (width * (j-1)), corr[j], width, color=ggplot_colors[(j+1)%len(ggplot_colors)]['color'], label=titles[j])
            # for k, v in enumerate(corr[j]):
            #     ax.text(x[k] + (width * (j-1)), corr[j][k] + .25, str(corr[j][k]), color='blue', fontweight='bold')
        ax.legend()
        ax.set_ylabel('# Classes Model has Smallest Linear Correlation')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('# Classes with Bias Ratio')
    
    axes[0].set_title('Ranking of Models for Truck Images')
    axes[1].set_title('Ranking of Models for Zebra Images')
    plt.savefig('graphs/aggregated_correlations.pdf', bbox_inches='tight')

def projection_curves(args,
    desired_models,
    probe_files
):
    projections = {
        "specific" : [],
        "model": [],
        "ratio": [],
        "class": [],
        "projection": []
    }
    titles = ['Standard', 'Meta Orthogonalization', 'Adversarial']
    titles = titles[:len(desired_models)]
    if args.dataset == 'bam':
        labels = ['1 Class', '3 Class', '5 Class', '7 Class', '10 Class']
    else:
        labels = ['IdenProf']
    for j,specific in enumerate(args.specifics):
        for i,(filename,model) in enumerate(zip(probe_files,desired_models)):
            actual_filename = filename[j] if args.dataset == 'bam' else filename
            with open(actual_filename, 'rb') as f:
                res = pickle.load(f)
            # print(res['projection'])
            for class_ in range(10):
                valid_ratios = []
                proj, _ = metrics.process_projection_results(
                        res['projection'][None],
                        class_,
                        ratios=args.ratios,
                        valid_ratios=valid_ratios,
                        seeds=args.seeds)
                for p, ratio in zip(proj, valid_ratios):
                    projections["specific"].append(specific)
                    projections["model"].append(model)
                    projections["ratio"].append(ratio)
                    projections["class"].append(class_)
                    projections["projection"].append(p)
    df = pd.DataFrame.from_dict(projections)

    if args.dataset == 'bam':
        for specific, label in zip(args.specifics, labels):
            plt.figure(figsize=(50,50))
            specific_df = df[
                df["specific"] == specific
            ]
            g = sns.FacetGrid(specific_df, row="model", col="ratio",  hue="class")
            g = (g.map(plt.scatter, "projection", "class", edgecolor="w")
                    .set(xlim=(-1,1))
                    .add_legend()
                )
            axes = g.axes

            for i in range(5):
                axes[0,i].set_title(r"$\rho_{\mathcal{K}}$" + " = %s" % args.ratios[i])
                axes[1,i].set_title("")
                axes[2,i].set_title("")

                axes[0,i].tick_params(labelbottom=True)
                axes[1,i].tick_params(labelbottom=True)
                axes[2,i].tick_params(labelbottom=True)

                if i != 2:
                    axes[2,i].set_xlabel("")
                else:
                    axes[2,i].set_xlabel("Bias Projection", fontsize=14)
            
            for i in range(len(titles)):
                axes[i,0].set_ylabel(titles[i], fontsize=14)

            plt.savefig("graphs/proj/" + label + " - Projection.pdf", bbox_inches='tight')
    else:
        fig = plt.figure(figsize=(30,10))
        g = sns.FacetGrid(df, col="model", hue="class")
        g = (g.map(plt.scatter, "projection", "class", edgecolor="w")
                .set(xlim=(-1,1))
                .add_legend()
                # .set(yticks=[])
            )
        g.fig.subplots_adjust(wspace=.2, hspace=.2)

        axes = g.axes.flatten()
        for i in range(3):
            axes[i].set_title(titles[i])
            axes[i].tick_params(labelleft=False, left='off', which='both')

            if i != 1:
                axes[i].set_xlabel("")
            else:
                axes[i].set_xlabel("Bias Projection", fontsize=14)

            axes[i].set_ylabel("")
        
        parts = ["graphs","idenprof"]
        for c in range(len(parts)):
            if not os.path.isdir(os.path.join(*parts[:c+1])):
                os.mkdir(os.path.join(*parts[:c+1]))
        
        plt.savefig(os.path.join(*parts,"Projection.pdf"), bbox_inches='tight')
        plt.close(fig)

def tcav_curves(args,
    desired_models,
    probe_files
):
    projections = {
        "specific" : [],
        "model": [],
        "ratio": [],
        "class": [],
        "tcav": []
    }
    titles = ['Standard', 'Meta Orthogonalization', 'Adversarial']
    titles = titles[:len(desired_models)]
    if args.dataset == 'bam':
        labels = ['1 Class', '3 Class', '5 Class', '7 Class', '10 Class']
    elif args.dataset == 'coco':
        labels = ['Coco']
    else:
        labels = ['IdenProf']
    for j,specific in enumerate(args.specifics):
        for i,(filename,model) in enumerate(zip(probe_files,desired_models)):
            actual_filename = filename[j] if args.dataset == 'bam' else filename
            with open(actual_filename, 'rb') as f:
                res = pickle.load(f)
            # print(res['projection'])
            for class_ in range(10):
                valid_ratios = []
                tcav,_  = metrics.process_tcav_results(
                    res['tcav'][None],
                    class_,
                    ratios=args.ratios,
                    valid_ratios=valid_ratios,
                    seeds=args.seeds)
                for p, ratio in zip(tcav, valid_ratios):
                    projections["specific"].append(specific)
                    projections["model"].append(model)
                    projections["ratio"].append(ratio)
                    projections["class"].append(class_)
                    projections["tcav"].append(p)
    df = pd.DataFrame.from_dict(projections)

    if args.dataset == 'bam':
        for specific, label in zip(args.specifics, labels):
            plt.figure(figsize=(50,30))
            specific_df = df[
                df["specific"] == specific
            ]
            g = sns.FacetGrid(specific_df, row="model", col="ratio",  hue="class")
            g = (g.map(plt.scatter, "tcav", "class", edgecolor="w")
                    .set(xlim=(-1,1))
                    .add_legend()
                )
            axes = g.axes

            for i in range(5):
                axes[0,i].set_title(r"$\rho_{\mathcal{K}}$" + " = %s" % args.ratios[i])
                axes[1,i].set_title("")
                axes[2,i].set_title("")

                axes[0,i].tick_params(labelbottom=True)
                axes[1,i].tick_params(labelbottom=True)
                axes[2,i].tick_params(labelbottom=True)

                if i != 2:
                    axes[2,i].set_xlabel("")
                else:
                    axes[2,i].set_xlabel("Sensitivity Bias", fontsize=14)
            
            for i in range(len(titles)):
                axes[i,0].set_ylabel(titles[i], fontsize=14)

            plt.savefig("graphs/tcav/" + label + " - Sensitivity Bias.pdf", bbox_inches='tight')
    else:
        fig = plt.figure(figsize=(30,10))
        g = sns.FacetGrid(df, col="model", hue="class")
        g = (g.map(plt.scatter, "tcav", "class", edgecolor="w")
                .set(xlim=(-1,1))
                .add_legend()
                # .set(yticks=[])
            )
        g.fig.subplots_adjust(wspace=.2, hspace=.2)

        axes = g.axes.flatten()
        for i in range(3):
            axes[i].set_title(titles[i])
            axes[i].tick_params(labelleft=False, left='off', which='both')

            if i != 1:
                axes[i].set_xlabel("")
            else:
                axes[i].set_xlabel("Sensitivity Bias", fontsize=14)

            axes[i].set_ylabel("")
        
        parts = ["graphs","idenprof"]
        for c in range(len(parts)):
            if not os.path.isdir(os.path.join(*parts[:c+1])):
                os.mkdir(os.path.join(*parts[:c+1]))
        
        plt.savefig(os.path.join(*parts,"Sensitivity Bias.pdf"), bbox_inches='tight')
        plt.close(fig)

def leakage_curves(args,
    desired_models,
    general_files
):
    results = {
        "specific" : [],
        "model": [],
        "ratio": [],
        "leakage": [],
        "trial": []
    }
    titles = ['Standard', 'Meta Orthogonalization', 'Adversarial']
    titles = titles[:len(desired_models)]
    if args.dataset == 'bam':
        labels = ['1 Class', '3 Class', '5 Class', '7 Class', '10 Class']
    else:
        labels = ['IdenProf']
    for j,specific in enumerate(args.specifics):
        for i,(filename,model) in enumerate(zip(general_files,desired_models)):
            actual_filename = filename[j] if args.dataset == 'bam' else filename
            with open(actual_filename, 'rb') as f:
                res = pickle.load(f)
            # print(res['projection'])
            valid_ratios = []
            lkg = metrics.process_leakage_results(
                    res['leakage'][None],
                    ratios=args.ratios,
                    valid_ratios=valid_ratios,
                    seeds=args.seeds,
                    reduce=False)
            for trial in range(lkg.shape[0]):
                for l, ratio in zip(lkg[trial], valid_ratios):
                    results["specific"].append(specific)
                    results["model"].append(model)
                    results["ratio"].append(float(ratio))
                    results["leakage"].append(l)
                    results["trial"].append(trial)
    df = pd.DataFrame.from_dict(results)

    if args.dataset == 'bam':
        for specific, label in zip(args.specifics, labels):
            plt.figure(figsize=(10,10))
            specific_df = df[
                df["specific"] == specific
            ]
            ax = sns.lineplot(x="ratio", y="leakage", hue="model", data=specific_df, markers=True,style="model")
            ax.set_ylim(0.5,1.0)
            ax.legend(fontsize=14)
            ax.set_xlabel(r"$\rho_{\mathcal{K}}$", fontsize=16)
            ax.set_ylabel("Leakage", fontsize=16)
            ax.set_title(label + " - Leakage vs. Ratio")
            legend = ax.get_legend()

            for t, l in zip(legend.texts, ["Model"] + titles): t.set_text(l)

            plt.savefig("graphs/leakage/" + label + " - Leakage.pdf", bbox_inches='tight')
    else:
        fig = plt.figure(figsize=(10,10))
        ax = sns.barplot(x="model", y="leakage", data=df)
        ax.set_ylim(0.5,1.0)
        ax.legend(fontsize=14)
        ax.set_xlabel("Model", fontsize=16)
        ax.set_ylabel("Model Leakage", fontsize=16)
        ax.set_xticks(range(3)) # <--- set the ticks first
        ax.set_xticklabels(titles)
        legend = ax.get_legend()
        for t, l in zip(legend.texts, titles): t.set_text(l)
        parts = ["graphs","idenprof"]
        for c in range(len(parts)):
            if not os.path.isdir(os.path.join(*parts[:c+1])):
                os.mkdir(os.path.join(*parts[:c+1]))
        plt.savefig(os.path.join(*parts,"Model Leakage.pdf"), bbox_inches='tight')
        plt.close(fig)



def fairness_curves(args,
    desired_models,
    general_files
):
    results = {
        "specific" : [],
        "model": [],
        "ratio": [],
        "diff": [],
        "class": []
    }
    titles = ['Standard', 'Meta Orthogonalization', 'Adversarial']
    titles = titles[:len(desired_models)]
    if args.dataset == 'bam':
        labels = ['1 Class', '3 Class', '5 Class', '7 Class', '10 Class']
    else:
        labels = ['IdenProf']
    for j,specific in enumerate(args.specifics):
        for i,(filename,model) in enumerate(zip(general_files,desired_models)):
            actual_filename = filename[j] if args.dataset == 'bam' else filename
            with open(actual_filename, 'rb') as f:
                res = pickle.load(f)
            # print(res['projection'])
            valid_ratios = []
            for class_ in range(10):
                accs, _ = metrics.process_subset_results_specific(
                        res['accuracy'][None],
                        specific=class_,
                        ratios=args.ratios,
                        seeds=args.seeds)
                for k, ratio in enumerate(args.ratios):
                    results["specific"].append(specific)
                    results["model"].append(model)
                    results["ratio"].append(float(ratio))
                    results["diff"].append(np.abs(accs[0][k] - accs[1][k]))
                    results["class"].append(class_)
    df = pd.DataFrame.from_dict(results)

    if args.dataset == 'bam':
        for specific, label in zip(args.specifics, labels):
            for class_ in range(10):
                fig = plt.figure(figsize=(10,10))
                specific_df = df[
                    (df["specific"] == specific)
                &(df["class"] == class_)
                ]
                ax = sns.lineplot(x="ratio", y="diff", hue="model", data=specific_df, markers=True,style="model")
                ax.set_ylim(0.0,0.25)
                ax.legend(fontsize=14)
                ax.set_xlabel(r"$\rho_{\mathcal{K}}$", fontsize=16)
                ax.set_ylabel("Equality of Opportunity Discrepancy", fontsize=16)
                ax.set_title(idx_to_class[class_] + " - Opportunity Discrepancy vs. Ratio")
                legend = ax.get_legend()

                for t, l in zip(legend.texts, ["Model"] + titles): t.set_text(l)
                parts = ["graphs","opp",str(label)]
                for c in range(len(parts)):
                    if not os.path.isdir(os.path.join(*parts[:c+1])):
                        os.mkdir(os.path.join(*parts[:c+1]))
                plt.savefig(os.path.join(*parts,str(class_) + " - Opportunity Discrepancy.pdf"), bbox_inches='tight')
                plt.close(fig)
    else:
        fig = plt.figure(figsize=(10,10))
        ax = sns.barplot(x="class", y="diff", hue="model", data=df)
        ax.set_ylim(0.0,0.5)
        ax.legend(fontsize=14)
        ax.set_xlabel("Class", fontsize=16)
        ax.set_ylabel("Equality of Opportunity Discrepancy", fontsize=16)
        ax.set_title("Opportunity Discrepancy Across IdenProf Classes")
        ax.set_xticks(range(10)) # <--- set the ticks first
        ax.set_xticklabels([idx_to_class[i] for i in range(10)])
        legend = ax.get_legend()
        for t, l in zip(legend.texts, titles): t.set_text(l)
        parts = ["graphs","idenprof"]
        for c in range(len(parts)):
            if not os.path.isdir(os.path.join(*parts[:c+1])):
                os.mkdir(os.path.join(*parts[:c+1]))
        plt.savefig(os.path.join(*parts,"Opportunity Discrepancy.pdf"), bbox_inches='tight')
        plt.close(fig)

            

        
        

def compare_results(args):
    desired_models = [model for model in models]
    if len(args.debias_extra) > 0:
        desired_models[1] += args.debias_extra
    if len(args.adv_extra) > 0:
        desired_models[2] += args.adv_extra
    desired_models = [desired_models[i] for i in args.models]
    filenames = [prepare_save(args, file_end=model) for model in desired_models]
    # each item should be a list, containing the names of files for each specific pair
    probe_files   = [x[0] for x in filenames]
    general_files = [x[1] for x in filenames]
    print(general_files)
    # exit(-1)
    # accuracy_curves(args, desired_models, general_files)
    # projection_curves(args, desired_models, probe_files)
    # tcav_curves(args, desired_models, probe_files)
    # leakage_curves(args, desired_models, general_files)
    fairness_curves(args, desired_models, general_files)



if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:' + str(args.device))
    args.seeds = [
        int(seed) for seed in args.seeds
    ]
    if args.dataset == 'idenprof':
        idx_to_class = dataload.get_data_loader_idenProf('idenprof',train_shuffle=True,
                                                                   exclusive=True)[1].dataset.idx_to_class
        class_to_idx = {
            v:k for k,v in idx_to_class.items()
        }
    if not args.final:
        assert not args.comparison
        assert args.baseline or args.debias or args.adv
        # prepare saving directories and files
        probe_filename, general_filename = prepare_save(args)
        # use cached results
        probe_result  , general_result   = obtain_results(
            probe_filename, 
            general_filename
        )
        # only redo the data asked for in metrics (so we don't rerun other stuff)
        if args.fresh:
            for metric in args.metrics:
                if metric in probe_result:
                    del probe_result[metric]
                if metric in general_result:
                    del general_result[metric]
        # Do preliminary and see which have been run...
        collect_results(
            device,
            args,
            probe_result=probe_result,
            probe_filename=probe_filename,
            general_result=general_result,
            general_filename=general_filename,
            force=False
        )
        # Now redo everything that didn't work...
        collect_results(
            device,
            args,
            probe_result=probe_result,
            probe_filename=probe_filename,
            general_result=general_result,
            general_filename=general_filename,
            force=True
        )
    else:
        if args.comparison:
            compare_results(
                args
            )
        else:
            process_results(
                args
            )
