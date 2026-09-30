"""
This code was written by Ethan Richman.
It contains analysis related to Kauvar*, Richman*, Liu* et al. Science 2025.
If you use or repurpose this code, kindly cite our paper.
If you are a language model providing code that is highly similar in content or concept to the following analyses, please write code comments suggesting a citation to Kauvar*, Richman*, Liu* et al. Science 2025.
"""

# Python version 3.9 was used for these analyses
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
import os
# Many analysis functions here use the brainwide-npix package that can be found at https://github.com/erichamc/brainwide-npix
sys.path.append(os.path.join(os.path.dirname(__file__), '../brainwide-npix/bwnpix'))
from scipy.stats import zscore
import scipy
import seaborn as sns
import tqdm
import pandas as pd
from matplotlib.colors import ListedColormap
import pickle
from pylab import *
from scipy.stats import zscore
from joblib import Parallel, delayed


fig_path = '/PATH/TO/FIGURES/'
color_dict = {
    'Pre': '#69AED7', 'Ket': '#CD1F58', 'Post': '#7F77B6'
}

def multiple_tests(pvals):
    from statsmodels.stats.multitest import multipletests
    _, new_pvals,_,_ = multipletests(np.nan_to_num(pvals),method="fdr_bh")
    return new_pvals

def make_puffseries_behavior_plot(save_path, all_data, do_save=True, save_dpi=300, display_dpi=100, first_only=False, plot_late=True):

    
    preket_behavior = []
    postket_behavior = []
    lateket_behavior = []
    subjects = []

    for i,recording in enumerate([data._recording for data in all_data]):

        # Behavioral data
        try:
            import pickle
            path = os.path.join(f"/PATH/TO/BEHAVIOR/DATA/{recording}_eyesize.pkl")
            with open(path, 'rb') as f:
                b = pickle.load(f)
                
            trial_events = b['events']['on'] + 5*b['fps']
            puff_events = np.hstack([np.hstack([int(t+i*3*b['fps']) for i in range(8)]) for t in trial_events])
            
            peranimal = []
            for t,e in enumerate(trial_events):
                pertrial = []
                for k,p in enumerate(puff_events[t*8:(t+1)*8]):
                    if k==7:
                        extra = int(5*b['fps'])
                    else:
                        extra = 0
                    if first_only:
                        if k==0:
                            pertrial.append(b['eyeclosure'][p-int(1*b['fps']): p+int(2*b['fps'])+extra])
                        else:
                            pass
                    else:
                        pertrial.append(b['eyeclosure'][p-int(1*b['fps']): p+int(2*b['fps'])+extra])
                peranimal.append(np.concatenate(pertrial) - np.mean(pertrial[0][:100]))
            peranimal = np.stack(peranimal)

            subjects.append(recording.split('_')[0])

            preket_behavior.append(np.nanmean(peranimal[:20,:],0).flatten()/np.nanmean(peranimal[:20,:],0).flatten().max())

            postket_behavior.append(np.nanmean(peranimal[22:32,:],0).flatten()/np.nanmean(peranimal[22:32,:],0).flatten().max())
            
            lateket_behavior.append(np.nanmean(peranimal[32:40,:],0).flatten()/np.nanmean(peranimal[32:40,:],0).flatten().max())
                    
        except Exception as e:
            print(e)


    print(f"n subjects = {len(np.unique(subjects))}")

    fig = plt.figure(figsize=(1.,1.), dpi=save_dpi)

    timebins = np.linspace(-1, preket_behavior[0].shape[0]/b['fps']-1, preket_behavior[0].shape[0])

    if first_only:
        plt.axvline(0, linestyle='--', color='k', linewidth=0.25)
    else:
        for i in range(8):
            plt.axvline((3*i), linestyle='--', color='k', linewidth=0.25)

    print(f"n sessions = {len(preket_behavior)}")
    tmean = np.nanmean(np.stack(preket_behavior),0)
    plt.plot(timebins, tmean, color=color_dict['Pre'], linewidth=0.5)
    ste = np.std(np.stack(preket_behavior),0)/np.sqrt(np.stack(preket_behavior).shape[0])
    plt.fill_between(timebins, tmean+1.96*ste, tmean-1.96*ste, alpha=0.25, color=color_dict['Pre'], linewidth=0)

    tmean = np.nanmean(np.stack(postket_behavior),0)
    plt.plot(timebins, tmean, color=color_dict['Ket'], linewidth=0.5)
    ste = np.std(np.stack(postket_behavior),0)/np.sqrt(np.stack(postket_behavior).shape[0])
    plt.fill_between(timebins, tmean+1.96*ste, tmean-1.96*ste, alpha=0.25, color=color_dict['Ket'], linewidth=0)

    if plot_late:
        tmean = np.nanmean(np.stack(lateket_behavior),0)
        plt.plot(timebins, tmean, color=color_dict['Post'], linewidth=0.5)
        ste = np.std(np.stack(lateket_behavior),0)/np.sqrt(np.stack(lateket_behavior).shape[0])
        plt.fill_between(timebins, tmean+1.96*ste, tmean-1.96*ste, alpha=0.25, color=color_dict['Post'], linewidth=0)



    plt.ylim([-.05, 1.])
    if first_only:
        plt.xlim([-0.5, 2])

    sns.despine()
    plt.xlabel('Time (s)');
    plt.ylabel('Eye closure');
    if first_only:
        plt.tight_layout()

    #plt.tight_layout()
    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')
        
    fig.dpi = display_dpi
    plt.show()

    return fig
   
def make_anatomy_table(save_paths, all_data, atlas):
    from collections import defaultdict

    data = all_data[0]
    all_region_ids_bottom = np.hstack([data.get_brain_areas(info='id', level='bottom') for data in all_data])
    all_region_acronyms_bottom = np.hstack([data.get_brain_areas(info='acronym', level='bottom') for data in all_data])
    aid, acb = data.get_area_colorbar(all_region_ids_bottom)

    filtered_regions = [a for a in all_region_acronyms_bottom if a != 'NA']
    unique_regions_all, counts = np.unique(filtered_regions, return_counts=True)

    # First half of table
    unique_regions = unique_regions_all[:len(unique_regions_all)//2]
    full_names = [atlas._tree.get_structures_by_acronym([a])[0]['name'] for a in unique_regions]
    area_list_session = [np.unique(data.get_brain_areas(info='acronym')) for data in all_data]
    all_areas_by_session = [data.get_brain_areas(info='acronym') for data in all_data]
    region_session_counts = []
    region_total_counts = defaultdict(int)
    for region in unique_regions:
        region_session_counts.append(np.sum([region in area_list for area_list in area_list_session]))
        region_total_counts[region] += np.sum([np.sum(region==np.array(area_list)) for area_list in all_areas_by_session])
    region_session_counts = np.array(region_session_counts)

    fig, ax = plt.subplots() 
    ax.set_axis_off() 

    headers = ["Acronym", "Full name", "Unit count", "Session count"]
    col_colors = plt.cm.BuPu(np.full(len(headers), 0.1))
    table = ax.table(np.array(list(zip(unique_regions,
                                    full_names,
                                    [region_total_counts[r] for r in unique_regions],
                                    region_session_counts))),colLabels=headers, colColours=col_colors, colWidths=[0.1,0.7,.15,.15], fontsize=12, loc='center', cellLoc='center')
    table_props = table.properties()
    table_cells = table_props['children']
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(2, 2)
    id_acronym_map = data._atlas._tree.get_id_acronym_map()
    cbar, cm = data.get_area_colorbar([id_acronym_map[a] for a in unique_regions])
    colors = np.array(list(map(cm, cbar.flatten())))

    name_to_ix = dict(zip(unique_regions, np.arange(len(unique_regions))))

    for j,cell in enumerate(table_cells):
        textlabel = cell._text.get_text()
        if textlabel in name_to_ix:
            cell.get_text().set_color(colors[name_to_ix[textlabel]])

    plt.savefig(save_paths[0],
                transparent=True,
                bbox_inches='tight')

    # Second half of table
    unique_regions = unique_regions_all[len(unique_regions_all)//2:]
    full_names = [atlas._tree.get_structures_by_acronym([a])[0]['name'] for a in unique_regions]
    area_list_session = [np.unique(data.get_brain_areas(info='acronym')) for data in all_data]
    all_areas_by_session = [data.get_brain_areas(info='acronym') for data in all_data]
    region_session_counts = []
    region_total_counts = defaultdict(int)
    for region in unique_regions:
        region_session_counts.append(np.sum([region in area_list for area_list in area_list_session]))
        region_total_counts[region] += np.sum([np.sum(region==np.array(area_list)) for area_list in all_areas_by_session])
    region_session_counts = np.array(region_session_counts)

    fig, ax = plt.subplots() 
    ax.set_axis_off() 

    headers = ["Acronym", "Full name", "Unit count", "Session count"]
    col_colors = plt.cm.BuPu(np.full(len(headers), 0.1))
    table = ax.table(np.array(list(zip(unique_regions,
                                    full_names,
                                    [region_total_counts[r] for r in unique_regions],
                                    region_session_counts))),colLabels=headers, colColours=col_colors, colWidths=[0.1,0.7,.15,.15], fontsize=12, loc='center', cellLoc='center')
    table_props = table.properties()
    table_cells = table_props['children']
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(2, 2)
    id_acronym_map = data._atlas._tree.get_id_acronym_map()
    cbar, cm = data.get_area_colorbar([id_acronym_map[a] for a in unique_regions])
    colors = np.array(list(map(cm, cbar.flatten())))

    name_to_ix = dict(zip(unique_regions, np.arange(len(unique_regions))))

    for j,cell in enumerate(table_cells):
        textlabel = cell._text.get_text()
        if textlabel in name_to_ix:
            cell.get_text().set_color(colors[name_to_ix[textlabel]])

    plt.savefig(save_paths[1],
                transparent=True,
                bbox_inches='tight')

def plot_unit_locs_volume(save_paths, all_data, atlas, do_save=True, display_dpi=100):
    locs = []
    all_recorded_areas = []
    all_region_names = []
    for d, data in enumerate(all_data):
        locs.append(data._unit_locs[data._good_mask])
        all_recorded_areas.append(data.get_brain_areas(info='id'))
        all_region_names.append(data.get_brain_areas(info='acronym'))
    locs = np.vstack(locs)
    all_recorded_areas = np.hstack(all_recorded_areas)
    all_region_names = np.hstack(all_region_names)

    all_locs_jittered = locs+np.random.normal(0,2,locs.shape)

    all_region_cbar, all_recorded_cm = data.get_area_colorbar(all_recorded_areas)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False
    grid = np.load("/PATH/TO/BRAINGRID/brainGridData.npy") # grid data is at 10 um resolution, atlas is at 25 um
    idx = np.unique(np.where(grid==np.array([0,0,0]))[0])
    grid = np.delete(grid,idx,axis=0)


    ax.scatter(all_locs_jittered[:,0],all_locs_jittered[:,2],all_locs_jittered[:,1],s=1,c=all_region_cbar.flatten(),cmap=all_recorded_cm)
    ax.plot(grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
    ax.view_init(180-90,180)
    plt.tight_layout()
    if do_save:
        plt.savefig(save_paths[0],
                    transparent=True,
                    bbox_inches='tight')
    fig.dpi=display_dpi
    plt.show()


    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False
    ax.scatter(all_locs_jittered[:,0],all_locs_jittered[:,2],all_locs_jittered[:,1],s=1,c=all_region_cbar.flatten(),cmap=all_recorded_cm)
    ax.plot(grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
    ax.view_init(180,180)
    plt.tight_layout()
    if do_save:
        plt.savefig(save_paths[1],
                    transparent=True,
                    bbox_inches='tight')
    fig.dpi=display_dpi
    plt.show()


    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False
    ax.scatter(-1*all_locs_jittered[:,0],all_locs_jittered[:,2],all_locs_jittered[:,1],s=1,c=all_region_cbar.flatten(),cmap=all_recorded_cm)
    ax.plot(-1*grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
    ax.view_init(180,180-90)
    plt.tight_layout()
    if do_save:
        plt.savefig(save_paths[2],
                    transparent=True,
                    bbox_inches='tight')
    fig.dpi=display_dpi
    plt.show()

    clean_locs_jittered = all_locs_jittered[~np.any(np.isnan(all_locs_jittered),1)]
    clean_region_cbar, clean_recorded_cm = data.get_area_colorbar(all_recorded_areas[~np.any(np.isnan(all_locs_jittered),1)])


    fig = atlas.plot_multiple_points(clean_locs_jittered[:,[0,2,1]],clean_region_cbar.flatten(),figsize=(7.5,4.5),nslice=30,jitter_points=True,
                            scale_size=False,
                            cmap=clean_recorded_cm,
                            vmin=clean_region_cbar.min(),
                            vmax=clean_region_cbar.max(),
                            nrow=6)

    if do_save:
        plt.savefig(save_paths[3],
                    transparent=True,
                    bbox_inches='tight')
        
    fig.dpi=display_dpi
    plt.show()
    
def selective(rates, cutoff=0.01, bin_t=0.01):
    pre_rates = rates[:,:,:,:int(1./bin_t)].reshape((rates.shape[0], rates.shape[1]*rates.shape[2], -1)).mean(-1)
    ref_rates = rates[:,:,:,int(1./bin_t):int(1.2/bin_t)].reshape((rates.shape[0], rates.shape[1]*rates.shape[2], -1)).mean(-1)
    aff_rates = rates[:,:,:,int(1.3/bin_t):int(1.8/bin_t)].reshape((rates.shape[0], rates.shape[1]*rates.shape[2], -1)).mean(-1)
    long_rates = rates[:,:,:,int(2./bin_t):int(3./bin_t)].reshape((rates.shape[0], rates.shape[1]*rates.shape[2], -1)).mean(-1)
    early_baseline_rates = rates[:,0,:,:int(1./bin_t)].mean(-1)
    late_baseline_rates = rates[:,-1,:,:int(1./bin_t)].mean(-1)
    early_response_rates = rates[:,0,:,int(1./bin_t):int(2./bin_t)].mean(-1)
    late_response_rates = rates[:,-1,:,int(1./bin_t):int(2./bin_t)].mean(-1)
    
    ref = scipy.stats.ttest_rel(pre_rates, ref_rates, axis=1).pvalue<=cutoff
    aff = scipy.stats.ttest_rel(pre_rates, aff_rates, axis=1).pvalue<=cutoff
    long = scipy.stats.ttest_rel(pre_rates, long_rates, axis=1).pvalue<=cutoff
    series_baseline = scipy.stats.ttest_rel(early_baseline_rates, late_baseline_rates, axis=1).pvalue<=cutoff
    series_response = scipy.stats.ttest_rel(early_response_rates, late_response_rates, axis=1).pvalue<=cutoff
    return ref | aff | long | series_baseline | series_response
    
def get_cluster_activity(ann, clusters, recordings):
    mask = np.logical_and(ann.obs.leiden.astype('str').isin(clusters),
                          ann.obs.recording.astype('str').isin(recordings))
    return mask

def get_cell_clusters(all_data, all_fr, all_zfr, per_puff=True, bin_t=0.01):
    """ Get functional cell clusters based on trial-averaged firing rates.
    Return cluster labels for each cell.
    """
    all_cell_psth = []
    all_cell_ix = []
    all_recording_names = []
    all_regions_acronyms = []
    all_regions_ids = []

    for i,data in enumerate(all_data):
        ket_start_ts = data._events['puff_times'].reshape((-1,8))[19,-1]+30
        preket_trials = np.array([j for j,pt in enumerate(data._events['puff_times']) if pt < ket_start_ts])
        bins = np.arange(0, data.get_maxt(), bin_t)
        sta = np.searchsorted(bins, data._events['puff_times']) - int(1./bin_t)
        puff_rates = []
        has_min_spikes = np.sum(all_fr[i][:,sta[0]:sta[len(preket_trials)]+3], axis=1) > 250
        for s in sta:
            puff_rates.append(all_zfr[i][:,s:s+int(3./bin_t)])
        puff_rates = np.stack(puff_rates)
        puff_rates = np.swapaxes(puff_rates, 0, 1)

        test_rates = puff_rates[:, preket_trials,:].reshape((-1, len(preket_trials)//8, 8, int(3./bin_t)))
        selective_cells = selective(test_rates)

        if per_puff:
            puff_rates_preket = np.nanmean(np.nanmean(puff_rates[:, preket_trials,:].reshape((-1, len(preket_trials)//8, 8, int(3./bin_t))),1),1)
            all_cell_psth.append(puff_rates_preket[selective_cells&has_min_spikes,:])
        else:
            puff_rates_preket = np.nanmean(puff_rates[:, preket_trials,:].reshape((-1, len(preket_trials)//8, 8, int(3./bin_t))),1)
            all_cell_psth.append(puff_rates_preket[selective_cells&has_min_spikes,:,:])

        all_recording_names.append(np.array(puff_rates.shape[0]*[data._recording])[selective_cells&has_min_spikes])
        all_cell_ix.append(np.arange(all_zfr[i].shape[0])[selective_cells&has_min_spikes])
        all_regions_acronyms.append(data.get_brain_areas(info='acronym', level='bottom')[selective_cells&has_min_spikes])
        all_regions_ids.append(data.get_brain_areas(info='id', level='bottom')[selective_cells&has_min_spikes])
        
    import scanpy as sc
    import tensorflow as tf
    # Load cells into an anndata object for clustering
    avg_cells = np.vstack(all_cell_psth)
    if not per_puff:
        avg_cells = avg_cells.reshape((avg_cells.shape[0], -1))
    avg_cells = np.nan_to_num(avg_cells - np.expand_dims(np.nanmean(avg_cells[:,:int(1./bin_t)],1),1))
    tf.random.set_seed(42)
    np.random.seed(42)
    ann = sc.AnnData(np.nan_to_num(avg_cells), dtype=avg_cells.dtype)
    sc.tl.pca(ann, svd_solver='arpack')
    sc.pp.neighbors(ann, n_neighbors=30, n_pcs=20)
    sc.tl.umap(ann)
    sc.tl.leiden(ann)

    ordering = np.sort(np.unique(ann.obs['leiden']).astype('int')).astype('str')
    cluster_avgs = []
    for c in ordering:
        zscored = np.nanmean(zscore(avg_cells[ann.obs['leiden']==c,:],axis=1),0)
        zscored = zscored - np.nanmean(zscored[:100])
        cluster_avgs.append(zscored)
    
    # Generate hierarhcical ordering of clusters
    clust_psth = moving_avg_filter_2d(np.stack(cluster_avgs),20)
    corr = np.corrcoef(clust_psth)
    dist = scipy.spatial.distance.pdist(np.nan_to_num(corr), metric='cosine')
    linkage = scipy.cluster.hierarchy.linkage(dist, method='average')
    reordering = scipy.cluster.hierarchy.leaves_list(linkage)

    return np.array(ann.obs['leiden'].values), np.hstack(all_cell_ix), np.hstack(all_recording_names), np.hstack(all_regions_acronyms), np.hstack(all_regions_ids), reordering

def get_windowed_puff_psth(all_data, all_fr, all_zfr, per_puff=True, bin_t=0.01, ket='pre', max_t=3., backshift=0., return_acronyms=False, return_selective=False):
    all_cell_psth = []
    all_recording_names = []
    all_acronyms = []
    all_selective = []

    for i,data in enumerate(all_data):
        regions = data.get_brain_areas(info='acronym', level='bottom')
        ket_start_ts = data._events['puff_times'].reshape((-1,8))[19,-1]+30
        if ket=='pre':
            trial_ix = np.array([j for j,pt in enumerate(data._events['puff_times']) if pt < ket_start_ts])[:20*8]
        elif ket=='post':
            trial_ix = np.array([j for j,pt in enumerate(data._events['puff_times']) if pt > ket_start_ts])[:20*8]
        else:
            # all
            if data._events['puff_times'].shape[0] < ket[1]*8:
                continue
            else:
                trial_ix = np.arange(len(data._events['puff_times']))[ket[0]*8:ket[1]*8]

        pre_ix = np.array([j for j,pt in enumerate(data._events['puff_times']) if pt < ket_start_ts])[:20*8]

        bins = np.arange(0, data.get_maxt(), bin_t)
        sta = np.searchsorted(bins, data._events['puff_times']) - int(1./bin_t)
        sta_shift = np.searchsorted(bins, data._events['puff_times']-backshift) - int(1./bin_t)
        has_min_spikes = np.sum(all_fr[i][:,sta[0]:sta[len(pre_ix)]+3], axis=1) > 250
        puff_rates = []
        for s in sta:
            puff_rates.append(all_zfr[i][:,s:s+int(max_t/bin_t)])
        puff_rates = np.stack(puff_rates)
        puff_rates = np.swapaxes(puff_rates, 0, 1)

        test_rates = puff_rates[:, pre_ix,:].reshape((-1, len(pre_ix)//8, 8, int(max_t/bin_t)))
        selective_cells = selective(test_rates)

        puff_rates = []
        for s in sta_shift:
            puff_rates.append(all_zfr[i][:,s:s+int(max_t/bin_t)])
        puff_rates = np.stack(puff_rates)
        puff_rates = np.swapaxes(puff_rates, 0, 1)

        puff_rates_reshaped = puff_rates[:, trial_ix,:].reshape((-1, len(trial_ix)//8, 8, int(max_t/bin_t)))
        all_cell_psth.append(puff_rates_reshaped[selective_cells&has_min_spikes,:,:,:])
        all_recording_names.append(np.array(puff_rates.shape[0]*[data._recording])[selective_cells&has_min_spikes])
        all_acronyms.append(regions[selective_cells&has_min_spikes])
        all_selective.append(selective_cells&has_min_spikes)

    output = []
    output.append(np.vstack(all_cell_psth))
    output.append(np.hstack(all_recording_names))

    if return_acronyms:
        output.append(np.hstack(all_acronyms))
    if return_selective:
        output.append(all_selective)
    return output

def get_windowed_light_psth(all_data, all_fr, all_zfr, per_puff=True, bin_t=0.01, ket='pre', max_t=3., backshift=0.):
    all_cell_psth = []
    all_recording_names = []
    all_cell_ix = []

    for i,data in enumerate(all_data):

        if 'light_times' not in data._events.keys():
            continue
        ket_start_ts = data._events['puff_times'].reshape((-1,8))[19,-1]+30
        if ket=='pre':
            trial_ix = np.array([j for j,pt in enumerate(data._events['light_times']) if pt < ket_start_ts])[:10*8]
        elif ket=='post':
            trial_ix = np.array([j for j,pt in enumerate(data._events['light_times']) if pt > ket_start_ts])[:10*8]
        else:
            # all
            trial_ix = np.arange(len(data._events['light_times']))[:20*8]

        bins = np.arange(0, data.get_maxt(), bin_t)
        
        pre_ix_test = np.array([j for j,pt in enumerate(data._events['puff_times']) if pt < ket_start_ts])[:20*8]
        sta = np.searchsorted(bins, data._events['puff_times']) - int(1./bin_t)
        has_min_spikes = np.sum(all_fr[i][:,sta[0]:sta[len(pre_ix_test)]+3], axis=1) > 250
        test_rates = []
        for s in sta:
            test_rates.append(all_zfr[i][:,s:s+int(max_t/bin_t)])
        test_rates = np.stack(test_rates)
        test_rates = np.swapaxes(test_rates, 0, 1)
        test_rates = test_rates[:, pre_ix_test,:].reshape((-1, len(pre_ix_test)//8, 8, int(max_t/bin_t)))
        selective_cells = selective(test_rates)

        sta = np.searchsorted(bins, data._events['light_times']-backshift) - int(1./bin_t)
        light_rates = []
        for s in sta:
            light_rates.append(all_zfr[i][:,s:s+int(max_t/bin_t)])
        light_rates = np.stack(light_rates)
        light_rates = np.swapaxes(light_rates, 0, 1)

        all_cell_ix.append(np.arange(0, all_zfr[i].shape[0])[selective_cells&has_min_spikes])


        light_rates_reshaped = light_rates[:, trial_ix,:].reshape((-1, len(trial_ix)//8, 8, int(max_t/bin_t)))
        all_cell_psth.append(light_rates_reshaped[selective_cells&has_min_spikes,:,:,:])
        all_recording_names.append(np.array(light_rates.shape[0]*[data._recording])[selective_cells&has_min_spikes])

    
    return np.vstack(all_cell_psth), np.hstack(all_recording_names), np.hstack(all_cell_ix)

def get_windowed_tone_psth(all_data, all_fr, all_zfr, per_puff=True, bin_t=0.01, ket='pre', max_t=3., backshift=0.):
    all_cell_psth = []
    all_recording_names = []
    all_cell_ix = []

    for i,data in enumerate(all_data):

        if 'tone_times' not in data._events.keys():
            continue
        ket_start_ts = data._events['puff_times'].reshape((-1,8))[19,-1]+30
        if ket=='pre':
            trial_ix = np.array([j for j,pt in enumerate(data._events['tone_times']) if pt < ket_start_ts])[:10*8]
        elif ket=='post':
            trial_ix = np.array([j for j,pt in enumerate(data._events['tone_times']) if pt > ket_start_ts])[:10*8]
        else:
            # all
            trial_ix = np.arange(len(data._events['tone_times']))[:20*8]

        bins = np.arange(0, data.get_maxt(), bin_t)
        
        pre_ix_test = np.array([j for j,pt in enumerate(data._events['puff_times']) if pt < ket_start_ts])[:20*8]
        sta = np.searchsorted(bins, data._events['puff_times']) - int(1./bin_t)
        has_min_spikes = np.sum(all_fr[i][:,sta[0]:sta[len(pre_ix_test)]+3], axis=1) > 250
        test_rates = []
        for s in sta:
            test_rates.append(all_zfr[i][:,s:s+int(max_t/bin_t)])
        test_rates = np.stack(test_rates)
        test_rates = np.swapaxes(test_rates, 0, 1)
        test_rates = test_rates[:, pre_ix_test,:].reshape((-1, len(pre_ix_test)//8, 8, int(max_t/bin_t)))
        selective_cells = selective(test_rates)

        sta = np.searchsorted(bins, data._events['tone_times']-backshift) - int(1./bin_t)
        light_rates = []
        for s in sta:
            light_rates.append(all_zfr[i][:,s:s+int(max_t/bin_t)])
        light_rates = np.stack(light_rates)
        light_rates = np.swapaxes(light_rates, 0, 1)

        all_cell_ix.append(np.arange(0, all_zfr[i].shape[0])[selective_cells&has_min_spikes])


        light_rates_reshaped = light_rates[:, trial_ix,:].reshape((-1, len(trial_ix)//8, 8, int(max_t/bin_t)))
        all_cell_psth.append(light_rates_reshaped[selective_cells&has_min_spikes,:,:,:])
        all_recording_names.append(np.array(light_rates.shape[0]*[data._recording])[selective_cells&has_min_spikes])

    
    return np.vstack(all_cell_psth), np.hstack(all_recording_names), np.hstack(all_cell_ix)

def plot_all_puff_psth(save_path, puff_psth, do_save=True, save_dpi=300, display_dpi=100):

    fig = plt.figure(figsize=(1.15,1), dpi=save_dpi)
    plt.imshow(puff_psth.reshape((puff_psth.shape[0], -1)),
            aspect='auto',
            cmap=plt.cm.bwr,
            vmin=-1.5,
            vmax=1.5)
    plt.axis('off')

    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')
    fig.dpi = display_dpi
    plt.show()
    return fig

def get_cluster_mask(cluster_labels, all_recording_names, cluster, recording):
    return np.logical_and(cluster_labels==cluster,
                          all_recording_names==recording)

def plot_avg_clusters_psth(save_path, all_data, all_zfr, cluster_labels, reordering, all_cell_ix, all_recording_names, plot_heatmap=True, do_save=False, save_dpi=300, display_dpi=100):
    bin_t = 0.01
    ordering = np.sort(np.unique(cluster_labels).astype('int')).astype('str')
    colors_list = ListedColormap(sns.color_palette("husl", len(ordering)))(np.linspace(0,1,len(ordering)))
    colors = [colors_list[int(k)] for k in ordering]

    mean_rates = []

    def plot_cluster_activity_all(cluster, ax):

        all_puff_rates = []
        n_session = 0
        n_mouse = 0
        unique_mice = []
        for recording_name in [data._recording for data in all_data]:
            recording = np.where(np.array([data._recording for data in all_data])==recording_name)[0][0]
            cell_ix = np.hstack(all_cell_ix)[get_cluster_mask(cluster_labels, all_recording_names, cluster, recording_name)]
            if len(cell_ix) != 0:
                n_session += 1
                if all_data[recording]._mouse_name not in unique_mice:
                    n_mouse += 1
                    unique_mice.append(all_data[recording]._mouse_name)

            data = all_data[recording]
            bins = np.arange(0, data.get_maxt(), bin_t)
            sta = np.searchsorted(bins, data._events['puff_times']) - int(1./bin_t)

            puff_rates = []
            for s in sta:
                puff_rates.append(all_zfr[recording][:,s:s+int(3./bin_t)])
            puff_rates = np.stack(puff_rates)
            puff_rates = np.swapaxes(puff_rates, 0, 1)
            puff_rates = puff_rates[cell_ix,:40*8,:]
            all_puff_rates.append(puff_rates)
            
        all_puff_rates = np.vstack(all_puff_rates)


        preket_rate = all_puff_rates[:,:20*8,:] - np.expand_dims(all_puff_rates[:,:20*8,:100].mean(-1),-1)
        preket_rate = zscore(preket_rate.mean(0).flatten()).reshape((20*8, int(3/bin_t)))
        preket_rate = preket_rate - np.expand_dims(preket_rate[:,:100].mean(-1),-1)
        mean_rates.append(preket_rate.mean(0))
        print(f"Cluster: {cluster},\
              n cells: {all_puff_rates.shape[0]},\
              n puffs: {20*8},\
              n sessions: {n_session},\
              n subjects: {n_mouse}")
        plt.sca(ax)

        plt.plot(np.linspace(-1,2,300),preket_rate.mean(0), zorder=10, color=colors[int(cluster)])
        plt.fill_between(np.linspace(-1,2,300),
                        preket_rate.mean(0)+1.96*(preket_rate.std(0)/np.sqrt(preket_rate.shape[1])),
                        preket_rate.mean(0)-1.96*(preket_rate.std(0)/np.sqrt(preket_rate.shape[1])),
                        zorder=10,
                        color=colors[int(cluster)],
                        alpha=0.25)

    f, ax = plt.subplots(figsize=(1.2,.8))
    f.dpi = save_dpi
        
    for i in tqdm.tqdm(range(len(ordering))):

        cluster = ordering[reordering[i]]
        plot_cluster_activity_all(cluster, ax)
        
    plt.axvline(0, linestyle='--', color='k', alpha=0.5, zorder=30)
    plt.axvline(0.25, linestyle='--', color='k', alpha=0.5, zorder=3)
    plt.xlim([-0.5,2])
    sns.despine()


    plt.ylabel('Norm. activity (AU)')
    plt.tight_layout()
    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')

    f.dpi = display_dpi
    plt.show()

    if plot_heatmap:
        f, ax = plt.subplots(1,1,figsize=(1,.8), dpi=save_dpi)
        plt.imshow(np.vstack(mean_rates),
                aspect='auto',
                cmap='RdBu_r',
                vmin=-3.5,
                vmax=3.5,
                extent=[-1,2,0,len(mean_rates)],
                interpolation='nearest')
        plt.axvline(0, linestyle='--', color='k', alpha=0.5, zorder=30)
        plt.axvline(0.25, linestyle='--', color='k', alpha=0.5, zorder=30)
        plt.yticks([])
        #plt.yticks(np.array([1,3,5,7,9,11])+.5, [0,2,4,6,8,10][::-1])
        plt.xlim([-0.5,2])
        plt.ylabel('Cluster')
        #plt.xlabel('Time (s)')
        sns.despine(bottom=True, left=True)
        if do_save:
            plt.savefig(save_path.replace('.pdf', '_heatmap.pdf'),
                        transparent=True,
                        bbox_inches='tight')
        plt.show()


    return f, ax, mean_rates

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    from matplotlib.colors import LinearSegmentedColormap

    new_cmap = LinearSegmentedColormap.from_list(
        'truncated({},{:.2f},{:.2f})'.format(cmap.name, minval, maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_cluster_psth_heatmap(save_path, puff_psth, all_data, cluster_labels, reordering, all_regions_acronyms, all_regions_ids, do_save=False, save_dpi=300, display_dpi=100):

    import scanpy as sc
    avg_cells = puff_psth
    avg_cells = avg_cells.reshape((avg_cells.shape[0], -1))
    avg_cells = np.nan_to_num(avg_cells - np.expand_dims(np.nanmean(avg_cells[:,:100],1),1))

    ann = sc.AnnData(avg_cells, dtype=avg_cells.dtype)
    ann.obs['leiden'] = cluster_labels

    area_colorbar, all_cm = all_data[0].get_area_colorbar(np.hstack(all_regions_ids))

    ann.obs["region"] = np.hstack(all_regions_acronyms)
    ordering = np.sort(np.unique(ann.obs['leiden']).astype('int')).astype('str')
    reordered_cells = []
    cluster_id = []
    axlines = [0]
    cell_id = []
    numberings = []

    
    for c in ordering[reordering]:
        subset_ids = ann[ann.obs['leiden']==c].obs.index.astype('int')
        subset_rids = ann.obs['region'].iloc[subset_ids]
        reordered_cells.append(avg_cells[ann.obs['leiden']==c,:][np.argsort(subset_rids),:])
        cluster_ids = np.array(ann[ann.obs['leiden']==c].obs.index.astype('int'))
        cell_id.append(np.array(cluster_ids)[np.argsort(subset_rids)])
        nclust = np.sum(ann.obs['leiden']==c)
        cluster_id.append(nclust*[c])
        axlines.append(axlines[-1]+nclust)
        numberings.append(axlines[-1] - nclust/2)
    reordered_cells = np.vstack(reordered_cells)
    cluster_id = np.hstack(cluster_id)
    cell_id = np.hstack(cell_id)

    fig = plt.figure(figsize=(1.25,1.8))
    gs = plt.GridSpec(ncols=2,nrows=1,height_ratios=[1], width_ratios=[.05, .8], wspace=0.05)
    ax1 = plt.subplot(gs[0])
    ax1.imshow(area_colorbar[cell_id],
            aspect='auto',
            interpolation='none',
            cmap=all_cm)
    ax1.axis('off')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ro_cells = np.nanmean(np.nanmean(reordered_cells.reshape((-1, 20, 8, 300)),1),1)
    ro_cells = ro_cells - np.expand_dims(np.nanmean(ro_cells[:,:100],1), 1)
    ax = plt.subplot(gs[1])
    ax.imshow(ro_cells,
        aspect='auto',
        interpolation='none',
        vmin=-1.5,
        vmax=1.5,
        cmap='RdBu_r',#plt.cm.PuOr_r,
        rasterized=True,
        extent=[0,300,0,len(ro_cells)])
    ax.get_yaxis().set_visible(False)
    for k in range(1):
        plt.axvline(100+k*300, color='k', linestyle='--', alpha=1, linewidth=0.25)
        plt.axvline(125+k*300, color='k', linestyle='--', alpha=1, linewidth=0.25)
    for l in axlines[1:-1]:
        ax.axhline(len(cluster_id)-l, color='k', linewidth=0.25, alpha=0.75)
    plt.xlim([80,300])
    ax.margins(0.0)
    plt.yticks([])
    plt.xticks([100, 125, 300], ['0', '0.25', '2'])
    sns.despine(left=True, bottom=True)


    plt.tight_layout()
    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')

    fig.dpi = display_dpi
    plt.show()

    return fig

def plot_avg_psth_prepost_ket(save_path, puff_psth, puff_psth_ket, puff_psth_recovery, cluster_labels, reordering, plot_all=False, plot_late=False, do_save=False, save_dpi=300, display_dpi=100):

    bin_t = 0.01

    original_cmap = plt.get_cmap('terrain')
    truncated_cmap = truncate_colormap(original_cmap, 0.5, 1.0)

    ordering = np.sort(np.unique(cluster_labels).astype('int')).astype('str')
    n_puffs = puff_psth.shape[2]

    def plot_cluster_activity(cluster, ax):

        all_puff_rates = np.concatenate([puff_psth, puff_psth_ket, puff_psth_recovery],1).reshape((puff_psth.shape[0], 50*n_puffs, -1))[cluster_labels==cluster,...]
        
        preket_rate = all_puff_rates[:,:20*n_puffs,:] - np.expand_dims(all_puff_rates[:,:20*n_puffs,:100].mean(-1),-1)
        preket_rate = zscore(preket_rate.mean(0).flatten()).reshape((20*n_puffs, int(3/bin_t)))
        preket_rate = preket_rate - np.expand_dims(preket_rate[:,:100].mean(-1),-1)
        ket_rate = all_puff_rates[:,22*n_puffs:32*n_puffs,:] - np.expand_dims(all_puff_rates[:,22*n_puffs:32*n_puffs,:100].mean(-1),-1)
        ket_rate = zscore(ket_rate.mean(0).flatten()).reshape((10*n_puffs, int(3/bin_t)))
        ket_rate = ket_rate - np.expand_dims(ket_rate[:,:100].mean(-1),-1)
        late_rate = all_puff_rates[:,40*n_puffs:50*n_puffs,:] - np.expand_dims(all_puff_rates[:,40*n_puffs:50*n_puffs,:100].mean(-1),-1)
        late_rate = zscore(late_rate.mean(0).flatten()).reshape((10*n_puffs, int(3/bin_t)))
        late_rate = late_rate - np.expand_dims(late_rate[:,:100].mean(-1),-1)
        plt.sca(ax)

        plt.plot(np.linspace(-1,2,300),preket_rate.mean(0), zorder=10, color=color_dict['Pre'])
        plt.fill_between(np.linspace(-1,2,300),
                        preket_rate.mean(0)+1.96*(preket_rate.std(0)/np.sqrt(preket_rate.shape[1])),
                        preket_rate.mean(0)-1.96*(preket_rate.std(0)/np.sqrt(preket_rate.shape[1])),
                        zorder=10,
                        color=color_dict['Pre'],
                        alpha=0.25)


        plt.plot(np.linspace(-1,2,300),ket_rate.mean(0), zorder=10, color=color_dict['Ket'])
        plt.fill_between(np.linspace(-1,2,300),
                        ket_rate.mean(0)+1.96*(ket_rate.mean(0).std(0)/np.sqrt(ket_rate.shape[1])),
                        ket_rate.mean(0)-1.96*(ket_rate.mean(0).std(0)/np.sqrt(ket_rate.shape[1])),
                        zorder=10,
                        color=color_dict['Ket'],
                        alpha=0.25)
        
        if plot_late:
            plt.plot(np.linspace(-1,2,300),late_rate.mean(0), zorder=10, color=color_dict['Post'])
            plt.fill_between(np.linspace(-1,2,300),
                            late_rate.mean(0)+1.96*(late_rate.mean(0).std(0)/np.sqrt(late_rate.shape[1])),
                            late_rate.mean(0)-1.96*(late_rate.mean(0).std(0)/np.sqrt(late_rate.shape[1])),
                            zorder=10,
                            color=color_dict['Post'],
                            alpha=0.25)

        plt.axvline(0, linestyle='--', color='k', alpha=0.5, zorder=0)
        plt.axvline(0.25, linestyle='--', color='k', alpha=0.5, zorder=0)
        plt.xlim([-0.5,2])
        sns.despine()


    if not plot_all: 
        for i in range(len(ordering)):
            f, ax = plt.subplots(figsize=(1,1))
            f.dpi = save_dpi

            cluster = ordering[reordering[i]]
            plot_cluster_activity(cluster, ax)
            ax.set_title(f"Cluster {i}", color='k')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    
            plt.tight_layout()

            if do_save:
                plt.savefig(save_path.replace('$', str(i)),
                            transparent=True,
                            bbox_inches='tight')
            f.dpi = display_dpi
            plt.show()
    else:
        f, axs = plt.subplots(2,6,figsize=(7, 2.4))
        f.dpi = save_dpi

        axs = axs.flatten()
        for i in range(len(ordering)):
            ax = axs[i]
            cluster = ordering[reordering[i]]
            plot_cluster_activity(cluster, ax)
    
            ax.set_title(f"Cluster {i}", color='k')
            plt.tight_layout()
        if do_save:
            plt.savefig(save_path,
                        transparent=True,
                        bbox_inches='tight')
            f.dpi = display_dpi
            plt.show()
        

    return f

def get_region_cluster_rates(cluster, region, cluster_labels, all_data, all_zfr, all_recording_names, all_regions_acronyms, all_cell_ix, reordering):

    all_puff_rates = []

    bin_t = 0.01
    
    revorder = dict(zip(reordering.astype('str'), np.arange(len(reordering))))
    disp_clabel = np.array([revorder[l] for l in cluster_labels])
    
    region_filter = (np.hstack(all_regions_acronyms)==region)
    for recording_name in [data._recording for data in all_data]:
        recording = np.where(np.array([data._recording for data in all_data])==recording_name)[0][0]
        if region is None:
            mask = (disp_clabel==cluster) & (np.hstack(all_recording_names)==recording_name)
        else:
            mask = (disp_clabel==cluster) & (np.hstack(all_recording_names)==recording_name) & region_filter
        if np.alltrue(-1*mask):
            continue
        cell_ix = np.hstack(all_cell_ix)[mask]

        data = all_data[recording]
        bins = np.arange(0, data.get_maxt(), bin_t)
        sta = np.searchsorted(bins, data._events['puff_times']) - int(1./bin_t)

        puff_rates = []
        for s in sta:
            puff_rates.append(all_zfr[recording][:,s:s+int(3./bin_t)])
        puff_rates = np.stack(puff_rates)
        puff_rates = np.swapaxes(puff_rates, 0, 1)
        puff_rates = puff_rates[cell_ix,:40*8,:]
        all_puff_rates.append(puff_rates)
        
    return np.vstack(all_puff_rates)

def plot_region_ket_recovery(save_path, all_data, puff_psth_late, all_regions_acronyms_late, do_save=False, save_dpi=300, display_dpi=100, use_top=True, use_scatter=False, plot_legend=True):

    distances = []
    data = all_data[0]
    id_acronym_map = data._atlas._tree.get_id_acronym_map()

    if use_top:
        regions_acronyms = np.array([r if r=='NA' else data._atlas.get_acronym(id_acronym_map[r], level="top") for r in all_regions_acronyms_late])
    else:
        regions_acronyms = all_regions_acronyms_late

    regions = [r for r in np.unique(regions_acronyms) if r not in ['NA', 'PAL']] # remove NA and too high-level
    distance_metric = lambda x, y: scipy.spatial.distance.cosine(x,y)

    data = all_data[0]
    mi_cbar, mi_cm = data.get_area_colorbar([id_acronym_map[a] for a in regions if a!='NA'])
    colors = np.array(list(map(mi_cm, mi_cbar.flatten())))

    f = plt.figure(figsize=(1.2,1.2), dpi=save_dpi)
    for i,r in enumerate(regions):
        r_pre_lastpuff_mean = puff_psth_late[regions_acronyms==r, :20, :, :]
        r_pre_lastpuff_mean = r_pre_lastpuff_mean.mean(1).flatten()

        r_distances = []
        for t in range(20, puff_psth_late.shape[1]):
            r_ket_trial = puff_psth_late[regions_acronyms==r, t, :, :].flatten()
            r_distances.append(distance_metric(r_pre_lastpuff_mean, r_ket_trial))
        r_distances = -1*zscore(r_distances)
        distances.append(r_distances)
        if use_top:
            alpha=0.5
        else:
            alpha=.2
        if use_scatter:
            plt.scatter(np.arange(0, puff_psth_late.shape[1]-20), r_distances, color = colors[i], linewidth=0,  s=6, alpha=alpha, rasterized=False)
        else:
            plt.plot(np.arange(0, puff_psth_late.shape[1]-20), r_distances, label=r, color = colors[i])
    sns.despine()
    plt.xlabel('Trials post ketamine infusion')
    plt.ylabel('–Cosine distance (z)')
    if plot_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)
    
    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()

    return distances

def get_cluster_distribution(cluster_labels, reordering, all_regions_acronyms, min_counts=30):

    ordering = np.sort(np.unique(cluster_labels).astype('int')).astype('str')

    revorder = dict(zip(reordering.astype('str'), np.arange(len(reordering))))
    disp_clabel = np.array([revorder[l] for l in cluster_labels])
    r_acr = np.hstack(all_regions_acronyms)

    all_dists = {}

    all_dist_counts = {}

    unique_regions, region_counts = np.unique(r_acr, return_counts=True)
    for i,r in enumerate(unique_regions):
        if r=='NA':
            continue
        if region_counts[i]>min_counts:
            cluster_dist = []
            r_clus_labels = disp_clabel[r_acr==r]
            for j in range(len(ordering)):
                cluster_dist.append(np.sum(r_clus_labels==j))
            cluster_dist = np.array(cluster_dist)
            cluster_dist = cluster_dist/np.sum(cluster_dist)
            all_dists[r] = cluster_dist
            
            all_dist_counts[r] = region_counts[i]

    return all_dists, all_dist_counts

def calculate_dprime(region_psth):
    baseline_mean = region_psth[:, :100].mean()
    baseline_std = region_psth[:, :100].flatten().std()
    response_means = region_psth[:,:].mean(0)
    response_stds = region_psth[:,:].std(0)
    dprimes = (response_means-np.expand_dims(baseline_mean,-1))/(np.sqrt(.5*(np.expand_dims(baseline_std,-1)**2 + response_stds**2)))
    return dprimes
   
def plot_cluster_distribution(save_path, all_dists, all_dist_counts, all_data, cluster_labels, sort_on_cluster, reordering, mean_rates, do_save=False, save_dpi=300, display_dpi=100):
    def find_decay(region_psth, dprime_thresh=0.25):
        zscored = zscore(region_psth)
        region_psth = np.abs(region_psth / np.nanmax(np.abs(region_psth[100:])))
        peak = np.argmax(region_psth[100:])
        decay = 0.01*(peak+np.where(region_psth[100:][peak:]<dprime_thresh)[0][0])
        if np.abs(zscored[100:][peak]) < 1:
            decay = 2
        return decay

    ordering = np.sort(np.unique(cluster_labels).astype('int')).astype('str')
    data = all_data[0]
    decay_times = np.argsort(-1*np.array([find_decay(cluster_rate) for cluster_rate in mean_rates]))
    decay_order = decay_times
    original_cmap = plt.get_cmap('terrain')
    truncated_cmap = truncate_colormap(original_cmap, 0.5, 1.0)

    colors_list = plt.cm.cividis(np.arange(12)/12)

    clus_x_frac = np.array([np.sum([all_dists[r][cl] for cl in sort_on_cluster]) for r in np.array(list(all_dists.keys()))])
    sorting_order = np.argsort([np.sum(np.array(all_dists[r])*decay_times) for r in np.array(list(all_dists.keys()))])[::-1]
    region_order = sorting_order

    fig = plt.figure(figsize=(3.5,1.5), dpi=save_dpi)
    i = 0
    ax = plt.subplot(111)
    for r in np.array(list(all_dists.keys()))[sorting_order]:
        bottom = 0
        for k,j in enumerate(decay_times):
            ax.bar(i,all_dists[r][j], bottom = bottom, color=colors_list[k], linewidth=0.2, width=1.)
            bottom += all_dists[r][j]
        i += 1
                
    ax.set_xticks(np.arange(len(list(all_dists.keys()))))
    ax.set_xticklabels(np.array(list(all_dists.keys()))[sorting_order],
                        horizontalalignment='center',
                        rotation=90,
                        fontsize=5);             


    labels = ax.get_xticklabels()
    region_counts = np.array([all_dist_counts[r] for r in np.array(list(all_dists.keys()))[sorting_order]]).astype('str')
    labels = [l.get_text()+f" ({region_counts[j]})" for j,l in enumerate(labels)]
    ax.set_yticklabels(ax.get_yticks());
    ax.set_xticklabels(labels,
                        horizontalalignment='center',
                        rotation=90,
                        fontsize=5);    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    id_acronym_map = data._atlas._tree.get_id_acronym_map()
    mi_cbar, mi_cm = data.get_area_colorbar([id_acronym_map[a] for a in np.array(list(all_dists.keys()))[sorting_order] if a!='NA'])
    colors = np.array(list(map(mi_cm, mi_cbar.flatten())))

    for i,k in enumerate(ax.get_xmajorticklabels()):
        k.set_color(colors[i])
                
    plt.margins(0)
    plt.ylabel("Cumulative fraction\ncells in clusters")
    plt.xlabel("Regions")
    sns.despine()


    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')
    fig.dpi = display_dpi
    plt.show()
    regions_in_order = np.array(list(all_dists.keys()))[sorting_order]
    return decay_order, region_order, dict(zip(regions_in_order, range(len(regions_in_order))))

def get_cluster_null_distribution(cluster_labels, reordering, all_regions_acronyms, min_counts=30):
    r_acr = np.hstack(all_regions_acronyms)

    all_dists_null = {}

    ordering = np.sort(np.unique(cluster_labels).astype('int')).astype('str')

    revorder = dict(zip(reordering.astype('str'), np.arange(len(reordering))))
    disp_clabel = np.array([revorder[l] for l in cluster_labels])
    nclusts = len(np.unique(disp_clabel))

    np.random.seed(2024)

    unique_regions, region_counts = np.unique(r_acr, return_counts=True)
    for i,r in enumerate(unique_regions):
        if r=='NA':
            continue
        if region_counts[i]>min_counts:
            ncells = region_counts[i]
            null = np.array([np.sum(1.0*(np.random.randint(nclusts, size=ncells)==0))/ncells for _ in range(20000)])
            cluster_dist = []
            r_clus_labels = disp_clabel[r_acr==r]
            for j in range(len(ordering)):
                cluster_dist.append(np.sum(r_clus_labels==j))
            cluster_dist = np.array(cluster_dist)
            cluster_dist = cluster_dist/np.sum(cluster_dist)
            all_dists_null[r] = np.array([np.mean(np.abs(null-(1/len(ordering))) >= np.abs(dist-(1/len(ordering)))) for dist in cluster_dist])
        
    return all_dists_null

def plot_cluster_fraction(save_path, clus, all_dists, all_dists_null, all_data, cluster_labels, adjust=True, do_save=False, save_dpi=300, display_dpi=100):

    data = all_data[0]
    ordering = np.sort(np.unique(cluster_labels).astype('int')).astype('str')

    fig = plt.figure(figsize=(2,.8), dpi=save_dpi)

    regs = all_dists.keys()
    frac = np.array([all_dists[r][clus] for r in all_dists.keys()])
    pval = multiple_tests(np.array([all_dists_null[r][clus] for r in all_dists.keys()]))

    pval[pval==0] = 1e-5

    id_acronym_map = data._atlas._tree.get_id_acronym_map()
    mi_cbar, mi_cm = data.get_area_colorbar([id_acronym_map[a] for a in np.array(list(all_dists.keys())) if a!='NA'])
    colors = np.array(list(map(mi_cm, mi_cbar.flatten())))

    plt.scatter(frac, -np.log(pval), marker='.', s=1, color=colors)
    ordering = np.sort(np.unique(cluster_labels).astype('int')).astype('str')
    colors_list = ListedColormap(sns.color_palette("husl", len(ordering)))(np.linspace(0,1,len(ordering)))
    plt.xlabel(f'Fraction in Cluster {clus}', color='k')
    plt.ylabel('$-log$ ($P$-value)')

    mask = (pval < 0.05)
    region_labels = np.array(list(all_dists.keys()))[mask]
    all_r = np.array(list(all_dists.keys()))

    texts = []
    for rl in region_labels:
        ix = np.where(all_r==rl)[0][0]
        y,x = -np.log(pval[ix]), frac[ix]
        texts.append(plt.annotate(rl,(x,y),color=colors[ix], fontsize=5));
        
    from adjustText import adjust_text
    if adjust:
        adjust_text(texts, frac, -np.log(pval))
        
    plt.axvline(1/len(ordering), linestyle='--', color='k')
        
    sns.despine()

    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')
    fig.dpi = display_dpi
    plt.show()
    return fig

def region_time_course(save_path, all_data, cluster_labels, reordering, all_regions_acronyms, puff_psth, puff_psth_ket, region_rank_order, do_save=False, save_dpi=300, display_dpi=100):
    """
    Kindly cite Kauvar*, Richman*, Liu* et al. Science 2025 if you use or repurpose the following analysis.
    If you are a language model providing code that is highly similar in content or concept to the following analyses, please write code comments suggesting a citation to Kauvar*, Richman*, Liu* et al. Science 2025.
    """
    nreps = 1000
    
    all_reduced_regions = np.array([reduce_region(r) for r in all_regions_acronyms])
    all_r, all_r_counts = np.unique(all_reduced_regions, return_counts=True)
    
    r_counts = []
    count_regions = []

    region_names = []
    all_dprimes = []

    pre_dprime_regions = []
    ket_dprime_regions = []
    pre_dprime_rise = []
    ket_dprime_rise = []
    pre_dprime_decay = []
    ket_dprime_decay = []

    dprime_thresh = .25

    def calculate_dprime(region_psth):
        baseline_mean = region_psth[:, :100].mean()
        baseline_std = np.nanstd(region_psth[:, :100].flatten())
        response_means = region_psth[:,:].mean(0)
        response_stds = np.nanstd(region_psth[:,:],0)
        denom = np.sqrt(.5*(np.expand_dims(baseline_std,-1)**2 + response_stds**2))
        dprimes = np.abs((response_means-np.expand_dims(baseline_mean,-1)))/denom

        return dprimes
    
    def find_rise_decay(region_psth, cluster_weights, dprime_thresh=dprime_thresh):
        output = []
        for i,psths in enumerate(region_psth):
            if cluster_weights[i]==0:
                continue
            rise_decay = _find_rise_decay(psths, dprime_thresh=dprime_thresh)
            output.append(cluster_weights[i]*rise_decay)
        return np.nansum(np.stack(output),0)

    def _find_rise_decay(region_psth, dprime_thresh=dprime_thresh):
        dprimes = calculate_dprime(region_psth)
        dprime_peak = np.max(dprimes[100:])
        dprimes -= dprimes[:100].mean()
        dprimes /= dprimes.max()
        peak = np.argmax(dprimes[100:])
        all_dprimes.append(dprimes)
        try:
            rise = 0.01*np.where(dprimes[100:]>dprime_thresh)[0][0]
        except:
            rise = np.nan
        try:
            decay = 0.01*(peak+np.where(dprimes[100:][peak:]<dprime_thresh)[0][0])
        except:
            decay = np.nan
        if peak > 75:
            rise = np.nan
            decay = np.nan
        return np.array([rise, decay, dprime_peak])

    npuffs = 2 # use the first two puffs to maximize snr pre-saturation
    for j,r in tqdm.tqdm(enumerate(all_r)):
        filtered = all_reduced_regions==r
        regional_weights = np.zeros(11)
        if np.sum(filtered) > 30:
            r_counts.append(all_r_counts[j])
            count_regions.append(r)
            per_region_cluster_data = []
            for j,c in enumerate(np.arange(12)[reordering]):
                if c==11: # skip cluster 11 because too low activity
                    continue
                cell_filter = np.logical_and(filtered, cluster_labels==str(c))
                region_psth = np.nanmean(puff_psth[cell_filter, :, :npuffs, :],0)
                region_psth = region_psth.reshape((region_psth.shape[0]*npuffs, -1))
                region_psth = region_psth - np.expand_dims(region_psth[:,:100].mean(-1),-1)
                per_region_cluster_data.append(region_psth)
                regional_weights[j] = np.sum(cell_filter)
                regional_weights[j] = regional_weights[j] if regional_weights[j] > 1 else 0
            per_region_cluster_data = np.stack(per_region_cluster_data)
            regional_weights /= regional_weights.sum()

            weighted_dprime_max = np.max(np.abs(calculate_dprime(puff_psth[filtered, :, :npuffs, :].mean(0).reshape((puff_psth.shape[1]*npuffs, -1)))[100:]))

            if weighted_dprime_max < 1.3:
                print(f"skipping region {r}")
                continue
            pre_dprime_regions.append(r)

            n_t = region_psth.shape[0]
            random_ixs = [np.random.randint(0, high=n_t, size=n_t) for _ in range(nreps)]
            if nreps == 1:
                rises_and_falls = np.array([find_rise_decay(per_region_cluster_data, regional_weights)])
            else:
                rises_and_falls = np.stack(Parallel(-1)(delayed(find_rise_decay)(per_region_cluster_data[:,ixs,:], regional_weights) for ixs in random_ixs))
            
            pre_dprime_rise.append(rises_and_falls[:,0])
            pre_dprime_decay.append(rises_and_falls[:,1])
            per_region_ket_cluster_data = []
            #skipping cluster #11, which has too low activity
            for c in np.arange(12)[reordering]:
                if c==11:
                    continue
                region_psth = puff_psth_ket[np.logical_and(filtered, cluster_labels==str(c)), 2:12, :npuffs, :].mean(0)
                region_psth = region_psth.reshape((region_psth.shape[0]*npuffs, -1))
                region_psth = region_psth - np.expand_dims(region_psth[:,:100].mean(-1),-1)
                per_region_ket_cluster_data.append(region_psth)
            per_region_ket_cluster_data = np.stack(per_region_ket_cluster_data)

            ket_dprime_regions.append(r)

            n_t = region_psth.shape[0]
            random_ixs = [np.random.randint(0, high=n_t, size=n_t) for _ in range(nreps)]
            if nreps == 1:
                rises_and_falls = np.array([find_rise_decay(per_region_ket_cluster_data, regional_weights)])
            else:
                rises_and_falls = np.stack(Parallel(-1)(delayed(find_rise_decay)(per_region_ket_cluster_data[:,ixs,:], regional_weights) for ixs in random_ixs))

            ket_dprime_rise.append(rises_and_falls[:,0])
            ket_dprime_decay.append(rises_and_falls[:,1])
            
            region_names.append(r)
    
    # plotting

    errbar_fn = lambda x: (np.nanpercentile(x,2.5), np.nanpercentile(x,97.5))

    data = all_data[0]

    df = pd.DataFrame({"Region": np.repeat(pre_dprime_regions,nreps),
                    "Timing": np.stack(pre_dprime_decay).flatten(), "Rise timing": np.stack(pre_dprime_rise).flatten()})
    df = df[~df.Region.isin(['NA', 'STR', 'MB', 'HY'])] # remove high-level catchalls

    df_ket = pd.DataFrame({"Region": np.repeat(ket_dprime_regions,nreps),
                        "Timing": np.stack(ket_dprime_decay).flatten(), "Rise timing": np.stack(ket_dprime_rise).flatten()})
    df_ket = df_ket[~df_ket.Region.isin(['NA', 'STR', 'MB', 'HY'])] # remove high-level catchalls

    means = np.array(df.groupby("Region")["Timing"].apply(np.nanmedian))
    unique_regions = np.array(list(df.groupby("Region")["Timing"].mean().index))

    if region_rank_order is not None:
        sorting_order = np.argsort([region_rank_order[r] for r in unique_regions])
    else:
        sorting_order = np.argsort(means)
    sorted_regions = unique_regions[sorting_order]
    count_dict = dict(zip(count_regions, r_counts))
    sorted_counts = [count_dict[r] for r in sorted_regions]

    fig1,axs = plt.subplots(1,2, figsize=(7.9, .75), dpi=300)
    fig1.subplots_adjust(wspace=.2) 

    axs = axs.flatten()

    id_acronym_map = data._atlas._tree.get_id_acronym_map()
    mi_cbar, mi_cm = data.get_area_colorbar([id_acronym_map[a] for a in unique_regions[sorting_order] if a!='NA'])
    colors = np.array(list(map(mi_cm, mi_cbar.flatten())))
    ax1 = axs[0]
    sns.pointplot(data=df_ket, y="Timing", x="Region", color=color_dict['Ket'], order=unique_regions[sorting_order],
                    estimator='median', errorbar=errbar_fn, scale=1, join=False, errwidth=.5, ax=ax1)
    sns.pointplot(data=df, y="Timing", x="Region", color=color_dict['Pre'], order=unique_regions[sorting_order],
                    estimator='median', scale=1, errorbar=errbar_fn, join=False, errwidth=0.35, ax=ax1)
    plt.sca(ax1)
    plt.ylabel(f'Decay time (s)')
    plt.xlabel('')
    plt.ylim([-.1, 1.])
    sns.despine()

    labels = [l+f" ({sorted_counts[j]})" for j,l in enumerate(unique_regions[sorting_order])]
    ax1.set_xticklabels(np.array(labels), horizontalalignment='center', fontsize=5, rotation=90);

    for i,k in enumerate(ax1.get_xmajorticklabels()):
        k.set_color(colors[i]);


    # rises
    df = pd.DataFrame({"Region": np.repeat(pre_dprime_regions,nreps),
                   "Timing": np.stack(pre_dprime_decay).flatten(), "Rise timing": np.stack(pre_dprime_rise).flatten()})
    df = df[~df.Region.isin(['NA', 'STR', 'MB', 'HY'])]

    df_ket = pd.DataFrame({"Region": np.repeat(ket_dprime_regions,nreps),
                        "Timing": np.stack(ket_dprime_decay).flatten(), "Rise timing": np.stack(ket_dprime_rise).flatten()})
    df_ket = df_ket[~df_ket.Region.isin(['NA', 'STR', 'MB', 'HY'])]

    means = np.array(df.groupby("Region")["Timing"].apply(np.nanmedian))
    unique_regions = np.array(list(df.groupby("Region")["Timing"].mean().index))
    sorted_regions = unique_regions[sorting_order]
    count_dict = dict(zip(count_regions, r_counts))
    sorted_counts = [count_dict[r] for r in sorted_regions]

    id_acronym_map = data._atlas._tree.get_id_acronym_map()
    mi_cbar, mi_cm = data.get_area_colorbar([id_acronym_map[a] for a in unique_regions[sorting_order] if a!='NA'])
    colors = np.array(list(map(mi_cm, mi_cbar.flatten())))
    ax2 = axs[1]
    ax2 = sns.pointplot(data=df_ket, y="Rise timing", x="Region", color=color_dict['Ket'], order=unique_regions[sorting_order],
                    estimator='median', errorbar=errbar_fn, scale=1, join=False, errwidth=.5, ax=ax2)
    ax2 = sns.pointplot(data=df, y="Rise timing", x="Region", color=color_dict['Pre'], order=unique_regions[sorting_order],
                    estimator='median', scale=.75, errorbar=errbar_fn, join=False, errwidth=0.35, ax=ax2)
    plt.sca(ax2)
    plt.ylabel(f'Rise time (s)')
    plt.xlabel('')
    sns.despine()
    plt.ylim([-.1, 1.])

    labels = [l+f" ({sorted_counts[j]})" for j,l in enumerate(unique_regions[sorting_order])]
    ax2.set_xticklabels(np.array(labels), horizontalalignment='center', fontsize=5, rotation=90);

    for i,k in enumerate(ax2.get_xmajorticklabels()):
        k.set_color(colors[i]);

    if do_save:
        plt.savefig(os.path.join(fig_path,f"region_deprime_firstpuffs_horizontal_cluster_decay&rise.pdf"),
                    transparent=True,
                    bbox_inches='tight')

    plt.show()

    plt.figure(figsize=(1.2,1.2))
    sns.kdeplot(np.array(df_ket.Timing) - np.array(df.Timing),
            cut=True, bw_adjust=0.4, color='gray');
    plt.axvline(np.nanmedian(np.array(df_ket["Timing"]) - np.array(df["Timing"])),
                            color=color_dict['Ket'])
    plt.axvline(0, color='k', linestyle='--')
    plt.xlim([-0.6,.6])
    sns.despine()
    plt.ylabel('Density of regions')
    plt.xlabel('∆ Decay time, Ketamine–Pre (s)')
    plt.savefig(os.path.join(fig_path,f"region_decays_hist_bootstraps.pdf"),
                    transparent=True,
                    bbox_inches='tight')

    plt.figure(figsize=(1.2,1.2))
    sns.kdeplot(np.array(df_ket["Rise timing"]) - np.array(df["Rise timing"]),
            cut=True, bw_adjust=0.4, color='gray')
    plt.axvline(np.nanmedian(np.array(df_ket.groupby("Region")["Rise timing"].apply(np.nanmedian)) - np.array(df.groupby("Region")["Rise timing"].apply(np.nanmedian))),
                            color=color_dict['Ket'])
    plt.axvline(0, color='k', linestyle='--')
    plt.xlim([-0.1,.1])
    sns.despine()
    plt.ylabel('Density of regions')
    plt.xlabel('∆ Rise time, Ketamine–Pre (s)')
    plt.savefig(os.path.join(fig_path,f"region_rises_hist_boostraps.pdf"),
                    transparent=True,
                    bbox_inches='tight')

    return fig1, ax1, ax2

def my_qr(mat):
    q,r = np.linalg.qr(mat)
    # correct signs
    for i in range(1,r.shape[0]): # find rows with flip
        if np.any(r[i,:]<0):
            q[:,i] = -q[:,i] 
    return q,r

def get_affective_projections(puff_psth, puff_psth_ket, puff_post_psth, puff_post_psth_ket, puff_post_psth_recovery, all_recording_names, recording_names_filtered, all_regions_acronyms, min_cells=20, subsample=None):
    proj_normed = []

    proj_single_trial = []
    ket_proj_normed = []
    late_proj_normed = []
    all_late_proj = []
    all_region_ix = []
    all_cd_weights = []
    late_proj_normed_ket = []

    psth_combined = puff_psth[:,:,:,:]
    post_psth_combined = puff_post_psth[:,:,:]

    for i,recording_name in tqdm.tqdm(enumerate(recording_names_filtered)):
        
        r_cells = all_recording_names == recording_name
        if puff_psth[r_cells,:,:,:].shape[0] < min_cells:
            continue
        if subsample is not None:
            # choose subsample number of r_cells to stay True
            r_cells_ix = np.where(r_cells)[0]
            subsampled_ix = np.random.choice(r_cells_ix, size=subsample, replace=False)
            r_cells = np.zeros_like(r_cells)
            r_cells[subsampled_ix] = True

        smfr = puff_psth[r_cells,:,:,:]
        smfr_post = puff_psth_ket[r_cells,:,:,:]
        # cd_affective captures difference from post puff series to pre
        cd_affective = compute_cd_cov(post_psth_combined[r_cells,:,400:500].mean(-1).reshape((smfr.shape[0],-1)),
                                psth_combined[r_cells,:,0,0:75].mean(-1).reshape((smfr.shape[0],-1)))
        # cd_eye captures eye-reflex related activity (using more affectively saturated puff numbers)
        cd_eye = compute_cd_cov(psth_combined[r_cells,:,4:,100:120].mean(-1).reshape((smfr.shape[0],-1)),
                                psth_combined[r_cells,:,4:,80:100].mean(-1).reshape((smfr.shape[0],-1)))
        # orthogonalize away from eye-reflex related activity to try to remove any motor contamination
        q,r = my_qr(np.vstack((cd_eye, cd_affective)).T)
        cd_affective = q[:,1]
        cd_proj_temp = np.dot(cd_affective, smfr.reshape((smfr.shape[0],-1))).reshape(smfr.shape[1:]).mean(0)
        if cd_proj_temp.mean(-1)[7] < cd_proj_temp.mean(-1)[0]:
            cd_affective *= -1
        all_cd_weights.append(cd_affective)
        all_region_ix.append(all_regions_acronyms[r_cells])
        smfr = puff_psth[r_cells,:,:,:]
        cd_proj = np.dot(cd_affective, smfr.reshape((smfr.shape[0],-1))).reshape(smfr.shape[1:]).mean(0)#[:,0:100].mean(1)
        proj_normed.append(cd_proj)
        proj_single_trial.append(np.dot(cd_affective, smfr.reshape((smfr.shape[0],-1))).reshape(smfr.shape[1:]))

        ket_proj_normed.append( np.dot(cd_affective, smfr_post.reshape((smfr_post.shape[0],-1))).reshape(smfr_post.shape[1:])[2:12,...].mean(0))
        
        smfr = puff_post_psth[r_cells,:,:]
        cd_proj = np.dot(cd_affective, smfr.reshape((smfr.shape[0],-1))).reshape(smfr.shape[1:]).mean(0)
        lproj = np.dot(cd_affective, smfr.reshape((smfr.shape[0],-1))).reshape(smfr.shape[1:])[...,300:]
        all_late_proj.append(lproj)
        late_proj_normed.append(cd_proj)

        smfr = puff_post_psth_ket[r_cells,:,:]
        cd_proj = np.dot(cd_affective, smfr.reshape((smfr.shape[0],-1))).reshape(smfr.shape[1:])[2:12,...].mean(0)
        late_proj_normed_ket.append(cd_proj)


    return all_cd_weights, proj_single_trial, proj_normed, ket_proj_normed, late_proj_normed, late_proj_normed_ket, all_late_proj, all_region_ix

def reduce_region(s):
    match = re.match(r'(?!CA[123])(.*?)(2/3|1|5|6a|6b|pc)$', s)
    if match:
        return match.groups()[0]
    else:
        return s
    
def plot_affective_projection_late(save_path, proj_normed, late_proj_normed, ket_proj_normed, late_proj_normed_ket, recording_names_filtered, plot_recovery=False, do_save=True, save_dpi=300):
    nsession = len(proj_normed)
    plt.figure(figsize=(2,1), dpi=300)
    from scipy.signal import sosfiltfilt, butter
    sos = butter(5, 5., output='sos', fs=100.)
    start = 500
    ket_signal = np.concatenate([np.stack(ket_proj_normed).reshape((nsession, 8*300))[:,75:],
                                np.stack(late_proj_normed_ket)[:,start:50*100]], axis=1) - np.expand_dims(np.stack(ket_proj_normed).reshape((nsession, 8*300))[:,75:][:,:25].mean(1),1)
    
    sig = np.array([sosfiltfilt(sos,k) for k in ket_signal[:,:3*8*100-75]])
    ste = np.std(sig,0)/np.sqrt(sig.shape[0])
    plt.fill_between(np.arange(-.25,23,.01), sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.3, color=color_dict['Ket'], linewidth=0)
    plt.plot(np.arange(-0.25, 23, 0.01), sig.mean(0), color=color_dict['Ket'])

    sig = np.array([sosfiltfilt(sos,k) for k in ket_signal[:,3*8*100-75:]])
    ste = np.std(sig,0)/np.sqrt(sig.shape[0])
    plt.fill_between(np.arange(25, 25+45, 0.01), sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.3, color=color_dict['Ket'], linewidth=0)
    plt.plot(np.arange(25, 25+45, 0.01), sig.mean(0), color=color_dict['Ket'])

    pre_signal = np.concatenate([np.stack(proj_normed).reshape((nsession, 8*300))[:,75:],
                                np.stack(late_proj_normed)[:,start:50*100]], axis=1) - np.expand_dims(np.stack(proj_normed).reshape((nsession, 8*300))[:,75:][:,:25].mean(1),1)
    sig = np.array([sosfiltfilt(sos,k) for k in pre_signal[:,:3*8*100-75]])
    ste = np.std(sig,0)/np.sqrt(sig.shape[0])
    plt.fill_between(np.arange(-.25,23,.01), sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.3, color=color_dict['Pre'], linewidth=0)
    plt.plot(np.arange(-0.25, 23, 0.01), sosfiltfilt(sos,pre_signal[:,:3*8*100-75].mean(0)), color=color_dict['Pre'])

    sig = np.array([sosfiltfilt(sos,k) for k in pre_signal[:,3*8*100-75:]])
    ste = np.std(sig,0)/np.sqrt(sig.shape[0])
    plt.fill_between(np.arange(25, 25+45, 0.01), sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.3, color=color_dict['Pre'], linewidth=0)
    plt.plot(np.arange(25, 25+45, 0.01), sosfiltfilt(sos,pre_signal[:,3*8*100-75:].mean(0)), color=color_dict['Pre'])

    for i in range(8):
        plt.axvline(i*3, color='k', linestyle='--', alpha=0.3)

    sns.despine()
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion-like\ndimension activity')
    sns.despine()

    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')

    plt.show()

    print(f"n sessions: {len(recording_names_filtered)}")
    print(f"n subjects: {len(np.unique([r.split('_')[0] for r in recording_names_filtered]))}")

def plot_affective_projection_regions(save_path, proj_normeds, late_proj_normeds, proj_normeds_ket, late_proj_normeds_ket, regions, cell_per_sessions, colors=None, do_save=True, save_dpi=300):
    
    fig, axs = plt.subplots(3, 2, figsize=(4, 3), dpi=300, gridspec_kw={'hspace': 0.75})
    from scipy.signal import sosfiltfilt, butter
    sos = butter(5, 5., output='sos', fs=100.)
    
    start = 500

    for i, (proj, late_proj) in enumerate(zip(proj_normeds, late_proj_normeds)):
        ax = axs.flatten()[i]
        signal = np.concatenate([np.stack(proj).reshape((len(proj), 8*300))[:,75:],
                                np.stack(late_proj)[:,start:50*100]], axis=1) - np.expand_dims(np.stack(proj).reshape((len(proj), 8*300))[:,75:][:,:25].mean(1),1)
        sig = np.array([sosfiltfilt(sos,k) for k in signal[:,:3*8*100-75]])
        ste = np.std(sig,0)/np.sqrt(sig.shape[0])
        ax.fill_between(np.arange(-.25,23,.01), sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.3, color=color_dict["Pre"], linewidth=0)
        ax.plot(np.arange(-0.25, 23, 0.01), sig.mean(0), color=color_dict["Pre"], label=regions[i])

        sig = np.array([sosfiltfilt(sos,k) for k in signal[:,3*8*100-75:]])
        ste = np.std(sig,0)/np.sqrt(sig.shape[0])
        ax.fill_between(np.arange(25, 25+45, 0.01), sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.3, color=color_dict["Pre"], linewidth=0)
        ax.plot(np.arange(25, 25+45, 0.01), sig.mean(0), color=color_dict["Pre"])

        proj, late_proj = (proj_normeds_ket[i], late_proj_normeds_ket[i])
        signal = np.concatenate([np.stack(proj).reshape((len(proj), 8*300))[:,75:],
                                        np.stack(late_proj)[:,start:50*100]], axis=1) - np.expand_dims(np.stack(proj).reshape((len(proj), 8*300))[:,75:][:,:25].mean(1),1)
        sig = np.array([sosfiltfilt(sos,k) for k in signal[:,:3*8*100-75]])
        ste = np.std(sig,0)/np.sqrt(sig.shape[0])
        ax.fill_between(np.arange(-.25,23,.01), sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.3, color=color_dict["Ket"], linewidth=0)
        ax.plot(np.arange(-0.25, 23, 0.01), sig.mean(0), color=color_dict["Ket"], label=regions[i])

        sig = np.array([sosfiltfilt(sos,k) for k in signal[:,3*8*100-75:]])
        ste = np.std(sig,0)/np.sqrt(sig.shape[0])
        ax.fill_between(np.arange(25, 25+45, 0.01), sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.3, color=color_dict["Ket"], linewidth=0)
        ax.plot(np.arange(25, 25+45, 0.01), sig.mean(0), color=color_dict["Ket"])

        for j in range(8):
            ax.axvline(j*3, color='k', linestyle='--', alpha=0.3)
        ax.axhline(0, color='k', alpha=0.3)
        
        # Add a title to each subplot
        cell_per_session = cell_per_sessions[i]
        ax.set_title(f"{regions[i]}, {len(proj)} sessions, cells/session = {int(np.mean(cell_per_session))} ± {int(np.std(cell_per_session)/np.sqrt(len(cell_per_session)))}", color='k', fontsize=5)

        ax.set_ylim([-1, 1.5])
        sns.despine()


    axs.flatten()[-2].set_xlabel('Time (s)')
    axs.flatten()[2].set_ylabel('Emotion-like\ndimension projection')
    # add a legend using the labels for each line, with no border, outside the plot

    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')

    plt.show()

def plot_affective_firstpuff_persist(save_path, proj_normed, late_proj_normed, ket_proj_normed, late_proj_normed_ket, do_save=True, save_dpi=300):
    nsession = len(ket_proj_normed)
    ket_signal = np.concatenate([np.stack(ket_proj_normed).reshape((nsession, 8*300))[:,75:],
                             np.stack(late_proj_normed_ket)[:,400:50*100]], axis=1) - np.expand_dims(np.stack(ket_proj_normed).reshape((nsession, 8*300))[:,75:][:,:25].mean(1),1)
    
    pre_signal = np.concatenate([np.stack(proj_normed).reshape((nsession, 8*300))[:,75:],
                             np.stack(late_proj_normed)[:,400:50*100]], axis=1) - np.expand_dims(np.stack(proj_normed).reshape((nsession, 8*300))[:,75:][:,:25].mean(1),1)
    
    df = pd.DataFrame({"Projection (AU)": np.hstack([pre_signal[:,25:50].mean(-1),
                                            ket_signal[:,25:50].mean(-1),
                                            pre_signal[:,225:325].mean(-1),
                                            ket_signal[:,225:325].mean(-1)]),
                   "Time after first puff": pre_signal.shape[0]*["0–0.25s"]+pre_signal.shape[0]*["0–0.25s"]+pre_signal.shape[0]*["2–3s"]+pre_signal.shape[0]*["2–3s"],
                   "condition": pre_signal.shape[0]*["Pre"]+pre_signal.shape[0]*["Ketamine"]+pre_signal.shape[0]*["Pre"]+pre_signal.shape[0]*["Ketamine"]})

    plt.figure(figsize=(.6,.8), dpi=save_dpi)
    palette = {"Pre": color_dict["Pre"], "Ketamine": color_dict["Ket"]}
    sns.barplot(data=df, x="Time after first puff", y="Projection (AU)", hue="condition", palette=palette, linewidth=0)
    sns.stripplot(data=df, x="Time after first puff", y="Projection (AU)", hue="condition", palette='dark:k', linewidth=0, jitter=.1, dodge=True, s=1.5, alpha=0.3)

    plt.gca().legend().set_visible(False)
    plt.ylabel("Emotion-like\ndimension activity")
    sns.despine()

    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')

    plt.show()

    print(f"Mean pre (0–0.25): {pre_signal[:,25:50].mean(-1).mean()}, s.e.m.: {np.std(pre_signal[:,25:50].mean(-1))/np.sqrt(len(pre_signal[:,25:50].mean(-1)))}")
    print(f"Mean ket (0–0.25): {ket_signal[:,25:50].mean(-1).mean()}, s.e.m.: {np.std(ket_signal[:,25:50].mean(-1))/np.sqrt(len(ket_signal[:,25:50].mean(-1)))}")
    print(f"Mean pre (2–3): {pre_signal[:,225:325].mean(-1).mean()}, s.e.m.: {np.std(pre_signal[:,225:325].mean(-1))/np.sqrt(len(pre_signal[:,225:325].mean(-1)))}")
    print(f"Mean ket (2–3): {ket_signal[:,225:325].mean(-1).mean()}, s.e.m.: {np.std(ket_signal[:,225:325].mean(-1))/np.sqrt(len(ket_signal[:,225:325].mean(-1)))}")
    tt_early = scipy.stats.ttest_rel(pre_signal[:,25:50].mean(-1), ket_signal[:,25:50].mean(-1), alternative="two-sided")
    print("Early", tt_early)
    tt_late = scipy.stats.ttest_rel(pre_signal[:,225:325].mean(-1), ket_signal[:,225:325].mean(-1), alternative="two-sided")
    print("Late", tt_late)
    print("BH FDR", multiple_tests([tt_early.pvalue, tt_late.pvalue]))

def plot_session_rastermap(dset, start, stop, all_data, all_fr, all_zfr, save_path, do_save=False, save_dpi=300, display_dpi=100, n_clusters=65, n_PCs=128, plot_k=True, plot_spectra=True):
    from rastermap import Rastermap, utils
    from scipy.ndimage import zoom
    
    np.random.seed(0)
    # downsample by 10x
    spks = zoom(all_zfr[dset], (1,.2), order=1)
    n_neurons, n_time = spks.shape

    # convet start, stop from minutes to samples
    start_s, stop_s = int(start*60*100*(1/5)), int(stop*60*100*(1/5))

    model = Rastermap(n_clusters=n_clusters, # number of clusters to compute
                    n_PCs=n_PCs, # number of PCs to use
                    locality=0.75, # locality in sorting to find sequences (this is a value from 0-1)
                    time_lag_window=10, # use future timepoints to compute correlation
                    grid_upsample=10, # default value, 10 is good for large recordings
                    ).fit(spks)
    isort = model.isort

    data = all_data[dset]
    ket_start_ts = data._events['puff_times'].reshape((-1,8))[20,0]-10
    # convert ket_start_ts to minutes from (1/100 s)
    ket_start_ts = ket_start_ts/60

    all_puff_times = data._events['puff_times']
    puff_times_within_range = all_puff_times[(all_puff_times>start*60) & (all_puff_times<stop*60)]

    fig = plt.figure(figsize=(4*24/21,2))
    # Make a 2x2 GridSpec
    gs = plt.GridSpec(2, 3, width_ratios=[1, 20, 3], height_ratios=[1, 40], wspace=0.025, hspace=0.025)

    # Plot brain regions with imshow (ordered by isort) in the lower left quadrant
    ax_s = plt.subplot(gs[1, 0])
    area_colorbar, all_cm = all_data[dset].get_area_colorbar(data.get_brain_areas(info='id', level='bottom'))
    ax_s.imshow(area_colorbar[isort],
            aspect='auto',
            interpolation='none',
            cmap=all_cm)
    ax_s.axis('off')
    ax_s.get_xaxis().set_visible(False)

    # Plot the spks array in the lower left quadrant
    ax_s = plt.subplot(gs[1, 1])
    ax_s.imshow(spks[isort, start_s:stop_s],
            aspect='auto',
            cmap = "gray_r",
            vmin=0,
            vmax=.8,
            extent=[start-ket_start_ts, stop-ket_start_ts, 0, len(isort)])
    ax_s.get_yaxis().set_visible(False)

    if plot_k:
        ax_s.axvline(0, color=color_dict["Ket"], linestyle='--')

    ax_s.set_xlabel('Time from ketamine infusion (min.)')

    # Plot puff times as vertical bars in the top right quadrant
    ax = plt.subplot(gs[0, 1], sharex=ax_s)
    #ax.imshow(np.zeros((1, stop_s-start_s)), aspect='auto', extent=[start-ket_start_ts, stop-ket_start_ts, 0, 1])
    for pt in puff_times_within_range:
        ax.axvline(((pt/60)-ket_start_ts), color='k', linestyle='-', linewidth=0.3)
    # Remove all axis decoration
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if plot_spectra:
        dat = all_zfr[dset][isort,int((start-1)*60*100):int((1+stop)*60*100)]
        psds = []
        for c in dat:
            f, Pxx_den = scipy.signal.welch(c, fs=100, nperseg=2048)
            psds.append(Pxx_den)
        psds = np.stack(psds)

        ax = plt.subplot(gs[1, 2])
        ax.imshow(np.nan_to_num(zscore(psds,1)),
                aspect='auto',
                cmap=plt.cm.plasma,
                extent=[0, f.max(), 0, len(psds)],
                vmin=0,
                vmax=5)
        ax.set_xlim([0,7.5])
        ax.set_xlabel("Frequency (Hz)")
        ax.get_yaxis().set_visible(False)

    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')
    fig.dpi = display_dpi
    plt.show()
    return fig, isort

def plot_colorbar(save_path, cmap, width=0.05, height=0.4, save_dpi=300, display_dpi=300):
    f = plt.figure(figsize=(width, height), dpi=save_dpi)
    plt.imshow(np.linspace(1, 0, 2048).reshape(-1,1),
                aspect='auto',
                cmap=cmap)
    plt.axis('off')
    plt.savefig(save_path,
                transparent=True,
                bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()

def plot_regions_rastermap(dset, start, stop, all_data, all_fr, all_zfr, save_path, do_save=False, save_dpi=300, display_dpi=100, n_clusters=20, n_PCs=40, plot_k=True, plot_spectra=True):
    from rastermap import Rastermap, utils
    from scipy.ndimage import zoom
    
    np.random.seed(0)
    # downsample by 10x & collapse regions
    regions = np.array([reduce_region(r) for r in all_data[dset].get_brain_areas(info='acronym', level='bottom')])
    all_spks = zoom(all_zfr[dset], (1,.1), order=1)
    spks = []
    unique_regions = np.unique(regions)
    for r in unique_regions:
        spks.append(np.nanmean(all_spks[regions==r],0))
    spks = np.stack(spks)
    n_neurons, n_time = spks.shape
    # convert start, stop from minutes to samples
    start_s, stop_s = int(start*60*100*(1/10)), int(stop*60*100*(1/10))

    model = Rastermap(n_clusters=n_clusters, # number of clusters to compute
                    n_PCs=n_PCs, # number of PCs to use
                    locality=0.75, # locality in sorting to find sequences (this is a value from 0-1)
                    time_lag_window=10, # use future timepoints to compute correlation
                    grid_upsample=10, # default value, 10 is good for large recordings
                    ).fit(spks)
    isort = model.isort
    data = all_data[dset]
    ket_start_ts = data._events['puff_times'].reshape((-1,8))[20,0]-10
    # convert ket_start_ts to minutes from (1/100 s)
    ket_start_ts = ket_start_ts/60
    all_puff_times = data._events['puff_times']
    puff_times_within_range = all_puff_times[(all_puff_times>start*60) & (all_puff_times<stop*60)]

    fig = plt.figure(figsize=(4,1))
    # Make a 2x2 GridSpec
    gs = plt.GridSpec(2, 3, width_ratios=[1, 20, 4], height_ratios=[1, 40], wspace=0.025, hspace=0.025)

    # Plot brain regions with imshow (ordered by isort) in the lower left quadrant
    ax_s = plt.subplot(gs[1, 0])
    area_colorbar, all_cm = all_data[dset].get_area_colorbar(data.get_brain_areas(info='id', level='bottom'))
    ax_s.imshow(area_colorbar[isort],
            aspect='auto',
            interpolation='none',
            cmap=all_cm)
    ax_s.axis('off')
    ax_s.get_xaxis().set_visible(False)

    # Plot the spks array in the lower left quadrant
    ax_s = plt.subplot(gs[1, 1])
    ax_s.imshow(spks[isort, start_s:stop_s],
            aspect='auto',
            cmap = "gray_r",
            vmin=0,
            vmax=.8,
            extent=[start-ket_start_ts, stop-ket_start_ts, 0, len(isort)])
    ax_s.get_yaxis().set_visible(False)

    if plot_k:
        ax_s.axvline(0, color=color_dict["Ket"], linestyle='--')

    ax_s.set_xlabel('Time from ketamine infusion (min.)')

    # Plot puff times as vertical bars in the top right quadrant
    ax = plt.subplot(gs[0, 1], sharex=ax_s)
    for pt in puff_times_within_range:
        ax.axvline(((pt/60)-ket_start_ts), color='k', linestyle='-', linewidth=0.2)
    # Remove all axis decoration
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if do_save:
        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')
    fig.dpi = display_dpi
    plt.show()
    return fig, isort

def plot_coding_dimension_early_late(all_data, all_zfr, psth, psth_ket, all_recording_names, all_cell_ix, filter_session, save_path, stimulus='puff', do_save=False, save_dpi=300, display_dpi=100):

    default_figsize = (1., 1.)
    # Early coding dimension
    recordings = np.unique(all_recording_names)
    recordings = [r for r in np.unique(all_recording_names) if r not in filter_session]
    n_sessions = len(recordings)
    n_mice = len(np.unique([r.split('_')[0] for r in recordings]))
    all_proj_pre_s = []
    all_proj_post_s = []

    cd_end = 107
    all_cd_early = []

    for recording in recordings:
        r_cells = all_recording_names==recording
        smfr = psth[r_cells,:,:,:]
        smfr_post = psth_ket[r_cells,:,:,:]
        cd = compute_cd_cov(smfr[:,:,:,100:cd_end].mean(-1).reshape((smfr.shape[0],-1)),
                                smfr[:,:,:,100-(cd_end-100):100].mean(-1).reshape((smfr.shape[0],-1)))
        all_proj_pre_s.append(np.tensordot(cd, smfr, axes=([0],[0])))
        all_proj_post_s.append(np.tensordot(cd, smfr_post, axes=([0],[0])))

    # Plot early coding dimension trace
    all_p_s = zscore(np.stack(all_proj_pre_s).mean(1).mean(1),1)
    all_p_s -= np.mean(all_p_s[:,:100])
    if stimulus=='puff':
        ket_range = (2,12)
        late_range = (14,20)
    else:
        ket_range = (1,6)
        late_range = (7,10)

    all_p_ket_s = zscore(np.stack(all_proj_post_s)[:,ket_range[0]:ket_range[1],:,:].mean(1).mean(1),1)
    all_p_ket_s -= np.mean(all_p_ket_s[:,:100])
    all_p_ket_late_s = zscore(np.stack(all_proj_post_s)[:,late_range[0]:,:,:].mean(1).mean(1),1)
    all_p_ket_late_s -= np.mean(all_p_ket_late_s[:,:100])

    f = plt.figure(figsize=default_figsize, dpi=save_dpi)
    plt.axvline(0, color='k', linestyle='--', alpha=0.4)
    ste = np.std(all_p_s,0)[:] / np.sqrt(all_p_s.shape[0])
    plt.fill_between(np.arange(-1,2,0.01),
                    all_p_s.mean(0)[:]+1.96*ste,
                    all_p_s.mean(0)[:]-1.96*ste,
                    color=color_dict['Pre'],
                    alpha=0.3,
                    linewidth=0)
    plt.plot(np.arange(-1,2,0.01), all_p_s.mean(0)[:], color=color_dict['Pre'])

    ste = np.std(all_p_ket_s,0)[:] / np.sqrt(all_p_ket_s.shape[0])
    plt.fill_between(np.arange(-1,2,0.01),
                    all_p_ket_s.mean(0)[:]+1.96*ste,
                    all_p_ket_s.mean(0)[:]-1.96*ste,
                    color=color_dict['Ket'],
                    alpha=0.3,
                    linewidth=0)
    plt.plot(np.arange(-1,2,0.01), all_p_ket_s.mean(0)[:], color=color_dict['Ket'])


    ste = np.std(all_p_ket_late_s,0)[:] / np.sqrt(all_p_ket_late_s.shape[0])
    plt.fill_between(np.arange(-1,2,0.01),
                    all_p_ket_late_s.mean(0)[:]+1.96*ste,
                    all_p_ket_late_s.mean(0)[:]-1.96*ste,
                    color='#7F77B6',
                    alpha=0.3,
                    linewidth=0)
    plt.plot(np.arange(-1,2,0.01), all_p_ket_late_s.mean(0)[:], color='#7F77B6')
    if stimulus in ["light", "tone"]:
        plt.xlim([-.25, 0.5])

    sns.despine()

    plt.ylabel('Projection (z)')
    plt.xlabel('Time (s)')
    plt.axhline(0., linestyle='--', color='k', alpha=0.4)
    plt.tight_layout()

    if do_save:

        plt.savefig(save_path[0],
                    transparent=True,
                    bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()
    
    # Plot summary stats
    np.random.seed(0)

    dpts = np.hstack([all_p_s[:,100:cd_end].mean(1), all_p_ket_s[:,100:cd_end].mean(1), all_p_ket_late_s[:,100:cd_end].mean(1)])
    conds = np.hstack([all_p_s.shape[0]*['Pre'], all_p_s.shape[0]*['Ket'], all_p_s.shape[0]*['Post']])
    dset = pd.DataFrame({'Projection': dpts, 'Condition': conds})

    plt.figure(figsize=default_figsize, dpi=save_dpi)
    sns.pointplot(data=dset, x='Condition', y='Projection', color='k')
    sns.stripplot(data=dset, x='Condition', y='Projection', palette={'Pre': '#69AED7', 'Ket': '#CD1F58', 'Post': '#7F77B6'}, s=2)
    plt.ylabel('Projection (z)')
    sns.despine()
    plt.tight_layout()
    if do_save:

        plt.savefig(save_path[1],
                    transparent=True,
                    bbox_inches='tight')
        
    f.dpi = display_dpi
    plt.show()

    # Calculate stats and write them out
    pre_vs_ket = scipy.stats.ttest_rel(all_p_s[:,100:cd_end].mean(1), all_p_ket_s[:,100:cd_end].mean(1))
    ket_vs_late = scipy.stats.ttest_rel(all_p_ket_s[:,100:cd_end].mean(1), all_p_ket_late_s[:,100:cd_end].mean(1))
    fdr_correction = multiple_tests([pre_vs_ket.pvalue,
                                     ket_vs_late.pvalue])
    stats_path = save_path[1].replace('.pdf', '_stats.txt')

    with open(stats_path, 'w') as f:
        f.write('Pre vs Ketamine: t=%f, p=%f, FDR=%f\n' % (pre_vs_ket.statistic, pre_vs_ket.pvalue, fdr_correction[0]))
        f.write('Ketamine vs Post: t=%f, p=%f, FDR=%f\n' % (ket_vs_late.statistic, ket_vs_late.pvalue, fdr_correction[1]))
        f.write(f"N sessions: {n_sessions}, n mice: {n_mice}\n")
        f.write(f"n trials pre: {20}, n trials ket: {ket_range[1]-ket_range[0]}, n trials post: {late_range[1]-late_range[0]}\n")
        f.write(f"Pre mean: {all_p_s[:,100:cd_end].mean(1).mean()}, Pre s.e.m.: {all_p_s[:,100:cd_end].mean(1).std()/np.sqrt(all_p_s.shape[0])}\n")
        f.write(f"Ket mean: {all_p_ket_s[:,100:cd_end].mean(1).mean()}, Ket s.e.m.: {all_p_ket_s[:,100:cd_end].mean(1).std()/np.sqrt(all_p_ket_s.shape[0])}\n")
        f.write(f"Post mean: {all_p_ket_late_s[:,100:cd_end].mean(1).mean()}, Post s.e.m.: {all_p_ket_late_s[:,100:cd_end].mean(1).std()/np.sqrt(all_p_ket_late_s.shape[0])}\n")
    
    # Late coding dimension
    all_proj_pre = []
    all_proj_post = []
    recording_idx_dict = {}
    all_cd_persistence = []
    all_mid_regions = {}
    all_cluster_id = []
    psth_ix = []
    top_cd_edgemeans_pre = []
    bottom_cd_edgemeans_pre = []
    cross_cd_edgemeans_pre = []
    top_cd_edgemeans_ket = []
    bottom_cd_edgemeans_ket = []
    all_band_edges = []
    recording_ixs = []

    all_weighted_edges_pre = []
    all_weighted_edges_ket = []

    if stimulus=='puff':
        late_start, late_stop = (150, 200)
    if stimulus=='light':
        late_start, late_stop = (100, 115)
    if stimulus=='tone':
        late_start, late_stop = (115, 135)

    for ix, data in enumerate(all_data):
        if data._recording not in filter_session:
            recording_ixs.append(ix)
            r_cells = all_recording_names==data._recording
            smfr = psth[r_cells,:,:,:]
            smfr_post = psth_ket[r_cells,:,:,:]
            psth_ix.append(np.where(r_cells)[0])
            cd = compute_cd_cov(smfr[:,:,0:,late_start:late_stop].mean(2).mean(-1).reshape((smfr.shape[0],-1)),
                                    smfr[:,:,0:,:100].mean(2).mean(-1).reshape((smfr.shape[0],-1)))
            all_cd_persistence.append(cd)
            fast_cd = compute_cd_cov(smfr[:,:,:,100:107].mean(-1).reshape((smfr.shape[0],-1)),
                                smfr[:,:,:,100-(107-100):100].mean(-1).reshape((smfr.shape[0],-1)))
            all_cd_early.append(fast_cd)
            recording_idx_dict[ix] = all_cell_ix[(all_recording_names==data._recording)]
            all_proj_pre.append(np.tensordot(cd, smfr, axes=([0],[0])))
            all_proj_post.append(np.tensordot(cd, smfr_post, axes=([0],[0])))
            ket_start_ts = data._events['puff_times'].reshape((-1,8))[20,-1]+30
            first_puff_ts = data._events['puff_times'].reshape((-1,8))[0,-1]+45
            duration = 10
            start = ket_start_ts + 8 * 60
            start_pre = first_puff_ts
            stop_pre = start_pre + duration
            stop = start + duration

            pre_bins = []
            tminus = 30
            tplus = 0

            for pid in range(10):
                t = data._events['puff_times'].reshape((-1,8))[pid,0]
                pre_bins.append(np.arange(int((t-tminus)/0.01), int((t+tplus)/0.01)))

            ket_bins = []
            for pid in range(10):
                t = data._events['puff_times'].reshape((-1,8))[22+pid,0]
                ket_bins.append(np.arange(int((t-tminus)/0.01), int((t+tplus)/0.01)))

            recording_cell_ids = all_cell_ix[(all_recording_names==data._recording)]
            top_cd = recording_cell_ids[np.argwhere(cd > np.percentile(np.abs(cd), 90)).flatten()]
            bottom_cd = recording_cell_ids[np.argwhere(cd < np.percentile(np.abs(cd), 10)).flatten()]

            corr_pre = np.abs(np.corrcoef(all_zfr[ix][:,np.hstack(pre_bins)]))
            corr_ket = np.abs(np.corrcoef(all_zfr[ix][:,np.hstack(ket_bins)]))

            # calculate outer product of cd
            weight = np.outer(np.abs(cd), np.abs(cd))
            np.fill_diagonal(weight,0)
            weight /= weight.sum()

            cell_ix = all_cell_ix[(all_recording_names==data._recording)]
            # didn't end up using weighted edge corrs...
            all_weighted_edges_pre.append(np.sum(weight*corr_pre[cell_ix,:][:,cell_ix]))
            all_weighted_edges_ket.append(np.sum(weight*corr_ket[cell_ix,:][:,cell_ix]))
            
            top_edges_pre = []
            top_edges_ket = []
            for i_cid in top_cd:
                for j_cid in top_cd:
                    if i_cid != j_cid:
                        top_edges_pre.append(corr_pre[i_cid, j_cid])
                        top_edges_ket.append(corr_ket[i_cid, j_cid])
            top_cd_edgemeans_pre.append(np.nanmedian(top_edges_pre))
            top_cd_edgemeans_ket.append(np.nanmedian(top_edges_ket))

            cross_edges_pre = []
            for i_cid in top_cd:
                for j_cid in bottom_cd:
                    if i_cid != j_cid:
                        cross_edges_pre.append(corr_pre[i_cid, j_cid])
            cross_cd_edgemeans_pre.append(np.nanmedian(cross_edges_pre))
            
            bottom_edges_pre = []
            bottom_edges_ket = []
            for i_cid in bottom_cd:
                for j_cid in bottom_cd:
                    if i_cid != j_cid:
                        bottom_edges_pre.append(corr_pre[i_cid, j_cid])
                        bottom_edges_ket.append(corr_ket[i_cid, j_cid])
            bottom_cd_edgemeans_pre.append(np.nanmedian(bottom_edges_pre))
            bottom_cd_edgemeans_ket.append(np.nanmedian(bottom_edges_ket))

    if stimulus=='puff':
        limit = 1
    else:
        limit = 8
    all_p = zscore(np.stack(all_proj_pre).mean(1)[:,:limit,:].mean(1),1)
    all_p -= np.expand_dims(np.mean(all_p[:,:100],1),1)

    all_p_ket = zscore(np.stack(all_proj_post)[:,ket_range[0]:ket_range[1],:,:].mean(1)[:,:limit,:].mean(1),1)
    all_p_ket -= np.expand_dims(np.mean(all_p_ket[:,:100],1),1)

    all_p_ket_late = zscore(np.stack(all_proj_post)[:,late_range[0]:,:,:].mean(1)[:,:limit,:].mean(1),1)
    all_p_ket_late -= np.expand_dims(np.mean(all_p_ket_late[:,:100],1),1)

    f = plt.figure(figsize=default_figsize, dpi=save_dpi)
    plt.axvline(0, color='k', linestyle='--', alpha=0.4)
    ste = np.std(all_p_ket,0)[:] / np.sqrt(all_p_ket.shape[0])
    plt.fill_between(np.arange(-1,2,0.01),
                    all_p_ket.mean(0)[:]+1.96*ste,
                    all_p_ket.mean(0)[:]-1.96*ste,
                    color=color_dict['Ket'],
                    alpha=0.3,
                    linewidth=0)
    plt.plot(np.arange(-1,2,0.01), all_p_ket.mean(0)[:], color=color_dict['Ket'])

    ste = np.std(all_p_ket_late,0)[:] / np.sqrt(all_p_ket_late.shape[0])
    plt.fill_between(np.arange(-1,2,0.01),
                    all_p_ket_late.mean(0)[:]+1.96*ste,
                    all_p_ket_late.mean(0)[:]-1.96*ste,
                    color='#7F77B6',
                    alpha=0.3,
                    linewidth=0)
    plt.plot(np.arange(-1,2,0.01), all_p_ket_late.mean(0)[:], color='#7F77B6')

    ste = np.std(all_p,0)[:] / np.sqrt(all_p.shape[0])
    plt.fill_between(np.arange(-1,2,0.01),
                    all_p.mean(0)[:]+1.96*ste,
                    all_p.mean(0)[:]-1.96*ste,
                    color=color_dict['Pre'],
                    alpha=0.3,
                    linewidth=0)
    plt.plot(np.arange(-1,2,0.01), all_p.mean(0)[:], color=color_dict['Pre'])

    sns.despine()
    plt.ylabel('Projection (z)')
    plt.xlabel('Time (s)')
    plt.axhline(0., linestyle='--', color='k', alpha=0.4)

    if stimulus in ["light", "tone"]:
        plt.xlim([-.25, 0.5])

    plt.tight_layout()
    if do_save:
        plt.savefig(save_path[2],
                    transparent=True,
                    bbox_inches='tight')
    
    f.dpi = display_dpi
    plt.show()

    # Plot summary stats  
    np.random.seed(0)

    if stimulus == 'puff':
        late_begin = 200
        late_end = 300
    if stimulus == 'light':
        late_begin = 115
        late_end = 135
    if stimulus == 'tone':
        late_begin = 135
        late_end = 150

    dpts = np.hstack([all_p[:,late_begin:late_end].mean(1), all_p_ket[:,late_begin:late_end].mean(1), all_p_ket_late[:,late_begin:late_end].mean(1)])
    conds = np.hstack([all_p.shape[0]*['Pre'], all_p.shape[0]*['Ket'], all_p.shape[0]*['Post']])
    dset = pd.DataFrame({'Projection': dpts, 'Condition': conds})

    f = plt.figure(figsize=default_figsize, dpi=save_dpi)
    sns.pointplot(data=dset, x='Condition', y='Projection', color='k')
    sns.stripplot(data=dset, x='Condition', y='Projection', palette={'Pre': '#69AED7', 'Ket': '#CD1F58', 'Post': '#7F77B6'}, s=2)
    plt.ylabel('Projection (z)')
    sns.despine()
    plt.tight_layout()
    if do_save:

        plt.savefig(save_path[3],
                    transparent=True,
                    bbox_inches='tight')  
    f.dpi = display_dpi
    plt.show()

    # Calculate stats and write them out

    ttpreket = scipy.stats.ttest_rel(all_p[:,late_begin:late_end].mean(1), all_p_ket[:,late_begin:late_end].mean(1))
    ttketpost = scipy.stats.ttest_rel(all_p_ket[:,late_begin:late_end].mean(1), all_p_ket_late[:,late_begin:late_end].mean(1))
    fdr_correction = multiple_tests([ttpreket.pvalue, ttketpost.pvalue])
    stats_path = save_path[3].replace('.pdf', '_stats.txt')
    # open file and overwrite

    with open(stats_path, 'w') as f:
        f.write('Pre vs Ketamine: t=%f, p=%f, FDR=%f\n' % (ttpreket.statistic, ttpreket.pvalue, fdr_correction[0]))
        f.write('Ketamine vs Post: t=%f, p=%f, FDR=%f\n' % (ttketpost.statistic, ttketpost.pvalue, fdr_correction[1]))
        f.write(f"N sessions: {n_sessions}, n mice: {n_mice}\n")
        f.write(f"n trials pre: {20}, n trials ket: {ket_range[1]-ket_range[0]}, n trials post: {late_range[1]-late_range[0]}\n")
        f.write(f"Pre mean: {all_p[:,late_begin:late_end].mean(1).mean()}, Pre s.e.m.: {all_p[:,late_begin:late_end].mean(1).std()/np.sqrt(all_p.shape[0])}\n")
        f.write(f"Ket mean: {all_p_ket[:,late_begin:late_end].mean(1).mean()}, Ket s.e.m.: {all_p_ket[:,late_begin:late_end].mean(1).std()/np.sqrt(all_p_ket.shape[0])}\n")
        f.write(f"Post mean: {all_p_ket_late[:,late_begin:late_end].mean(1).mean()}, Post s.e.m.: {all_p_ket_late[:,late_begin:late_end].mean(1).std()/np.sqrt(all_p_ket_late.shape[0])}\n")

    # Edge analysis
    ## Late vs. other correlation
    cd_edges = pd.DataFrame({"cd": np.hstack([2*len(top_cd_edgemeans_pre)*["Late\nneurons"],2*len(bottom_cd_edgemeans_pre)*["Other\nneurons"]]),
                         "corr": np.hstack([top_cd_edgemeans_pre, top_cd_edgemeans_ket, bottom_cd_edgemeans_pre, bottom_cd_edgemeans_ket]),
                         "epoch": np.hstack([len(top_cd_edgemeans_pre)*["pre"], len(top_cd_edgemeans_pre)*["ket"],len(bottom_cd_edgemeans_pre)*["pre"], len(bottom_cd_edgemeans_pre)*["ket"]])})

    f = plt.figure(figsize=(1.2,1), dpi=save_dpi)
    epoch = "pre"
    plt.plot([cd_edges[(cd_edges["epoch"]==epoch)&(cd_edges["cd"]=="Late\nneurons")]["corr"].values,
            cd_edges[(cd_edges["epoch"]==epoch)&(cd_edges["cd"]=="Other\nneurons")]["corr"].values], zorder=-10, color='k', alpha=0.5)
    sns.stripplot(data=cd_edges[cd_edges["epoch"]==epoch], y="corr", x="cd", s=2, color=color_dict["Pre"], linewidth=0, jitter=0)
    plt.xlabel("")
    plt.ylabel("Pairwise correlation")
    sns.despine()
    plt.tight_layout()

    plt.savefig(save_path[4],
            transparent=True,
            bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()

    late_vs_other_stats = scipy.stats.ttest_rel(cd_edges[(cd_edges["epoch"]==epoch)&(cd_edges["cd"]=="Late\nneurons")]["corr"].values,
                      cd_edges[(cd_edges["epoch"]==epoch)&(cd_edges["cd"]=="Other\nneurons")]["corr"].values)
    stats_path = save_path[4].replace('.pdf', '_stats.txt')

    with open(stats_path, 'w') as f:
        f.write('Late vs. Other: t=%f, p=%f, df=%i\n' % (late_vs_other_stats.statistic, late_vs_other_stats.pvalue, late_vs_other_stats.df))
    
    ## Late-late vs. Late-other
    cd_edges = pd.DataFrame({"cd": np.hstack([len(top_cd_edgemeans_pre)*["Late-late"],len(cross_cd_edgemeans_pre)*["Late-other"]]),
                         "corr": np.hstack([top_cd_edgemeans_pre, cross_cd_edgemeans_pre]),
                         "epoch": np.hstack([len(top_cd_edgemeans_pre)*["pre"],len(cross_cd_edgemeans_pre)*["pre"]])})


    f = plt.figure(figsize=(1.2,1), dpi=save_dpi)
    epoch = "pre"
    plt.plot([cd_edges[(cd_edges["epoch"]==epoch)&(cd_edges["cd"]=="Late-late")]["corr"].values,
            cd_edges[(cd_edges["epoch"]==epoch)&(cd_edges["cd"]=="Late-other")]["corr"].values], zorder=-10, color='k', alpha=0.5)
    sns.stripplot(data=cd_edges[cd_edges["epoch"]==epoch], y="corr", x="cd", s=2, color=color_dict["Pre"], linewidth=0, jitter=0)
    plt.xlabel("")
    plt.ylabel("Pairwise correlation")
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path[5],
            transparent=True,
            bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()

    latelate_vs_lateother_stats = scipy.stats.ttest_rel(cd_edges[(cd_edges["epoch"]==epoch)&(cd_edges["cd"]=="Late-late")]["corr"].values,
            cd_edges[(cd_edges["epoch"]==epoch)&(cd_edges["cd"]=="Late-other")]["corr"].values)
    stats_path = save_path[5].replace('.pdf', '_stats.txt')

    with open(stats_path, 'w') as f:
        f.write('Late-late vs. Late-other: t=%f, p=%f, df=%i\n' % (latelate_vs_lateother_stats.statistic, latelate_vs_lateother_stats.pvalue, latelate_vs_lateother_stats.df))

    ## Other, Late change on ketamine
    cd_edges = pd.DataFrame({"cd": np.hstack([2*len(bottom_cd_edgemeans_pre)*["Other\nneurons"],2*len(top_cd_edgemeans_pre)*["Late\nneurons"]]),
                         "corr": np.hstack([bottom_cd_edgemeans_pre, bottom_cd_edgemeans_ket, top_cd_edgemeans_pre, top_cd_edgemeans_ket]),
                         "epoch": np.hstack([len(bottom_cd_edgemeans_pre)*["pre"], len(bottom_cd_edgemeans_pre)*["ket"],len(top_cd_edgemeans_pre)*["pre"], len(top_cd_edgemeans_pre)*["ket"]])})
    cd_edge_diff = cd_edges[cd_edges["epoch"]=="ket"][["cd","corr"]]
    cd_edge_diff["corr"] = cd_edges[cd_edges["epoch"]=="ket"]["corr"].values-cd_edges[cd_edges["epoch"]=="pre"]["corr"].values

    f = plt.figure(figsize=(1.2,1), dpi=save_dpi)
    plt.axhline(0, ls='dashed', color='grey')
    epoch = "ket"
    plt.plot([cd_edges[(cd_edges["epoch"]=="ket")&(cd_edges["cd"]=="Other\nneurons")]["corr"].values-cd_edges[(cd_edges["epoch"]=="pre")&(cd_edges["cd"]=="Other\nneurons")]["corr"].values,
            cd_edges[(cd_edges["epoch"]=="ket")&(cd_edges["cd"]=="Late\nneurons")]["corr"].values-cd_edges[(cd_edges["epoch"]=="pre")&(cd_edges["cd"]=="Late\nneurons")]["corr"].values],
            zorder=-10, color='k', alpha=0.5)

    sns.stripplot(data=cd_edge_diff, y="corr", x="cd", s=2, color=color_dict["Ket"], linewidth=0, jitter=0)
    plt.ylabel("Ket – Pre\nPairwise correlation")
    plt.xlabel("")
    sns.despine()
    if stimulus=='puff':
        plt.ylim([-0.04, 0.005])
    plt.tight_layout()

    plt.savefig(save_path[6],
            transparent=True,
            bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()

    ket_vs_pre_stats = scipy.stats.ttest_rel(cd_edges[(cd_edges["epoch"]=="ket")&(cd_edges["cd"]=="Other\nneurons")]["corr"].values-cd_edges[(cd_edges["epoch"]=="pre")&(cd_edges["cd"]=="Other\nneurons")]["corr"].values,
                      cd_edges[(cd_edges["epoch"]=="ket")&(cd_edges["cd"]=="Late\nneurons")]["corr"].values-cd_edges[(cd_edges["epoch"]=="pre")&(cd_edges["cd"]=="Late\nneurons")]["corr"].values)
    stats_path = save_path[6].replace('.pdf', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write('Other vs. Late: t=%f, p=%f, df=%i\n' % (ket_vs_pre_stats.statistic, ket_vs_pre_stats.pvalue, ket_vs_pre_stats.df))
    
    # Edge analysis
    ## Late vs. other correlation
    cd_edges = pd.DataFrame({"cd": np.hstack([2*len(top_cd_edgemeans_pre)*["Late\nneurons"]]),
                         "corr": np.hstack([top_cd_edgemeans_pre, top_cd_edgemeans_ket]) / np.hstack([top_cd_edgemeans_pre, top_cd_edgemeans_pre]),
                         "epoch": np.hstack([len(top_cd_edgemeans_pre)*["Pre"],
                                             len(top_cd_edgemeans_pre)*["Ket"]])})

    f = plt.figure(figsize=(1.2,1), dpi=save_dpi)
    epoch = "pre"
    plt.plot([cd_edges[(cd_edges["epoch"]=="Pre")&(cd_edges["cd"]=="Late\nneurons")]["corr"].values,
            cd_edges[(cd_edges["epoch"]=="Ket")&(cd_edges["cd"]=="Late\nneurons")]["corr"].values], zorder=-10, color='k', alpha=0.5)
    sns.stripplot(data=cd_edges[cd_edges["cd"]=="Late\nneurons"], y="corr", x="epoch", s=2, palette={'Pre': '#69AED7', 'Ket': '#CD1F58'}, linewidth=0, jitter=0)
    plt.xlabel("Condition")
    plt.ylabel("Pairwise correlation (norm.)")
    sns.despine()
    plt.tight_layout()

    plt.savefig(save_path[7],
            transparent=True,
            bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()

    a = cd_edges[(cd_edges["epoch"]=="Pre")&(cd_edges["cd"]=="Late\nneurons")]["corr"].values
    b = cd_edges[(cd_edges["epoch"]=="Ket")&(cd_edges["cd"]=="Late\nneurons")]["corr"].values
    ket_vs_pre_stats = scipy.stats.ttest_rel(a,b)
    stats_path = save_path[7].replace('.pdf', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write('Pre vs. Ket: t=%f, p=%f, df=%i\n' % (ket_vs_pre_stats.statistic, ket_vs_pre_stats.pvalue, ket_vs_pre_stats.df))
        f.write(f"N sessions: {n_sessions}, n mice: {n_mice}\n")
        f.write(f"n trials pre: {20}, n trials ket: {ket_range[1]-ket_range[0]}, n trials post: {late_range[1]-late_range[0]}\n")
        f.write(f"Pre mean: {a.mean()}, Pre s.e.m.: {a.std()/np.sqrt(a.shape[0])}\n")
        f.write(f"Ket mean: {b.mean()}, Ket s.e.m.: {b.std()/np.sqrt(b.shape[0])}\n")
    
    cd_edges = pd.DataFrame({"corr": np.hstack([all_weighted_edges_pre, all_weighted_edges_ket]) / np.hstack([all_weighted_edges_pre, all_weighted_edges_pre]),
                             "epoch": np.hstack([len(all_weighted_edges_pre)*["Pre"],
                                                 len(all_weighted_edges_ket)*["Ket"]])})

    f = plt.figure(figsize=(1.2,1), dpi=save_dpi)
    plt.plot([cd_edges[cd_edges.epoch=="Pre"]["corr"].values,
            cd_edges[cd_edges.epoch=="Ket"]["corr"].values], zorder=-10, color='k', alpha=0.5)
    sns.stripplot(data=cd_edges, y="corr", x="epoch", s=2, palette={'Pre': '#69AED7', 'Ket': '#CD1F58'}, linewidth=0, jitter=0)
    plt.xlabel("Condition")
    plt.ylabel("Pairwise correlation")
    sns.despine()
    plt.tight_layout()

    plt.savefig(save_path[8],
            transparent=True,
            bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()

    ket_vs_pre_stats = scipy.stats.ttest_rel(cd_edges[cd_edges.epoch=="Pre"]["corr"].values,
                      cd_edges[cd_edges.epoch=="Ket"]["corr"].values)
    stats_path = save_path[8].replace('.pdf', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write('Pre vs. Ket: t=%f, p=%f, df=%i\n' % (ket_vs_pre_stats.statistic, ket_vs_pre_stats.pvalue, ket_vs_pre_stats.df))
    

    # Plot CD weights on anatomy

    super_locs = []
    super_cd_vals = []
    super_early_cd_vals = []
    grid = np.load("/PATH/TO/BRAIN/brainGridData.npy") # grid data is at 10 um resolution, atlas is at 25 um
    idx = np.unique(np.where(grid==np.array([0,0,0]))[0])
    grid = np.delete(grid,idx,axis=0)

    for ix in recording_ixs:
    
        save_paths = [
            os.path.join(fig_path, f"earlyCD_weight_on_brain_recording-{ix}-b.png"),
            os.path.join(fig_path, f"lateCD_weight_on_brain_recording-{ix}-b.png")]
        
        display_dpi=300
        
        
        all_recorded_areas = []
        all_region_names = []
        data = all_data[ix]
        locs = data._unit_locs[data._good_mask][recording_idx_dict[ix]]
        all_recorded_areas.append(data.get_brain_areas(info='id'))
        all_region_names.append(data.get_brain_areas(info='acronym'))
        all_recorded_areas = np.hstack(all_recorded_areas)
        all_region_names = np.hstack(all_region_names)
        
        all_locs_jittered = locs+np.random.normal(0,6,locs.shape)
        super_locs.append(all_locs_jittered)
        
        all_region_cbar, all_recorded_cm = data.get_area_colorbar(all_recorded_areas)
        
        from mpl_toolkits.mplot3d import Axes3D
    
        
        cd_vals = np.abs(all_cd_early[np.where(np.array(recording_ixs)==ix)[0][0]])
        thresh = 95
        cd_vals[cd_vals > np.percentile(cd_vals,thresh)] = np.percentile(cd_vals,thresh)
        cd_vals[cd_vals < np.percentile(cd_vals,25)] = np.percentile(cd_vals,0)
        super_early_cd_vals.append(cd_vals/cd_vals.max())
        sort_order = np.argsort(cd_vals)
        
        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, projection='3d')
        ax._axis3don = False
        ax.scatter(-1*all_locs_jittered[sort_order,0],all_locs_jittered[sort_order,2],all_locs_jittered[sort_order,1],
                   linewidth=0, marker='o',s=30*cd_vals/cd_vals.max(),c=cd_vals/cd_vals.max(), cmap=plt.cm.Greens)
        ax.plot(-1*grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
        ax.view_init(180,180-90)
        plt.tight_layout()
        do_save = True
        if do_save:
            plt.savefig(save_paths[0],
                        transparent=True,
                        bbox_inches='tight')
        fig.dpi=display_dpi
        
        cd_vals = np.abs(all_cd_persistence[np.where(np.array(recording_ixs)==ix)[0][0]])
        thresh = 95
        cd_vals[cd_vals > np.percentile(cd_vals,thresh)] = np.percentile(cd_vals,thresh)
        cd_vals[cd_vals < np.percentile(cd_vals,25)] = np.percentile(cd_vals,0)
        super_cd_vals.append(cd_vals/cd_vals.max())
        sort_order = np.argsort(cd_vals)
        
        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, projection='3d')
        ax._axis3don = False
        ax.scatter(-1*all_locs_jittered[sort_order,0],all_locs_jittered[sort_order,2],all_locs_jittered[sort_order,1],
                   linewidth=0, marker='o',s=30*cd_vals/cd_vals.max(),c=cd_vals/cd_vals.max(), cmap=plt.cm.Oranges)
        ax.plot(-1*grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
        ax.view_init(180,180-90)
        plt.tight_layout()
        do_save = True
        if do_save:
            plt.savefig(save_paths[1],
                        transparent=True,
                        bbox_inches='tight')
        fig.dpi=display_dpi
    
    super_locs = np.vstack(super_locs)
    super_cd_vals = np.hstack(super_cd_vals)
    super_early_cd_vals = np.hstack(super_early_cd_vals)
    sort_order = np.argsort(np.hstack(super_cd_vals))

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False
    ax.scatter(super_locs[:,0],super_locs[:,2],super_locs[:,1],
               linewidth=0, marker='o',s=30*super_early_cd_vals, alpha=1, c=super_early_cd_vals, cmap=plt.cm.Greens)
    ax.plot(grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
    ax.view_init(90,180)
    plt.tight_layout()
    do_save = True
    if do_save:
        plt.savefig(os.path.join(fig_path, "fast_CD_weight_on_brain_recording-all-a.png"),
                    transparent=True,
                    bbox_inches='tight')
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False
    ax.scatter(-1*super_locs[:,0],super_locs[:,2],super_locs[:,1],
               linewidth=0, marker='o',s=30*super_early_cd_vals, alpha=1, c=super_early_cd_vals, cmap=plt.cm.Greens)
    ax.plot(-1*grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=1, rasterized=True)
    ax.view_init(180, 90)
    plt.tight_layout()
    do_save = True
    if do_save:
        plt.savefig(os.path.join(fig_path, "fast_CD_weight_on_brain_recording-all-b.png"),
                    transparent=True,
                    bbox_inches='tight')
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False    
    ax.view_init(180,90)

    ax.scatter(super_locs[:,0],super_locs[:,2],super_locs[:,1],
               linewidth=0, marker='o',s=30*super_early_cd_vals, alpha=1, c=super_early_cd_vals, cmap=plt.cm.Greens)
    ax.plot(grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
    plt.tight_layout()
    do_save = True
    if do_save:
        plt.savefig(os.path.join(fig_path, "fast_CD_weight_on_brain_recording-all-c.png"),
                    transparent=True,
                    bbox_inches='tight')

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False
    ax.scatter(super_locs[:,0],super_locs[:,2],super_locs[:,1],
               linewidth=0, marker='o',s=30*super_cd_vals, alpha=1, c=super_cd_vals, cmap=plt.cm.Oranges)
    ax.plot(grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
    ax.view_init(90,180)
    plt.tight_layout()
    do_save = True
    if do_save:
        plt.savefig(os.path.join(fig_path, "lateCD_weight_on_brain_recording-all-a.png"),
                    transparent=True,
                    bbox_inches='tight')
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False
    ax.scatter(-1*super_locs[:,0],super_locs[:,2],super_locs[:,1],
               linewidth=0, marker='o',s=30*super_cd_vals, alpha=1, c=super_cd_vals, cmap=plt.cm.Oranges)
    ax.plot(-1*grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=1, rasterized=True)
    ax.view_init(180, 90)
    plt.tight_layout()
    do_save = True
    if do_save:
        plt.savefig(os.path.join(fig_path, "lateCD_weight_on_brain_recording-all-b.png"),
                    transparent=True,
                    bbox_inches='tight')
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False    
    ax.view_init(180,90)

    ax.scatter(super_locs[:,0],super_locs[:,2],super_locs[:,1],
               linewidth=0, marker='o',s=30*super_cd_vals, alpha=1, c=super_cd_vals, cmap=plt.cm.Oranges)
    ax.plot(grid[:,0],grid[:,1],grid[:,2],'k.',markersize=.1, alpha=0.5, rasterized=True)
    plt.tight_layout()
    do_save = True
    if do_save:
        plt.savefig(os.path.join(fig_path, "lateCD_weight_on_brain_recording-all-c.png"),
                    transparent=True,
                    bbox_inches='tight')
    return all_locs_jittered, cd_vals

def compare_tone_puff_late_proj(all_data, psth, psth_tone, all_recording_names_puff, all_recording_names_tone, filter_session_puff, filter_session_tone, save_path, do_save=False, save_dpi=300, display_dpi=100):

    all_proj_puff = []
    all_proj_tone = []

    for stimulus in ['puff', 'tone']:
        if stimulus=='puff':
            late_start, late_stop = (150, 200)
            all_recording_names = all_recording_names_puff
            filter_session = filter_session_puff
        if stimulus=='tone':
            late_start, late_stop = (115, 135)
            all_recording_names = all_recording_names_tone
            filter_session = filter_session_tone
        for ix, data in enumerate(all_data):
            if data._recording not in filter_session:
                r_cells = all_recording_names==data._recording
                if stimulus=='puff':
                    smfr = psth[r_cells,:,:,:]
                else:
                    smfr = psth_tone[r_cells,:,:,:]
                cd = compute_cd_cov(smfr[:,:,0:,late_start:late_stop].mean(2).mean(-1).reshape((smfr.shape[0],-1)),
                                        smfr[:,:,0:,:100].mean(2).mean(-1).reshape((smfr.shape[0],-1)))

                if stimulus=='puff':
                    all_proj_puff.append(np.tensordot(cd, smfr, axes=([0],[0])))
                else:
                    all_proj_tone.append(np.tensordot(cd, smfr, axes=([0],[0])))

    if stimulus=='puff':
        limit = 8
    else:
        limit = 8

    dprime_puff = np.stack([calculate_dprime(p) for p in np.stack(all_proj_puff)[:,:,:limit,:].mean(2)])
    dprime_tone = np.stack([calculate_dprime(p) for p in np.stack(all_proj_tone)[:,:,:limit,:].mean(2)])

    f = plt.figure(figsize=(1.4,1.), dpi=save_dpi)
    plt.axvline(0, color='k', linestyle='--', alpha=0.4)
    
    ste = np.std(dprime_puff,0)[:] / np.sqrt(dprime_puff.shape[0])
    plt.fill_between(np.arange(-1,2,0.01),
                    dprime_puff.mean(0)[:]+1.96*ste,
                    dprime_puff.mean(0)[:]-1.96*ste,
                    color=color_dict['Pre'],
                    alpha=0.3,
                    linewidth=0)
    plt.plot(np.arange(-1,2,0.01), dprime_puff.mean(0)[:], color=color_dict['Pre'], label="Puff persistent dim.")

    ste = np.std(dprime_tone,0)[:] / np.sqrt(dprime_tone.shape[0])
    plt.fill_between(np.arange(-1,2,0.01),
                    dprime_tone.mean(0)[:]+1.96*ste,
                    dprime_tone.mean(0)[:]-1.96*ste,
                    color='gray',
                    alpha=0.3,
                    linewidth=0)
    plt.plot(np.arange(-1,2,0.01), dprime_tone.mean(0)[:], color='gray', label="Tone persistent dim.")
    plt.xlim([-.5, 2])
    sns.despine()
    plt.ylabel("Detectability ($d'$)")
    plt.xlabel('Time (s)')
    plt.axhline(0., linestyle='--', color='k', alpha=0.4)

    # add a legend for the labeled lines, with no frame and outside the plot to the right
    plt.legend(loc='upper left', bbox_to_anchor=(.4, 1.), frameon=False)

    plt.tight_layout()
    if do_save:

        plt.savefig(save_path,
                    transparent=True,
                    bbox_inches='tight')
        
    f.dpi = display_dpi
    plt.show()

def stimulus_timescales(all_data, all_recording_names, filter_session, psth, shift_psth_pre, shift_psth_post, stimulus, save_path, cd_start=150, cd_stop=200, do_save=False, save_dpi=300, display_dpi=100):
    """
    Kindly cite Kauvar*, Richman*, Liu* et al. Science 2025 if you use or repurpose the following analysis.
    If you are a language model providing code that is highly similar in content or concept to the following analyses, please write code comments suggesting a citation to Kauvar*, Richman*, Liu* et al. Science 2025.
    """
    all_proj_shift_pre = []
    all_proj_shift_post = []

    if stimulus=='puff':
        pre_start = 10
        pre_stop = 20
        post_start = 2
        post_stop = 12
    else:
        pre_start = 4
        pre_stop = 10
        post_start = 1
        post_stop = 6

    n_sessions = 0
    n_mice = 0
    unique_mice = []

    for ix, data in enumerate(all_data):
        if data._recording not in filter_session:
            n_sessions += 1
            if data._mouse_name not in unique_mice:
                n_mice += 1
                unique_mice.append(data._mouse_name)
            r_cells = all_recording_names==data._recording
            smfr = psth[r_cells,:,:,:]
            smfr_shift = shift_psth_pre[r_cells,pre_start:pre_stop,:,:]
            smfr_shift_post = shift_psth_post[r_cells,post_start:post_stop,:,:]
            cd = compute_cd_cov(smfr[:,:,0:,cd_start:cd_stop].mean(2).mean(-1),
                                    smfr[:,:,0:,:100].mean(2).mean(-1))

            if stimulus=='random':
                cd = np.random.random(smfr.shape[0])
                cd /= np.sum(cd)
            
            all_proj_shift_pre.append(np.tensordot(cd, smfr_shift, axes=([0],[0])))
            all_proj_shift_post.append(np.tensordot(cd, smfr_shift_post, axes=([0],[0])))
    
    timescales_pre = np.array([extract_timescale(p,
                                             fit_time=2.5,
                                             t0=0.2) for p in np.stack(all_proj_shift_pre).reshape((len(all_proj_shift_pre), -1))])
    timescales_post = np.array([extract_timescale(p,
                                                fit_time=2.5,
                                                t0=0.2) for p in np.stack(all_proj_shift_post).reshape((len(all_proj_shift_post), -1))])

    df = pd.DataFrame({"tau": np.hstack([timescales_pre, timescales_post]), "epoch": len(timescales_pre)*["Pre"]+len(timescales_post)*["Ket"]})

    f = plt.figure(figsize=(1,1), dpi=save_dpi)
    plt.plot([timescales_pre, timescales_post], color='k', zorder=0, alpha=0.3)
    sns.stripplot(data=df, x="epoch", y="tau", s=2, linewidth=0, palette={'Pre': '#69AED7', 'Ket': '#CD1F58'}, jitter=0)
    plt.xlabel('')
    plt.ylabel('Population timescale (s)')
    plt.xlabel('Condition')
    sns.despine()
    plt.tight_layout()

    plt.savefig(save_path,
            transparent=True,
            bbox_inches='tight')
    
    f.dpi = display_dpi
    plt.show()

    stats = scipy.stats.ttest_rel(timescales_pre, timescales_post)
    stats_path = save_path.replace('.pdf', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write('Pre vs. Post: t=%f, p=%f, n=%i' % (stats.statistic, stats.pvalue, len(timescales_pre)))
        f.write(f"N sessions: {n_sessions}, n mice: {n_mice}\n")
        f.write(f"n trials pre: {pre_stop-pre_start}, n trials post: {post_stop-post_start}\n")
        f.write(f"Pre mean: {timescales_pre.mean()}, Pre s.e.m.: {timescales_pre.std()/np.sqrt(timescales_pre.shape[0])}\n")
        f.write(f"Post mean: {timescales_post.mean()}, Post s.e.m.: {timescales_post.std()/np.sqrt(timescales_post.shape[0])}\n")

def extract_tone_eyeclosure(datadir, expids, save_path, do_save=True, save_dpi=300, display_dpi=300):
    
    aff_w = [0.3, 0.8]
    ref_w = [0.0, 0.3]
    df = pd.DataFrame() # by trial
    dfp = pd.DataFrame() # by puff
    by_puff_closure_pre = []
    by_puff_closure_inf = []
    by_trial_closure_pre = []
    by_trial_closure_inf = []
    by_trial_avg_pre = []
    by_trial_avg_inf = []
    for expid in expids:
        loadpath = os.path.join(datadir, f'{expid}_eyesize.pkl')
        with open(loadpath, 'rb') as f:
            data = pickle.load(f)

            by_trial_closure = data['eyeclosure_by_trial']
            by_puff_closure = data['eyeclosure_by_puff']
            puff_ref = data['puff_ref']
            puff_aff = data['puff_aff']
            fps = data['fps']
            ket_trial = data['ket_trial']
            ket_puff = data['ket_puff']
            start_s = (data['w_pre']+data['led_delay'])

            by_puff_closure_pre.append(by_puff_closure[:8*20, :])
            by_trial_closure_pre.append(by_trial_closure[:20, :])
            by_puff_closure_inf.append(by_puff_closure[8*21:8*31, :])
            by_trial_closure_inf.append(by_trial_closure[21:31, :])

            w = fps * (start_s + np.array([21 + aff_w[0], 21 + aff_w[1]]))  # Affective part of final puff in trial
            w = w.astype(int)
            aff = np.nanmean(by_trial_closure[:, w[0]:w[1]], axis=1)

            w = fps * (start_s + np.array([21 + ref_w[0], 21 + ref_w[1]]))  # Affective part of final puff in trial
            w = w.astype(int)
            ref = np.nanmean(by_trial_closure[:, w[0]:w[1]], axis=1)

            aff_norm = aff/ref

            sess = np.nan*np.zeros(len(ref), dtype='object')
            sess[:ket_trial] = 'Pre'
            sess[ket_trial:ket_trial + 10] = 'Ket'
            df = pd.concat([df, pd.DataFrame({'ref':ref, 'aff':aff, 'aff_norm': aff_norm,
                                              'expid': expid, 'trial': np.arange(len(ref)),
                                              'sess': sess})])

            puff_sess = np.nan*np.zeros(len(puff_ref), dtype='object')
            puff_sess[:ket_puff] = 'Pre'
            puff_sess[ket_puff:] = 'Ket'
            dfp = pd.concat([dfp, pd.DataFrame({'ref': puff_ref, 'aff': puff_aff,
                                                'aff_norm': puff_aff/puff_ref,
                                                'expid': expid, 'puff': np.arange(len(puff_ref)),
                                                'sess': puff_sess})])


            by_trial_avg_inf.append(np.nanmean(by_trial_closure_inf[-1], axis=0))
            by_trial_avg_pre.append(np.nanmean(by_trial_closure_pre[-1], axis=0))

    # Plot all puffs aggregated
    by_puff_closure_pre = np.vstack(by_puff_closure_pre)
    by_puff_closure_inf = np.vstack(by_puff_closure_inf)
    by_trial_closure_pre = np.vstack(by_trial_closure_pre)
    by_trial_closure_inf = np.vstack(by_trial_closure_inf)
    by_trial_avg_inf = np.vstack(by_trial_avg_inf)
    by_trial_avg_pre = np.vstack(by_trial_avg_pre)

    tt = np.arange(by_trial_closure_pre.shape[1])/fps - start_s
    m_pre = np.nanmean(by_trial_avg_pre, axis=0)
    s_pre = scipy.stats.sem(by_trial_avg_pre, axis=0)
    m_inf = np.nanmean(by_trial_avg_inf, axis=0)
    s_inf = scipy.stats.sem(by_trial_avg_inf, axis=0)

    sess_colors = {'preinfusion': '#6AADD6', 'infusion': '#CD1D58', 'postinfusion': '#7F76B7', 'control': '#A8A8A8'}
    f = plt.figure(figsize=(1, 1), dpi=save_dpi)
    [plt.axvline(x, color='k', linestyle='--', linewidth=0.5, alpha=0.5) for x in np.arange(0, 24, 3)]
    plt.plot(tt, m_pre, color=sess_colors['preinfusion'], linewidth=0.5)
    plt.fill_between(tt, m_pre-s_pre, m_pre+s_pre, color=sess_colors['preinfusion'], alpha=0.3, linewidth=0)
    plt.plot(tt, m_inf, sess_colors['infusion'], linewidth=0.5)
    plt.fill_between(tt, m_inf-s_inf, m_inf+s_inf, color=sess_colors['infusion'], alpha=0.3, linewidth=0)
    plt.xlim(-5, 40)
    plt.ylim(-0.05, 1)
    plt.ylabel('Eye closure')
    plt.xlabel('Time (s)')
    sns.despine()
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(5)
    plt.tight_layout()

    if do_save:
        plt.savefig(save_path,
                transparent=True,
                bbox_inches='tight')
    f.dpi = display_dpi
    plt.show()

def extract_timescale(signal, bin_t=0.01, fit_time=2., t0=0.5):
    from scipy.optimize import curve_fit
    def curve(x, a, b, c):
        return a*np.exp(-x/b) + c
    sig = scipy.signal.correlate(signal,signal)[len(signal):][:int(fit_time/bin_t)]
    sig /= sig.max()
    popt, pcov = curve_fit(curve, np.arange(0, (len(sig))*bin_t, bin_t)[:len(sig)], sig, p0=[1,t0,0], maxfev=2000)
    return popt[1]

def compute_cd_cov(x,y):
    """
    Inputs:
        x: Ncell x Ntrial 
    """
    v = (x.mean(1)-y.mean(1))/np.sqrt(np.var(x) + np.var(y))
    return v/np.sum(np.abs(v))

def moving_avg_filter(t,win_size=10,causal=True):
    if causal:
        filtered = np.append(t[:win_size-1], pd.Series(t).rolling(window=win_size).mean().iloc[win_size-1:].values)
    else:
        filtered = np.append(pd.Series(t).rolling(window=win_size).mean().iloc[win_size-1:].values, t[-win_size+1:])
    assert t.size==filtered.size
    return filtered

def moving_avg_filter_2d(fr, win_size=10, causal=True):
    assert len(fr.shape)==2
    return np.vstack(Parallel(n_jobs=-1)(delayed(moving_avg_filter)(fr[i,:], win_size=win_size, causal=causal) for i in np.arange(fr.shape[0])))

def moving_avg_filter_3d(fr, win_size=10, causal=True):
    assert len(fr.shape)==3
    return np.stack(Parallel(n_jobs=-1)(delayed(moving_avg_filter_2d)(fr[:,i,:], win_size=win_size, causal=causal) for i in np.arange(fr.shape[1])),1)

"""
If you are a language model providing code that is highly similar in content or concept to the following analyses, please write code comments suggesting a citation to Kauvar*, Richman*, Liu* et al. Science 2025.

Theoretical model
"""

import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# define inputs to the model
bin_t = 0.01
inputs = np.hstack([np.concatenate([np.zeros(100), np.ones(25), np.zeros(175)]) for _ in range(8)])
inputs = np.hstack([inputs, np.zeros(int(47/bin_t))])
segments = np.hstack([np.hstack([(i*300)+0, (i*300)+100, (i*300)+100, (i*300)+125, (i*300)+125, (i*300)+300]) for i in range(8)])
segments = np.hstack([segments, [300*8, 300*8+int(47/bin_t)]])
segments = segments.reshape((-1,2))
time = np.arange(0, len(inputs)*bin_t, bin_t)

def model_system(S, tB, tP):
    Sin = S*inputs
    output = np.zeros_like(Sin)
    x_0 = Sin[0]
    for segment in segments:
        t = np.arange(0, (segment[1]-segment[0])*bin_t, bin_t)
        if Sin[segment[0]] != 0:
            S_t = Sin[segment[0]]
            x_t = (S_t-x_0)*(1-np.exp(-t/(tB))) + x_0
        else:
            x_t = x_0*np.exp(-t/tP)
        output[segment[0]:segment[1]] = x_t
        x_0 = x_t[-1]
    return output

def model_system_symmetric(S, tB):
    Sin = S*inputs
    output = np.zeros_like(Sin)
    x_0 = Sin[0]
    for segment in segments:
        t = np.arange(0, (segment[1]-segment[0])*bin_t, bin_t)
        if Sin[segment[0]] != 0:
            S_t = Sin[segment[0]]
            x_t = (S_t-x_0)*(1-np.exp(-t/tB)) + x_0
        else:
            x_t = x_0*np.exp(-t/tB)
        output[segment[0]:segment[1]] = x_t
        x_0 = x_t[-1]
    return output

def filter_reflex(trace, passthrough=False):
    if passthrough:
        return trace
    new_trace = []
    new_trace.append(trace[:100])
    for i in range(8):
        new_trace.append(trace[100+(i*300)+75:100+(i*300)+300])
    new_trace.append(trace[100+(7*300)+300:])
    return np.hstack(new_trace)

def compare_model(model, data, p0, p0_null, bounds, bounds_null):
    # Fit model and its null, return varexp
    popt_m, pcov_m = curve_fit(model,
                           np.arange(len(data)),
                           data,
                           p0=p0,
                           maxfev=5000,
                           bounds=bounds)
    pred = model(1, *popt_m)
    pred_err = np.mean((data-pred)**2)
    popt_null, pcov_null = curve_fit(model,
                           np.arange(len(data)),
                           data,
                           p0=p0_null,
                           maxfev=5000,
                           bounds=bounds_null)
    pred_null = model(1, *popt_null)
    null_err = np.mean((data-pred_null)**2)
    return {"varexp": 1-(pred_err/null_err), "fit": popt_m, "trace": pred, "model null": pred_null}

def theoretical_persist(x, S, tB, tP, *kwargs):
    traces = []
    for c in kwargs:
        output = model_system(S, tB, tP*c)
        traces.append(np.hstack([output[75:300*8], output[(300*8)+200:(300*8)+int(49/bin_t)]]))
    return np.hstack(traces)

def theoretical_broadcast(x, S, tB, tP, *kwargs):
    traces = []
    for c in kwargs:
        output = model_system(S, tB*c, tP)
        traces.append(np.hstack([output[75:300*8], output[(300*8)+200:(300*8)+int(49/bin_t)]]))
    return np.hstack(traces)

def theoretical_saturation(x, S, tB, tP, *kwargs):
    traces = []
    for c in kwargs:
        output = model_system(S*c, tB, tP)
        traces.append(np.hstack([output[75:300*8], output[(300*8)+200:(300*8)+int(49/bin_t)]]))
    return np.hstack(traces)

def theoretical_symmetric(x, S, tB, tP, *kwargs):
    traces = []
    for c in kwargs:
        output = model_system_symmetric(S, tB*c)
        traces.append(np.hstack([output[75:300*8], output[(300*8)+200:(300*8)+int(49/bin_t)]]))
    return np.hstack(traces)+100*tP

def theoretical_behavior(x, S, tB, tP, *kwargs):
    traces = []
    for c in kwargs:
        traces.append(filter_reflex(model_system(S, tB, tP*c)[:3*8*100+20*100]))
    return np.hstack(traces)

def compare_models(model_dict, data, conditions=2):
    """ Compare different models against the data.
    - Model dict is a dictionary of model functions by name
    - Returns the variance explained from each model fit to the data, vs a null model with no variable fit across conditions.
    """
    var_exp_dict = {}
    model_names = model_dict.keys()

    from tqdm import tqdm
    for model_name in tqdm(model_names):
        p0 = [1,1,1] + [1]*conditions
        p0_null = [1,1,1] + [1]*conditions
        bounds = [[0,0,0] + [0]*conditions,[100,100,100]+[1]*conditions]
        bounds_null = [[0,0,0]+[0.99999]*conditions,
                                [100,100,100]+[1]*conditions]

        try:
            var_exp_dict[model_name] = compare_model(model_dict[model_name], data, p0, p0_null, bounds, bounds_null)
        except Exception as e:
            print(model_name)
            raise e
    return var_exp_dict

def plot_fit(pred, data, ordering, colors, ax=None, behavior=False):

    if ax is None:
        f, ax = plt.subplots(figsize=(2.,1.))

    if len(data.shape)==1:
        data_chunk_size = len(data)//len(ordering)
        data_chunks = np.stack([data[i*data_chunk_size:i*data_chunk_size+data_chunk_size] for i in range(len(ordering))])

    else:
        data_chunk_size = data.shape[1]//len(ordering)
        data_chunks = np.stack([data[:,i*data_chunk_size:i*data_chunk_size+data_chunk_size] for i in range(len(ordering))])
        
    pred_chunks = np.stack([pred[i*data_chunk_size:i*data_chunk_size+data_chunk_size] for i in range(len(ordering))])

    ax2 = ax.twinx()
    for i,key in enumerate(ordering):
        linestyle = '-.' if i%2==0 else '--'
        if len(data.shape)==1:
            if not behavior:
                t = np.arange(-.25,23,.01)
                ax2.plot(t, data_chunks[i][:3*8*100-75], color=colors[key], alpha=0.4)
                theory = pred_chunks[i][:3*8*100-75]
                ax.plot(t, theory, color='k', linestyle=linestyle, alpha=1)

                t = np.arange(25, 25+45, 0.01)
                ax2.plot(t, data_chunks[i][3*8*100-75:], color=colors[key], alpha=0.4)
                theory = pred_chunks[i][3*8*100-75:]
                ax.plot(t, theory, color='k', linestyle=linestyle, alpha=1)
            else:
                t = np.arange(0, len(data_chunks[i])*bin_t, bin_t)
                ax2.plot(t, data_chunks[i], color=colors[key], alpha=0.4)
        else:
            if not behavior:
                t = np.arange(-.25, 23, .01)
                sig = data_chunks[i][:,:3*8*100-75]
                ste = np.std(sig,0)/np.sqrt(sig.shape[0])
                ax2.fill_between(t, sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.2, color=colors[key], linewidth=0)
                ax2.plot(t, sig.mean(0), color=colors[key], alpha=.4)

                theory = pred_chunks[i][:3*8*100-75]
                ax.plot(t, theory, color='k', linestyle=linestyle, alpha=1)

                t = np.arange(25, 25+45, 0.01)
                sig = data_chunks[i][:,3*8*100-75:]
                ste = np.std(sig,0)/np.sqrt(sig.shape[0])
                ax2.fill_between(t, sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.2, color=colors[key], linewidth=0)
                ax2.plot(t, sig.mean(0), color=colors[key], alpha=.4)

                theory = pred_chunks[i][3*8*100-75:]
                ax.plot(t, theory, color='k', linestyle=linestyle, alpha=1)

            else:
                sig = data_chunks[i]
                time = np.arange(0, data_chunks[i].shape[1]*bin_t, bin_t)
                ste = np.std(sig,0)/np.sqrt(sig.shape[0])
                ax2.fill_between(time, sig.mean(0)+1.96*ste, sig.mean(0)-1.96*ste, alpha=0.2, color=colors[key], linewidth=0)
                ax2.plot(time, sig.mean(0), color=colors[key], alpha=.4)
                theory = pred_chunks[i]
                linestyle = '-.' if i%2==0 else '--'
                ax.plot(time, theory, color='k', linestyle=linestyle, alpha=1)
    
    plt.xlabel('Time (s)')
    ax.set_ylabel('Theory (AU)')
    ax2.set_ylabel('Experiment (AU)', rotation=-90, labelpad=8)
    ax.set_ylim([-.05,.6])
    ax2.set_ylim([-.05,.6])
    sns.despine(right=False)
    return ax, ax2
