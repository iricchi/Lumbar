#!/usr/bin/python

"""
This script is used to generate the static functional connectome analyses voxel-wise.

This is done on all subjects.

To change some outputs look for "INPUT HERE" comments.

Author: Ilaria Ricchi
"""
#from numba import jit
import numpy as np
import time
import os
import sys
from glob import glob
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from nibabel import load, save, Nifti1Image

import pandas as pd
import matplotlib
import matplotlib.cm as cm

import subprocess
#from joblib import Parallel, delayed

from scipy.signal import butter, filtfilt, cheby2, lfilter
import pandas as pd
import plotly.express as px
from statsmodels.tsa import stattools
from scipy.stats import norm

from utils import *

global N_JOBS, DATA_PARENTPATH, FUNC

N_JOBS=10
DATA_PARENTPATH='/media/miplab-nas2/Data2/SpinalCord/Spinal_fMRI_Lumbar'
FUNC='func'   # name of the folder where functional data is present

### TO DO :
# - CORRECT THE 3D NII IMAGE
# - SLICES CENTER

## Functions for sFC per slice
def sFC_perslice(seed,sub=''):
    if len(sub) == 0:
        subj_paths =  glob(os.path.join(DATA_PARENTPATH,'LU_*',FUNC))  # GB has been excluded
        # subj_paths = [sub for sub in subj_paths if sub.split('/')[-2] not in ['LU_GB']]
    else:
        subj_paths = [os.path.join(DATA_PARENTPATH,sub,FUNC)]

    print("### Info: Starting static FC per slice...") 

    start = time.time()

    for sub in subj_paths:
        ## INPUT HERE Wn if u want to change window of filtering
        _sFC_perslice_persub(sub,seed)

    print("### Info: static FC per slice  done in %.3f s" %(time.time() - start ))

def _sFC_perslice_persub(sps, seed, Wn=[0.01,0.13]):

    print("### Info: subject path: ", sps)
    sub = sps.split('/')[-2]
    # Load data
    sub_path_func, sub_path_anat, mask_path, seed_mask = _load_images_persub(sps)
    img = nib.load(sub_path_func)
    img_anat = nib.load(sub_path_anat)
    mask = nib.load(mask_path)
    seeds = nib.load(seed_mask)

    img_filt = temporal_filter2_vol(img,Wn)  # for analysis
    mask_array = mask.get_fdata()   # 3D
    seeds_array = seeds.get_fdata()
    img_data = img_anat.get_fdata() # for plotting
    f_img_data = img_filt.get_fdata()
    # Given a specific input seed (VL, VR, DL, DR)
    seeds_xyz = _load_seeds(seeds_array,seed)
    masked_img3d = cv2.bitwise_and(mask_array,f_img_data[:,:,:,1])
    print("### Info: data loaded.")

    # generate png figures maps overlayed slice per slice (z)
    # iterate over the slices
    dict_corrs_all = dict() 
     
    for xyz in seeds_xyz:
        print("xyz ",xyz)
        
        masked_img = cv2.bitwise_and(mask_array[:,:,xyz[-1]],f_img_data[:,:,xyz[-1],1])
        mask_x, mask_y = np.where(masked_img!=0)
        assert xyz[0] in mask_x
        assert xyz[1] in mask_y
        
        X = np.zeros((len(mask_x), f_img_data.shape[-1]))
        i = 0
        list_xy = []
        # iterate only over the pixels that belong to the mask
        for x, y in zip(mask_x, mask_y):
            X[i,:] = f_img_data[x,y,xyz[-1],:]   # take all timepoints
            list_xy.append((x,y))
            if x == xyz[0] and y == xyz[1]:
                print("seed index:", i)
                seed_ind = i
            
            i+=1

        # compute corr with respect to the seed
        corr_vals = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            R_corr = np.corrcoef(X[seed_ind,:],X[i,:])
            corr_vals[i] = R_corr[0,1]

        # create dictionary to easy access info
        dict_corrs = dict(zip(list_xy, corr_vals))
        dict_corrs_all[xyz[-1]] = dict_corrs
        print("Plotting z %s" %str(xyz[-1]))
        # create image per slice
        _plot_simplemap_overlayed(img_data, xyz, dict_corrs, [sub, seed])
    
    print("Generate 3D plot ...")
    # generate nii in 3D 
    img_corr, outpath = _generate_3d_heatmaps(masked_img3d,seeds_xyz,dict_corrs_all, [sub, seed]) 
    out = Nifti1Image(img_corr, header=img.header, affine=img.affine)
    save(out, outpath) 

    print("Done.")
    return

def _load_images_persub(sub_path):

    sub_path_func = os.path.join(sub_path, 'mfmri_denoised.nii.gz')
    sub_path_anat = os.path.join(sub_path,'mfmri_mean.nii.gz')
    mask_path = os.path.join(sub_path,'Segmentation','mask_sco.nii.gz')
    seed_mask = os.path.join(sub_path,'mfmri_mean_seeds.nii.gz')

    return sub_path_func, sub_path_anat, mask_path, seed_mask

def _load_seeds(seeds_array, seed):
    
    seeds_xyz = 0
    if seed == 'VR':
        seeds_xyz = np.array(np.where(seeds_array==1)).T    
    elif seed == 'VL':
        seeds_xyz = np.array(np.where(seeds_array==2)).T
    elif seed == 'DR':
        seeds_xyz = np.array(np.where(seeds_array==3)).T    
    elif seed == 'DL':
        seeds_xyz = np.array(np.where(seeds_array==4)).T

    return seeds_xyz

def _plot_simplemap_overlayed(img_data, seed_xyz, dict_corrs, sub_atype):

    sub = sub_atype[0]
    atype = sub_atype[1]

    slice_xy_mri = img_data[:, :, seed_xyz[-1]]
    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot(111)
    ax.imshow(slice_xy_mri.T, cmap='gray', origin='lower')
    ax.set_xlim([60,95])
    ax.set_ylim([10,40])

    # highlight seed
    rect = patches.Rectangle((seed_xyz[0]-0.5, seed_xyz[1]-0.5), 1, 1, linewidth=7, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    corr_vals = np.array(list(dict_corrs.values()))
    # generate heatmap for pixels
    minc = min(corr_vals)
    maxc = max(corr_vals)
    norm = matplotlib.colors.Normalize(vmin=minc, vmax=maxc, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    for vox in dict_corrs.keys():
        v = dict_corrs[vox]
        c = mapper.to_rgba(v) # color
        rect = patches.Rectangle((vox[0]-0.5, vox[1]-0.5), 1, 1, linewidth=2, edgecolor='none', facecolor=c, alpha=0.6)
        ax.add_patch(rect)

    fig.colorbar(mapper)
    fig.savefig(os.path.join(DATA_PARENTPATH, 'sFC_results',sub,atype,'heatmap_ps_z_%s') %str(seed_xyz[-1]))
    plt.close(fig)

def _generate_3d_heatmaps(masked_img,seed_xyz,dict_corrs_all, sub_atype, vertical=False):

    sub = sub_atype[0]
    atype = sub_atype[1]
    # create image
    img_corr = -10*np.ones(masked_img.shape)

    mask_x, mask_y, mask_z = np.where(masked_img!=0)

    if vertical:
        print("### Info: this could take a while ...")
        print("Vertical analysis: generating 3D correlations, fixing z...")
        for xyz in dict_corrs_all.keys():
            img_corr[xyz[0], xyz[1], xyz[-1]] = dict_corrs_all[xyz]

        seed_xyz_str = (' '.join(map(str,seed_xyz))).replace(' ','-')
        outfilepath = os.path.join(DATA_PARENTPATH, 'sFC_results',sub,atype,'heatmaps_ps_vertically_seed%s.nii.gz') % seed_xyz_str
    else:
        for x, y, z in zip(mask_x, mask_y, mask_z):
            # take z location to 
            if z in seed_xyz[:,2]:
                dict_corrs = dict_corrs_all[z]
                for xy in dict_corrs.keys():
                    # take the slice correlations
                    img_corr[xy[0], xy[1], z] = dict_corrs[xy]

        outfilepath = os.path.join(DATA_PARENTPATH, 'sFC_results',sub,atype,'heatmaps_ps_allz.nii.gz')

    print("### Info: 3D output generated.")
    
    return img_corr, outfilepath

## Functions for sFC vertically 
def sFC_vertically(seed,sub=''):


    if len(sub) == 0:
        subj_paths =  glob(os.path.join(DATA_PARENTPATH,'LU_*',FUNC))
        subj_paths = [sub for sub in subj_paths if sub.split('/')[-2] not in ['LU_GB','LU_VG']]
    else:
        subj_paths = [os.path.join(DATA_PARENTPATH,sub,FUNC)]

    print("### Info: Starting static FC vertically...") 

    start = time.time()

    ### INPUT HERE
    ## if you want to change Wn for filter: declare Wn and add it as input down here
    for sps in subj_paths:
        _sFC_vertical_persub(sps,seed)

    print("### Info: static FC vertically  done in %.3f s" %(time.time() - start ))

def _sFC_vertical_persub(sps, seed, Wn=[0.01,0.13]):

    print("### Info: subject path: ", sps)
    sub = sps.split('/')[-2]
    # select from the location of the seed a random seed:
    # fixing seed just to generate the same seed and not create too many images
    # for testing the code, but change seed to generate new images or comment
    rng = np.random.default_rng(56974)
    rng.random(42)
    
    # Load data
    sub_path_func, sub_path_anat, mask_path, seed_mask = _load_images_persub(sps)
    img = nib.load(sub_path_func)
    img_anat = nib.load(sub_path_anat)
    mask = nib.load(mask_path)
    seeds = nib.load(seed_mask)

    img_filt = temporal_filter2_vol(img,Wn)  # for analysis
    mask_array = mask.get_fdata()
    seeds_array = seeds.get_fdata()
    img_data = img_anat.get_fdata() # for plotting
    f_img_data = img_filt.get_fdata()
    # Given a specific input seed (VL, VR, DL, DR)
    seeds_xyz = _load_seeds(seeds_array,seed)
    selected_seed = seeds_xyz[np.random.randint(seeds_xyz.shape[0])]
    print("### Info: data loaded.")
    
    masked_img3d = cv2.bitwise_and(mask_array,f_img_data[:,:,:,1])
    mask_x, mask_y, mask_z = np.where(masked_img3d!=0)
    
    i = 0
    X = np.zeros((len(mask_x), f_img_data.shape[-1]))
    list_xyz = []
    for x,y,z in zip(mask_x,mask_y,mask_z):
        X[i,:] = f_img_data[x, y, z, :]
        list_xyz.append((x,y,z))
        if x==selected_seed[0] and y==selected_seed[1] and z==selected_seed[2]:
            print("seed index:", i)
            seed_ind = i
        i+=1

    # compute corr with respect to the seed
    corr_vals = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        R_corr = np.corrcoef(X[seed_ind,:],X[i,:])
        corr_vals[i] = R_corr[0,1]

    # create dictionary to easy access info
    dict_corrs = dict(zip(list_xyz, corr_vals))
    heatmap3d, outpath = _generate_3d_heatmaps(masked_img3d,selected_seed,dict_corrs, [sub, seed], True)
    out = Nifti1Image(heatmap3d, header=img.header, affine=img.affine)
    save(out, outpath) 

# seed-to-seed functional connectivity matrices
def seed2seed_allsubs(Wn=[0.01,0.13]):
    subj_paths =  glob(os.path.join(DATA_PARENTPATH,'LU_*',FUNC))
    subj_paths = [sub for sub in subj_paths if sub.split('/')[-2] not in ['LU_GB','LU_VG','LU_NG']]

    # recheck VG mask not matching
    # MP removed because Dorsal left are missing

    zscores_ave_all = np.zeros((4,4))
    corrs_ave_all = np.zeros((4,4))
    zscores_all_persubj = []
    zscores_all = []
    for sps in subj_paths:
        print("### Info: subject path: ", sps)
        sub = sps.split('/')[-2]
        # Load data
        print("Loading data from subject %s" % sub)
        sub_path_func, sub_path_anat, mask_path, seed_mask = _load_images_persub(sps)
        img = nib.load(sub_path_func)
        img_anat = nib.load(sub_path_anat)
        mask = nib.load(mask_path)
        seeds = nib.load(seed_mask)

        ## INPUT HERE Wn if you want to change filter window
        img_filt = temporal_filter2_vol(img,Wn)  # for analysis
        # img_filt = online_filter(img)  # online method similar to AFNI
        mask_array = mask.get_fdata()
        seeds_array = seeds.get_fdata()
        img_data = img_anat.get_fdata() # for plotting
        f_img_data = img_filt.get_fdata()
        
        # Take all seeds
        vr_xyz = np.array(np.where(seeds_array==1)).T
        vl_xyz = np.array(np.where(seeds_array==2)).T
        dr_xyz = np.array(np.where(seeds_array==3)).T
        dl_xyz = np.array(np.where(seeds_array==4)).T

        # compute average among the 4 regions
        vr = ts_to_average(vr_xyz,f_img_data)
        vl = ts_to_average(vl_xyz,f_img_data)
        dr = ts_to_average(dr_xyz,f_img_data)
        dl = ts_to_average(dl_xyz,f_img_data)

        Xave = np.vstack([vr,vl,dr,dl])
        corr_ave = np.corrcoef(Xave)
        corrs_ave_all += corr_ave
        # rhos = np.zeros(Xave.shape[0])
        # N_volumes = Xave.shape[1]
        # for i in range(Xave.shape[0]):
        #     rhos[i] = stattools.acf(Xave[i,:],nlags=1,fft=True)[1]  # first order
        # print(rhos)
        # dof_voxelwise = N_volumes * (1-rhos**2)/(1+rhos**2) -3
        zscores_ave = np.arctanh(corr_ave) # * np.sqrt(dof_voxelwise)

        zscores_all.extend(zscores_ave[np.triu_indices(zscores_ave.shape[0],1)])
        zscores_all_persubj.append(zscores_ave[np.triu_indices(zscores_ave.shape[0],1)])

        zscores_ave_all += zscores_ave
        print(zscores_ave_all)
        # plot single subject correlations
        _plot_FCmatrix_seed2seed(corr_ave, sub, 'corr_mat')
        # plot single subject zscores
        _plot_FCmatrix_seed2seed(zscores_ave, sub, 'z-scores_mat')

        all_seeds_xyz = np.vstack([vr_xyz,vl_xyz,dr_xyz,dl_xyz])
        print("Generating seed-to-seed connectivity matrix...")
        # generate seed-to-seed connectivity matrix per subject
        X = np.zeros((all_seeds_xyz.shape[0], f_img_data.shape[-1]))
        i = 0
        all_seeds_xyz_list = []
        zs = []
        for xyz in all_seeds_xyz:
            all_seeds_xyz_list.append(tuple(xyz))
            X[i, :]  = f_img_data[xyz[0], xyz[1], xyz[2], :]
            # all time points
            i+=1
            zs.append(str(xyz[2]))

        mat_corr_seeds = np.corrcoef(X)
        ### INPUT HERE
        # Change here flag if you want to generate not sorted matrix
        sort = True
        mat_corr = _plot_FCmatrix(mat_corr_seeds,[zs,vr.shape[0]],sub,sorted=sort)

    # normalize by the number of subject
    corrs_ave_all = corrs_ave_all/len(subj_paths)
    
    # normalize and correct by degree of freedom
    zscores_ave_all = zscores_ave_all/len(subj_paths) 

    # test significance:
    rho = np.arctanh(0.6)
    sr = 1/(np.sqrt(len(subj_paths) -3))
    pvals = 2*(1-norm.pdf(zscores_ave_all,rho,sr))

    # dof_subjects = len(subj_paths) - 3
    # zscores_ave_all = zscores_ave_all * np.sqrt(dof_subjects)
    
    # plot all_sub 4 regions zscores
    _plot_FCmatrix_seed2seed(zscores_ave_all, 'all', 'averaged_zscores')
    _plot_FCmatrix_seed2seed(pvals, 'all', 'averaged_pvals')
    _plot_FCmatrix_seed2seed(corrs_ave_all, 'all', 'averaged_corrs')
    # boxplots of the zscores distributions across subjects
    zscores_all_persubj = np.array(zscores_all_persubj).T
    _plot_boxplot_distribution(zscores_all_persubj, 'zscores_distributions_asubs')
    # 1 boxplot all subjects 
    _plot_boxplot_distribution(zscores_all, 'zscores_distributions_all')

def _plot_FCmatrix_seed2seed(mat_corr, sub, name):
    
    # for now we use 4x4 only GM
    assert mat_corr.shape[0] == 4

    fig = plt.figure(dpi=100)
    if name == 'averaged_corrs':
        plt.imshow(mat_corr,vmax=0.3)
    else:
        plt.imshow(mat_corr)
    plt.colorbar()
    plt.xticks([0,1,2,3],['VR','VL', 'DR', 'DL'], fontsize=15)
    plt.yticks([0,1,2,3],['VR','VL', 'DR', 'DL'], fontsize=15)
    
    if sub == 'all':
        fig.savefig(os.path.join(DATA_PARENTPATH, 'sFC_results', name))
    else:
        fig.savefig(os.path.join(DATA_PARENTPATH, 'sFC_results',sub, name))

    plt.close(fig)


def _plot_boxplot_distribution(values,name):
    df = pd.DataFrame(values)
    fig =  px.violin(df, box=True, points='all',title="z-scores distribution")
    fig.write_image(os.path.join(DATA_PARENTPATH,'sFC_results',"%s.png" %name))
    

def _plot_FCmatrix(mat_corr_seeds, ticks_info, sub, sorted=True):

    ## generate ticks
    tot_len = len(mat_corr_seeds)
    num_seeds = tot_len/4 # 4 seeds location
    start = int(np.floor(num_seeds/2))
    sec = start+num_seeds
    thi = sec+num_seeds
    last = thi+num_seeds


    figure=plt.figure(dpi=200)

    if sorted:
        zs = ticks_info[0]
        num = ticks_info[1]
        ticks = []
        ticks.extend(['VR']*num)
        ticks.extend(['VL']*num)
        ticks.extend(['DR']*num)
        ticks.extend(['DL']*num)
        zticks = []
        for t, z in zip(ticks,zs):
            zticks.append(t+z)

        idx_s=np.argsort(zticks)
        mat_corr=mat_corr_seeds[:,idx_s]
        mat_corr=mat_corr[idx_s,:]
        
        plt.xticks([start,sec,thi,last],['DL','DR', 'VL', 'VR'], fontsize=10 )
        plt.yticks([start,sec,thi,last],['DL','DR', 'VL', 'VR'], fontsize=10 )
        filename = 'FCmatrix_sortedZ'

    else:
        mat_corr = mat_corr_seeds
        plt.xticks([start,sec,thi,last],['VR','VL', 'DR', 'DL'], fontsize=10 )
        plt.yticks([start,sec,thi,last],['VR','VL', 'DR', 'DL'], fontsize=10 )
        filename = 'FCmatrix'
    
    plt.imshow(mat_corr)
    plt.colorbar()

    figure.savefig(os.path.join(DATA_PARENTPATH, 'sFC_results',sub,filename))
    plt.close(figure)

if __name__ == '__main__':
    
    if '--sub' in sys.argv:
        subjects_list = []
        sub_ind = sys.argv.index('--sub') + 1
        sub = sys.argv[sub_ind]
        subjects_list.append(sub)
        specific_subject = True
    else:
        subjects_list = [sub.split('/')[-1] for sub in glob(os.path.join(DATA_PARENTPATH, 'LU*')) \
                         if sub.split('/')[-1] not in ["LU_VS", "LU_NG"]] 
        specific_subject = False

    if '--seed' in sys.argv:
        # specified seed analysis
        atype_ind = sys.argv.index('--seed') + 1
        atype = sys.argv[atype_ind]
        assert atype in ['VL','VR','DL', 'DR']
        # make output directories
        for sub in subjects_list:
            os.makedirs(os.path.join(DATA_PARENTPATH, 'sFC_results', sub, atype), exist_ok=True)
        
        if '--perslice' in sys.argv:
            if specific_subject:
                sFC_perslice(atype, sub)
            else:
                sFC_perslice(atype)
        elif '--vertical' in sys.argv:
            if specific_subject:
                sFC_vertically(atype,sub)
            else:
                sFC_vertically(atype)

    elif '--seed2seed' in sys.argv:

        # compute FC matrix and average across subjects 
        seed2seed_allsubs()
        # compute also the region 4x4

        
        
       