#!/usr/bin/python

"""
This script contains the classes to perform static functional connectivity.

Author: Ilaria Ricchi
"""
# from re import sub
from random import random, seed

import numpy as np
import time
import os
import sys
from glob import glob
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp
from tqdm import tqdm

import nibabel as nib
from nilearn import plotting
from nilearn.signal import clean
import matplotlib.pyplot as plt
from matplotlib import patches

# import cv2
from nibabel import load, save, Nifti1Image

import pandas as pd
import matplotlib
import matplotlib.cm as cm


from scipy.signal import butter, filtfilt, cheby2, lfilter
from scipy.ndimage import gaussian_filter

global ROIS_EXEP

# DEFINE GLOBAL VARIABLE FOR ROIs
ROIS_EXEP = {
    "LU_AF" : [26,20,13,12,9] , # 26, 11 ,9 missing seeds
    "LU_AR" : [22,18,13,12,10,9],
    "LU_AT" : [7,8,9,17, 18, 19, 21,25],
    "LU_BN" : [25],
    "LU_EM"  : [21,9],
    "LU_EP" : [26,16,6,5,4,3],
    "LU_FB" : [14,15],
    "LU_GL" : [19,9,8,7],
    "LU_GP" : [11,10],
    "LU_IR" : [7,6,8],
    "LU_MC" : [12,11,10,9,8],
    "LU_MD" : [23,9,6,5,4],
    "LU_ML" : [26,24,12,11,10,9],
    "LU_MP" : [4,7,26],
    "LU_MT" : [12,13,14,15,16,17,20,21,24,25],
    "LU_NB" : [7,6,5,4,3],
    "LU_NG" : [22,13,12,11,10],
    "LU_NK" : [26,25,22,20,18,17,8,7,6],
    "LU_NS" : [9,7],
    "LU_SA" : [22,21,20,17,15,8,7,6],
    "LU_SL" : [11],
    "LU_SM" : [25,24,11,8],
    "LU_VG" : [25,16,],
    "LU_YF" : [12,11,10,9,8]

}


### DEFINITION OF CLASSES TO PERFORM STATIC FUNCTIONAL CONNECTIVITY :
# sFCsorConnectivity performs 
# 1) seeds connectivity on the average signal
# 2) ROIs connectivity  that have been registered from template to the native space

########################################################################
# ********** sFC seeds based **********
########################################################################
class sFCsorConnectivity(object):
    """ sFCSeedConnectivity performs static functional connectivity at the slice level or ROI.
    1) Slice by slice fixing seed (seed vs all voxels)
    2) Slice by slice using the manual seeds 
    3) Averaged seeds signals manual seeds
    3) Slice by slice using the automatic ROIs (template)
    4) Averaged ROI signals (template)
    """

    def __init__(self, data_path, 
                 subset_subs=[], 
                 subIDs_to_exclude=[],
                 method="average",
                 auto=True,
                 TR = 2.5,
                 tempfilt = None,
                 out_folder = "sFC_results"
                 ):

        """ Mandatory input is the data_path. 
        Optionals are:
        1) the list of subjects IDs,
        2) subIDs_to_exclude that collects all the IDs of subjects to exclude,
        3) the seeds of interest initialized to 'VL','VR','DL', 'DR'.
        """
        super(sFCsorConnectivity, self).__init__()
        self.data_path = data_path
        self.subset_subs = subset_subs
        self.subIDs_to_exclude = subIDs_to_exclude
        self.method = method
        self.auto = "ROIs" if auto else "seeds"
        self.TR = TR
        self.addnamestr = ''   # init filename additional part
        
        self.__init_configs()
        self.low_cutoff = None
        self.high_cutoff = None
        if tempfilt == "HP":
            self.low_cutoff = 1/100
        elif tempfilt == "BP13":
            self.low_cutoff = 0.01
            self.high_cutoff = 0.13
        elif tempfilt == "BP17":
            self.low_cutoff = 0.01
            self.high_cutoff = 0.17
            
        self.out_folder = os.path.join(out_folder,tempfilt,self.method,self.auto)
         # make outputdir
        os.makedirs(os.path.join(self.data_path, self.out_folder), exist_ok=True)

    ########################################################################
    # ********** Basic functions definition **********
    ########################################################################

    def __init_configs(self):
        """Initialze variables"""
        
        # self.bp_temporal_filter = False
        # self.hp_temporal_filter = False
        self.func = "func"   # name of the functional folder
        self.mask_file = "mask_sco.nii.gz"
        self.seeds_names = ["VR","VL","DR","DL"]
        self.n_jobs = -2
        
        if len(self.subset_subs) == 0:
            self.subjects_paths = glob(os.path.join(self.data_path, "LU_*"))
        else:
            self.subjects_paths = [os.path.join(self.data_path, sub) for sub in self.subset_subs]
        
        # Exclude subjects:
        if len(self.subIDs_to_exclude) != 0:
            self.subjects_paths = [sub for sub in self.subjects_paths if sub.split('/')[-1] not in self.subIDs_to_exclude]

    def __load_data_persub(self, sub_path, fname='mfmri_st_corr_denoised', aname='mfmri_st_corr_mean'):
        """Load data per subject."""
        sub_path_func = os.path.join(sub_path,  self.func, fname+'.nii.gz') # denoised fmri to perform sFC
        sub_path_anat = os.path.join(sub_path, self.func, aname+'.nii.gz')   # anat being the better quality image as background
        mask_path = os.path.join(sub_path, self.func, 'Segmentation', self.mask_file)
        ## OLD VERSION seeds manually placed
        #  seed_mask = os.path.join(sub_path, self.func, 'mfmri_mean_seeds.nii.gz')
        seed_mask = os.path.join(sub_path, self.func, 'gm_rois.nii.gz')

        
        img = nib.load(sub_path_func)
        img_anat = nib.load(sub_path_anat)
        mask = nib.load(mask_path)
        seeds = nib.load(seed_mask)
        
        # get arrays
        self.img_filt = img
        self.mask_array = mask.get_fdata()   # 3D
        self.grey_matter_mask = nib.load(os.path.join(sub_path,self.func,'GM.nii.gz')).get_fdata()
        self.img_data = img_anat.get_fdata() # for plotting
        self.f_img_data = self.img_filt.get_fdata()
        self.seeds_array = seeds.get_fdata()
        seeds_VR = self.__load_seeds('VR')
        self.zslices = np.unique(seeds_VR[:,2])
        ### Apply temporal filter on timeserieses (fMRI) only if no denoising applied (pure motion corrected)
        if fname in ["mfmri", "mfmri_st_corr"]:
            rf_img_data = self.f_img_data.reshape([self.f_img_data.shape[0]*self.f_img_data.shape[1]*self.f_img_data.shape[2],self.f_img_data.shape[-1]])
            filt_img_data = clean(rf_img_data,standardize=False,low_pass=self.high_cutoff,high_pass=self.low_cutoff,t_r=self.TR)
            self.f_img_data = filt_img_data.reshape([self.f_img_data.shape[0],self.f_img_data.shape[1],self.f_img_data.shape[2],self.f_img_data.shape[-1]])


    def __load_seeds(self,seed):
        seeds_xyz = 0
        if seed == 'VR':
            seeds_xyz = np.array(np.where(self.seeds_array==1)).T    
        elif seed == 'VL':
            seeds_xyz = np.array(np.where(self.seeds_array==2)).T
        elif seed == 'DR':
            seeds_xyz = np.array(np.where(self.seeds_array==3)).T    
        elif seed == 'DL':
            seeds_xyz = np.array(np.where(self.seeds_array==4)).T

        return seeds_xyz

    def __load_rois(self,roi,sub):
        rois_xyz = 0
        subname = sub.split('/')[-1].split('_')[-1]
        roi_loc = nib.load(os.path.join(sub, f'func/ROIs/{roi}_{subname}.nii.gz')).get_fdata()
        rois_xyz = np.array(np.where(roi_loc!=0)).T

        return rois_xyz

    def __ts_to_average(self, seeds_xyz):
        # from time serieses to mean
        X = np.zeros((seeds_xyz.shape[0], self.f_img_data.shape[-1]))
        for i, xyz in enumerate(seeds_xyz):
            X[i,:] = self.f_img_data[xyz[0], xyz[1], xyz[2], :]
            i+=1
        
        return np.mean(X,0)
    
    def __exclude_slices(self, sub, xyz_list):

        new_xyz = []
        for xyz in xyz_list:
            if xyz[2] not in ROIS_EXEP[sub]:
                new_xyz.append(xyz)
        
        return np.array(new_xyz)

        
    ########################################################################
    # ********** Per slice approach : seed 2 voxels **********
    ########################################################################

    def _sFC_seed2voxels_persub(self, 
                                subpath, 
                                soi,
                                fname='mfmri_st_corr_denoised'):
        """Given the specific subject path (subpath) and the seed of interest (soi) """
        # make output folder
        sub = subpath.split('/')[-1]
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub, soi), exist_ok=True)
        print("### Info: Loading subject: ", subpath)
        start = time.time()
        self.__load_data_persub(subpath,fname)
        
        self.seeds_xyz = self.__load_seeds(soi)      

        # average only the seeds that are on the same slice
        slices = np.unique(self.seeds_xyz[:,2])
        dict_corrs_all = dict()
        for z in slices:
            slices_oi = self.seeds_xyz[np.where(self.seeds_xyz[:,2]==z)]
            
            ts_ave = self.__ts_to_average(slices_oi)
            # go through slices and correlate the above seeds time series with all voxels
            mask_x, mask_y = np.where(self.mask_array[:,:,z]!=0)
            #list_xy = []
            #X = np.zeros((len(mask_x), self.f_img_data.shape[-1]))
            #i = 0
            # iterate over all the pixels that belong to the mas
            list_xy = []
            corrs = []
            for x,y in zip(mask_x,mask_y):
                if x in slices_oi[:,0] and y in slices_oi[:,1]:
                    corr = 1  # becuase it belongs to the seeds and the signals have been averaged
                else:
                    corr = np.corrcoef(ts_ave,self.f_img_data[x,y,z,:])[0,1]
                
                #dict_corrs[(x,y)]  = corr
                corrs.append(corr)
                list_xy.append((x,y))
            
            dict_corrs = dict(zip(list_xy, corrs))
            dict_corrs_all[z] = dict_corrs

            self._plot_simplemap_overlayed(z, dict_corrs, [sub, soi])
        
        #seeds_ts = dict(zip(slices,ts_ave))

        # consider only seeds of interest
        # but there are multiple seeds in each slice so we'll take only one of each
        # seeds_xyz = []
        # seen_z = set()
        # for xyz in self.seeds_xyz:
        #     z = xyz[-1]
        #     if z not in seen_z:
        #         seeds_xyz.append(xyz)
        #         seen_z.add(z)
        
        # print("### Info: subject data loaded in %.2f s." %(time.time()-start))

        # # Generate png figures maps overlayed slice per slice (z)
        # # iterate over the slices
        # dict_corrs_all = dict() 

        # # select elements in the mask

        # for xyz in self.seeds_xyz:
        #     # for each

        #     # print("Seed [x y z]: ",xyz)
        #     # masked_img = cv2.bitwise_and(self.mask_array[:,:,xyz[-1]],self.img_data[:,:,xyz[-1]])
        #     mask_x, mask_y = np.where(self.mask_array[:,:,xyz[-1]]!=0)
        #     assert xyz[0] in mask_x
        #     assert xyz[1] in mask_y
            
        #     i = 0
        #     list_xy = []
        #     X = np.zeros((len(mask_x), self.f_img_data.shape[-1]))
        #     # iterate only over the pixels that belong to the mask
        #     for x, y in zip(mask_x, mask_y):
        #         X[i,:] = self.f_img_data[x,y,xyz[-1],:]   # take all timepoints
        #         list_xy.append((x,y))
        #         if x == xyz[0] and y == xyz[1]:
        #             # print("seed index:", i)
        #             seed_ind = i
                
        #         i+=1

        #     # compute corr with respect to the seed
        #     corr_vals = np.zeros(X.shape[0])
        #     print(X.shape)
        #     for j in range(X.shape[0]):
        #         R_corr = np.corrcoef(X[seed_ind,:],X[j,:])
        #         corr_vals[j] = R_corr[0,1]

        #     # create dictionary to easy access info
        #     dict_corrs = dict(zip(list_xy, corr_vals))
        #     dict_corrs_all[xyz[-1]] = dict_corrs

        #     # (UN)COMMENT
        #     # plot heatmap on every slice
        #     self._plot_simplemap_overlayed(xyz, dict_corrs, [sub, soi])


        print("### Info: generate 3D map in nifti ...")
        # generate nii in 3D 
        print("dict_corrs_all.keys()", dict_corrs_all.keys())
        print("dict_corrs_all.values()",dict_corrs_all.values())
        img_corr3d, outpath = self._generate_3d_heatmaps(dict_corrs_all,[sub, soi]) 
        out = Nifti1Image(img_corr3d, header=self.img_filt.header, affine=self.img_filt.affine)
        save(out, outpath) 

    def _plot_simplemap_overlayed(self, z, dict_corrs, sub_atype):

        sub = sub_atype[0]
        atype = sub_atype[1]
        
        slice_xy_mri = self.img_data[:, :, z]
        fig = plt.figure(figsize=(12,10))
        ax = plt.subplot(111)
        ax.imshow(slice_xy_mri.T, cmap='gray', origin='lower',zorder=1)
        ax.set_xlim([55,93])
        ax.set_ylim([10,35])

        #for xy, cor in dict_corrs.items():
            # if cor == 1:
            #     # it's the seed so highlight it in red
            #     rect = patches.Rectangle((xy[0]-0.5, xy[1]-0.5), 1, 1, linewidth=7, edgecolor='r', facecolor='none',zorder=4)
            #     ax.add_patch(rect)
        
        corr_vals = np.array(list(dict_corrs.values()))  # used for min and max values in case 
        # corr_vals = corr_vals[corr_vals<0.99]
        minc = 0
        maxc = 0.5
        # maxc = np.max(corr_vals)

        contour_level = 0.5 # to drawn the outlnes 

        norm = matplotlib.colors.Normalize(vmin=minc, vmax=maxc, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        for vox,v in dict_corrs.items():
            c = mapper.to_rgba(v) # color
            rect = patches.Rectangle((vox[0]-0.5, vox[1]-0.5), 1, 1, linewidth=2, edgecolor='none', facecolor=c, alpha=0.6,zorder=2)
            ax.add_patch(rect)

        # add outlines / contour
        # ax.contour(self.grey_matter_mask[:,:,z].T, levels=[contour_level],colors='white',linewidths=2,linestyles='dotted',zorder=3)

        fig.colorbar(mapper)        
        fig.savefig(os.path.join(self.data_path, self.out_folder,sub, atype,f'heatmap_ps_z_{z}'))
        plt.close(fig)

    def _generate_3d_heatmaps(self, dict_corrs_all, sub_atype):

        # masked_img3d = cv2.bitwise_and(self.mask_array,self.img_data)
        sub = sub_atype[0]
        atype = sub_atype[1]
        # initialize 3d image
        img_corr3d = np.zeros(self.mask_array.shape)

        mask_x, mask_y, mask_z = np.where(self.mask_array!=0)
                
        # in the case of slice per slice
        for x, y, z in zip(mask_x, mask_y, mask_z):
            # take z location to 
            if z in np.unique(self.seeds_xyz[:,2]):
                dict_corrs = dict_corrs_all[z]
                print(dict_corrs)
                for xy in dict_corrs.keys():
                    # take the slice correlations
                    img_corr3d[xy[0], xy[1], z] = dict_corrs[xy]

        outfilepath = os.path.join(self.data_path, self.out_folder,sub, atype,'heatmaps_ps_allz.nii.gz') 
        
        print("### Info: 3D output generated.")

        return img_corr3d, outfilepath

    def corr_seed2vox_perslice(self, soi,
                     fname='mfmri_denoised', 
                     ):
        print("### Info: sFC per slice running in parallel...")
        # with parallel_backend("loky", inner_max_num_threads=2):
        
        Parallel(n_jobs=self.n_jobs,
                verbose=10,
                backend="multiprocessing")(delayed(self._sFC_seed2voxels_persub)(sub,soi,fname)\
                for sub in self.subjects_paths)





    ########################################################################
    # ********** Slices approach - seeds **********
    ########################################################################

    def _seed2seed_slices_persub(self, sub_path, 
                                fname='mfmri_st_corr_denoised'):

        if len(fname.split('_')) > 4:
            # in the case we are not seeing only the mfmri_st_corr_denoised but other combinations
            addname = os.path.join(*fname.split('_')[4:])
            self.addnamestr = "_"+addname.replace('/','')

        # make directories per subj
        sub = sub_path.split('/')[-1]
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub,'FC_slices_img'), exist_ok=True)

        print("### Info: Loading subject: ", sub_path)
        start = time.time()
        self.__load_data_persub(sub_path,fname)
        print("### Info: subject data loaded in %.2f s." %(time.time()-start))

        FC_z = {}
        FC_z1 = {}
        FC_z2 = {}
        FC_ave = np.zeros((len(self.seeds_names),len(self.seeds_names)))
        for z in self.zslices:
            # initialize
            X = np.zeros((len(self.seeds_names),self.f_img_data.shape[-1]))  # n seeds x T
            for i in range(len(self.seeds_names)):
                seeds = self.__load_seeds(self.seeds_names[i])
                # take only the seeds of the specific slice
                rois = np.array([list(xyz) for xyz in seeds if xyz[2] == z])
                #print(rois)
                # NEW part average the signals across the rois:
                X[i,:] = self.__ts_to_average(rois)
                #print(X)
                # commented
                #X[i,:] = self.f_img_data[rois[0],rois[1],rois[2],:]

            
            FC_z[z] = np.corrcoef(X)
            FC_z1[z] = np.corrcoef(X[:,:int(np.round(X.shape[-1]/2))])
            FC_z2[z] = np.corrcoef(X[:,int(np.round(X.shape[-1]/2)):])
            FC_ave += FC_z[z]
            # plot single slice
            fig=plt.figure(dpi=100)
            plt.imshow(FC_z[z])
            plt.colorbar()
            fig.savefig(os.path.join(self.data_path, self.out_folder, sub,'FC_slices_img',f'FC_z{z}'))
            plt.close(fig)

        FC_ave /= len(self.zslices)
        # save FC all  
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'FC_zslices'+self.addnamestr), FC_z)
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'FC_zslices_1'), FC_z1)
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'FC_zslices_2'), FC_z2)
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'FC_mat'), FC_ave)

        # plot the subject FC
        fig = plt.figure(dpi=150)
        plt.imshow(FC_ave,vmax=0.3)
        plt.colorbar()
        plt.xticks(np.arange(len(self.seeds_names)),self.seeds_names , fontsize=15)
        plt.yticks(np.arange(len(self.seeds_names)),self.seeds_names , fontsize=15)
        fig.savefig(os.path.join(self.data_path, self.out_folder, sub, 'FC_s2s_slice_'+self.addnamestr))
        plt.close(fig)

    ########################################################################
    # ********** Average approach - seeds **********
    ########################################################################

    def _seed2seed_ave_persub(self, sub_path, 
                             fname='mfmri_st_corr_denoised' 
                             ):

        if len(fname.split('_')) > 4:
            # in the case we are not seeing only the mfmri_denoised but other combinations
            
            addname = os.path.join(*fname.split('_')[1:])
            self.addnamestr = "_"+addname.replace('/','')

        # make directories per subj
        sub = sub_path.split('/')[-1]
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub), exist_ok=True)

        
        print(f"### Info: Loading subject {sub}... ")
        self.__load_data_persub(sub_path,fname)
        print(f"### Info: Loading seeds... ")

        vr_xyz = self.__load_seeds(self.seeds_names[0])
        vl_xyz = self.__load_seeds(self.seeds_names[1])
        dr_xyz = self.__load_seeds(self.seeds_names[2])
        dl_xyz = self.__load_seeds(self.seeds_names[3])

        vr = self.__ts_to_average(vr_xyz)
        vl = self.__ts_to_average(vl_xyz)
        dr = self.__ts_to_average(dr_xyz)
        dl = self.__ts_to_average(dl_xyz)

        Xave = np.vstack([vr,vl,dr,dl])
        # save Xave
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'Xave'),Xave)

        corr_ave = np.corrcoef(Xave)
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'FC_mat'),corr_ave)
        fig = plt.figure(dpi=150)
        plt.imshow(corr_ave,vmax=0.3)
        plt.colorbar()
        plt.xticks(np.arange(len(self.seeds_names)), self.seeds_names , fontsize=15)
        plt.yticks(np.arange(len(self.seeds_names)), self.seeds_names , fontsize=15)
        fig.savefig(os.path.join(self.data_path, self.out_folder, sub, 'FC_s2s_ave_'+self.addnamestr))
        plt.close(fig)


    ########################################################################
    # ********** Slices approach - ROIs **********
    ########################################################################

    def _roi2roi_slices_persub(self, sub_path, 
                               fname='mfmri_st_corr_denoised'):
        
        sub = sub_path.split('/')[-1]
        if len(fname.split('_')) > 4:
            # in the case we are not seeing only the mfmri_st_corr_denoised but other combinations
            addname = os.path.join(*fname.split('_')[4:])
            self.addnamestr = "_"+addname.replace('/','')
            
        self.__load_data_persub(sub_path,fname)

        vr_xyz = self.__load_rois(self.seeds_names[0], sub_path)
        vl_xyz = self.__load_rois(self.seeds_names[1], sub_path)
        dr_xyz = self.__load_rois(self.seeds_names[2], sub_path)
        dl_xyz = self.__load_rois(self.seeds_names[3], sub_path)
        
        # exclude ROIs that have missing points
        vr_xyz = self.__exclude_slices(sub, vr_xyz)
        vl_xyz = self.__exclude_slices(sub, vl_xyz)
        dr_xyz = self.__exclude_slices(sub, dr_xyz)
        dl_xyz = self.__exclude_slices(sub, dl_xyz)

        ROIs_list = [vr_xyz, vl_xyz, dr_xyz, dl_xyz]
        zslices = np.unique(vr_xyz[:,2])
        
        # make directories per subj
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub,'FC_rois_img'), exist_ok=True)

        print("### Info: Loading subject: ", sub_path)
        start = time.time()
        
        print("### Info: subject data loaded in %.2f s." %(time.time()-start))

        FC_z = {}
        FC_ave = np.zeros((len(self.seeds_names),len(self.seeds_names)))
        for z in zslices:
            # initialize
            X = np.zeros((len(self.seeds_names),self.f_img_data.shape[-1]))  # n seeds x T
            for i in range(len(self.seeds_names)):
                xyz = ROIs_list[i]

                # take only the seeds of the specific slice
                X[i,:] = self.__ts_to_average(xyz[np.where(xyz[:,2]==z)])

            FC_z[z] = np.corrcoef(X)
            FC_ave += FC_z[z]
         
            # plot single slice
            fig=plt.figure(dpi=100)
            plt.imshow(FC_z[z])
            plt.colorbar()
            fig.savefig(os.path.join(self.data_path, self.out_folder, sub,'FC_rois_img',f'FC_z{z}'))
            plt.close(fig)

        FC_ave /= len(zslices)
        # save FC all  
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'FC_zslices'+self.addnamestr), FC_z)
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'FC_mat'), FC_ave)

        # plot the subject FC
        fig = plt.figure(dpi=150)
        plt.imshow(FC_ave,vmax=0.3)
        plt.colorbar()
        plt.xticks(np.arange(len(self.seeds_names)),self.seeds_names , fontsize=15)
        plt.yticks(np.arange(len(self.seeds_names)),self.seeds_names , fontsize=15)
        fig.savefig(os.path.join(self.data_path, self.out_folder, sub, 'FC_r2r_slice_'+self.addnamestr))
        plt.close(fig)

    ########################################################################
    # ********** Average approach - rois **********
    ########################################################################                  

    def _roi2roi_ave_persub(self, sub_path,
                            fname='mfmri_st_corr_denoised'
                           ):

        if len(fname.split('_')) > 2:
            # in the case we are not seeing only the mfmri_denoised but other combinations
            
            addname = os.path.join(*fname.split('_')[1:])
            self.addnamestr = "_"+addname.replace('/','')

        # make directories per subj
        sub = sub_path.split('/')[-1]
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub), exist_ok=True)

        
        print(f"### Info: Loading subject {sub}... ")
        self.__load_data_persub(sub_path,fname)
        print(f"### Info: Loading seeds... ")

        vr_xyz = self.__load_rois(self.seeds_names[0], sub_path)
        vl_xyz = self.__load_rois(self.seeds_names[1], sub_path)
        dr_xyz = self.__load_rois(self.seeds_names[2], sub_path)
        dl_xyz = self.__load_rois(self.seeds_names[3], sub_path)

        # exclude ROIs that have missing points
        vr_xyz = self.__exclude_slices(sub, vr_xyz)
        vl_xyz = self.__exclude_slices(sub, vl_xyz)
        dr_xyz = self.__exclude_slices(sub, dr_xyz)
        dl_xyz = self.__exclude_slices(sub, dl_xyz)

        vr = self.__ts_to_average(vr_xyz)
        vl = self.__ts_to_average(vl_xyz)
        dr = self.__ts_to_average(dr_xyz)
        dl = self.__ts_to_average(dl_xyz)

        Xave = np.vstack([vr,vl,dr,dl])
        # save Xave
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'Xave'),Xave)

        corr_ave = np.corrcoef(Xave)
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'FC_mat'),corr_ave)
        fig = plt.figure(dpi=150)
        plt.imshow(corr_ave,vmax=0.3)
        plt.colorbar()
        plt.xticks(np.arange(len(self.seeds_names)), self.seeds_names , fontsize=15)
        plt.yticks(np.arange(len(self.seeds_names)), self.seeds_names , fontsize=15)
        fig.savefig(os.path.join(self.data_path, self.out_folder, sub, 'FC_r2r_ave_'+self.addnamestr))
        plt.close(fig)

    ########################################################################
    # ********** RUN IN PARALLEL ACROSS SUBJECTS **********
    ########################################################################  

    def run_allsubs_parallel(self, fname="mfmri_st_corr_denoised"):
        # run in Parallel the seed2seed correlation computation per subject
        
        if self.auto == "seeds" and self.method == "slices":
            func_to_use = self._seed2seed_slices_persub
            #name_to_load = "FC_s2s_slice.npy"
        elif self.auto == "seeds" and self.method == "average":
            func_to_use = self._seed2seed_ave_persub
            #name_to_load = "FC_s2s_ave.npy"
        elif self.auto == "ROIs" and self.method == "slices":
            func_to_use = self._roi2roi_slices_persub
            #name_to_load = "FC_r2r_slice.npy"
        elif self.auto == "ROIs" and self.method == "average":
            func_to_use = self._roi2roi_ave_persub
            #name_to_load = "FC_r2r_ave.npy"

        Parallel(n_jobs=self.n_jobs,
                 verbose=10)(delayed(func_to_use)(sub_path,fname) \
                 for sub_path in self.subjects_paths)
        
        if len(fname.split('_')) > 4: #"st" in fname.split('_'):
            # in the case we are not seeing only the mfmri_denoised but other combinations
            addname = os.path.join(*fname.split('_')[4:])
            self.addnamestr = "_"+addname.replace('/','')
            
        # go through all saved correlation matrices and compute a population average
        corrs_ave_all = np.zeros((len(self.seeds_names),len(self.seeds_names)))  # 4x4
        for i_s in tqdm(range(len(self.subjects_paths))):
            sub_path = self.subjects_paths[i_s]
            sub = sub_path.split('/')[-1]

            corrs_ave_all += np.load(os.path.join(self.data_path, self.out_folder,sub, 'FC_mat.npy'))

        # normalize by the number of subject
        corrs_ave_all = corrs_ave_all/len(self.subjects_paths)

        np.save(os.path.join(self.data_path, self.out_folder, f'FC_ave_all_{self.auto}_{self.method}'), corrs_ave_all)
        # plot full average population
        fig = plt.figure(dpi=150)
        plt.imshow(corrs_ave_all,vmax=0.3)
        plt.colorbar()
        plt.xticks(np.arange(len(self.seeds_names)),self.seeds_names , fontsize=15)
        plt.yticks(np.arange(len(self.seeds_names)),self.seeds_names , fontsize=15)
        fig.savefig(os.path.join(self.data_path, self.out_folder, f'FC_ave_all_{self.auto}_{self.method}'))
        plt.close(fig)

        return corrs_ave_all
