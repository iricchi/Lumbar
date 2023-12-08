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


### DEFINITION OF CLASSES TO PERFORM STATIC FUNCTIONAL CONNECTIVITY :
# 2 classes are defined: 
# 1) sFCSeedConnectivity: performs sFC slice-wise, vertically and seed2seed 
# 2) sFCParcellation: perform sFC but averaging across ROIs that have been registered
#    to template. In this way sFC is comupted with a parcelation format.

########################################################################
# ********** sFC seeds based **********
########################################################################
class sFCSeedConnectivity(object):
    """ sFCSeedConnectivity performs static functional connectivity at the slice level.
    1) Slice by slice fixing seed (seed vs all voxels)
    2) Seed of 1 slice with respect to all voxels in +- 2 slices in z (vertically)
    3) Seed-to-seed connectivity
    """

    def __init__(self, data_path, 
                 subset_subs=[], 
                 subIDs_to_exclude=[],
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
        super(sFCSeedConnectivity, self).__init__()
        self.data_path = data_path
        self.subset_subs = subset_subs
        self.subIDs_to_exclude = subIDs_to_exclude
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
            
        self.out_folder = os.path.join(out_folder,tempfilt)
         # make outputdir
        os.makedirs(os.path.join(self.data_path, self.out_folder), exist_ok=True)

    def __init_configs(self):
        """Initialze variables"""
        
        # self.bp_temporal_filter = False
        # self.hp_temporal_filter = False
        self.func = "func"   # name of the functional folder
        self.mask_file = "mask_sco.nii.gz"
        self.n_jobs = -2
        
        if len(self.subset_subs) == 0:
            self.subjects_paths = glob(os.path.join(self.data_path, "LU_*"))
        else:
            self.subjects_paths = [os.path.join(self.data_path, sub) for sub in self.subset_subs]
        
        # Exclude subjects:
        if len(self.subIDs_to_exclude) != 0:
            self.subjects_paths = [sub for sub in self.subjects_paths if sub.split('/')[-1] not in self.subIDs_to_exclude]

    def temporal_filter2_vol(self, img):
        """ img is the input img loaded with nibabel, Wn is the widnow for the bandpass:
    #     [0.01,0.13]. It's one value in case of a high pass filter (cutoff freq).
    #     """
        
        f_img_data = img.get_fdata()
        if isinstance(self.Wn,list):
            filtertype = 'bandpass'
        elif isinstance(self.Wn,float):
            filtertype = 'highpass'

        B,A = butter(1,self.Wn,filtertype,fs=1./self.TR)
        filtered_data = filtfilt(B,A,f_img_data)
        img_out = Nifti1Image(filtered_data, img.affine, img.header)
        
        return img_out
    

    # def temporal_filter2_vol(self, img, Wn=[0.01,0.13],TR=2.5):
    #     """ img is the input img loaded with nibabel, Wn is the widnow for the bandpass:
    #     [0.01,0.13]
    #     """
    #     f_img_data = img.get_fdata()
    #     #rf_img_data = f_img_data.reshape(f_img_data.shape[0]*f_img_data.shape[1]*f_img_data.shape[2],f_img_data.shape[-1])
    #     b,a = cheby2(4, 30, Wn, 'bandpass', fs=1./TR)
    #     filtered_data = filtfilt(b,a,f_img_data)
    #     # filtered_data = filtered_data.reshape(f_img_data.shape[0],f_img_data.shape[1],f_img_data.shape[2],f_img_data.shape[-1])
    #     img_out = Nifti1Image(filtered_data, img.affine, img.header)
        
    #     return img_out

    # def temporal_HP_filter2_vol(self, img, HPCs=1/100,TR=2.5,chebyfilt=True):
    #     """ img is the input img loaded with nibabel, Wn is the widnow for the bandpass:
    #     [0.01,0.13]
    #     """

    #     f_img_data = img.get_fdata()
    #     #rf_img_data = f_img_data.reshape(f_img_data.shape[0]*f_img_data.shape[1]*f_img_data.shape[2],f_img_data.shape[-1])

        
    #     b,a = cheby2(4, 30, HPCs, 'highpass', fs=1./TR)
    #     filtered_data = filtfilt(b,a,f_img_data)
    #     # else:
    #     #     # proceed with the gaussian filter
    #     #     sigma = (HPCs)/(np.sqrt(8*np.log2(2))/TR)
    #     #     lowcomp = gaussian_filter(rf_img_data,sigma)
    #     #     filtered_data = rf_img_data - lowcomp

    #     # filtered_data = filtered_data.reshape(f_img_data.shape[0],f_img_data.shape[1],f_img_data.shape[2],f_img_data.shape[-1])
    #     img_out = Nifti1Image(filtered_data, img.affine, img.header)
        
    #     return img_out

    def __load_data_persub(self, sub_path, fname='mfmri_denoised', aname='mfmri_mean'):
        """Load data per subject.
        - sub_path is the """
        sub_path_func = os.path.join(sub_path,  self.func, fname+'.nii.gz') # denoised fmri to perform sFC
        sub_path_anat = os.path.join(sub_path, self.func, aname+'.nii.gz')   # anat being the better quality image as background
        mask_path = os.path.join(sub_path, self.func, 'Segmentation', self.mask_file)
        seed_mask = os.path.join(sub_path, self.func, 'mfmri_mean_seeds.nii.gz')

        img = nib.load(sub_path_func)
        img_anat = nib.load(sub_path_anat)
        mask = nib.load(mask_path)
        seeds = nib.load(seed_mask)
        
        # get arrays
        self.img_filt = img
        self.mask_array = mask.get_fdata()   # 3D
        seeds_array = seeds.get_fdata()
        self.img_data = img_anat.get_fdata() # for plotting
        self.f_img_data = self.img_filt.get_fdata()
        ### Apply temporal filter on timeserieses (fMRI) only if no denoising applied (pure motion corrected)
        if fname == "mfmri":
            rf_img_data = self.f_img_data.reshape([self.f_img_data.shape[0]*self.f_img_data.shape[1]*self.f_img_data.shape[2],self.f_img_data.shape[-1]])
            filt_img_data = clean(rf_img_data,standardize=False,low_pass=self.high_cutoff,high_pass=self.low_cutoff,t_r=self.TR)
            self.f_img_data = filt_img_data.reshape([self.f_img_data.shape[0],self.f_img_data.shape[1],self.f_img_data.shape[2],self.f_img_data.shape[-1]])
        

        # self.nf_img_data = img.get_fdata()   # not filtered for comparisons
        # Given a specific input seed (VL, VR, DL, DR)
        self.seeds_array = seeds_array   # store all seeds


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

    ###### Per slice:

    def _sFC_perslice_persub(self, 
                             subpath, 
                             soi,
                             fname='mfmri_denoised', 
                             aname='mfmri_mean'):
        """Given the specific subject path (subpath) and the seed of interest (soi) """
        # make output folder
        sub = subpath.split('/')[-1]
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub, soi), exist_ok=True)
        print("### Info: Loading subject: ", subpath)
        start = time.time()
        self.__load_data_persub(subpath,fname,aname)
        
        self.seeds_xyz = self.__load_seeds(soi)  # consider only seeds of interest
        
        print("### Info: subject data loaded in %.2f s." %(time.time()-start))

        # Generate png figures maps overlayed slice per slice (z)
        # iterate over the slices
        dict_corrs_all = dict() 
        for xyz in self.seeds_xyz:
            # print("Seed [x y z]: ",xyz)
            
            # masked_img = cv2.bitwise_and(self.mask_array[:,:,xyz[-1]],self.img_data[:,:,xyz[-1]])
            mask_x, mask_y = np.where(self.mask_array[:,:,xyz[-1]]!=0)
            assert xyz[0] in mask_x
            assert xyz[1] in mask_y
            
                    
            i = 0
            list_xy = []
            X = np.zeros((len(mask_x), self.f_img_data.shape[-1]))
            # iterate only over the pixels that belong to the mask
            for x, y in zip(mask_x, mask_y):
                X[i,:] = self.f_img_data[x,y,xyz[-1],:]   # take all timepoints
                list_xy.append((x,y))
                if x == xyz[0] and y == xyz[1]:
                    # print("seed index:", i)
                    seed_ind = i
                
                i+=1

            # compute corr with respect to the seed
            corr_vals = np.zeros(X.shape[0])
            print(X.shape)
            for j in range(X.shape[0]):
                R_corr = np.corrcoef(X[seed_ind,:],X[j,:])
                corr_vals[j] = R_corr[0,1]

            # create dictionary to easy access info
            dict_corrs = dict(zip(list_xy, corr_vals))
            dict_corrs_all[xyz[-1]] = dict_corrs

            # (UN)COMMENT
            # plot heatmap on every slice
            self._plot_simplemap_overlayed(xyz, dict_corrs, [sub, soi])

        print("### Info: generate 3D map in nifti ...")
        # generate nii in 3D 
        img_corr3d, outpath = self._generate_3d_heatmaps(dict_corrs_all, '',[sub, soi]) 
        out = Nifti1Image(img_corr3d, header=self.img_filt.header, affine=self.img_filt.affine)
        save(out, outpath) 

    def _plot_simplemap_overlayed(self, seed_xyz, dict_corrs, sub_atype):

        sub = sub_atype[0]
        atype = sub_atype[1]

        slice_xy_mri = self.img_data[:, :, seed_xyz[-1]]
        fig = plt.figure(figsize=(12,10))
        ax = plt.subplot(111)
        ax.imshow(slice_xy_mri.T, cmap='gray', origin='lower')
        ax.set_xlim([55,93])
        ax.set_ylim([10,35])

        # highlight seed
        rect = patches.Rectangle((seed_xyz[0]-0.5, seed_xyz[1]-0.5), 1, 1, linewidth=7, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        corr_vals = np.array(list(dict_corrs.values()))
        # generate heatmap for pixels
        ## uncomment this if you want to have min max of the correlations
        # minc = min(corr_vals)
        # maxc = max(corr_vals) 
        ## standardized min-max
        minc = 0
        maxc = 0.7


        norm = matplotlib.colors.Normalize(vmin=minc, vmax=maxc, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        for vox in dict_corrs.keys():
            v = dict_corrs[vox]
            c = mapper.to_rgba(v) # color
            rect = patches.Rectangle((vox[0]-0.5, vox[1]-0.5), 1, 1, linewidth=2, edgecolor='none', facecolor=c, alpha=0.6)
            ax.add_patch(rect)

        fig.colorbar(mapper)
        fig.savefig(os.path.join(self.data_path, self.out_folder,sub, atype,'heatmap_ps_z_%s') %str(seed_xyz[-1]))
        plt.close(fig)

    def _generate_3d_heatmaps(self, dict_corrs_all, selected_seed, sub_atype, vertical=False):

        # masked_img3d = cv2.bitwise_and(self.mask_array,self.img_data)
        sub = sub_atype[0]
        atype = sub_atype[1]
        # initialize 3d image
        img_corr3d = np.zeros(self.mask_array.shape)

        mask_x, mask_y, mask_z = np.where(self.mask_array!=0)
        seed_xyz_str = (' '.join(map(str,selected_seed))).replace(' ','-')

        if vertical:
            # looking at correlations across 
            print("### Info: this could take long ...")
            for xyz in dict_corrs_all.keys():
                img_corr3d[xyz[0], xyz[1], xyz[-1]] = dict_corrs_all[xyz]

            outfilepath = os.path.join(self.data_path, self.out_folder,sub, atype,'heatmaps_ps_vertically_seed_%s.nii.gz') % seed_xyz_str
        else:
            # in the case of slice per slice
            for x, y, z in zip(mask_x, mask_y, mask_z):
                # take z location to 
                if z in self.seeds_xyz[:,2]:
                    dict_corrs = dict_corrs_all[z]
                    for xy in dict_corrs.keys():
                        # take the slice correlations
                        img_corr3d[xy[0], xy[1], z] = dict_corrs[xy]

            outfilepath = os.path.join(self.data_path, self.out_folder,sub, atype,'heatmaps_ps_allz.nii.gz') 
        
        print("### Info: 3D output generated.")

        return img_corr3d, outfilepath

    def sFC_perslice(self, soi,
                     fname='mfmri_denoised', 
                     aname='mfmri_mean'):
        print("### Info: sFC per slice running in parallel...")
        # with parallel_backend("loky", inner_max_num_threads=2):
        
        Parallel(n_jobs=self.n_jobs,
                verbose=10,
                backend="multiprocessing")(delayed(self._sFC_perslice_persub)(sub,soi,fname,aname)\
                for sub in self.subjects_paths)


    ###### Vertically :

    def _inspect_seeds_persub(self, sps,
                              fname='mfmri_denoised', 
                              aname='mfmri_mean'):
        """This function is used externaly in JN to interactivally see the specific
           subject seeds and decide on the slice to consider for the vertical approach."""
        # specific path of the subject
        subject_path = os.path.join(self.data_path, sps)
        # load data of the subject
        self.__load_data_persub(subject_path,fname,aname)
        print("### Info: data loaded.")

    def _sFC_vertical_persub_z(self, sps, 
                               soi, z_s,
                               fname='mfmri_denoised', 
                               aname='mfmri_mean'):
        
        if not hasattr(self, 'img_filt'):
            print(f"### Info: loading subject {sps}...")
            subject_path = os.path.join(self.data_path, sps)
            # load data per subject
            self.__load_data_persub(subject_path,fname,aname)

        if not hasattr(self, 'seeds_xyz'):
            print("### Info: loading seeds ...")
            self.seeds_xyz = self.__load_seeds(soi)

        if z_s==-99:
            # select seed randomly
            selected_seed = self.seeds_xyz[np.random.randint(self.seeds_xyz.shape[0])]
        else:
            ind_selected = np.where(self.seeds_xyz[:,-1]==z_s)[0][0]
            selected_seed = self.seeds_xyz[ind_selected]
        
        print(f"### Info: looking for subject {sps}, seed located in {soi} at z = {z_s}...")
        print("Seed selected: ", selected_seed)

        # masked_img3d = cv2.bitwise_and(self.mask_array,self.img_data)
        mask_x, mask_y, mask_z = np.where(self.mask_array!=0)

        i = 0
        X = np.zeros((len(mask_x), self.f_img_data.shape[-1]))
        list_xyz = []
        for x,y,z in zip(mask_x,mask_y,mask_z):
            X[i,:] = self.f_img_data[x, y, z, :]
            list_xyz.append((x,y,z))
            if x==selected_seed[0] and y==selected_seed[1] and z==selected_seed[2]:
                print("seed index:", i)
                seed_ind = i
            i+=1

        # compute corr wrt the seed
        corr_vals = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            R_corr = np.corrcoef(X[seed_ind,:],X[j,:])
            corr_vals[j] = R_corr[0,1]

        # create dictionary to easy access info
        dict_corrs = dict(zip(list_xyz, corr_vals))
        heatmap3d, outpath = self._generate_3d_heatmaps(dict_corrs, selected_seed, [sps, soi], True)
        out = Nifti1Image(heatmap3d, header=self.img_filt.header, affine=self.img_filt.affine)
        save(out, outpath) 

    def sFC_vertical_all_subjects(self, soi,
                                  fname='mfmri_denoised', 
                                  aname='mfmri_mean'):
        # z will be picked randomly so z= -99
        
        print("### Info: sFC per across all voxels vertically running in parallel...")
        # with parallel_backend("loky", inner_max_num_threads=2):
        Parallel(n_jobs=self.n_jobs,
                verbose=100,
                backend="multiprocessing")(delayed(self._sFC_vertical_persub_z)(sub.split('/')[-1],soi,23,fname,aname)\
                for sub in self.subjects_paths)
        
    ###### Seed2seed:

    def __ts_to_average(self, seeds_xyz):
        # from time serieses to mean
        X = np.zeros((seeds_xyz.shape[0], self.f_img_data.shape[-1]))
        i = 0 
        for xyz in seeds_xyz:
            X[i,:] = self.f_img_data[xyz[0], xyz[1], xyz[2], :]
            i+=1
        
        return np.mean(X,0)

    def _seed2seed_persub(self, sub_path, 
                          sort, 
                          randomize,
                          fname='mfmri_denoised', 
                          aname='mfmri_mean'):

        if len(fname.split('_')) > 2:
            # in the case we are not seeing only the mfmri_denoised but other combinations
            
            addname = os.path.join(*fname.split('_')[1:])
            self.addnamestr = "_"+addname.replace('/','')

        # make directories per subj
        sub = sub_path.split('/')[-1]
        os.makedirs(os.path.join(self.data_path, self.out_folder, sub), exist_ok=True)

        seeds = ['VR','VL','DR','DL']
        
        print(f"### Info: Loading subject {sub}... ")
        self.__load_data_persub(sub_path,fname,aname)
        print(f"### Info: Loading seeds... ")
        if randomize:
            seeds = np.random.permutation(seeds)

        vr_xyz = self.__load_seeds(seeds[0])
        vl_xyz = self.__load_seeds(seeds[1])
        dr_xyz = self.__load_seeds(seeds[2])
        dl_xyz = self.__load_seeds(seeds[3])

        vr = self.__ts_to_average(vr_xyz)
        vl = self.__ts_to_average(vl_xyz)
        dr = self.__ts_to_average(dr_xyz)
        dl = self.__ts_to_average(dl_xyz)

        Xave = np.vstack([vr,vl,dr,dl])
        # save Xave
        np.save(os.path.join(self.data_path, self.out_folder, sub, 'Xave'),Xave)

        corr_ave = np.corrcoef(Xave)
        # corrs_ave_all += corr_ave
        
        if randomize:
            np.save(os.path.join(self.data_path, self.out_folder,sub, 'corr_mat_rand'), corr_ave)

        # plot single subject correlations
        # plotthis = False
        # if plotthis:
        if not randomize:
            self._plot_FCmatrix_seed2seed(corr_ave, sub, 'corr_mat')

            # then consider all seeds
            all_seeds_xyz = np.vstack([vr_xyz,vl_xyz,dr_xyz,dl_xyz])
            print("Generating seed-to-seed connectivity matrix...")
            # generate seed-to-seed connectivity matrix per subject
            X = np.zeros((all_seeds_xyz.shape[0], self.f_img_data.shape[-1]))
            i = 0
            all_seeds_xyz_list = []
            zs = []
            for xyz in all_seeds_xyz:
                all_seeds_xyz_list.append(tuple(xyz))
                X[i, :]  = self.f_img_data[xyz[0], xyz[1], xyz[2], :]
                # all time points
                i+=1
                zs.append(str(xyz[2]))

            mat_corr_seeds = np.corrcoef(X)
            mat_corr = self.__plot_FCmatrix(mat_corr_seeds,[zs,vr.shape[0]],sub,sorted=sort)

            # # plot debug:
            # plt.figure()
            # plt.plot(self.nf_img_data[xyz[0], xyz[1], xyz[2], :],'b-')
            # plt.plot(self.f_img_data[xyz[0], xyz[1], xyz[2], :],'r-')
            # plt.ylabel('Signal intensity', fontsize=15)
            # plt.xlabel('Time (nr. of volumes)', fontsize=15)
            # #plt.show()
            # plt.savefig(os.path.join(self.data_path, self.out_folder, "Debug_timecourses"))
            # plt.close()

    def seed2seed_allsubsP(self, fname="mfmri_denoised", sort=True, randomize=False):
        # run in Parallel the seed2seed correlation computation per subject
        # in this way each subject correlation matrix is stored
        
        ### USE only the list of subjects that have that file (e.g. outliers 1 subject doesnt)
        # if fname == "mfmri_denoised_outl":
        #     # these three have not outliers, for the other combinations, it should be 
        #     subjects_paths = [sub for sub in self.subjects_paths if sub.split('/')[-1] not in ["LU_VG", "LU_BN", "LU_NB"]]    
        # else:
        #     subjects_paths = self.subjects_paths
        subjects_paths = self.subjects_paths
        
        Parallel(n_jobs=self.n_jobs,
                 verbose=10)(delayed(self._seed2seed_persub)(sub_path, sort, randomize,fname) \
                 for sub_path in subjects_paths)

        if len(fname.split('_')) > 2 and not randomize: #"st" in fname.split('_'):
            # in the case we are not seeing only the mfmri_denoised but other combinations
            addname = os.path.join(*fname.split('_')[1:])
            self.addnamestr = "_"+addname.replace('/','')
            
        # go through all saved correlation matrices and compute a population average
        if randomize:
            load_cormat_name = 'corr_mat_rand' 
        else:
            load_cormat_name = 'corr_mat'

        corrs_ave_all = np.zeros((4,4))
        for i_s in tqdm(range(len(subjects_paths))):
            sub_path = subjects_paths[i_s]
            sub = sub_path.split('/')[-1]

            corrs_ave_all += np.load(os.path.join(self.data_path, self.out_folder,sub, f'{load_cormat_name}{self.addnamestr}.npy'))

        # normalize by the number of subject
        corrs_ave_all = corrs_ave_all/len(self.subjects_paths)
        if not randomize:
            self._plot_FCmatrix_seed2seed(corrs_ave_all, 'all', 'averaged_corrs')

        return corrs_ave_all


    def _plot_FCmatrix_seed2seed(self, mat_corr, sub, name):
    
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
            fig.savefig(os.path.join(self.data_path, self.out_folder, name+self.addnamestr))
            np.save(os.path.join(self.data_path, self.out_folder,f'corrs_ave_all{self.addnamestr}'), mat_corr)
        else:
            fig.savefig(os.path.join(self.data_path, self.out_folder,sub, name+self.addnamestr))
            np.save(os.path.join(self.data_path, self.out_folder,sub, f'corr_mat{self.addnamestr}'), mat_corr)

        plt.close(fig)

    def __plot_FCmatrix(self, mat_corr_seeds, ticks_info, sub, sorted=True):
        """THis plotting function is a bit useless in terms of interpretability. It simply
        plots the seed2seed matrix that changes size according to the number of seeds considered."""

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
            filename = f'FCmatrix_sortedZ{self.addnamestr}'

        else:
            mat_corr = mat_corr_seeds
            plt.xticks([start,sec,thi,last],['VR','VL', 'DR', 'DL'], fontsize=10 )
            plt.yticks([start,sec,thi,last],['VR','VL', 'DR', 'DL'], fontsize=10 )
            filename = f'FCmatrix{self.addnamestr}'
        
        plt.imshow(mat_corr,vmax=0.3)
        plt.colorbar()

        figure.savefig(os.path.join(self.data_path, self.out_folder,sub,filename))
        plt.close(figure)


    # def seed2seed_allsubs(self, sort=True, randomize=False):
    #     # sort input changes only the seed2seed matrix (all seed correlation plotted) 
    #     # if True, it sorts by z 
    #     seeds = ['VR','VL','DR','DL']
    #     corrs_ave_all = np.zeros((4,4))

    #     for i_s in range(len(self.subjects_paths)):

    #         sub_path = self.subjects_paths[i_s]
    #         sub = sub_path.split('/')[-1]
    #         print(f"### Info: Loading subject {sub}... ")
    #         self.__load_data_persub(sub_path)
    #         print(f"### Info: Loading seeds... ")
    #         if randomize:
    #             seeds = np.random.permutation(seeds)

    #         vr_xyz = self.__load_seeds(seeds[0])
    #         vl_xyz = self.__load_seeds(seeds[1])
    #         dr_xyz = self.__load_seeds(seeds[2])
    #         dl_xyz = self.__load_seeds(seeds[3])

    #         vr = self.__ts_to_average(vr_xyz)
    #         vl = self.__ts_to_average(vl_xyz)
    #         dr = self.__ts_to_average(dr_xyz)
    #         dl = self.__ts_to_average(dl_xyz)

    #         Xave = np.vstack([vr,vl,dr,dl])
    #         corr_ave = np.corrcoef(Xave)
    #         corrs_ave_all += corr_ave
    #         # plot single subject correlations
    #         if not randomize:
    #             self._plot_FCmatrix_seed2seed(corr_ave, sub, 'corr_mat')

    #         # consider all seeds
    #         all_seeds_xyz = np.vstack([vr_xyz,vl_xyz,dr_xyz,dl_xyz])
    #         print("Generating seed-to-seed connectivity matrix...")
    #         # generate seed-to-seed connectivity matrix per subject
    #         X = np.zeros((all_seeds_xyz.shape[0], self.f_img_data.shape[-1]))
    #         i = 0
    #         all_seeds_xyz_list = []
    #         zs = []
    #         for xyz in all_seeds_xyz:
    #             all_seeds_xyz_list.append(tuple(xyz))
    #             X[i, :]  = self.f_img_data[xyz[0], xyz[1], xyz[2], :]
    #             # all time points
    #             i+=1
    #             zs.append(str(xyz[2]))

    #         mat_corr_seeds = np.corrcoef(X)
    #         if not randomize:
    #             mat_corr = self.__plot_FCmatrix(mat_corr_seeds,[zs,vr.shape[0]],sub,sorted=sort)
        
    #     # normalize by the number of subject
    #     corrs_ave_all = corrs_ave_all/len(self.subjects_paths)
    #     if not randomize:
    #         self._plot_FCmatrix_seed2seed(corrs_ave_all, 'all', 'averaged_corrs')

    #     return corrs_ave_all

########################################################################
# ********** sFC parcellation/atlas based **********
########################################################################

### DEFINITION OF CLASS TO PERFORM STATIC FUNCTIONAL CONNECTIVITY WITH A PARCELLATION 
class sFCParcellation(object):
    """
    sFCParcellation performs static functional connectivity at the parcellation level: so given
    some areas of interest from the template, it will perform the analyses on the normalized images.

    NOTE: The fmri images need to be normalized with spline interpolation.

    """

    def __init__(self, data_path, 
                 parcellation,
                 subset_subs=[], 
                 subIDs_to_exclude=[],
                 TR = 2.5,
                 out_folder = "sFC_results",
                 PAM50_path = "/home/iricchi/sct_5.3.0/data/PAM50/template/PAM50_t2_lumbar_crop.nii.gz"
                 ):

        """ Mandatory input is the data_path and the parcellation path that contains the list of 
        nii files (masks) for the regions of interest.
        
        NOTE: the variable parcellation must contain the full path!

        Optionals are:
        1) the list of subjects IDs,
        2) subIDs_to_exclude that collects all the IDs of subjects to exclude,
        3) TR,
        4) PAM50 template path. 
        """
        super(sFCParcellation, self).__init__()
        self.data_path = data_path
        self.parcellation = parcellation
        self.subset_subs = subset_subs
        self.subIDs_to_exclude = subIDs_to_exclude
        self.out_folder = out_folder
        self.TR = TR
        self.PAM50_path = PAM50_path
        self.__init_configs()

    def __init_configs(self):
        """Initialze variables"""
        self.func = "func"   # name of the functional folder
        # self.temporal_filter = True
        self.n_jobs = -2
        
        if len(self.subset_subs) == 0:
            self.subjects_paths = glob(os.path.join(self.data_path, "LU_*"))
        else:
            self.subjects_paths = [os.path.join(self.data_path, sub) for sub in self.subset_subs]
        
        # Exclude subjects:
        if len(self.subIDs_to_exclude) != 0:
            self.subjects_paths = [sub for sub in self.subjects_paths if sub.split('/')[-1] not in self.subIDs_to_exclude]

        # make outputdir
        if len(self.parcellation.split('/')[-1]) == 0:
            self.parc_name = self.parcellation.split('/')[-2]
        else:
            self.parc_name = self.parcellation.split('/')[-1]
        
        print(f"### Info: using parcellation {self.parc_name}")
        self.out_dir_path = os.path.join(self.data_path, self.out_folder,'Template_res',self.parc_name)
        os.makedirs(self.out_dir_path, exist_ok=True)

        # List of ROIs nii files
        self.mask_rois = glob(os.path.join(self.parcellation, "*.nii.gz"))
        print(f"### Info: Using {len(self.mask_rois)} regions in the parcellation...")

        self.img_data = nib.load(self.PAM50_path).get_fdata()
        

    # def temporal_filter2_vol(self, Wn=[0.01,0.13]):
    #     """img is the input img loaded with nibabel, Wn is the widnow for the bandpass:
    #     [0.01,0.13]
    #     """
    #     print("### Info: caching the img in memory... This might take a while ...")
    #     start = time.time()
    #     f_img_data = self.img.get_fdata()
    #     print(f"### Info: img loaded in {time.time()-start} s")
        
    #     rf_img_data = f_img_data.reshape(f_img_data.shape[0]*f_img_data.shape[1]*f_img_data.shape[2],f_img_data.shape[-1])
    #     b,a = cheby2(4, 30, Wn, 'bandpass', fs=1./self.TR)
    #     filtered_data = filtfilt(b,a,rf_img_data)
    #     filtered_data = filtered_data.reshape(f_img_data.shape[0],f_img_data.shape[1],f_img_data.shape[2],f_img_data.shape[-1])
    #     img_out = Nifti1Image(filtered_data, self.img.affine, self.img.header)
    #     self.f_img_data = f_img_data

    #     return img_out


    def __load_data_persub(self, sub_path, fname='mfmri_denoised_n_spline.nii.gz'):

        print(f"### Info: Loading subject {sub_path} ...")
        sub_path_func = os.path.join(sub_path,  self.func, fname) # denoised fmri to perform sFC
        self.img = nib.load(sub_path_func)

        
        ### Apply temporal filter on timeserieses (fMRI)
        # if self.temporal_filter:
        #     img_filt = self.temporal_filter2_vol()
        # else:
        #     img_filt = np.asanyarray(self.img.dataobj)

        # get arrays
        self.f_img_data = self.img.get_fdata()
        print(f"Subject loaded: {sub_path}.")


    def __roi_to_average_ts(self, masked_roi):

        # masked_img3d = cv2.bitwise_and(masked_roi,self.img_data)
        mask_x, mask_y, mask_z = np.where(masked_roi!=0)
        # from time serieses to mean
        X = np.zeros((len(mask_x), self.f_img_data.shape[-1]))
        i = 0 

        print("### Info: computing average of voxels in the ROI...")
        for x,y,z in zip(mask_x, mask_y, mask_z):
            X[i,:] = self.f_img_data[x, y, z, :]
            i+=1

        return np.mean(X,0)

    def _compute_sFC_parc(self, sub_path, fname='mfmri_denoised_n_spline.nii.gz'):
        
        # Load data
        self.__load_data_persub(sub_path, fname)

        sub = sub_path.split("/")[-1]
        print(f"### Info: computing subject {sub}")

        ## init matrix
        X = np.zeros((len(self.mask_rois),self.f_img_data.shape[-1]))
        ## Iterate over the parcellation ROIs
        i = 0
        roi_names = []
        for roi in self.mask_rois:
            roi_names.append(roi.split("/")[-1].split(".")[0])
            roi_array = nib.load(roi).get_fdata()
            X[i,:] = self.__roi_to_average_ts(roi_array)
            i += 1
        
        self.roi_names = roi_names

        corr_mat = np.corrcoef(X)
        # save and plot
        figure=plt.figure(dpi=200)
        plt.imshow(corr_mat)
        plt.colorbar()
        plt.xticks(np.arange(len(self.mask_rois)), roi_names, fontsize=10)
        plt.yticks(np.arange(len(self.mask_rois)), roi_names, fontsize=10)
        os.makedirs(os.path.join(self.out_dir_path, sub), exist_ok=True)
        figure.savefig(os.path.join(self.out_dir_path, sub, "corr_mat"))
        plt.close(figure)

        # save matrix
        np.save(os.path.join(self.out_dir_path, sub, 'corr_mat'), corr_mat)

    def sFC_parcelled_sub(self,fname='mfmri_denoised_n_spline.nii.gz'):
        
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._compute_sFC_parc)(sub_path,fname) \
                 for sub_path in self.subjects_paths)


    def sFC_parcelled_allsub(self, sorted=True):

        print("### Info: Loading subjects correlations and AVERAGING subjects...")
        corrs_ave_all = np.zeros((len(self.mask_rois),len(self.mask_rois),len(self.subjects_paths)))
        for i_s in tqdm(range(len(self.subjects_paths))):
            sub_path = self.subjects_paths[i_s]
            sub = sub_path.split('/')[-1]

            corrs_ave_all[:,:,i_s]= np.load(os.path.join(self.out_dir_path, sub, 'corr_mat.npy'))

        roi_names = []
        for roi in self.mask_rois:
            roi_names.append(roi.split("/")[-1].split(".")[0])

        return corrs_ave_all, roi_names