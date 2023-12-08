#!/usr/bin/python

""" This script contains useful function used in sFC_analysis.py.

Author: Ilaria Ricchi
"""
import numpy as np
from scipy.signal import butter, filtfilt, cheby2, lfilter
from nibabel import Nifti1Image

### List Filters functions

def temporal_filter(timeseries, Wn):
    b,a = butter(1,Wn,'bandpass')
    return filtfilt(b,a,mid_vox_ts)



def temporal_filter2(timeseries, Wn, order=4, dB=20):
    b,a = cheby2(order, dB, Wn, 'bandpass')
    return filtfilt(b,a,timeseries)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def online_filter(img, LP_freq=0.01,HP_freq=0.13):

    TR = 2.5
    sampling_rate = 1./TR
    f_img_data = img.get_fdata()
    timepoints = f_img_data.shape[-1]
     
    F = np.zeros((timepoints))
    lowidx = timepoints // 2 + 1 # "/" replaced by "//"
    lowidx = int(np.round(HP_freq / sampling_rate * timepoints)) # "np.round(..." replaced by "int(np.round(..."
    highidx = 0
    highidx = int(np.round(LP_freq / sampling_rate * timepoints)) # same
    F[highidx:lowidx] = 1
    F = ((F + F[::-1]) > 0).astype(int)
    if np.all(F == 1):
        filtered_data = f_img_data
    else:
        filtered_data = np.real(np.fft.ifftn(np.fft.fftn(f_img_data) * F))
    
    img_out = Nifti1Image(filtered_data, img.get_affine(),
                            img.get_header())
    return img_out


## THIS IS THE SELECTED ONE!
def temporal_filter2_vol(img, Wn):
    """img is the input img loaded with nibabel, Wn is the widnow for the bandpass:
    [0.01,0.13]
    """
    f_img_data = img.get_fdata()
    rf_img_data = f_img_data.reshape(f_img_data.shape[0]*f_img_data.shape[1]*f_img_data.shape[2],f_img_data.shape[-1])
    b,a = cheby2(4, 30, Wn, 'bandpass', fs=1./2.5)
    filtered_data = filtfilt(b,a,rf_img_data)
    filtered_data = filtered_data.reshape(f_img_data.shape[0],f_img_data.shape[1],f_img_data.shape[2],f_img_data.shape[-1])
    img_out = Nifti1Image(filtered_data, img.affine, img.header)
    
    return img_out


## Compute the average of voxels from the same seed region

def ts_to_average(seeds_xyz, f_img_data):

    X = np.zeros((seeds_xyz.shape[0], f_img_data.shape[-1]))
    i = 0 
    for xyz in seeds_xyz:
        X[i,:] = f_img_data[xyz[0], xyz[1], xyz[2], :]
        i+=1
    
    return np.mean(X,0)

