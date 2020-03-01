#!/usr/bin/env python
# coding: utf-8

#importing the necessary modules
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split

#defining functions to calculate required features 
def power(epoch,n):
    return np.sum(epoch*epoch)/n

def mean(epoch,n):
    return np.sum(epoch)/n

def energy(epoch,n):
    return np.sum(epoch*epoch)

#Creating an empty dataframe with features defined above for each band and each channel
df_columns=[]
channels=["C3","C4"]
bands = ["beta","mu"]
features = [
    "Power",
    "Mean",
    "Energy",
]
for bd in bands:
    for ch in channels:
        for feat in features:
            df_columns.append(ch+"_"+bd+"_"+feat)
df_columns.append("label")
df = pd.DataFrame(columns=df_columns)

#looping over runs : 4,8, and 12 for each subject for left fist-right fist motor imagery
runs=[4,8,12]
for subject in range(1,110):
    #loading data and concatenating each user's runs into a single raw file
    raw_fnames = mne.datasets.eegbci.load_data(subject=subject,runs=runs)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])

    #remove trailing dots from channel names and pick channels C3 and C4 for analysis
    raw.rename_channels(lambda x: x.strip('.'))
    raw = raw.pick_channels(ch_names=['C3','C4'])
    
    raw = raw.set_eeg_reference('average', projection=False) #apply common average reference to remove noise

    #band-passing raw through mu band
    raw_mu = raw.copy()
    raw_mu.filter(8., 12., fir_design='firwin', skip_by_annotation='edge',)

    #band passing raw through beta band
    raw_b = raw.copy()
    raw_b.filter(16., 24., fir_design='firwin', skip_by_annotation='edge')

    #choosing events left hand and right hand motor imagery
    events = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')

    #epochs in beta band
    epochs_b = mne.Epochs(raw_b, events[0], events[1],tmax=4.0, proj=True, picks=picks,
                          baseline=None, preload=True)
    #epochs in mu band
    epochs_mu = mne.Epochs(raw_mu, events[0],events[1],tmax=4.0, proj=True, picks=picks,
                           baseline=None, preload=True)
    
    #recropping epochs to get data from 1s to 3s in each epoch
    ep_b = epochs_b.copy().crop(1.0,3.0)
    ep_m = epochs_mu.copy().crop(1.0,3.0)

    #seperating left and right labels
    left_b = epochs_b['T1'].get_data()
    right_b = epochs_b['T2'].get_data()
    left_mu = epochs_mu['T1'].get_data()
    right_mu = epochs_mu['T2'].get_data()
    
    #defining feature dictionary
    features = {
    "C3_beta_Power":0,
    "C3_beta_Mean":0,
    "C3_beta_Energy":0,
    "C4_beta_Power": 0,
    "C4_beta_Mean":0,
    "C4_beta_Energy":0,
    "C3_mu_Power":0,
    "C3_mu_Mean":0,
    "C3_mu_Energy":0,
    "C4_mu_Power":0,
    "C4_mu_Mean":0,
    "C4_mu_Energy":0,
    "label":False,
    }
    
    #adding rows to dataframe
    for i in range(left_b.shape[0]):
        n = left_b[i][0].size
        features['C3_beta_Power'] = power(left_b[i][0],n)
        features['C3_beta_Mean'] = mean(left_b[i][0],n)
        features['C3_beta_Energy'] = energy(left_b[i][0],n)
        features['C4_beta_Power'] = power(left_b[i][1],n)
        features['C4_beta_Mean'] = mean(left_b[i][1],n)
        features['C4_beta_Energy'] = energy(left_b[i][1],n)
        features['C3_mu_Power'] = power(left_mu[i][0],n)
        features['C3_mu_Mean'] = mean(left_mu[i][0],n)
        features['C3_mu_Energy'] = energy(left_mu[i][0],n)
        features['C4_mu_Power'] = power(left_mu[i][1],n)
        features['C4_mu_Mean'] = mean(left_mu[i][1],n)
        features['C4_mu_Energy'] = energy(left_mu[i][1],n)
        features['label'] = False
        df = df.append(features,ignore_index=True)
    
    for i in range(right_b.shape[0]):
        n = right_b[i][0].size
        features['C3_beta_Power'] = power(right_b[i][0],n)
        features['C3_beta_Mean'] = mean(right_b[i][0],n)
        features['C3_beta_Energy'] = energy(right_b[i][0],n)
        features['C4_beta_Power'] = power(right_b[i][1],n)
        features['C4_beta_Mean'] = mean(right_b[i][1],n)
        features['C4_beta_Energy'] = energy(right_b[i][1],n)
        features['C3_mu_Power'] = power(right_mu[i][0],n)
        features['C3_mu_Mean'] = mean(right_mu[i][0],n)
        features['C3_mu_Energy'] = energy(right_mu[i][0],n)
        features['C4_mu_Power'] = power(right_mu[i][1],n)
        features['C4_mu_Mean'] = mean(right_mu[i][1],n)
        features['C4_mu_Energy'] = energy(right_mu[i][1],n)
        features['label'] = True
        df = df.append(features,ignore_index=True)

#shuffling pandas dataframe
df = df.sample(frac=1).reset_index(drop=True)

#train test split
train, test = train_test_split(df, test_size=0.2)

train.to_pickle('/home/chinmay/Projects/EEG_Classification/train.pkl')
test.to_pickle('/home/chinmay/Projects/EEG_Classification/test.pkl')
