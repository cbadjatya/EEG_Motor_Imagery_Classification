#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:44:23 2020

@author: chinmay
"""
#code for performing logistic regression on EEG-Data

import pandas as pd

#importing train and test data
train = pd.read_pickle('/home/chinmay/Projects/EEG_Classification/train.pkl')
test = pd.read_pickle('/home/chinmay/Projects/EEG_Classification/test.pkl')

