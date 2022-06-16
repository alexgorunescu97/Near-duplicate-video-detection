#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:26:11 2022

@author: alex
"""

from utils import *
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist, euclidean

cc_dataset = pk.load(open('output_data/datasets/cc_web_video.pickle', 'rb'))

cc_dataset_train = {}
cc_dataset_dev = {}

ind_train, ind_dev =  train_test_split(range(24), test_size=0.3, random_state=42)

queries_train, queries_dev = np.array(cc_dataset['queries'])[ind_train], np.array(cc_dataset['queries'])[ind_dev]
gt_train, gt_dev = np.array(cc_dataset['ground_truth'])[ind_train], np.array(cc_dataset['ground_truth'])[ind_dev]

cc_dataset_train['ground_truth'], cc_dataset_dev['ground_truth'] = gt_train, gt_dev
cc_dataset_train['queries'], cc_dataset_dev['queries'] = queries_train, queries_dev
cc_dataset_train['index'], cc_dataset_dev['index'] = cc_dataset['index'], cc_dataset['index']

with open('output_data/datasets/cc_dataset_train.pickle', 'wb') as handle:
    pk.dump(cc_dataset_train, handle)
    
with open('output_data/datasets/cc_dataset_dev.pickle', 'wb') as handle:
    pk.dump(cc_dataset_dev, handle)
    
    
    
triplets = []

features = np.load("output_data/vgg/cc_web_video_features.npy")  

for i, query in enumerate(gt_train):
    print(f"Starting processing for query {i}")
    query_key = queries_train[i]
    ndvr_videos = []
    distractors = []
    for video in query.keys():
        
        if query[video] in 'ESLMV':
            ndvr_videos.append(video)
        else:
            distractors.append(video)
    
    for j, other_query in enumerate(gt_train):
        
        if other_query != query:
            
            other_query_key = queries_train[j]
            
            for video in other_query.keys():
                
                distractors.append(video)
                
    distractors = list(set([distractor for distractor in distractors if distractor is not query_key and distractor not in ndvr_videos]))
    
    distractor_features = features[distractors]
    
    query_1 = features[query_key]
    
    print(f"Starting triplet extraction for query {i}")
    for ndvr_video in ndvr_videos:
        
        query_2 = features[ndvr_video]
        
        pair_distance = euclidean(query_1, query_2)
        negative_distances = cdist(np.array([query_1, query_2]), distractor_features, metric='euclidean')

        hard_negatives = np.where(negative_distances[0] < pair_distance)[0]
        triplets += [[query_key, ndvr_video, distractors[negative]] for negative in hard_negatives]

        hard_negatives = np.where(negative_distances[1] < pair_distance)[0]
        triplets += [[ndvr_video, query_key, distractors[negative]] for negative in hard_negatives]
        
np.save('cc_vgg_triplets', triplets)