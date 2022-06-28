#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:03:11 2022

@author: alex
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist, euclidean


with open("dataset/annotation.json", "r") as f:
    annotations = json.load(f)
    
annotations = [{k: annotations[k]} for k in annotations.keys()]

ids = list(np.loadtxt("dataset/youtube_ids.txt", dtype=str))

annotations_train, annotations_dev = train_test_split(annotations, test_size=0.5, random_state=42)


annotations_dev_json = {}
for query in annotations_dev:
    query_key = list(query.keys())[0]
    annotations_dev_json[query_key] = query[query_key]
    
with open('dataset/annotation_dev.json', 'w') as f:
    json.dump(annotations_dev_json, f, indent=1)


annotations_train_json = {}
for query in annotations_train:
    query_key = list(query.keys())[0]
    annotations_train_json[query_key] = query[query_key]
    
with open('dataset/annotation_train.json', 'w') as f:
    json.dump(annotations_train_json, f, indent=1)
    

triplets = []

vgg_features = np.load("vgg.npy")  

for i, query in enumerate(annotations_train):
    print(f"Starting processing for query {i}")
    query_key = list(query.keys())[0]
    ndvr_videos = list(set(query[query_key].get("ND", []) + query[query_key].get("DS", [])))
    distractors = query[query_key].get("DI", [])
    
    for other_query in annotations_train:
        
        if other_query != query:
            
            other_query_key = list(other_query.keys())[0]
            
            videos = other_query[other_query_key]
            
            for label in videos.keys():
                
                distractors += videos[label]
                
    distractors = list(set([distractor for distractor in distractors if distractor is not query_key and distractor not in ndvr_videos]))
    distractor_ids = [ids.index(distractor) for distractor in distractors]
    
    
    distractor_features = vgg_features[distractor_ids]
    query_key_ind = ids.index(query_key)
    
    query_1 = vgg_features[query_key_ind]
    
    print(f"Starting triplet extraction for query {i}")
    for ndvr_video in ndvr_videos:
        
        ndvr_ind = ids.index(ndvr_video)
        
        query_2 = vgg_features[ndvr_ind]
        
        pair_distance = euclidean(query_1, query_2)
        negative_distances = cdist(np.array([query_1, query_2]), distractor_features, metric='euclidean')

        hard_negatives = np.where(negative_distances[0] < pair_distance)[0]
        triplets += [[query_key_ind, ndvr_ind, distractor_ids[negative]] for negative in hard_negatives]

        hard_negatives = np.where(negative_distances[1] < pair_distance)[0]
        triplets += [[ndvr_ind, query_key_ind, distractor_ids[negative]] for negative in hard_negatives]
        
np.save('fivr_triplets_full', triplets)
    
    
    

                

                
    



    


    
    
    


  

    