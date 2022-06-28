# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 23:34:17 2022

@author: alexandru.gorunescu
"""

import os
import argparse
import numpy as np

from tqdm import tqdm
from scipy.spatial.distance import cdist, euclidean
from sklearn.model_selection import train_test_split
from utils.dnn_utils import load_dataset

excluded_negatives = [6171, 97284]

def triplet_generator_vcdb(dataset, core_dataset_ind, distractors_ind, vcdb_features, threshold):
    
    core_dataset = vcdb_features[:528]
    distractors = vcdb_features[528:]

    print('\nVCDB Triplet Generation')
    print('=======================')
    triplets = []
    for video_pair in tqdm(dataset['video_pairs']):
        if video_pair['videos'][0] in core_dataset_ind and video_pair['videos'][1] in core_dataset_ind and video_pair['overlap'][0] > threshold and video_pair['overlap'][1] > threshold:
            video1 = core_dataset[video_pair['videos'][0]]
            video2 = core_dataset[video_pair['videos'][1]]

            pair_distance = euclidean(video1, video2)
            negative_distances = cdist(np.array([video1, video2]), distractors, metric='euclidean')

            hard_negatives = np.where(negative_distances[0] < pair_distance)[0] + 528
            triplets += [[video_pair['videos'][0], video_pair['videos'][1], negative]
                         for negative in hard_negatives if negative not in excluded_negatives and negative in distractors_ind]

            hard_negatives = np.where(negative_distances[1] < pair_distance)[0] + 528
            triplets += [[video_pair['videos'][1], video_pair['videos'][0], negative]
                         for negative in hard_negatives if negative not in excluded_negatives and negative in distractors_ind]
            
    return triplets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--global_features', type=str, required=True, 
                        help="Path to the .npy file containing global features")
    parser.add_argument('-s', '--split', type=float, default=0.3,
                        help="Train-dev split")
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory where the generated files will be stored')
    parser.add_argument('-t', '--overlap_threshold', type=float, default=0.8,
                        help='Overlap threshold over which the video pairs in VCDB dataset'
                             'are considered positives. Default: 0.8')
    args = vars(parser.parse_args())
        
    dataset = load_dataset('vcdb')
    vcdb_features = np.load(args['global_features']).astype(np.float32)
    
    core_dataset_train_ind, core_dataset_dev_ind = train_test_split(range(528), test_size=args['split'], random_state=42)
    distractors_train_ind, distractors_dev_ind = train_test_split(range(528, vcdb_features.shape[0]), test_size=args['split'], random_state=42)
    
    triplets_train = triplet_generator_vcdb(dataset, core_dataset_train_ind, distractors_train_ind, vcdb_features, args['overlap_threshold'])
    triplets_dev = triplet_generator_vcdb(dataset, core_dataset_dev_ind, distractors_dev_ind, vcdb_features, args['overlap_threshold'])

    np.save(os.path.join(args['output_dir'], 'vcdb_triplets_train'), triplets_train)
    np.save(os.path.join(args['output_dir'], 'vcdb_triplets_dev'), triplets_dev)

    

