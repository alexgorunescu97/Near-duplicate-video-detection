#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:47:53 2022

@author: alex
"""

"""
Script that calculate video similarities for the videos in FIVR-200K dataset
"""

import os
import json
import argparse
import numpy as np
import scipy as sp
from model import DNN

from argparse import RawTextHelpFormatter
from sklearn.metrics import pairwise_distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', '--feature_file',
                        required=True,
                        help='File that contains the global features vectors of each video in the dataset.\n'
                             'The order of the feature vectors have to be the same with the videos contained '
                             'in the file provided to the \'--dataset_ids\' argument.\n'
                             'Only .npy and .mtx files are supported')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to load the trained DML model')
    parser.add_argument('-r', '--result_file',
                        required=True,
                        help='File where the results will be saved.')
    parser.add_argument('-a', '--annotations_file',
                        default='dataset/annotation.json',
                        help='File that contains the video annotations of the FIVR-200K dataset')
    parser.add_argument('-d', '--dataset_ids',
                        default='dataset/youtube_ids.txt',
                        help='File that contains the Youtube IDs of the videos in FIVR-200K dataset')
    parser.add_argument('-s', '--similarity_metric',
                        default='cosine',
                        help='Distance metric that will be used to calculate similarity.\n'
                             'The supported metrics can be found here:\n'
                             'https://scikit-learn.org/stable/modules/generated/'
                             'sklearn.metrics.pairwise_distances.html')
    args = parser.parse_args()

    # load global video features from given file
    print('Loading features from file:', args.feature_file)
    _, extension = os.path.splitext(args.feature_file)
    if extension == '.npy':
        features = np.load(args.feature_file)
    elif extension == '.mtx':
        features = sp.io.mmread(args.feature_file).tolil()
    else:
        raise Exception('Unknown file format of the provided feature file.'
                        'Please use only .npy or .mtx extensions.')

    # load the ids of the query videos and the dataset videos
    with open(args.annotations_file, 'r') as f:
        query_ids = list(json.load(f).keys())
    dataset = list(np.loadtxt(args.dataset_ids, dtype=str))
    
    
    # print('Loading model...')
    # model = DNN(features.shape[1],
    #             args.model_path,
    #             load_model=True,
    #             trainable=False)
    
    # print('Extract video embeddings...')
    # embeddings = model.embeddings(features)
    embeddings = features

    assert embeddings.shape[0] == len(dataset), 'Number of videos in the dataset is no equal to the ' \
                                              'number of embeddding vectors provided'

    # calculate similarities between each query and candidate video in FIVR-200K
    queries = embeddings[[dataset.index(q) for q in query_ids]]
    similarities = 1. - pairwise_distances(queries, embeddings, args.similarity_metric)

    # prepare result dictionary
    results = dict()
    print('Store results in file:', args.result_file)
    for j, query in enumerate(query_ids):
        query_results = dict(map(lambda v: (dataset[v], float(similarities[j, v])),
                                 np.where(similarities[j] > 0.0)[0]))
        del query_results[query]
        results[query] = query_results

    # save results in a json file
    with open(args.result_file, 'w') as f:
        json.dump(results, f, indent=1)