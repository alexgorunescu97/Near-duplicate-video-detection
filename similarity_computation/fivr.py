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
from tqdm import tqdm

from argparse import RawTextHelpFormatter
from sklearn.metrics import pairwise_distances
from models.dnn_model import DNN

def get_similarities(feature_file, result_file, annotations_file='dataset/annotation_dev.json', model=None, model_path=None, features=None, dataset_ids='dataset/youtube_ids.txt', dataset=None, similarity_metric="euclidean"):
    
    
    if features is None:
    
        # load global video features from given file
        print('Loading features from file:', feature_file)
        _, extension = os.path.splitext(feature_file)
        if extension == '.npy':
            features = np.load(feature_file)
        elif extension == '.mtx':
            features = sp.io.mmread(feature_file).tolil()
        else:
            raise Exception('Unknown file format of the provided feature file.'
                            'Please use only .npy or .mtx extensions.')

    # load the ids of the query videos and the dataset videos
    with open(annotations_file, 'r') as f:
        query_ids = list(json.load(f).keys())
        
    if dataset is None:
        dataset = list(np.loadtxt(dataset_ids, dtype=str))
    
    print('Extract video embeddings...')
    if model is not None:
        embeddings = model.embeddings(features)
    elif model_path:
        model = DNN(features.shape[1],
                    model_path,
                    load_model=True,
                    trainable=False)
        
        embeddings = np.zeros((features.shape[0], model.embedding_dim))
        for i, feature in enumerate(tqdm(features)):
            
            embedding = model.embeddings(np.reshape(feature, (-1, 4096)))
            embeddings[i] = embedding

    assert embeddings.shape[0] == len(dataset), 'Number of videos in the dataset is no equal to the ' \
                                              'number of embeddding vectors provided'

    # calculate similarities between each query and candidate video in FIVR-200K
    queries = embeddings[[dataset.index(q) for q in query_ids]]
    similarities = 1. - pairwise_distances(queries, embeddings, similarity_metric)

    # prepare result dictionary
    results = dict()
    for j, query in enumerate(query_ids):
        query_results = dict(map(lambda v: (dataset[v], float(similarities[j, v])),
                                 np.where(similarities[j] > 0.0)[0]))
        del query_results[query]
        results[query] = query_results
        
    
    if result_file:

        print('Store results in file:', result_file)
        # save results in a json file
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=1)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', '--feature_file',
                        required=True,
                        help='File that contains the global features vectors of each video in the dataset.\n'
                             'The order of the feature vectors have to be the same with the videos contained '
                             'in the file provided to the \'--dataset_ids\' argument.\n'
                             'Only .npy and .mtx files are supported')
    parser.add_argument('-r', '--result_file',
                        required=True,
                        help='File where the results will be saved.')
    parser.add_argument('-m', '--model_path', type=str,
                        help='Path to load the trained DML model')
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
    
    get_similarities(args.feature_file, args.result_file, annotations_file=args.annotations_file, model_path=args.model_path, dataset_ids=args.dataset_ids, similarity_metric=args.similarity_metric)
    
    