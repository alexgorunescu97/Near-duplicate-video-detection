# Copyright 2018 Giorgos Kordopatis-Zilos. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Implementation of the evaluation process based on CC_WEB_VIDEO dataset.
"""

from __future__ import division
from __future__ import print_function

import argparse

from utils import *
from models.dnn_model import DNN
from tqdm import tqdm
from scipy.spatial.distance import cdist


def calculate_similarities(queries, features, val_ind_map):
    """
      Function that generates video triplets from CC_WEB_VIDEO.

      Args:
        queries: indexes of the query videos
        features: global features of the videos in CC_WEB_VIDEO
      Returns:
        similarities: the similarities of each query with the videos in the dataset
    """
    similarities = []
    dist = np.nan_to_num(cdist(features[queries], features, metric='euclidean'))
    for i, v in enumerate(queries):
        sim = np.round(1 - dist[i] / dist.max(), decimals=6)
        similarities += [[(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s]) and (val_ind_map is None or s in val_ind_map)]]
    return similarities

def evaluate_cc_web(val_dataset, val_features, model, fusion='early', results_path=None, evaluation_features=None, positive_labels='ESLMV', val_ind_map=None, full_evaluation=True):

    print('Loading data...')
    cc_dataset = pk.load(open(val_dataset, 'rb'))
    cc_features = load_features(val_features)

    if fusion.lower() == 'early':
        print('Fusion type: Early')
        print('Extract video embeddings...')
        cc_embeddings = model.embeddings(cc_features)
    else:
        print('Fusion type: Late')
        print('Extract video embeddings...')

        assert evaluation_features is not None, \
            'Argument \'--evaluation_features\' must be provided for Late fusion'
        feature_files = load_feature_files(evaluation_features)

        cc_embeddings = np.zeros((len(cc_dataset['index']), model.embedding_dim))
        for i, video_id in enumerate(tqdm(cc_dataset['index'])):
            if video_id in feature_files:
                features = load_features(feature_files[video_id])
                embeddings = model.embeddings(normalize(features))
                embeddings = embeddings.mean(0, keepdims=True)
                cc_embeddings[i] = normalize(embeddings, zero_mean=False)

    print('\nEvaluation set file: ', val_features)
    print('Positive labels: ', positive_labels)

    print('\nEvaluation Results')
    print('==================')
    
    similarities = calculate_similarities(cc_dataset['queries'], cc_embeddings, val_ind_map)
    mAP_dml, pr_curve_dml, thresholds_cc_web_video_dml = evaluate(cc_dataset['ground_truth'], cc_dataset['index'], cc_dataset['queries'], similarities, results_path,
                                     positive_labels=positive_labels, all_videos=False, is_baseline=False)
    
    print('CC_WEB_VIDEO')
    print('DML mAP: ', mAP_dml)
    if full_evaluation:
    
        baseline_similarities = calculate_similarities(cc_dataset['queries'], cc_features, val_ind_map)
        mAP_base, pr_curve_base, thresholds_cc_web_video_base = evaluate(cc_dataset['ground_truth'], cc_dataset['index'], cc_dataset['queries'], baseline_similarities, results_path,
                                           positive_labels=positive_labels, all_videos=False, is_baseline=True)
        
        print('baseline mAP: ', mAP_base)
        plot_pr_curve(pr_curve_dml, pr_curve_base, 'CC_WEB_VIDEO')
    
        mAP_dml, pr_curve_dml, thresholds_cc_web_video_s_dml  = evaluate(cc_dataset['ground_truth'], cc_dataset['index'], cc_dataset['queries'], similarities, results_path,
                                         positive_labels=positive_labels, all_videos=True, is_baseline=False)
        mAP_base, pr_curve_base, thresholds_cc_web_video_s_base = evaluate(cc_dataset['ground_truth'], cc_dataset['index'], cc_dataset['queries'], baseline_similarities, results_path,
                                           positive_labels=positive_labels, all_videos=True, is_baseline=True)
        print('\nCC_WEB_VIDEO*')
        print('baseline mAP: ', mAP_base)
        print('DML mAP: ', mAP_dml)
        plot_pr_curve(pr_curve_dml, pr_curve_base, 'CC_WEB_VIDEO*')
    
    return mAP_dml    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-es', '--evaluation_set', type=str, required=True,
                        help='Path to the .npy file that contains the global '
                             'video vectors of the CC_WEB_VIDEO dataset')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to load the trained DML model')
    parser.add_argument('-f', '--fusion', type=str, default='Early',
                        help='Processed dataset. Options: Early and Late. Default: Early')
    parser.add_argument('-ef', '--evaluation_features', type=str,
                        help='Paths to the .npy files that contains the feature vectors '
                             'of the videos in the CC_WEB_VIDEO dataset. Each line of the '
                             'file have to contain the video id (name of the video file) '
                             'and the full path to the corresponding .npy file, separated '
                             'by a tab character (\\t)')
    parser.add_argument('-rs', '--results_path', type=str, required=True, help="Path for storing results")
    parser.add_argument('-pl', '--positive_labels', type=str, default='ESLMV',
                        help='Labels in CC_WEB_VIDEO datasets that '
                             'considered posetive. Default=\'ESLMV\'')
    args = vars(parser.parse_args())

    cc_features = load_features(args['evaluation_set'])
    
    print('Loading model...')
    model = DNN(cc_features.shape[1],
                args['model_path'],
                load_model=True,
                trainable=False)

    evaluate_cc_web('output_data/datasets/cc_web_video.pickle', args['evaluation_set'], model, fusion=args['fusion'], results_path=args['results_path'], evaluation_features=args['evaluation_features'], positive_labels=args['positive_labels'])

