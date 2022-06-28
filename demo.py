#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 22:22:06 2022

@author: alex
"""

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

from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import plotly.io as pio
import plotly.express as px
import pandas as pd

from tqdm import tqdm
from future.utils import lrange
from multiprocessing import Pool
from utils.cnn_utils import load_video
from utils import normalize
from scipy.spatial.distance import cdist
from models.dnn_model import DNN

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def feature_extraction_videos(model, cores, batch_sz, videos):
    """
      Function that extracts the intermediate CNN features
      of each video in a provided video list.

      Args:
        model: CNN network
        cores: CPU cores for the parallel video loading
        batch_sz: batch size fed to the CNN network
        videos: list of video to extract features
    """
    print('\nNumber of videos: ', len(videos))
    print('CPU cores: ', cores)
    print('Batch size: ', batch_sz)

    print('\nFeature Extraction Process')
    print('==========================')
    pool = Pool(cores)
    future_videos = dict()
    pbar = tqdm(lrange(np.max(list(videos.keys())) + 1), mininterval=1.0, unit='videos')
    global_features = []
    video_frames = []
    for video in pbar:
        if os.path.exists(videos[video]):
            video_name = os.path.splitext(os.path.basename(videos[video]))[0]
            if video not in future_videos:
                video_tensor = load_video(videos[video], model.desired_size)
            else:
                video_tensor = future_videos[video].get()
                del future_videos[video]

            # load videos in parallel
            for _ in lrange(cores - len(future_videos)):
                next_video = np.max(list(future_videos.keys())) + 1 \
                    if len(future_videos) else video + 1

                if next_video in videos and \
                    next_video not in future_videos and \
                        os.path.exists(videos[next_video]):
                    future_videos[next_video] = pool.apply_async(load_video,
                                                                 args=[videos[next_video], model.desired_size])

            video_frames.append(video_tensor.shape[0])
            # extract features
            video_features = model.extract(video_tensor, batch_sz)
            global_video_vector = get_global_vector(np.nan_to_num(video_features))
            
            global_features.append(global_video_vector)
    np.save('demo.npy', np.array(global_features).reshape(len(global_features), -1))    
    return np.array(global_features).reshape(len(global_features), -1),  np.array(video_frames)

def get_global_vector(features):
        X = normalize(features)
        X = X.mean(axis=0, keepdims=True)
        X = normalize(X)
        return X

def get_similarities_dict(queries, features, videos_info):
    """
      Function that generates video triplets from CC_WEB_VIDEO.

      Args:
        queries: indexes of the query videos
        features: global features of the videos in CC_WEB_VIDEO
        videos_info: dict of video info
      Returns:
        similarities: the similarities of each query with the videos in the dataset
    """
    similarities = {"Video": [], "Query": [], "Similarity": [], "Frames": []}
    dist = np.nan_to_num(cdist(features[queries], features, metric='euclidean'))
    for i, v in enumerate(queries):
        sim = np.round(1 - dist[i] / dist.max(), decimals=6)
        for d in sim.argsort():
            
            similarities["Video"].append(videos_info[d]["Name"])
            similarities["Query"].append(videos_info[v]["Name"])
            similarities["Similarity"].append(sim[d])
            similarities["Frames"].append(videos_info[d]["Frames"])
        # sim = np.round(1 - dist[i] / dist.max(), decimals=6)
        # similarities += [[(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]]
    return similarities

def get_video_info(videos, video_frames):
    
    video_info = {}
    
    for v in videos.keys():
        
        video_info[v] = {"Name": videos[v].split('/')[-1], "Frames": video_frames[v]}
        
    return video_info
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str, required=True,
                        help='Name of the network')
    parser.add_argument('-v', '--video_list', type=str, required=True,
                        help='List of videos to extract features')
    parser.add_argument('-cnn', '--cnn_model', type=str,
                        help='Path to the .ckpt file of the pre-trained CNN model.')
    parser.add_argument('-dnn', '--dnn_model', type=str,
                        help='Path to the .ckpt file of the pre-trained DNN model.')
    parser.add_argument('-c', '--cores', type=int, default=8,
                        help='Number of CPU cores for the parallel load of images or video')
    parser.add_argument('-b', '--batch_sz', type=int, default=32,
                        help='Number of the images fed to the CNN network at once')
    args = vars(parser.parse_args())


    if not args['cnn_model']:
        raise Exception('--cnn_model argument is not provided. It have to be provided when '
                        'Tensorflow framework is selected. Download: '
                        'https://github.com/tensorflow/models/tree/master/research/slim')
    elif '.ckpt' not in args['cnn_model']:
        raise Exception('--cnn_model argument is not a .ckpt file.')
    from models.cnn_model import CNN_tf
    model = CNN_tf(args['network'].lower(), args['cnn_model'])

    print('\nCNN model has been built and initialized')
    print('Architecture used: ', args['network'])
    
    videos = {i: video.strip() for i, video in enumerate(open(args['video_list']).readlines())}

    features, video_frames = feature_extraction_videos(model, args['cores'], args['batch_sz'], videos)
    
    features = np.load('demo.npy')
    
    triplet_model = DNN(features.shape[1],
                args['dnn_model'],
                load_model=True,
                trainable=False)
    
    embeddings = triplet_model.embeddings(features)
    
    video_info = get_video_info(videos, video_frames)
    
    similarities = get_similarities_dict([0, 3], embeddings, video_info)
    
    pio.renderers.default="browser"
    
    df = pd.DataFrame(similarities)

    fig = px.bar(df, y="Query", x="Frames", color="Similarity", orientation="h",
                  color_continuous_scale="Bluered_r", hover_name="Video")

    fig.show()
    
    