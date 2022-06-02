#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:41:41 2022

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
"""
Tensorflow implementation of the Deep Metric Learning training process.
"""

import os
import tqdm
import argparse
import numpy as np
import tensorflow as tf
import json

import matplotlib.pyplot as plt
from model import DNN
from calculate_similarities_fivr import get_similarities
from evaluate_fivr import evaluate
from future.utils import lrange

def plot_train_val_metrics(train_loss, epochs, results_path):
    
        fig_loss, ax_loss = plt.subplots(nrows=1, ncols=1)
        
        ax_loss.plot(epochs, train_loss)
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        
        loss_fig_path = os.path.join(results_path, 'loss.png')
        
        fig_loss.savefig(loss_fig_path)
        plt.close(fig_loss)
        
def train_dml_network(model, features, triplets, epochs, batch_sz, ids=None, val_set_ind=None, val_set=None, dropout_rate_1=1, dropout_rate_2=1):
    """
      Function that handles the training process.

      Args:
        model: the DML model
        features: the global video features
        triplets: the generated triplets for training
        epochs: the training epochs
        batch_sz: the batch size
    """
    
    EARLY_STOPPING_LIMIT = 10

    print('\nStart of DML Training')
    print('=====================')
    n_batch = triplets.shape[0] // batch_sz + 1
    max_val_map = [float('-inf'), 0]
    train_losses = []
    epochs_without_progress = 0
    
    for i in lrange(epochs):
        np.random.shuffle(triplets)
        pbar = tqdm.trange(n_batch,
                           desc='epoch {}'.format(i),
                           mininterval=1.0,
                           unit='batch')
        losses = np.zeros(n_batch)
        for j in pbar:
            triplet_batch = triplets[j * batch_sz: (j + 1) * batch_sz]
            train_batch = features[triplet_batch.reshape(-1)]

            _, loss, error = model.train(train_batch, dropout_rate_1, dropout_rate_2)
            losses[j] = loss

            pbar.set_postfix(loss=loss, error='{0:.2f}%'.format(error))
            
        mean_loss = np.mean(losses)
        train_losses.append(mean_loss)
        print(f'Training Loss for epoch {i}: {mean_loss}')
        
        if val_set is not None:
            ids_val = list(np.array(ids)[val_set_ind])
            features_val = features[val_set_ind]
            current_map = validate_dml_network(model, features_val, ids_val, val_set)
            if current_map > max_val_map[0]:
                max_val_map[0] = current_map
                max_val_map[1] = i
                epochs_without_progress = 0;
                model.save()
            else:
                epochs_without_progress += 1
            
            if epochs_without_progress == EARLY_STOPPING_LIMIT:
                print(f'Early stoppage at epoch {i}')
                break
        else:    
            model.save()
    
    return max_val_map, train_losses, i
        
        
def validate_dml_network(model, features, ids, val_set):
    """
      Function that handles the validating process.

      Args:
        model: the DML model
        features: the global video features
    """

    print('\nStart of DML Validating')
    print('=====================')
    
    
    results = get_similarities(None, None, model=model, features=features, dataset=ids)
    # run the evaluation process
    mAP, _ = evaluate(val_set, results, ['ND', 'DS'], ids, True)
    
    
    print('Total queries: {}\t\tmAP={:.4f}'.format(len(mAP), np.mean(mAP)))
    return np.mean(mAP)
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Path to the .npy file that contains the global '
                             'video vectors for the fivr dataset')
    parser.add_argument('-ttr', '--train_triplets', type=str, required=True,
                        help='Path to the .npy file that contains the training triplets')
    parser.add_argument('-vtr', '--validation_set', type=str,
                        help='Path to the file that contains the validation set')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Directory where the generated files will be stored')
    parser.add_argument('-es', '--evaluation_set', type=str,
                        help='Path to the .npy file that contains the global '
                             'video vectors of the evaluation set')
    parser.add_argument('-et', '--evaluation_triplets', type=str,
                        help='Path to the .npy file that contains the triplets '
                             'generated based on the evaluation set')
    parser.add_argument('-ij', '--injection', type=int, default=10000,
                        help='Number of injected triplets generated from the '
                             'evaluation set. It is only applied when the '
                             'evaluation_set is provided. Default: 10000, Max:10000')
    parser.add_argument('-l', '--layers', default='1000,500,250',
                        help='Number of neuron for each layer of the DML network, '
                             'separated by a comma \',\'. Default: 1000,500,250')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs to train the DML network. Default: 10')
    parser.add_argument('-b', '--batch_sz', type=int, default=1024,
                        help='Number of triplets fed every training iteration. '
                             'Default: 1024')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6,
                        help='Learning rate of the DML network. Default: 10^-6')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5,
                        help='Regularization parameter of the DML network. Default: 10^-5')
    parser.add_argument('-g', '--gamma', type=float, default=1.0,
                        help='Margin parameter of the distance between the two pairs of '
                             'every triplet. Default: 1.0')
    parser.add_argument('-vr', '--validation_results', default='', help="Path for validation results")
    parser.add_argument('-d', '--dataset_ids',
                        default='dataset/youtube_ids.txt',
                        help='File that contains the Youtube IDs of the videos in FIVR-200K dataset')
    
    args = vars(parser.parse_args())

    print('Train set file: ', args['dataset'])
    print('Train triplet file: ', args['train_triplets'])
    print('Validation set: ',  args['validation_set'] if args['validation_set'] else "No validation set")

    print('loading data...')
    dataset = np.load(args['dataset']).astype(np.float32)
    train_triplets = np.load(args['train_triplets'])

    if args.get('evaluation_set'):
        args['injection'] = np.min([args['injection'], 10000])
        print('Evaluation set file: ', args['evaluation_set'])
        print('Evaluation triplet file: ', args['evaluation_triplets'])
        print('Injected triplet: ', args['injection'])
        print('loading data...')
        evaluation_set = np.load(args['evaluation_set']).astype(np.float32)
        eval_triplets = np.load(args['evaluation_triplets']).astype(np.int) + len(dataset)
        np.random.shuffle(eval_triplets)
        dataset = np.concatenate([dataset, evaluation_set], axis=0)
        train_triplets = np.concatenate([train_triplets, eval_triplets[:args['injection']]], axis=0)

    try:
        layers = [int(l) for l in args['layers'].split(',') if l]
    except Exception:
        raise Exception('--layers argument is in wrong format. Specify the number '
                        'of neurons in each layer separated by a comma \',\'')
     
            
    if args['validation_set']:
        
        # load the ids of the query videos and the dataset videos
        with open(args["validation_set"], 'r') as f:
            val_set = json.load(f)
            
        ids = list(np.loadtxt(args["dataset_ids"], dtype=str))
        
        val_set_ind = []

        for query in val_set.keys():
            query_ind = ids.index(query)
            val_set_ind.append(query_ind)
            for label in val_set[query].keys():
                val_set_ind += [ids.index(video) for video in val_set[query][label]]

        with open('hyperparameters.txt', 'r') as f, open('conf.txt', 'a') as g:
            
            for i, line in enumerate(f.readlines()):
                
                tf.keras.backend.clear_session()
                
                line_split = line.split(';')
                layers_str, dropout_rate_1, dropout_rate_2, learning_rate, batch_size, weight_decay, gamma, reg = line_split[0], float(line_split[1]), float(line_split[2]), float(line_split[3]), int(line_split[4]), float(line_split[5]), float(line_split[6]), line_split[7] 
                
                model_path = os.path.join(args['model_path'], f'model_dev_fivr_{i + 33}')
                os.mkdir(model_path)
            
                model = DNN(dataset.shape[1],
                            os.path.join(model_path, f'model_dev_fivr_{i + 33}'),
                            hidden_layer_sizes=[int(l) for l in layers_str.split(',') if l],
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            gamma=gamma,
                            reg=reg.strip())
                
                max_val_map, train_loss, epochs = train_dml_network(model, dataset, train_triplets, args['epochs'], batch_size, ids=ids, val_set_ind=val_set_ind, val_set=val_set, dropout_rate_1=dropout_rate_1, dropout_rate_2=dropout_rate_2)
                
                
                print(f'Max validation mAP: {max_val_map[0]} at epoch {max_val_map[1]}')
                
                results_path = os.path.join(args['validation_results'], f'results_dev_fivr_{i + 33}')
                os.mkdir(results_path)
                r = open(os.path.join(results_path, 'results.txt'), 'a')
                r.write('Max validation mAP: {0:.2f}%'.format(max_val_map[0]))
                r.write(f'\nfivr-{i + 33}: vgg layers={layers_str}; epochs={max_val_map[1] + 1}; batch_size={batch_size};learning_rate={learning_rate};weight_decay={weight_decay};gamma={gamma};{reg.strip()}_reg;relu;drop_rate={dropout_rate_1}_{dropout_rate_2}')
                g.write(f'\nfivr{i + 33}: vgg layers={layers_str}; epochs={max_val_map[1] + 1}; batch_size={batch_size};learning_rate={learning_rate};weight_decay={weight_decay};gamma={gamma};{reg.strip()}_reg;relu;drop_rate={dropout_rate_2}_{dropout_rate_2}')
                r.close()
                
                plot_train_val_metrics(train_loss, np.arange(epochs + 1), results_path)
    else:
        batch_size = 2048
        model_path = os.path.join(args['model_path'], 'model_full')
        model = DNN(dataset.shape[1],
                    os.path.join(model_path, 'model_full'),
                    hidden_layer_sizes=[2000,1000,1000],
                    learning_rate=1e-5,
                    weight_decay=1e-5,
                    gamma=1,
                    reg='l2')
        
        _ = train_dml_network(model, dataset, train_triplets, 1, batch_size, dropout_rate_1=1, dropout_rate_2=1)
        