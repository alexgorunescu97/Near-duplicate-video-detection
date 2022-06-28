#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 23:02:26 2022

@author: alex
"""

import os
import tqdm
import argparse
import numpy as np
import tensorflow as tf
import pickle as pk

import matplotlib.pyplot as plt
from models.dnn_model import DNN
from similarity_computation.cc_web_video import evaluate_cc_web
from future.utils import lrange

CC_DATASET = 'output_data/datasets/cc_dataset_dev.pickle'

def plot_train_val_metrics(train_loss, epochs, results_path):
    
        fig_loss, ax_loss = plt.subplots(nrows=1, ncols=1)
        
        ax_loss.plot(epochs, train_loss)
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        
        loss_fig_path = os.path.join(results_path, 'loss.png')
        
        fig_loss.savefig(loss_fig_path)
        plt.close(fig_loss)
        
def train_dml_network(model, features, triplets, epochs, batch_sz, learning_rate, val_ind_map=None, val_set=None, dropout_rate_1=1, dropout_rate_2=1):
    """
      Function that handles the training process.

      Args:
        model: the DML model
        features: the global video features
        triplets: the generated triplets for training
        epochs: the training epochs
        batch_sz: the batch size
    """
    
    EARLY_STOPPING_LIMIT = 5

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

            _, loss, error = model.train(train_batch, dropout_rate_1, dropout_rate_2, learning_rate)
            losses[j] = loss

            pbar.set_postfix(loss=loss, error='{0:.2f}%'.format(error))
            
        mean_loss = np.mean(losses)
        train_losses.append(mean_loss)
        print(f'Training Loss for epoch {i}: {mean_loss}')
        
        if val_set is not None:
            current_map = validate_dml_network(model, features, val_set, val_ind_map)
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
        
        
def validate_dml_network(model, features, val_set, val_ind_map):
    """
      Function that handles the validating process.

      Args:
        model: the DML model
        features: the global video features
    """

    print('\nStart of DML Validating')
    print('=====================')
    
    mAP = evaluate_cc_web(CC_DATASET, val_set, model, val_ind_map=val_ind_map, full_evaluation=False)
    
    return mAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Path to the .npy file that contains the global '
                             'video vectors for the cc_web_videos dataset')
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
        
        val_set = pk.load(open(CC_DATASET, 'rb'))
            
        val_ind_map = {}
        
        for i, query in enumerate(val_set['ground_truth']):
            
            val_ind_map[val_set['queries'][i]] = True
            
            for video in query.keys():
                
                val_ind_map[video] = True

        with open('hyperparameters.txt', 'r') as f:
            
            for i, line in enumerate(f.readlines()):
                
                tf.keras.backend.clear_session()
                
                line_split = line.split(';')
                layers_str, dropout_rate_1, dropout_rate_2, learning_rate, batch_size, weight_decay, gamma, reg = line_split[0], float(line_split[1]), float(line_split[2]), float(line_split[3]), int(line_split[4]), float(line_split[5]), float(line_split[6]), line_split[7] 
                
                model_path = os.path.join(args['model_path'], f'model_dev_cc_{i}')
                os.mkdir(model_path)
            
                model = DNN(dataset.shape[1],
                            os.path.join(model_path, f'model_dev_cc_{i}'),
                            hidden_layer_sizes=[int(l) for l in layers_str.split(',') if l],
                            weight_decay=weight_decay,
                            gamma=gamma,
                            reg=reg.strip())
                
                max_val_map, train_loss, epochs = train_dml_network(model, dataset, train_triplets, args['epochs'], batch_size, learning_rate, val_set=args['dataset'], val_ind_map=val_ind_map, dropout_rate_1=dropout_rate_1, dropout_rate_2=dropout_rate_2)
                
                
                print(f'Max validation mAP: {max_val_map[0]} at epoch {max_val_map[1]}')
                
                results_path = os.path.join(args['validation_results'], f'results_dev_cc_{i}')
                os.mkdir(results_path)
                
                plot_train_val_metrics(train_loss, np.arange(epochs + 1), results_path)

        