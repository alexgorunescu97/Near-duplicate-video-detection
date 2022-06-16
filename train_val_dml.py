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

from __future__ import division
from __future__ import print_function

import os
import tqdm
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from model import DNN
from future.utils import lrange

def plot_train_val_metrics(train_acc, train_loss, val_acc, val_loss, epochs, results_path):
    
        fig_acc, ax_acc = plt.subplots(nrows=1, ncols=1)
        fig_loss, ax_loss = plt.subplots(nrows=1, ncols=1)
        
        ax_acc.plot(epochs, train_acc)
        ax_acc.plot(epochs, val_acc)
        ax_acc.set_title('Accuracy')
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend(['train', 'val'], loc='upper left')
        
        ax_loss.plot(epochs, train_loss)
        ax_loss.plot(epochs, val_loss)
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(['train', 'val'], loc='upper left')
        
        acc_fig_path = os.path.join(results_path, 'acc.png')
        loss_fig_path = os.path.join(results_path, 'loss.png')
        
        fig_acc.savefig(acc_fig_path)
        plt.close(fig_acc)
        fig_loss.savefig(loss_fig_path)
        plt.close(fig_loss)
        
def train_dml_network(model, features, triplets, epochs, batch_sz, val_triplets=None, dropout_rate_1=1, dropout_rate_2=1):
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
    min_val_loss_seen = float('inf')
    max_val_acc = [float('-inf'), 0]
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    epochs_without_progress = 0
    for i in lrange(epochs):
        np.random.shuffle(triplets)
        pbar = tqdm.trange(n_batch,
                           desc='epoch {}'.format(i),
                           mininterval=1.0,
                           unit='batch')
        losses = np.zeros(n_batch)
        accuracies = np.zeros(n_batch)
        for j in pbar:
            triplet_batch = triplets[j * batch_sz: (j + 1) * batch_sz]
            train_batch = features[triplet_batch.reshape(-1)]

            _, loss, error = model.train(train_batch, dropout_rate_1, dropout_rate_2)
            losses[j] = loss
            accuracies[j] = 100 - error

            pbar.set_postfix(loss=loss, error='{0:.2f}%'.format(error))
            
        mean_loss = np.mean(losses)
        mean_acc = np.mean(accuracies)
        train_accuracies.append(mean_acc)
        train_losses.append(mean_loss)
        print(f'Training Loss for epoch {i}: {mean_loss}')
        
        if val_triplets is not None:
            acc, val_loss = validate_dml_network(model, features, val_triplets, batch_sz)
            val_accuracies.append(acc)
            val_losses.append(val_loss)
            if acc > max_val_acc[0]:
                max_val_acc[0] = acc
                max_val_acc[1] = i
                model.save()
            
            print(f'Validation Loss for epoch {i}: {val_loss}')
            if val_loss < min_val_loss_seen:
                min_val_loss_seen = val_loss
                epochs_without_progress = 0
            else:
                epochs_without_progress += 1
            
            if epochs_without_progress == EARLY_STOPPING_LIMIT:
                print(f'Early stoppage at epoch {i}')
                break
        else:    
            model.save()
    
    return max_val_acc, min_val_loss_seen, train_accuracies, train_losses, val_accuracies, val_losses, i
        
        
def validate_dml_network(model, features, triplets, batch_sz):
    """
      Function that handles the validating process.

      Args:
        model: the DML model
        features: the global video features
        triplets: the generated triplets for validation
        batch_sz: the batch size
    """

    print('\nStart of DML Validating')
    print('=====================')
    n_batch = triplets.shape[0] // batch_sz + 1
    pbar = tqdm.trange(n_batch, desc="Validating model", mininterval=1.0)
    costs = np.zeros(n_batch)
    accuracy = np.zeros(n_batch)
    for j in pbar:
        triplet_batch = triplets[j * batch_sz : (j + 1) * batch_sz]
        val_batch = features[triplet_batch.reshape(-1)]
        
        cost, error = model.test(val_batch)
        costs[j] = cost
        accuracy[j] = 100 - error
        
        pbar.set_postfix(cost=cost, accuracy='{0:.2f}%'.format(100 - error))
    
    print('Mean Accuracy: {0:.2f}%'.format(np.mean(accuracy)))
    return np.mean(accuracy), np.mean(costs)
 

N_TRAIN_VAL_ITERATIONS = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Path to the .npy file that contains the global '
                             'video vectors for the vcdb dataset')
    parser.add_argument('-ttr', '--train_triplets', type=str, required=True,
                        help='Path to the .npy file that contains the training triplets')
    parser.add_argument('-sv', '--split_validation', type=str, default='y',
                        help='Enables or disables split validation')
    parser.add_argument('-vtr', '--validation_triplets', type=str,
                        help='Path to the .npy file that contains the validation triplets')
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
    parser.add_argument('-s', '--split', type=float, default=0.9,
                        help="Train-dev split"),
    parser.add_argument('-vr', '--validation_results', default='', help="Path for validation results")
    args = vars(parser.parse_args())

    print('Train set file: ', args['dataset'])
    print('Train triplet file: ', args['train_triplets'])
    print('Validation triplet file: ',  args['validation_triplets'] if args['validation_triplets'] else "No validation set")

    print('loading data...')
    dataset = np.load(args['dataset']).astype(np.float32)
    train_triplets = np.load(args['train_triplets']).astype(np.int)
    feature_type = args['dataset'].split('/')[1]

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
        

        
    if args['validation_triplets']:
        
        val_triplets = np.load(args['validation_triplets']).astype(np.int)
        
        with open('hyperparameters.txt', 'r') as f:
            
            for i, line in enumerate(f.readlines()):
                
                tf.keras.backend.clear_session()
                
                line_split = line.split(';')
                layers_str, dropout_rate_1, dropout_rate_2, learning_rate, batch_size, weight_decay, gamma, reg = line_split[0], float(line_split[1]), float(line_split[2]), float(line_split[3]), int(line_split[4]), float(line_split[5]), float(line_split[6]), line_split[7] 
                
                model_path = os.path.join(args['model_path'], f'model_dev_{i}')
                os.mkdir(model_path)
            
                model = DNN(dataset.shape[1],
                            os.path.join(model_path, f'model_dev_{i}'),
                            hidden_layer_sizes=[int(l) for l in layers_str.split(',') if l],
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            gamma=gamma,
                            reg=reg.strip())
                
                max_val_acc, min_val_loss, train_acc, train_loss, val_acc, val_loss, epochs = train_dml_network(model, dataset, train_triplets, args['epochs'], batch_size, val_triplets=val_triplets, dropout_rate_1=dropout_rate_1, dropout_rate_2=dropout_rate_2)
                
                
                print(f'Max validation accuracy: {max_val_acc[0]} at epoch {max_val_acc[1]}')
                
                results_path = os.path.join(args['validation_results'], f'results_dev_{i + 460}_{max_val_acc[1] + 1}')
                os.mkdir(results_path)
                
                plot_train_val_metrics(train_acc, train_loss, val_acc, val_loss, np.arange(epochs + 1), results_path)

        