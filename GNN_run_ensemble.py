#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:02:24 2024

@author: peterson
"""
# Importing necessary functions and modules
from src.data.x_exploration_data import explore_data
from src.data.train_test_split_normalization import train_test_split_normalization
from src.data.graphs_generation import graphs_generation
from src.models.train import train
from src.models.predict import predict
from src.visualization.plot_train_traj import plot_train_traj
from src.visualization.plot_future_prediction import plot_future_prediction
from src.utils.IEEE_eval import run_IEEE_eval

def run_GNN(run_num=""):
      # Setting up parameters for the experiment
      case = 3  # Experiment case number
      gnn = 'GAT'  # Type of Graph Neural Network: GCN, GAT, GGNN, RGGNN
      train_typ = 'window'  # Type of training: simple, multistep, window
      window_size = 20  # Size of the sliding window for data processing
      hidden_dim = 228  # Dimensionality of hidden layers in the neural network
      n_points_per_second = 50  # Number of data points per second
      total_time = 35  # Duration of total data in seconds
      validation_mode = False # True for training-validation run, False for training-testing run
      if validation_mode:
            train_time = 15  # Duration of training data in seconds
            validation_time = 5  # Duration of validation data in seconds
            num_future_snapshots = validation_time * n_points_per_second  # Number of future snapshots to predict
      else:
            train_time = 20  # Duration of training data in seconds
            test_time =  total_time - train_time # Duration of testing data in seconds
            num_future_snapshots = test_time * n_points_per_second  # Number of future snapshots to predict
      lr = 5.000000000000004e-05 # The learning rate
      num_epochs = 30 # The number of training epochs
      steps_ahead = 5 # Number of steps ahead for multistep training
      noise_std = 0.001 # Standard deviation of noise for multistep training
      smoothness_weight = 5 # Weight for smoothness term in multistep training

      # Explore the dataset
      explore_data(case)

      # Split the dataset into train and test sets, and normalize it
      if validation_mode:
            train_test_split_normalization(case,
                                                train_time=train_time,
                                                validation_mode=validation_mode,
                                                n_points_per_second=n_points_per_second,
                                                validation_time = validation_time)
      else:
            train_test_split_normalization(case,
                                                train_time=train_time,
                                                validation_mode=validation_mode,
                                                n_points_per_second=n_points_per_second)

      # Generate graphs for spatial segments
      graphs_generation(case,
                        gnn,
                        window_size,
                        validation_mode=validation_mode)

      # Train Graph Neural Networks (GNNs)
      train(case,
            hidden_dim,
            gnn=gnn,
            train_typ=train_typ,
            num_epochs=num_epochs,
            lr=lr,
            window_size=window_size,
            steps_ahead=steps_ahead,
            noise_std=noise_std,
            smoothness_weight=smoothness_weight
            )



      # Use trained GNNs to predict future snapshots
      predict(case,
            hidden_dim,
            gnn,
            window_size,
            num_future_snapshots,
            run_num)

      # Evaluate the results
      #plot_train_traj(case, gnn)  # Plot the training trajectories
      #plot_future_prediction(case, gnn)  # Plot the future predictions
      #run_IEEE_eval(case, gnn, validation_mode=validation_mode)  # Run evaluation based on IEEE standards

for i in np.linspace(0,9,10):
      run_num = int(i)

      run_GNN(run_num)


import numpy as np, pandas as pd, os

case = 3
gnn = 'GAT'


def ensemble_mean(case, gnn):
      base_dir = os.getcwd()
      case_folder = 'case_' + str(case)
      pred_folder = os.path.join(base_dir, 'reports', 'predictions', case_folder)
      T_dict = {}
      C_dict = {}
      for i in np.linspace(0,9,10):
            run_num = int(i)
            T_dict[run_num] =  pd.read_csv(f'{pred_folder}/T_{gnn}_{run_num}.csv')
            C_dict[run_num] =  pd.read_csv(f'{pred_folder}/c_{gnn}_{run_num}.csv')

      T_mean = pd.DataFrame(np.zeros((1751, 751)),index=T_dict[0].index, columns=T_dict[0].columns)
      C_mean = pd.DataFrame(np.zeros((1751, 751)),index=C_dict[0].index, columns=C_dict[0].columns)
      for i in np.linspace(0,9,10):
            run_num = int(i)
            T_mean = T_mean + T_dict[run_num]
            C_mean = C_mean + C_dict[run_num]
      T_mean = T_mean/10
      C_mean = C_mean/10
      pd.DataFrame(T_mean).to_csv(f'{pred_folder}/T_{gnn}_.csv', index=False, header=False)
      pd.DataFrame(C_mean).to_csv(f'{pred_folder}/c_{gnn}_.csv', index=False, header=False)


ensemble_mean(case, gnn)



run_IEEE_eval(case, gnn, validation_mode=False, ensemble_run=True, run_num="")  # Run evaluation based on IEEE standards

# mean of individual errors
# rel_err = []
# for i in np.linspace(0,9,10):
#       run_num = str(int(i))
#       rel_err_i = run_IEEE_eval(case, gnn, validation_mode=False, run_num=run_num)
#       rel_err.append(rel_err_i)
# print(np.mean(rel_err))


