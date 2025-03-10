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

import optuna
import torch

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline


SEED = 13
torch.manual_seed(SEED)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Setting up parameters for the experiment
case = 3  # Experiment case number
gnn = 'GAT'  # Type of Graph Neural Network: GCN, GAT, GGNN, RGGNN
train_typ = 'window'  # Type of training: simple, multistep, window
#window_size = 10  # Size of the sliding window for data processing
#hidden_dim = 128  # Dimensionality of hidden layers in the neural network
n_points_per_second = 50  # Number of data points per second
total_time = 35  # Duration of total data in seconds
validation_mode = True # True for training-validation run, False for training-testing run
if validation_mode:
      train_time = 15  # Duration of training data in seconds
      validation_time = 5  # Duration of validation data in seconds
      num_future_snapshots = validation_time * n_points_per_second  # Number of future snapshots to predict
else:
      train_time = 20  # Duration of training data in seconds
      test_time =  total_time - train_time # Duration of testing data in seconds
      num_future_snapshots = test_time * n_points_per_second  # Number of future snapshots to predict

num_epochs = 30
steps_ahead = 5
noise_std = 0.001
smoothness_weight = 5

def GNN(case, gnn, train_typ, window_size, hidden_dim, n_points_per_second, train_time, num_future_snapshots, num_epochs, lr, steps_ahead, noise_std, smoothness_weight):

      print("Type: ", gnn)
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
                  num_future_snapshots)

      # Evaluate the results
      plot_train_traj(case, gnn)  # Plot the training trajectories
      plot_future_prediction(case, gnn)  # Plot the future predictions

      rel_err = run_IEEE_eval(case, gnn, validation_mode=validation_mode)  # Run evaluation based on IEEE standards

      return rel_err


def objective(trial):

      window_size = trial.suggest_int(f'window_size', 2, 20, step=2)

      hidden_dim = trial.suggest_int(f'hidden_dimension', 16, 512, step=2)

      lr = trial.suggest_float(f'learning_rate', 0.00005, 0.001, log=True)

      rel_err = GNN(case, gnn, train_typ, window_size, hidden_dim, n_points_per_second, train_time, num_future_snapshots, num_epochs, lr, steps_ahead, noise_std, smoothness_weight)

      if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
      
      return rel_err


if __name__ == "__main__":

      study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=SEED))        # GPSampler, RandomSampler, GridSampler, TPESampler(default)
      study.optimize(objective, n_trials=30, show_progress_bar=True)


      pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
      completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

      print("Study Stats for: ", gnn)
      print("Number of finished trials: ", len(study.trials))
      print("Number of pruned trials: ", len(pruned_trials))
      print("Number of complete trials: ", len(completed_trials))

      print("Best trial value: ", study.best_trial.value)
      print("Best trial params: ", study.best_trial.params)

      
      # plot_optimization_history(study)
      # plot_intermediate_values(study)
      # plot_parallel_coordinate(study, params=["window_size", "hidden_dim", "lr"])
      # plot_contour(study, params=["window_size", "hidden_dim", "lr"])
      # plot_param_importances(study)
      # plot_rank(study)
      # plot_timeline(study)
      
      
