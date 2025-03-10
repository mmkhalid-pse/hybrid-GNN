# Train and test split
# author: Edgar Sanchez

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import scipy as sp
import pickle as pk
from scipy.linalg import svd as SVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ipdb
# Set seed for reproducibility
seed = 1
np.random.seed(seed)

def train_test_split_normalization(case, train_time, validation_mode, n_points_per_second, hybrid_mode="simple", validation_time=0):
    """
    Split the data into train and test sets and perform normalization.

    Parameters:
    - case (int): The case number.
    - train_time (int): Duration of training data in seconds.
    - validation_mode (Boolean): True for training-validation run, False for training-testing run.
    - n_points_per_second (int): Number of data points per second.
    - hybrid_mode (str, optional): Type of hybridization ('simple', 'derivative_informed', 'derivative_trained', 'physics_informed'). Defaults to 'simple'.
    - validation_time (int, optional): Duration of validation data in seconds. Defaults to 0.
    - seed (int, optional): Seed for reproducibility. Defaults to 1.

    Returns:
    None
    """

    # Define file names and case dictionary
    file_names = [
        'conversion_vec',
        #'derivatives_conversion_vec',
        #'derivatives_temperature_vec',
        'temperature_vec',
        'time_vec',
        'z_vec'
    ]
    cases_dict = {
        1: 'cold_start_up',
        2: 'warm_start_up',
        3: 'flow_change',
        4: 'control_change'
    }

    # Construct folder and case name based on the provided case number
    case_folder = 'case_' + str(case)
    case_name = cases_dict[case]
    base_dir = os.getcwd()

    # Create folder to store split data if it doesn't exist
    split_data_folder = os.path.join(base_dir, 'data', 'interim', case_folder)
    if not os.path.exists(split_data_folder):
        os.makedirs(split_data_folder)

    # Define train/test split parameters
    train_last_idx = train_time * n_points_per_second

    # Loop through each file
    for name in tqdm(file_names):
        file_path = os.path.join(base_dir, 'data', 'raw',
                                 case_folder, f'{name}_{case_name}.npy')
        data = np.load(file_path)

        # Split data into train and test sets
        if name == 'z_vec':
            if hybrid_mode == "derivative_trained":
                data = data[:-1]
            elif hybrid_mode == "physics_informed":
                data = data[:-1]
            else:
                data = data[0:-1:2]
            np.savetxt(f'{split_data_folder}/{name}.csv',
                       data, delimiter=',', fmt='%f')

        else:
            # Normalize Temperature and Conversion
            if name in ['temperature_vec']:
                T_min = 400  # expected lowest temperature
                T_max = 800  # high risk temperature
                data = (data - T_min) / (T_max - T_min)
                
            #     T_mu = 0.4533159615958609
            #     T_sigma = 0.13944152723688819
            #     data = (data - T_mu)/T_sigma
    
            # if name in ['conversion_vec']:
            #    X_mu = 0.8487244223653506
            #    X_sigma = 0.2238620396750383
            #    data = (data - X_mu)/X_sigma

            if name in ["conversion_vec", "temperature_vec"]:
                if hybrid_mode in ["derivative_trained", "derivative_informed"]:
                    data = data[:-1,:]
            # if name in ["derivatives_conversion_vec"]:
            #     dXdt_mu = 0.0024839903350116215
            #     dXdt_sigma = 0.007050165670661646
            #     data = (data - dXdt_mu)/dXdt_sigma
            
            # if name in ["derivatives_temperature_vec"]:
            #     dTdt_mu = -0.760526477008922
            #     dTdt_sigma = 2.107505797657783
            #     data = (data - dTdt_mu)/dTdt_sigma

            if validation_mode:
                
                validation_last_idx = (train_time + validation_time) * n_points_per_second
                try:
                    train_data = data[:, :train_last_idx]
                    validation_data = data[:, train_last_idx:validation_last_idx]

                except:
                    train_data = data[:train_last_idx]
                    validation_data = data[train_last_idx:validation_last_idx]

                # Save train and test data
                np.savetxt(f'{split_data_folder}/train_{name}.csv',
                        train_data, delimiter=',', fmt='%f')
                np.savetxt(f'{split_data_folder}/validation_{name}.csv',
                        validation_data, delimiter=',', fmt='%f')

            elif not validation_mode:
                try:
                    if hybrid_mode == "physics_informed":
                        train_data = data[0:-1:1, :train_last_idx]
                        test_data = data[0:-1:1, train_last_idx:]
                    else:
                        train_data = data[0:-1:2, :train_last_idx]
                        test_data = data[0:-1:2, train_last_idx:]

                except:
                    train_data = data[:train_last_idx]
                    test_data = data[train_last_idx:]

                # Save train and test data
                np.savetxt(f'{split_data_folder}/train_{name}.csv',
                        train_data, delimiter=',', fmt='%f')
                np.savetxt(f'{split_data_folder}/test_{name}.csv',
                        test_data, delimiter=',', fmt='%f')
    return
