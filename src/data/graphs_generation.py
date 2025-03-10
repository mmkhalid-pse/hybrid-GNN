# Train and test split
# author: Edgar Sanchez

import pandas as pd
import os
import torch
from src.utils.datasets import ReactorDataset_simple, ReactorDataset_window
import ipdb


def graphs_generation(case, gnn, window_size, validation_mode, hybrid_mode="simple", seed=1):
    """
    Generate graphs for spatial segments.

    Parameters:
    - case (int): The case number.
    - gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    - window_size (int): Size of the window for window-based dataset.
    - validation_mode (Boolean): True for training-validation run, False for training-testing run.
    - hybrid_mode (str, optional): Type of hybridization ('simple', 'derivative_informed', 'derivative_trained', 'physics_informed'). Defaults to 'simple'.
    - seed (int, optional): Seed for reproducibility. Defaults to 1.

    Returns:
    None
    """
    # Set seeds for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    base_dir = os.getcwd()
    case_folder = 'case_' + str(case)
    data_folder = os.path.join(base_dir, 'data', 'interim', case_folder)
    dataset_folder = os.path.join(base_dir, 'data', 'processed', case_folder)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Read spatial coordinates
    df_z = pd.read_csv(f'{data_folder}/z_vec.csv', header=None)

    # Read conversion and temperature data
    split = 'train'
    df_c = pd.read_csv(f'{data_folder}/{split}_conversion_vec.csv',
                       header=None)
    df_T = pd.read_csv(f'{data_folder}/{split}_temperature_vec.csv',
                       header=None)
    
    if hybrid_mode in ["derivative_trained", "derivative_informed"]:
        df_dcdt = pd.read_csv(f'{data_folder}/{split}_derivatives_conversion_vec.csv',
                                header=None)
        df_dTdt = pd.read_csv(f'{data_folder}/{split}_derivatives_temperature_vec.csv',
                                header=None)
    
    # Create reactor dataset
    if window_size == 1:
        reactor_dataset = ReactorDataset_simple(df_z=df_z,
                                                df_c=df_c,
                                                df_T=df_T)
    else:
        if hybrid_mode == "derivative_trained":
            reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                    df_c=df_c,
                                                    df_T=df_T,
                                                    window_size=window_size,
                                                    hybrid_mode=hybrid_mode,
                                                    df_dcdt=df_dcdt,
                                                    df_dTdt=df_dTdt)
        elif hybrid_mode == "derivative_informed":
            reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                    df_c=df_dcdt,
                                                    df_T=df_dTdt,
                                                    window_size=window_size)
            torch.save(reactor_dataset, f'{dataset_folder}/{split}_{gnn}_derivatives.pt')
            
            reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                    df_c=df_c,
                                                    df_T=df_T,
                                                    window_size=window_size)
            #torch.save(reactor_dataset, f'{dataset_folder}/{split}_{gnn}.pt')
            

        else:
            reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                    df_c=df_c,
                                                    df_T=df_T,
                                                    window_size=window_size)
    
    # Save reactor dataset
    torch.save(reactor_dataset, f'{dataset_folder}/{split}_{gnn}.pt')

    # Quick check of correct construction of datasets
    train_dataset = torch.load(f'data/processed/{case_folder}/train_{gnn}.pt')


    if validation_mode:
        try:
            # Read test conversion and temperature data if available
            split = 'validation'
            df_c = pd.read_csv(f'{data_folder}/{split}_conversion_vec.csv',
                            header=None)
            df_T = pd.read_csv(f'{data_folder}/{split}_temperature_vec.csv',
                            header=None)

            if hybrid_mode in ["derivative_trained", "derivative_informed"]:
                print("Code missing for hybrid model with validation run")
            # Create test reactor dataset
            if window_size == 1:
                reactor_dataset = ReactorDataset_simple(df_z=df_z,
                                                        df_c=df_c,
                                                        df_T=df_T)
            else:
                reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                        df_c=df_c,
                                                        df_T=df_T,
                                                        window_size=window_size)

            # Save test reactor dataset
            torch.save(reactor_dataset, f'{dataset_folder}/test_{gnn}.pt')
            test_dataset = torch.load(f'data/processed/{case_folder}/test_{gnn}.pt')
        except:
            print('No test data. Using the training dataset instead.')
            test_dataset = train_dataset

        print('-'*110)
        print(f'Graphs in training dataset: {len(train_dataset)}')
        print(f'Graphs in validation dataset: {len(test_dataset)}')
        print('-'*110)
    
    else:
        try:
            # Read test conversion and temperature data if available
            split = 'test'
            df_c = pd.read_csv(f'{data_folder}/{split}_conversion_vec.csv',
                                header=None)
            df_T = pd.read_csv(f'{data_folder}/{split}_temperature_vec.csv',
                                header=None)
            if hybrid_mode in ["derivative_trained", "derivative_informed"]:
                df_dcdt = pd.read_csv(f'{data_folder}/{split}_derivatives_conversion_vec.csv',
                                        header=None)
                df_dTdt = pd.read_csv(f'{data_folder}/{split}_derivatives_temperature_vec.csv',
                                header=None)

            # Create test reactor dataset
            if window_size == 1:
                reactor_dataset = ReactorDataset_simple(df_z=df_z,
                                                        df_c=df_c,
                                                        df_T=df_T)
            else:
                if hybrid_mode == "derivative_trained":
                    reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                            df_c=df_c,
                                                            df_T=df_T,
                                                            window_size=window_size,
                                                            hybrid_mode=hybrid_mode,
                                                            df_dcdt=df_dcdt,
                                                            df_dTdt=df_dTdt)
                elif hybrid_mode == "derivative_informed":
                    reactor_dataset  = ReactorDataset_window(df_z=df_z,
                                                            df_c=df_dcdt,
                                                            df_T=df_dTdt,
                                                            window_size=window_size)
                    torch.save(reactor_dataset, f'{dataset_folder}/{split}_{gnn}_derivatives.pt')
                    
                    reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                            df_c=df_c,
                                                            df_T=df_T,
                                                            window_size=window_size)
                    #torch.save(reactor_dataset, f'{dataset_folder}/{split}_{gnn}.pt')
                else:
                    reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                            df_c=df_c,
                                                            df_T=df_T,
                                                            window_size=window_size)

            # Save test reactor dataset
            torch.save(reactor_dataset, f'{dataset_folder}/{split}_{gnn}.pt')
            test_dataset = torch.load(f'data/processed/{case_folder}/test_{gnn}.pt')
        except:
            print('No test data. Using the training dataset instead.')
            test_dataset = train_dataset

        print('-'*110)
        print(f'Graphs in training dataset: {len(train_dataset)}')
        print(f'Graphs in test dataset: {len(test_dataset)}')
        print('-'*110)
    return
