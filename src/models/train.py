import torch
import os
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import pandas as pd
import random
from torch.optim.lr_scheduler import StepLR
from src.utils.gnn_architectures import GCN, GAT, GGNN, RGGNN
from src.utils.train_test_predict import train_simple, test_simple, train_multistep, train_window, test_window
import ipdb

def train(case, hidden_dim, gnn, train_typ, num_epochs, lr, window_size,
          steps_ahead, noise_std, smoothness_weight, hybrid_mode="simple", seed=1):
    """
    Train the specified graph neural network (GNN) model.

    Parameters:
    - case (int): The case number.
    - hidden_dim (int): The dimension of the hidden layers.
    - gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    - train_typ (str): The type of training ('simple', 'multistep', 'window').
    - num_epochs (int): The number of training epochs.
    - lr (float): The learning rate.
    - window_size (int): The size of the window for window-based training.
    - steps_ahead (int): Number of steps ahead for multistep training.
    - noise_std (float): Standard deviation of noise for multistep training.
    - smoothness_weight (float): Weight for smoothness term in multistep training.
    - hybrid_mode (str, optional): Type of hybridization ('simple', 'derivative_informed', 'derivative_trained', 'physics_informed'). Defaults to 'simple'.
    - seed (int, optional): Seed for reproducibility. Defaults to 1.

    Returns:
    None
    """
    # Set seed for reproducibility if provided
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    rs = RandomState(MT19937(SeedSequence(seed)))

    case_folder = 'case_' + str(case)

    train_dataset = torch.load(f'data/processed/{case_folder}/train_{gnn}.pt')
    test_dataset = torch.load(f'data/processed/{case_folder}/test_{gnn}.pt')

    if hybrid_mode == "derivative_informed":
        data2deriv_weight_factor = 0.5
        train_derivatives_dataset = None        #torch.load(f'data/processed/{case_folder}/train_{gnn}_derivatives.pt')
        test_derivatives_dataset = None     #torch.load(f'data/processed/{case_folder}/test_{gnn}_derivatives.pt')
    if hybrid_mode == "physics_informed":
        data2physics_weight_factor = 0.5

    if hybrid_mode == "derivative_trained":
        node_feature_dim = 4
    else:
        node_feature_dim = 2

    # Initialize model based on specified GNN architecture
    if gnn == 'GCN':
        model = GCN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim, window_size=window_size)
    elif gnn == 'GAT':
        model = GAT(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim, window_size=window_size)
    elif gnn == 'GGNN':
        model = GGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim, window_size=window_size)
    elif gnn == 'RGGNN':
        model = RGGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim, window_size=window_size)

    # Print model parameters
    print(f'\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # train_model_folder = f'models/{case_folder}'
    # model.load_state_dict(torch.load(f'{train_model_folder}/model_{gnn}_pi-gnn_25.pt'))

    train_traj = np.zeros(num_epochs)
    test_traj = np.zeros(num_epochs)

    # Define the learning rate scheduler, criterion, and optimizer
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print('-'*110)

        if train_typ == 'simple':
            if hybrid_mode != "simple":
                print("Code missing for hybrid model for simple training")
            train_mse = train_simple(train_dataset, model, lr,
                              criterion, device, optimizer)
            test_mse = test_simple(test_dataset, model, criterion, device)
        elif train_typ == 'multistep':
            if hybrid_mode != "simple":
                print("Code missing for hybrid model for multistep training")
            train_mse = train_multistep(train_dataset,
                                        model, lr,
                                        steps_ahead=steps_ahead,
                                        noise_std=noise_std,
                                        smoothness_weight=smoothness_weight)
            test_mse = test_simple(test_dataset, model, criterion, device)
        elif train_typ == 'window':

            if hybrid_mode == "derivative_informed":
                train_mse = train_window(train_dataset, model, criterion,
                                     device, optimizer, hybrid_mode, data2deriv_weight_factor, train_derivatives_dataset)
                test_mse = test_window(test_dataset, model, criterion, device, hybrid_mode, data2deriv_weight_factor, test_derivatives_dataset)
            elif hybrid_mode == "physics_informed":
                train_mse = train_window(train_dataset, model, criterion,
                                     device, optimizer, hybrid_mode, data2physics_weight_factor)
                test_mse = test_window(test_dataset, model, criterion, device, hybrid_mode, data2physics_weight_factor)
            else:
                train_mse = train_window(train_dataset, model, criterion,
                                        device, optimizer)
                test_mse = test_window(test_dataset, model, criterion, device)

        # save performance
        train_traj[epoch] = train_mse
        test_traj[epoch] = test_mse

        print(f'Epoch {epoch + 1} completed   Train_MSE: {train_mse} / Test_MSE: {test_mse}')
        train_model_folder = f'models/{case_folder}'
        torch.save(model.state_dict(), f'{train_model_folder}/model_{gnn}_simp-gnn_{epoch}.pt')
        

    # Save model
    train_model_folder = f'models/{case_folder}'
    if not os.path.exists(train_model_folder):
        os.makedirs(train_model_folder)
    if hybrid_mode == "simple":
        torch.save(model.state_dict(), f'{train_model_folder}/model_{gnn}_simp.pt')
    elif hybrid_mode == "derivative_informed":
        torch.save(model.state_dict(), f'{train_model_folder}/model_{gnn}_di-gnn.pt')
    elif hybrid_mode == "derivative_trained":
        torch.save(model.state_dict(), f'{train_model_folder}/model_{gnn}_pi-gnn.pt')

    df_training_traj = pd.DataFrame({
        'Train_rmse': np.sqrt(train_traj),
        'Test_rmse': np.sqrt(test_traj)
    })

    # Save training trajectory
    train_traj_folder = f'reports/training_traj/{case_folder}'
    if not os.path.exists(train_traj_folder):
        os.makedirs(train_traj_folder)

    df_training_traj.to_csv(f'{train_traj_folder}/train_traj_{gnn}.csv', index=False)
    return
