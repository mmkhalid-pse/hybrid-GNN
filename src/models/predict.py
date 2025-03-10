from src.utils.gnn_architectures import GCN, GAT, GGNN, RGGNN
from src.utils.train_test_predict import predict_future_snapshots, predict_future_snapshots_window, shift_time_snapshot
import os
import torch


def predict(case, hidden_dim, gnn, window_size, num_future_snapshots, hybrid_mode="simple", run_num="", seed=1):
    """
    Predict future snapshots using the trained GNN model.

    Parameters:
    - case (int): The case number.
    - hidden_dim (int): Dimension of the hidden layers in the GNN.
    - gnn (str): Type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    - window_size (int or None): Size of the window for window-based dataset, or None for simple dataset.
    - num_future_snapshots (int): Number of future snapshots to predict.
    - hybrid_mode (str, optional): Type of hybridization ('simple', 'derivative_informed', 'derivative_trained', 'physics_informed'). Defaults to 'simple'.
    - run_num (str, optional): Identifying number of ensemble run, defaults to empty string ('') for ensemble_run = False.
    - seed (int, optional): Seed for reproducibility. Defaults to 1.

    Returns:
    None
    """
    case_folder = 'case_' + str(case)

    # Set seeds for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Load the training dataset
    train_dataset = torch.load(f'data/processed/{case_folder}/train_{gnn}.pt')

    # Get the last training graph and its initial state
    if window_size is None:
        last_training_graph = train_dataset.get(len(train_dataset)-1)
        c_0 = last_training_graph.x[:, 0].cpu().numpy()
        T_0 = last_training_graph.x[:, 1].cpu().numpy()
    else:
        if hybrid_mode == "derivative_trained":
            window_size = int(window_size)
            last_training_graph, last_target = train_dataset.get(len(train_dataset)-1)
            # Update input graph to get the very last information from the training set
            shift_time_snapshot(last_training_graph.x, last_target)
            c_0 = last_target[:, 0].cpu().numpy()
            T_0 = last_target[:, 1].cpu().numpy()
            dcdt_0 = last_target[:, 2].cpu().numpy()
            dTdt_0 = last_target[:, 3].cpu().numpy()
        else:
            window_size = int(window_size)
            last_training_graph, last_target = train_dataset.get(len(train_dataset)-1)
            # Update input graph to get the very last information from the training set
            shift_time_snapshot(last_training_graph.x, last_target)
            c_0 = last_target[:, 0].cpu().numpy()
            T_0 = last_target[:, 1].cpu().numpy()

    if hybrid_mode == "derivative_trained":
        node_feature_dim = 4
    else:
        node_feature_dim = 2

    # Initialize the GNN model based on the chosen architecture
    if gnn == 'GCN':
        model = GCN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                    window_size=window_size)
    elif gnn == 'GAT':
        model = GAT(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                    window_size=window_size)
    elif gnn == 'GGNN':
        model = GGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                    window_size=window_size)
    elif gnn == 'RGGNN':
        model = RGGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                    window_size=window_size)

    # Load the trained model
    train_model_folder = f'models/{case_folder}'
    model.load_state_dict(torch.load(f'{train_model_folder}/model_{gnn}.pt'))

    # Predict future snapshots using the trained model
    if window_size is None:
        df_c_pred, df_T_pred = predict_future_snapshots(model,
                                                        last_training_graph,
                                                        num_future_snapshots)
    else:
        if hybrid_mode == "derivative_trained":
            df_c_pred, df_T_pred, df_dcdt_pred, df_dTdt_pred = predict_future_snapshots_window(model,
                                                                last_training_graph,
                                                                num_future_snapshots,
                                                                hybrid_mode)
        else:
            df_c_pred, df_T_pred = predict_future_snapshots_window(model,
                                                        last_training_graph,
                                                        num_future_snapshots,
                                                        hybrid_mode)

    # Create folder to store predictions
    pred_folder = f'reports/predictions/{case_folder}'
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
    
    if hybrid_mode == "derivative_trained":
        # Append initial graph in case of including training
        df_c_pred.insert(loc=0, column='initial', value=c_0)
        df_T_pred.insert(loc=0, column='initial', value=T_0)
        df_dcdt_pred.insert(loc=0, column='initial', value=dcdt_0)
        df_dTdt_pred.insert(loc=0, column='initial', value=dTdt_0)

        # Save predicted data
        df_c_pred.to_csv(f'{pred_folder}/c_{gnn}_{run_num}.csv', index=False)
        df_T_pred.to_csv(f'{pred_folder}/T_{gnn}_{run_num}.csv', index=False)
        df_dcdt_pred.to_csv(f'{pred_folder}/dcdt_{gnn}_{run_num}.csv', index=False)
        df_dTdt_pred.to_csv(f'{pred_folder}/dTdt_{gnn}_{run_num}.csv', index=False)
    else:
        # Append initial graph in case of including training
        df_c_pred.insert(loc=0, column='initial', value=c_0)
        df_T_pred.insert(loc=0, column='initial', value=T_0)

        # Save predicted data
        df_c_pred.to_csv(f'{pred_folder}/c_{gnn}_{run_num}.csv', index=False)
        df_T_pred.to_csv(f'{pred_folder}/T_{gnn}_{run_num}.csv', index=False)
    return
