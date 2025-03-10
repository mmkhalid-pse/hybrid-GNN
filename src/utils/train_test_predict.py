import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
import ipdb
import random
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

from src.physics.parameter import data
import src.physics.parameter as parameter
import src.physics.inlet_profiles as inlet_profiles

from src.physics.balances import PDE_sys
import src.physics.thermodynamics as thermo
#import src.physics.pressure_drop as pd
from src.physics.reaction_rate import reaction_rate_LHHW, reaction_rate_PL
from src.physics.effective_diffusion import effective_diffusion
from src.physics.utils import replace_out_of_bounds_values

import ipdb

# Set seed for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
rs = RandomState(MT19937(SeedSequence(seed)))

def train_simple(dataset, model, lr, criterion, device, optimizer):
    """
    Train the GNN model.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The training dataset.
    - model: The GNN model.
    - lr (float): Learning rate.
    - criterion: Loss function.
    - device: Device for training (CPU or GPU).
    - optimizer: Optimizer for updating model parameters.

    Returns:
    - float: Average training loss.
    """
    model.train()
    total_loss = 0
    for i in range(len(dataset) - 1):
        current_graph = dataset[i].to(device)
        next_graph = dataset[i + 1].to(device)

        optimizer.zero_grad()
        out = model(current_graph)
        loss = criterion(out, next_graph.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataset)

def test_simple(dataset, model, criterion, device):
    """
    Evaluate the GNN model on the test dataset.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The test dataset.
    - model: The trained GNN model.
    - criterion: Loss function.
    - device: Device for evaluation (CPU or GPU).

    Returns:
    - float: Average test loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(len(dataset) - 1):
            current_graph = dataset[i].to(device)
            next_graph = dataset[i + 1].to(device)

            out = model(current_graph)
            loss = criterion(out, next_graph.x)
            total_loss += loss.item()

    avg_loss = total_loss / (len(dataset) - 1)
    return avg_loss

def predict_future_snapshots(model, last_training_graph, num_future_snapshots):
    """
    Predict future snapshots based on the last training graph.

    Parameters:
    - model: The trained GNN model.
    - last_training_graph: The last graph from the training set (PyTorch Geometric Data object).
    - num_future_snapshots (int): The number of future time snapshots to predict.

    Returns:
    - df_conversion: DataFrame with predicted conversion values.
    - df_temperature: DataFrame with predicted temperature values.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    current_graph = last_training_graph.to(device)

    conversion_predictions = []
    temperature_predictions = []

    with torch.no_grad():
        for _ in tqdm(range(num_future_snapshots), desc="Predicting the future"):
            out_features = model(current_graph)
            conversion, temperature = out_features[:, 0], out_features[:, 1]

            conversion_predictions.append(conversion.cpu().numpy())
            temperature_predictions.append(temperature.cpu().numpy())

            current_graph.x = out_features

    df_conversion = pd.DataFrame(conversion_predictions).transpose()
    df_temperature = pd.DataFrame(temperature_predictions).transpose()

    return df_conversion, df_temperature

# Define smoothness regularization function
def smoothness_regularization(node_attr):
    derivatives = torch.gradient(node_attr, dim=0)[0]
    diff_of_derivatives = derivatives[1:] - derivatives[:-1]
    squared_diff = torch.sum(diff_of_derivatives ** 2, dim=0)
    return squared_diff

def train_multistep(dataset, model, lr, steps_ahead=10, noise_std=0.0001, smoothness_weight=0.001):
    """
    Train the GNN model for multi-step prediction.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The training dataset.
    - model: The GNN model.
    - lr (float): Learning rate.
    - steps_ahead (int): Number of steps ahead for multi-step prediction.
    - noise_std (float): Standard deviation of Gaussian noise to inject into input features.
    - smoothness_weight (float): Weight of smoothness regularization in the loss function.

    Returns:
    - float: Average training loss.
    """
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    total_loss = 0
    num_snapshots = len(dataset)
    for i in tqdm(range(num_snapshots - 1)):
        optimizer.zero_grad()
        multi_step_loss = 0

        current_graph = dataset[i].to(device)
        x_copy = current_graph.x.clone()
        for step in range(1, min(steps_ahead, num_snapshots - i)):

            noise = torch.randn_like(x_copy) * noise_std
            noisy_x = x_copy + noise
            noisy_graph = Data(noisy_x, current_graph.edge_index)

            out = model(noisy_graph)
            target_graph = dataset[i + step].to(device)

            loss = criterion(out, target_graph.x)
            smoothness_loss = smoothness_regularization(noisy_x)
            loss += torch.sum(smoothness_weight * smoothness_loss)
            multi_step_loss += loss

            x_copy = out

        multi_step_loss /= min(steps_ahead, num_snapshots - i)

        multi_step_loss.backward()
        optimizer.step()

        total_loss += multi_step_loss.item()

    average_loss = total_loss / (len(dataset) - steps_ahead)
    return average_loss

def shift_time_snapshot(tensor, new_snapshot):
    """
    Shift time snapshot by updating the input tensor.

    Parameters:
    - tensor (torch.Tensor): Input tensor containing previous snapshots.
    - new_snapshot (torch.Tensor): New snapshot to insert into the tensor.

    Returns:
    None
    """
    num_nodes = tensor.size(0)
    z = new_snapshot.size(0)

    num_chunks = num_nodes // z

    for i in range(1, num_chunks):
        start = (i-1) * z
        end = i * z
        tensor[start:end] = tensor[start + z:end + z]

    tensor[(num_chunks-1) * z:] = new_snapshot

def train_multistep_window(dataset, model, lr, steps_ahead=10, noise_std=0.0001, smoothness_weight=0.001):
    """
    Train the GNN model for multi-step prediction with window-based dataset.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The training dataset.
    - model: The GNN model.
    - lr (float): Learning rate.
    - steps_ahead (int): Number of steps ahead for multi-step prediction.
    - noise_std (float): Standard deviation of Gaussian noise to inject into input features.
    - smoothness_weight (float): Weight of smoothness regularization in the loss function.

    Returns:
    - float: Average training loss.
    """
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    total_loss = 0
    num_snapshots = len(dataset)
    for i in tqdm(range(num_snapshots - 1)):
        optimizer.zero_grad()
        multi_step_loss = 0

        current_graph, _ = dataset[i]
        current_graph = current_graph.to(device)
        x_copy = current_graph.x.clone()
        for step in range(1, min(steps_ahead, num_snapshots - i)):

            noise = torch.randn_like(x_copy) * noise_std
            noisy_x = x_copy + noise
            noisy_graph = Data(noisy_x, current_graph.edge_index)

            out = model(noisy_graph)

            _, target_x = dataset[i + step - 1]
            target_x = target_x.to(device)

            loss = criterion(out, target_x)
            smoothness_loss = smoothness_regularization(noisy_x)
            loss += torch.sum(smoothness_weight * smoothness_loss)
            multi_step_loss += loss

            shift_time_snapshot(x_copy, out)

        multi_step_loss /= min(steps_ahead, num_snapshots - i)

        multi_step_loss.backward()
        optimizer.step()

        total_loss += multi_step_loss.item()

    average_loss = total_loss / (len(dataset) - steps_ahead)
    return average_loss


def PDE_sys(t, x):
    """
    PDE system at current time snapshot.

    Parameters:
    - t (float): Time at the current time snapshot.
    - x (numpy.ndarray): Output array from the GNN model.

    Returns:
    - numpy.ndarray: substituted PDE system solution.
    """
    eps = np.finfo(float).eps

    simulation_params = {
        "T_cool": [250, 250],  # Initial and final wall temperatures in °C (227 °C < T_cool < 476 °C)
        "T_gas_in": 250,  # Inlet gas temperature in °C (set to 250)
        "pressure": 3e5,  # Reactor pressure in Pa (1e5 Pa < p < 10e5 Pa)
        "F_H2": 10*(4/10),  # Hydrogen flow rate in Nl/min (2.40 – 120 nL/min)
        "F_CO2": 10*(1/10),  # Carbon Dioxide flow rate in Nl/min (0.12 – 25   nL/min)
        "F_CH4": 0,  # Methane flow rate in Nl/min (0.12 – 25   nL/min)
        "F_Ar": 10*(5/10),  # Argon flow rate in Nl/min (0.16 – 50   nL/min)
        "start_up": True,  # Perform a cold start up or load in data for start
        "initial_values": "start_up",  # Initial conditions or file name
    }

    profile_params = {
        "choose_F_in_profile": 'jump',  # Flow rate profile type
        'wind_profile_start': 0,  # start time of the predefined wind profile
        "flow_rate_change": 1,  # set to 1 if not change is wanted
        "jump_start": 20,  # Start time for flow rate jump change in s
        "choose_T_cool_profile": 'log_saturation',  # Cooling temperature profile type
        "ramp_start": 0,  # Start time for temperature ramp
    }

    numerical_params = {
        "t_end": 35,  # End time of the simulation (seconds)
        "Nz": 1750,  # Number of spatial discretization points (385 are in the reactor) w/o inlet channel
        "Nz_all": 1750, # Number of spatial discretization points (385 are in the reactor)
        "Nt": 35*50,  # Number of time steps
        "calculate_derivatives": False,  # Toggle for derivative calculations
        "enable_profile_plot": True,  # Toggle for profile plotting
        "enable_dynamic_plot": False,  # Toggle for dynamic plotting
        "simulation_name": 'start_up',  # File name for simulation results
    }

    T_gas_in = simulation_params["T_gas_in"] + 273.15  # Convert to Kelvin
    T_cool_in = simulation_params["T_cool"][0] + 273.15  # Convert to Kelvin
    T_cool_out = simulation_params["T_cool"][1] + 273.15  # Convert to Kelvin
    flow_rates = np.array([simulation_params["F_H2"],
                            simulation_params["F_CO2"],
                            simulation_params["F_CH4"],
                            0,
                            simulation_params["F_Ar"]])
    flow_rates = flow_rates.astype(float)

    # Handle flow rates close to zero
    flow_rates = np.where(flow_rates < eps, eps, flow_rates)

    data.update({
        "T_gas_in": T_gas_in,
        "p_R": simulation_params["pressure"],
        "x_in": flow_rates / np.sum(flow_rates),
        "t_end": numerical_params["t_end"],
        "Nt": numerical_params["Nt"],
        "Nz": numerical_params["Nz"],
        "Nz_all": numerical_params["Nz_all"],
        "derivatives": numerical_params["calculate_derivatives"]
    })


    F_in_profile = inlet_profiles.choose_flow_rate_profile(
        profile_params["choose_F_in_profile"],
        data["t_end"],
        data["Nt"],
        np.sum(flow_rates),
        np.sum(flow_rates) * profile_params["flow_rate_change"],
        profile_params["jump_start"],
        profile_params["wind_profile_start"],
        numerical_params["enable_profile_plot"]
    )

    #F_in_profile = replace_out_of_bounds_values(F_in_profile, eps)


    T_cool_profile = inlet_profiles.choose_cooling_temperature_profile(
        profile_params["choose_T_cool_profile"],
        data["t_end"],
        data["Nt"],
        T_cool_in,
        T_cool_out,
        profile_params["ramp_start"],
        numerical_params["enable_profile_plot"]
    )


    data.update({
        "F_in_Nl_min": F_in_profile,
        "T_cool": T_cool_profile,
    })

    if np.sum(flow_rates) < 1e-10:
        print('Information: Flow rate is zero')
        F_in_profile_eps = np.ones_like(F_in_profile) * eps
        data.update({
            'x_in': np.array([eps, eps, 0, 0, 1]),
            "F_in_Nl_min": F_in_profile_eps
        })

    parameter.update_dependent_values(data)

    if simulation_params["start_up"] is True:
        iv_X_CO2 = np.ones((data["Nz"], 1)) * data["X_0"]
        x_values = np.linspace(0, 1, data["Nz"])
        iv_T = inlet_profiles.get_log_saturation(
            x_values, data["T_gas_in"], data["T_cool"][0], 10).reshape(-1, 1)
    else:
        try:
            results_folder = os.path.join(os.getcwd(), "results")
            file_name_X = 'conversion_' \
                + simulation_params["initial_values"] +  '.npy'
            iv_X_CO2 = np.load(os.path.join(results_folder,
                                            file_name_X))[1:, -1]
            file_name_T = 'temperature_' \
                + simulation_params["initial_values"] +  '.npy'
            iv_T = np.load(os.path.join(results_folder,
                                        file_name_T))[1:, -1]
        except FileNotFoundError:
            print('Data not available. Performing cold start up.')
            iv_X_CO2 = np.ones((data["Nz"], 1)) * data["X_0"]
            x_values = np.linspace(0, 1, data["Nz"])
            iv_T = inlet_profiles.get_log_saturation(x_values, data["T_gas_in"], data["T_cool"][0], 10).reshape(-1, 1)

    if not (500 <= T_gas_in <= 750):
        print('Warning: Inlet temperature is not between 500 and 750 K.')
    if not (1e5 <= simulation_params["pressure"] <= 10e5):
        print('Warning: Pressure is not between 1e5 and 10e5 Pa.')
    if (np.sum(flow_rates) > 20) or (np.sum(flow_rates) * profile_params["flow_rate_change"] > 20):
        print('Warning: Flow rate is greater than 20 Nl/min.')
    if not np.all((500-273.15 <= np.array(simulation_params["T_cool"])) & (np.array(simulation_params["T_cool"]) <= 750-273.15)):
        print('Warning: Cooling temperature is not between 500 and 750 K.')


    """Build up mass and energy balances for the methanation reaction."""
    X_CO2 = x[:,0]          #[0*data["Nz"]:1*data["Nz"]]
    tau = x[:,1]            #[1*data["Nz"]:2*data["Nz"]]
    
    T = tau*data["T_scale"]+data["T_scale_add"]
    X_CO2 = replace_out_of_bounds_values(X_CO2, eps, array_name='X_CO2_'+str(t))
    X_CO2 = replace_out_of_bounds_values(X_CO2,
                                         threshold_value=1,
                                         is_lower_bound=False,
                                         array_name='X_CO2_'+str(t))
    T = replace_out_of_bounds_values(T, data["T_min"], array_name="T_"+str(t))
    ###########################################################################
    # TIME-CHANGING PARAMATERS
    ###########################################################################
    T_cool = np.interp(t, data["t_points"], data["T_cool"])
    v_gas_in = np.interp(t, data["t_points"], data["v_gas_in"])

    ###########################################################################
    # INITIAL CONDITIONS
    ###########################################################################
    # pressure drop in Pa
    p_loss = 0 # pd.pressure_drop(data["T_gas_in"], data["x_in"], v_gas_in)
    p_R_loss = data["p_R"]-p_loss*data["zeta"]*data["L_R"]

    ###########################################################################
    # CONVERSIONS
    ###########################################################################
    # get part of fractions
    n_i = data["n_in"][:, None] \
        + data["nue"][:, None]*X_CO2[None, :]*data["n_in"][1]
    # Set negative values to eps
    n_i = replace_out_of_bounds_values(n_i, eps, array_name="n_i_"+str(t))
    # mole fraction
    n = np.sum(n_i, axis=0)
    x_i = (n_i/n)
    # M_gas
    M_gas = np.sum(x_i.T*data["Molar_Mass"], axis=1)
    # Gas densitity by ideal gas law in kg/m^3 (validiert)
    density_fluid = (p_R_loss*M_gas/data["R"]/(T)).T
    # Mass flow (axial mass flow remains constant) - quasistationary
    v_gas = v_gas_in*data["density_fluid_in"]/density_fluid

    ###########################################################################
    # THERMODYNAMICS
    ###########################################################################
    # get thermodynamics
    cp_fluid = thermo.get_cp_fluid(data["T_prop"], data["x_i_prop"])
    H_r = thermo.get_reaction_enthalpy(T)
    U, _, lambda_ax = thermo.get_heat_transfer_coeff(
        T, data["x_i_prop"], v_gas, density_fluid)

    ###########################################################################
    # REACTION RATE
    ###########################################################################
    # mass based reactino rate in  mol/(s*kg_cat)
    if np.min(x_i[0, :]) < 1e-4:
        r = np.zeros_like(T)
    else:
        r = reaction_rate_PL(T, x_i, p_R_loss)
    # molar Reaction Rate in mol/(s*mcat^3)
    if np.all(r == 0):
        r_eff = r
    else:
        r_int = (1-data["epsilon_core"])*data["density_cat_core"]*r
        eta = effective_diffusion(T, p_R_loss, x_i, v_gas, r_int)
        r_eff = data["cat_frac"]*(1-data["epsilon"])*eta*r_int
        
    ###########################################################################
    # BALANCES
    ###########################################################################
    # Initialize composition matrix
    dxdt = np.zeros((2, data["Nz"]))
    dz = data["D_zeta"] * data["L_R"]
    dz_2 = data["d_zeta"]*data["L_R"]

    # Mass Balance
    dxdt_mass_0 = -v_gas[0] * (X_CO2[0]-data["X_in"])/data["epsilon"]/dz[0] \
        + data["Molar_Mass"][1]*r_eff[0]/(data["w_in"][1]+eps)/density_fluid[0]/data["epsilon"]
    dxdt[0, 0] = dxdt_mass_0
    # inner control volumes
    dxdt_mass = -v_gas[1:data["Nz"]]*(X_CO2[1:data["Nz"]]-X_CO2[0:data["Nz"]-1]) \
        / data["epsilon"] / dz[1:data["Nz"]] \
        + data["Molar_Mass"][1]*r_eff[1:data["Nz"]] \
        / ((data["w_in"][1]+eps)*density_fluid[1:data["Nz"]]*data["epsilon"]+eps)
    dxdt[0, 1:data["Nz"]] = dxdt_mass

    # Pre-compute constants and repeated expressions
    lambda_ax_mean = 0.5 * (lambda_ax[:-1] + lambda_ax[1:])  # arithmetic mean
    lambda_ax_mean[lambda_ax_mean < 0] = 0
    # Expressions for Axial Dispersion
    Ax_Disp = np.zeros(data["Nz"])
    # Vectorized computation for the inner elements
    i = np.arange(1, data["Nz"]-1)
    Ax_Disp[i] = lambda_ax_mean[i]*(T[i+1]-T[i])/(dz_2[i]*dz[i]) \
        - lambda_ax_mean[i-1]*(T[i]-T[i-1])/(dz_2[i-1]*dz[i])
    # left BC
    Ax_Disp[0] = lambda_ax_mean[0] * (T[1] - T[0]) / (dz_2[0]*dz[0]) \
        -lambda_ax[0]*(T[-1]-T[-2])/(dz_2[0]*(dz[0]))
    # right BC
    Ax_Disp[-1] = 0-lambda_ax_mean[-2]*(T[-1]-T[-2])/(dz_2[-2]*(dz[-1]))


    # Kinetics for heat transfer (Wärmeübergang Gas-Wand)
    g_qw = 4 * U * (T - T_cool) / data["D_R"]

    # Calculating effective heat capacity
    rhocp_cat = ((data["r_core"]**3 / data["r_shell"]**3) * (1 - data["epsilon_core"]) * data["density_cat_core"] * data["cp_core"]
                 + (data["r_shell"]**3 - data["r_core"]**3) / data["r_shell"]**3 * (1 - data["epsilon_shell"]) * data["density_cat_shell"] * data["cp_shell"])
    rhocp_g = density_fluid*cp_fluid*data["epsilon"] \
        + rhocp_cat*(1-data["epsilon"])
    dxdt_T_0 = (-v_gas_in*data["density_fluid_in"]*cp_fluid[0]
                   * ((T[0]-data["T_gas_in"])/dz[0]))/rhocp_g[0] \
        - r_eff[0].T*H_r[0]/rhocp_g[0] \
        - g_qw[0]/rhocp_g[0] \
        + Ax_Disp[0]/rhocp_g[0]
    dxdt_tau_0 = dxdt_T_0/data["T_scale"]
    dxdt[1, 0] = dxdt_tau_0
    # ODEs inner control volumes
    dxdt_T = (-v_gas_in*data["density_fluid_in"]*cp_fluid[1:data["Nz"]]
                 * ((T[1:data["Nz"]]-T[0:data["Nz"]-1])/dz[1:data["Nz"]])) \
        / rhocp_g[1:data["Nz"]] \
        - r_eff[1:data["Nz"]]*H_r[1:data["Nz"]]/rhocp_g[1:data["Nz"]] \
        - g_qw[1:data["Nz"]]/rhocp_g[1:data["Nz"]]\
        + Ax_Disp[1:data["Nz"]]/rhocp_g[1:data["Nz"]]
    dxdt_tau = dxdt_T/data["T_scale"]
    dxdt[1, 1:data["Nz"]] = dxdt_tau
    # Flatten to create a long row vector with entries for each component in each control volume
    #dxdt = dxdt.flatten()
    return dxdt


def physics_window(i, out_tensor):
    """
    Substitute the values of current states into the PDE system.

    Parameters:
    - i (int): Index of the current time snapshot.
    - out_tensor (torch.tensor): Output tensor from the GNN model.

    Returns:
    - torch.tensor: substituted PDE system solution.
    """

    x_ = out_tensor.cpu().detach().numpy()


    t = np.load("./data/raw/case_3/time_vec_flow_change.npy")

    t_ = t[i,]

    PDE_sub_sol = PDE_sys(t_, x_)
    ipdb.set_trace()
    return torch.tensor(PDE_sub_sol.T).float()


def train_window(dataset, model, criterion, device, optimizer, hybrid_mode="simple", weight_factor=0.5, derivatives_dataset=None):
    """
    Train the GNN model with window-based dataset.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The training dataset.
    - model: The GNN model.
    - criterion: Loss function.
    - device: Device for training (CPU or GPU).
    - optimizer: Optimizer for updating model parameters.
    - hybrid_mode (str, optional): Type of hybridization ('simple', 'derivative_informed', 'derivative_trained', 'physics_informed'). Defaults to 'simple'.
    - weight_factor (float, optional): Weight factor for hybridization. Defaults to 0.5.
    - derivatives_dataset (torch_geometric.data.Dataset, optional): Dataset with derivatives for derivative-informed hybridization. Defaults to None.

    Returns:
    - float: Average training loss.
    """
    model.train()
    total_loss = 0
    for i in tqdm(range(len(dataset))):
        current_graph, target_x = dataset[i]
        current_graph = current_graph.to(device)
        target_x = target_x.to(device)
        optimizer.zero_grad()
        out = model(current_graph)
        if hybrid_mode == "derivative_informed":
            #deriv_saved_data, deriv_saved_tensor  = derivatives_dataset[i]
            N_w = weight_factor
            if i ==0:
                loss = criterion(out, target_x)
            else:
                previous_graph, previous_target_x = dataset[i-1]
                previous_graph = previous_graph.to(device)
                out_old = model(previous_graph)
                deriv_out = ( out - out_old ) / (0.020011435105774727*2)        #0.002285714285714285
                deriv_target_x = ( target_x - previous_target_x) / (0.020011435105774727 *2)    #0.002285714285714285
                loss = (N_w) * criterion(out, target_x) + (1-N_w) * criterion(deriv_out, deriv_target_x)#deriv_saved_tensor)
        elif hybrid_mode == "physics_informed":
            N_w = weight_factor
            if i ==0:
                loss = criterion(out, target_x)
            else:
                previous_graph, previous_target_x = dataset[i-1]
                previous_graph = previous_graph.to(device)
                out_old = model(previous_graph)
                deriv_out = ( out - out_old ) / (0.020011435105774727*2)
                physics_target_x = physics_window(i, out).to(device)
                loss = (N_w) * criterion(out, target_x) + (1-N_w) * criterion(deriv_out, physics_target_x)
        else:
            loss = criterion(out, target_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataset)

def test_window(dataset, model, criterion, device, hybrid_mode="simple", weight_factor=0.5, derivatives_dataset=None):
    """
    Evaluate the GNN model on the test dataset with window-based dataset.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The test dataset.
    - model: The trained GNN model.
    - criterion: Loss function.
    - device: Device for evaluation (CPU or GPU).
    - hybrid_mode (str, optional): Type of hybridization ('simple', 'derivative_informed', 'derivative_trained', 'physics_informed'). Defaults to 'simple'.
    - weight_factor (float, optional): Weight factor for hybridization. Defaults to 0.5.
    - derivatives_dataset (torch_geometric.data.Dataset, optional): Dataset with derivatives for derivative-informed hybridization. Defaults to None.

    Returns:
    - float: Average test loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            current_graph, target_x = dataset[i]
            current_graph = current_graph.to(device)
            target_x = target_x.to(device)
            out = model(current_graph)
            if hybrid_mode == "derivative_informed":
                #deriv_saved_data, deriv_saved_tensor  = derivatives_dataset[i]
                N_w = weight_factor
                if i ==0:
                    loss = criterion(out, target_x)
                else:
                    previous_graph, previous_target_x = dataset[i-1]
                    previous_graph = previous_graph.to(device)
                    out_old = model(previous_graph)
                    deriv_out = ( out - out_old ) / (0.020011435105774727*2)        #0.002285714285714285
                    deriv_target_x = ( target_x - previous_target_x) / (0.020011435105774727 *2)    #0.002285714285714285
                    loss = (N_w) * criterion(out, target_x) + (1-N_w) * criterion(deriv_out, deriv_target_x)#deriv_saved_tensor)
            elif hybrid_mode == "physics_informed":
                N_w = weight_factor
                if i ==0:
                    loss = criterion(out, target_x)
                else:
                    previous_graph, previous_target_x = dataset[i-1]
                    previous_graph = previous_graph.to(device)
                    out_old = model(previous_graph)
                    deriv_out = ( out - out_old ) / (0.020011435105774727*2)
                    physics_target_x = physics_window(i, out).to(device)
                    loss = (N_w) * criterion(out, target_x) + (1-N_w) * criterion(deriv_out, physics_target_x)
            else:
                loss = criterion(out, target_x)
            total_loss += loss.item()

    avg_loss = total_loss / (len(dataset))
    return avg_loss

def predict_future_snapshots_window(model, last_training_graph, num_future_snapshots, hybrid_mode="simple"):
    """
    Predict future snapshots based on the last training graph with window-based dataset.

    Parameters:
    - model: The trained GNN model.
    - last_training_graph: The last graph from the training set (PyTorch Geometric Data object).
    - num_future_snapshots (int): The number of future time snapshots to predict.
    - hybrid_mode (str, optional): Type of hybridization ('simple', 'derivative_informed', 'derivative_trained', 'physics_informed'). Defaults to 'simple'.

    Returns:
    - df_conversion: DataFrame with predicted conversion values.
    - df_temperature: DataFrame with predicted temperature values.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    current_graph = last_training_graph.to(device)

    conversion_predictions = []
    temperature_predictions = []
    if hybrid_mode == "derivative_trained":
        derivatives_conversion_predictions = []
        derivatives_temperature_predictions = []

    with torch.no_grad():
        for _ in tqdm(range(num_future_snapshots), desc="Predicting the future"):
            out_features = model(current_graph)

            if hybrid_mode == "derivative_trained":
                conversion, temperature, derivatives_conversion, derivatives_temperature = out_features[:, 0], out_features[:, 1], out_features[:, 2], out_features[:, 3]

                conversion_predictions.append(conversion.cpu().numpy())
                temperature_predictions.append(temperature.cpu().numpy())
                derivatives_conversion_predictions.append(derivatives_conversion.cpu().numpy())
                derivatives_temperature_predictions.append(derivatives_temperature.cpu().numpy())

                shift_time_snapshot(current_graph.x, out_features)

            else:
                conversion, temperature = out_features[:, 0], out_features[:, 1]

                conversion_predictions.append(conversion.cpu().numpy())
                temperature_predictions.append(temperature.cpu().numpy())

                shift_time_snapshot(current_graph.x, out_features)

    if hybrid_mode == "derivative_trained":
        df_conversion = pd.DataFrame(conversion_predictions).transpose()
        df_temperature = pd.DataFrame(temperature_predictions).transpose()
        df_derivatives_conversion = pd.DataFrame(derivatives_conversion_predictions).transpose()
        df_derivatives_temperature = pd.DataFrame(derivatives_temperature_predictions).transpose()

        return df_conversion, df_temperature, df_derivatives_conversion, df_derivatives_temperature

    else:
        df_conversion = pd.DataFrame(conversion_predictions).transpose()
        df_temperature = pd.DataFrame(temperature_predictions).transpose()

        return df_conversion, df_temperature
