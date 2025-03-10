import os
import numpy as np
import ipdb


# Function to explore the data for a specific case
def explore_data(case):
    """
    Explore the data for a specific case.

    Parameters:
    - case (int): The case number to explore.

    Returns:
    None
    """
    # List of file names for the data files
    file_names = [
        'conversion_vec',
        #'derivatives_conversion_vec',
        #'derivatives_temperature_vec',
        'temperature_vec',
        'time_vec',
        'z_vec'
    ]

    # Dictionary mapping case numbers to case names
    cases_dict = {
        1: 'cold_start_up',
        2: 'warm_start_up',
        3: 'flow_change',
        4: 'control_change'
    }

    # Construct folder and case name based on the provided case number
    case_folder = 'case_' + str(case)
    case_name = cases_dict[case]

    # Dictionary to store loaded data arrays
    data_dict = {}

    # Get the current working directory
    base_dir = os.getcwd()

    # Print separator for clarity
    print('-'*50)

    # Loop through each file name in the list
    for name in file_names:
        # Construct file path for the current data file
        file_path = os.path.join(base_dir, 'data', 'raw', case_folder,
                                 f'{name}_{case_name}.npy')
        # Load data from the file into a np array and store it in the data dictionary
        data_dict[name] = np.load(file_path)

        # Print the name of the data and its shape
        print(name, '   ', data_dict[name].shape)

        # If it's the time vector, print the last time point
        if name == 'time_vec':
            print('Last time point:', '   ', data_dict[name][-1], 's')

        # If it's the z vector, print the last space point
        if name == 'z_vec':
            print('Last space point:', '   ', data_dict[name][-1], 'm')

        # Print separator for clarity
        print('-'*50)
