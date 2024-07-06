import numpy as np
import random



def filter_unique_patients(data_dict):
    filtered_data = {}

    for dataset in ['train', 'test']:  # Loop over both 'train' and 'test' datasets
        seen_patients = set()  # Keep track of seen patient names
        filtered_data[dataset] = {key: [] for key in data_dict[dataset]}  # Initialize filtered data structure
        
        for i, patname in enumerate(data_dict[dataset]['x_patname']):  # Loop over patient names
            if patname not in seen_patients:  # Check if the patient name has already been seen
                seen_patients.add(patname)  # Add the patient name to the seen set
                for key in filtered_data[dataset]:  # Loop over all keys to add the corresponding data
                    filtered_data[dataset][key].append(data_dict[dataset][key][i])
        
        # Convert lists back to numpy arrays where necessary
        filtered_data[dataset]['x_path'] = np.array(filtered_data[dataset]['x_path'])
        filtered_data[dataset]['x_omic'] = np.array(filtered_data[dataset]['x_omic'])
        filtered_data[dataset]['x_grph'] = np.array(filtered_data[dataset]['x_grph'])
        filtered_data[dataset]['e'] = np.array(filtered_data[dataset]['e'])
        filtered_data[dataset]['t'] = np.array(filtered_data[dataset]['t'])
        filtered_data[dataset]['g'] = np.array(filtered_data[dataset]['g'])
    return filtered_data

def filter_unique_patients_ordered(data_dict):
    filtered_data = {}

    for dataset in ['train', 'test']:  # Loop over both 'train' and 'test' datasets
        seen_patients = set()  # Keep track of seen patient names
        filtered_data[dataset] = {key: [] for key in data_dict[dataset]}  # Initialize filtered data structure
        
        patient_indices = {}  # To store indices of each patient name

        for i, patname in enumerate(data_dict[dataset]['x_patname']):  # Loop over patient names
            if patname not in patient_indices:
                patient_indices[patname] = []
            patient_indices[patname].append(i)

        for patname, indices in patient_indices.items():
            random_index = random.choice(indices)  # Select a random instance for each unique patient
            seen_patients.add(patname)  # Add the patient name to the seen set
            for key in filtered_data[dataset]:  # Loop over all keys to add the corresponding data
                filtered_data[dataset][key].append(data_dict[dataset][key][random_index])
        
        # Convert lists back to numpy arrays where necessary
        filtered_data[dataset]['x_path'] = np.array(filtered_data[dataset]['x_path'])
        filtered_data[dataset]['x_omic'] = np.array(filtered_data[dataset]['x_omic'])
        filtered_data[dataset]['x_grph'] = np.array(filtered_data[dataset]['x_grph'])
        filtered_data[dataset]['e'] = np.array(filtered_data[dataset]['e'])
        filtered_data[dataset]['t'] = np.array(filtered_data[dataset]['t'])
        filtered_data[dataset]['g'] = np.array(filtered_data[dataset]['g'])
    return filtered_data
