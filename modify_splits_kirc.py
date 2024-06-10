import pickle
import pandas as pd
import os
import numpy as np

pkl_path = "./data/TCGA_KIRC/splits/KIRC_st_0_og.pkl"
data_cv = pickle.load(open(pkl_path, "rb"))
clinical_data = pd.read_csv(os.path.join("./data/TCGA_KIRC/", "kirc_tcga_pan_can_atlas_2018_clinical_data.tsv"), delimiter='\t')

# Filter the clinical data into just the columns I'm interested in
clinical_data = clinical_data[['Patient ID', 'Diagnosis Age', 'Sex', 'Buffa Hypoxia Score', 'Ragnum Hypoxia Score', 'Winter Hypoxia Score', 'Neoplasm Disease Stage American Joint Committee on Cancer Code', 'American Joint Committee on Cancer Metastasis Stage Code', 'Primary Lymph Node Presentation Assessment', 'Prior Diagnosis']]

# One hot-encode certain columns
clinical_data = pd.get_dummies(clinical_data, columns=['Sex'], drop_first=True)

# Fixing columns
stage_to_int = {
    'STAGE I': 1,
    'STAGE II': 2,
    'STAGE III': 3,
    'STAGE IV': 4
}

metastasis_to_int = {
    'M0': 0,  # No distant metastasis
    'M1': 1   # Distant metastasis present
}


lymph_presentation_to_int = {
    'No': 0,
    'Yes':1
}

tumor_stage_to_int = {
    'T0': 0,
    'T1': 1,
    'T2': 2,
    'T3': 3,
    'T4': 4
}

# Applying the mapping
clinical_data['Metastasis_Stage'] = clinical_data['American Joint Committee on Cancer Metastasis Stage Code'].replace(metastasis_to_int)
clinical_data['Grade'] = clinical_data['Neoplasm Disease Stage American Joint Committee on Cancer Code'].replace(stage_to_int)
clinical_data['Lymph_Presentation'] = clinical_data['Primary Lymph Node Presentation Assessment'].replace(lymph_presentation_to_int)

# Drop old column names
clinical_data.drop(columns=['American Joint Committee on Cancer Metastasis Stage Code', 
                            'Neoplasm Disease Stage American Joint Committee on Cancer Code', 
                            'Primary Lymph Node Presentation Assessment'], inplace=True)

clinical_data = clinical_data[clinical_data['Metastasis_Stage']!='MX']
clinical_data['Metastasis_Stage'] = clinical_data['Metastasis_Stage'].astype(float)

# In order to one-hot encode the 'Prior Diagnosis' column, we need to first clean it up
def encode_yes_no(value):
    # Convert the value to lowercase to make the search case-insensitive
    value = str(value).lower()
    if 'yes' in value:
        return 1
    elif 'no' in value:
        return 0
    else:
        return pd.NA 
clinical_data['Prior Diagnosis'] = clinical_data['Prior Diagnosis'].apply(encode_yes_no)

clinical_data = clinical_data.dropna()

# Add the clinical data to the splits
data_cv_splits = data_cv['split']
results = []

for k in range(1, 16):
    data = data_cv_splits[k]
    # Loop through train and test splits
    for split in data.keys():
        # Initialise everything
        patient_with_clin = []
        # omic_with_clin = []
        # path_with_clin = []
        # graph_with_clin = []
        # e_with_clin = []
        # t_with_clin = []
        # g_with_clin = []
        clin_data = []
        index_list = []
        n = -1

        patients = data[split]['x_patname']
        # Find out the patients with clinical data
        for patient in patients:
            n += 1
            if patient in clinical_data['Patient ID'].values:
                # Save relevant data to lists from existing pkl files
                patient_with_clin += [patient]
                # Save the clinical data to a list
                patient_data = clinical_data[clinical_data['Patient ID'] == patient].drop(columns=['Patient ID']).to_numpy().flatten()
                clin_data.append(patient_data)
                index_list.append(n)
        clin_matrix = np.vstack(clin_data)
        try:
            clin_matrix = clin_matrix.astype(np.float32)
        except ValueError as e:
            print("Array contains non-numeric data:", e)
    # Handle non-numeric data conversion here if needed
        # Replace the clinical data with the dictionary
        data[split]['x_patname'] = patient_with_clin
        data[split]['x_clin'] = clin_matrix
        data[split]['x_omic'] = np.take(data['train']['x_omic'],index_list, axis =0)
        data[split]['x_path'] = np.take(data['train']['x_path'],index_list, axis =0)
        data[split]['x_grph'] = np.take(data['train']['x_grph'],index_list, axis =0)
        data[split]['e'] = np.take(data['train']['e'],index_list, axis =0)
        data[split]['t'] = np.take(data['train']['t'],index_list, axis =0)
        data[split]['g'] = np.take(data['train']['g'],index_list, axis =0)

# Dump into pkl file
import os
import pickle

# Create the directory if it doesn't exist
os.makedirs('./data/TCGA_KIRC/splits/', exist_ok=True)

# Create the file and dump the data
with open('./data/TCGA_KIRC/splits/KIRC_st_0_clinical.pkl', 'wb') as file:
    pickle.dump(data_cv_splits, file)
