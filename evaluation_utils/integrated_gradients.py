## This script is used to generate the integrated gradients for the top 20 features for the entire dataset, astrocytoma wt, astrocytoma mut, and oligodendroglioma for TCGA-GBMLGG genomic SNN
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import numpy as np
import pickle
import torch
from training_utils.data_loaders import *
from data_utils.options import parse_args
from evaluation_utils.captum_data_retriever import retrieve_captum_data
import torch_geometric
print(torch_geometric.__version__)
from training_utils.result_plots import save_metric_logger, plots_train_vs_test
from data_utils.option_file_converter import parse_opt_file
from captum.attr import IntegratedGradients
from evaluation_utils.networks_captum import define_net
import matplotlib.pyplot as plt
from training_utils.utils import getCleanAllDataset
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

checkpoints_dir = "./checkpoints/TCGA_GBMLGG"
results_folder = "results_3"

def info_importer():
    """
    Function to import the data from the checkpoints directory based on the train.opt.txt file and additional settings
    """
    file_path = os.path.join(checkpoints_dir, setting, mode)
    opt = parse_opt_file(os.path.join(file_path, "train_opt.txt"))
    opt.use_rnaseq = 0
    # opt.input_size_omic = 80

    # Adding in changes away from default opmodel options
    opt.dataroot = './data/TCGA_GBMLGG'
    opt.verbose = 1
    opt.print_every = 1
    opt.checkpoints_dir = checkpoints_dir
    opt.vgg_features = 0
    opt.use_vgg_features = 0
    opt.gpu_ids = []

    if setting=="grad_15" and mode=='pathgraphomic_fusion':
        opt.model_name = opt.model

    if setting=="surv_15_rnaseq" and mode=="omic":
        opt.model_name = opt.model

    # RNASeq setting
    if "omic" in mode:
        opt.use_rnaseq = 1
        opt.input_size_omic = 320
    else:
        opt.use_rnaseq = 0
    #   opt.input_size_omic = 80


    # Added in code to print changed attributes
    for attr, value in vars(opt).items():
        print(f"{attr} = {value}")

    # Set device to MPS if GPU is available
    device = (
        torch.device("mps:{}".format(opt.gpu_ids[0]))
        if opt.gpu_ids
        else torch.device("cpu")
    )
    print("Using device:", device)

    # 1 if the string grad is found in opt.task, 0 otherwise
    ignore_missing_histype = 1 if "grad" in opt.task else 0
    ignore_missing_moltype = 1 if "omic" in opt.mode else 0

    use_patch, roi_dir = (
        ("_patch_", "all_st_patches_512") if opt.use_vgg_features else ("_", "all_st")
    )

    # Use_rnaseq defaults to 0
    use_rnaseq = "_rnaseq" if opt.use_rnaseq else ""

    data_cv_path = "%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl" % (
        opt.dataroot,
        roi_dir,
        ignore_missing_moltype,
        ignore_missing_histype,
        opt.use_vgg_features,
        use_rnaseq,
    )
    print("Loading %s" % data_cv_path)

    data_cv = pickle.load(open(data_cv_path, "rb"))

    # Extracts the cv_splits from the data
    data_cv_splits = data_cv['cv_splits']
    k = 1
    data = data_cv_splits[k]
    return data, opt, device, k

# Model on which to do gradient attribution
setting = "surv_15_rnaseq"
mode = "omic"

data, opt, device, k = info_importer()

# Load in the trained model
checkpoint = torch.load(
    os.path.join(
        opt.checkpoints_dir,
        opt.exp_name,
        opt.model_name,
        "%s_%d.pt" % (opt.model_name, k),
    )
)

model = define_net(opt, k)
model.eval()

# Load the model state dict
model.load_state_dict(checkpoint["model_state_dict"])   
X_patname = data['train']['x_patname']
X_patname_test = data['test']['x_patname']
pat_comb = np.concatenate((X_patname, X_patname_test), axis=0)

X_omic = data['train']['x_omic']
X_omic_test = data['test']['x_omic']
omic_comb = np.concatenate((X_omic, X_omic_test), axis=0)

attributions =   []
for i in range(omic_comb.shape[0]): # Loop through rows
    row = omic_comb[i, :]
    input_tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
    input_tensor.requires_grad = True

    ## Captum
    ig = IntegratedGradients(model)
    # model.to(device)

    # Calculate attributions
    out = ig.attribute(input_tensor.to(device), n_steps=100, target=0)
    out = out.detach().numpy()
    attributions.append(out)

attributions = np.vstack(attributions)


def formatting_attributes(attributions, omic_comb):
    """
    Function to format the attributions and omic data for easy plotting"""
    # Get names of the attributions
    metadata, all_dataset = getCleanAllDataset(use_rnaseq=True)
    dataset_filtered = all_dataset.iloc[:, 7:]
    column_names = dataset_filtered.columns
    attributions_df = pd.DataFrame(attributions, columns=column_names)

    # Compute the mean abs attributions for each feature
    mean_attributions = np.mean(abs(attributions_df), axis=0) # Mean for each column
    # Get the indices of the top 20 attributions
    top_20_indices = np.argsort(mean_attributions)[-20:]
    top_features = np.array(column_names)[top_20_indices] # Get the names of the top 20 features

    # Get the data for the top 20 features and attributions
    top_features_data = omic_comb[:, top_20_indices]
    top_attributions = attributions[:, top_20_indices] 

    # Put it into a df for easy use
    top_attributions_df = pd.DataFrame(top_attributions, columns=top_features)
    top_features_data_df = pd.DataFrame(top_features_data, columns=top_features)

    return top_attributions_df, top_features_data_df, top_features

# # Make a label mapping
label_mapping = {
    'PTEN': 'PTEN (CNV)',
    'PROKR2_rnaseq': 'PROKR2 (RNA)',
    '10q': '10q (CNV)',
    '10p': '10p (CNV)',
    'CDKN2A': 'CDKN2A (CNV)',
    'TUBA3C_rnaseq': 'TUBA3C (RNA)',
    'CDKN2B': 'CDKN2B (CNV)',
    'idh mutation': 'IDH (mut)',
    'AFF2_rnaseq': 'AFF2 (RNA)',
    'FAM47A_rnaseq': 'FAM47A (RNA)',
    'SAMD9_rnaseq': 'SAMD9 (RNA)',
    'ACAN_rnaseq': 'ACAN (RNA)',
    'FLG_rnaseq': 'FLG (RNA)',
    'MUC5B_rnaseq': 'MUC5B (RNA)',
    'RPL5_rnaseq': 'RPL5 (RNA)',
    'LRP1B_rnaseq': 'LRP1B (RNA)',
    'EPPK1_rnaseq': 'EPPK1 (RNA)',
    'ANO2_rnaseq': 'ANO2 (RNA)',
    'LRFN5_rnaseq': 'LRFN5 (RNA)',
    'MYC': 'MYC (CNV)',
    'EBF1': 'EBF1 (CNV)',
    'FGFR2': 'FGFR2 (CNV)',
    'BRCA2_rnaseq': 'BRCA2 (RNA)',
    '7q': '7q (CNV)',
    '20p': '20p (CNV)',
    'HRNR_rnaseq': 'HRNR (RNA)',
    'USH2A_rnaseq': 'USH2A (RNA)',
    'TSHZ2_rnaseq': 'TSHZ2 (RNA)',
    'ZNF208_rnaseq': 'ZNF208 (RNA)',
    'ARFGEF3_rnaseq': 'ARFGEF3 (RNA)'

}

# Plotting the integrated gradients
def plot_captum(top_attributions_df, top_features_data_df, top_features, save_name):
    plt.figure(figsize=(10, 8))

    for i, feature in enumerate(top_features):
        attribution_values = top_attributions_df[feature] 
        feature_values = top_features_data_df[feature]
        # Normalize feature values to range [0, 1]
        normalized_feature_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())

        normalized_feature_values = normalized_feature_values.fillna(0.5)  # Fill NaN with 0.5

        # Define the color dictionary
        cdict = {
            'red':   [(0.0, 1.0, 1.0),  # At position 0.0, the red component is 1.0 (full intensity)
                    (1.0, 0.0, 0.0)], # At position 1.0, the red component is 0.0 (no intensity)
            
            'green': [(0.0, 0.0, 0.0),  # At position 0.0, the green component is 0.0 (no intensity)
                    (1.0, 0.0, 0.0)], # At position 1.0, the green component is 0.0 (no intensity)

            'blue':  [(0.0, 0.0, 0.0),  # At position 0.0, the blue component is 0.0 (no intensity)
                    (1.0, 1.0, 1.0)]  # At position 1.0, the blue component is 1.0 (full intensity)
        }

        # Create the colormap
        cmap = LinearSegmentedColormap('RedBlue', cdict)
        jitter = np.random.normal(0, 0.03, size=top_attributions_df[feature].shape[0])
        y_values = np.ones(top_attributions_df.shape[0]) * i + jitter
        colors = cmap(normalized_feature_values) # Create colors based on the normalized feature values
        plt.scatter(attribution_values, y_values, c=colors, alpha=0.5, label=feature, s=10)

    # Apply the mapping
    new_labels = [label_mapping.get(label, label) for label in top_features]

    plt.yticks(ticks=range(len(top_features)), labels=new_labels)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('Integrated Gradient Attribution')
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca(), orientation='vertical', label='Relative Feature Value')
    eval_folder = 'evaluation'
    plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'{save_name}.png')
    plt.show()
    plt.savefig(plot_filepath)
    return None

top_attributions_df, top_features_data_df, top_features = formatting_attributes(attributions, omic_comb)
plot_captum(top_attributions_df, top_features_data_df, top_features, 'integrated_gradients_all')

metadata, all_dataset = getCleanAllDataset(use_rnaseq=True)
astro_wt = all_dataset[all_dataset.iloc[:, 0]=='idhwt_ATC']['TCGA ID']
astro_mut =  all_dataset[all_dataset.iloc[:, 0]=='idhmut_ATC']['TCGA ID']
oligodendro = all_dataset[all_dataset.iloc[:, 0]=='ODG']['TCGA ID']

# Find relevant indices for attributions and data for attributions and feature
astro_wt_idx = [i for i, pat in enumerate(pat_comb) if pat in astro_wt.values]
astro_mut_idx = [i for i, pat in enumerate(pat_comb) if pat in astro_mut.values]
oligodendro_idx = [i for i, pat in enumerate(pat_comb) if pat in oligodendro.values]

# Filter the features based on patname
astro_wt_data = omic_comb[astro_wt_idx]
astro_mut_data = omic_comb[astro_mut_idx]
oligodendro_data = omic_comb[oligodendro_idx]

# Filter the attributions based on patname
astro_wt_attributions = attributions[astro_wt_idx]
astro_mut_attributions = attributions[astro_mut_idx]
oligodendro_attributions = attributions[oligodendro_idx]

top_attributions_astro_wt, top_features_data_astro_wt, top_features_astro_wt = formatting_attributes(astro_wt_attributions, astro_wt_data)
plot_captum(top_attributions_astro_wt, top_features_data_astro_wt, top_features_astro_wt, 'integrated_gradients_astro_wt')

top_attributions_astro_mut, top_features_data_astro_mut, top_features_astro_mut = formatting_attributes(astro_mut_attributions, astro_mut_data)
plot_captum(top_attributions_astro_mut, top_features_data_astro_mut, top_features_astro_mut, 'integrated_gradients_astro_mut')

top_attributions_odg, top_features_data_odg, top_features_odg= formatting_attributes(oligodendro_attributions, oligodendro_data)
plot_captum(top_attributions_odg, top_features_data_odg, top_features_odg, 'integrated_gradients_odg')

