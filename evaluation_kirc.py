# %%
from core.utils_data import getCleanGBMLGG
import pandas as pd
import numpy as np
from tqdm import tqdm
from core.utils_analysis_new import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
# from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
from lifelines import KaplanMeierFitter
import pickle
from scipy.stats import zscore

# Making the loss curves nicer
# Load in the saved model information from the checkpoints
import math
import os
import os

import math
import pandas as pd
import matplotlib.pyplot as plt


# %%
#### LOSS & METRIC PLOTS - SURVIVAL ####
# List of model names
model_names = ['path', 'clin', 'omic', 'pathomic_fusion', 'clinomic_fusion',  'pathclinomic_fusion']

# Define relabeling dictionaries
model_mappings = {
    'omic': 'Genomic SNN',
    'path': 'Histology CNN',
    'clin': 'Clinical SNN',
    'pathomic_fusion': 'Pathomic Fusion',
    'clinomic_fusion': 'Clinomic Fusion',
    'pathclinomic_fusion': 'Pathclinomic',
    'omicomic_fusion': 'Omicomic Fusion', 
    'pathpath_fusion': 'Pathpath Fusion',  
    'clinclin_fusion': 'Graphgraph Fusion'  
}
metric_mapping = {
    'loss': 'Loss',
    'cindex': 'C-index',
    'surv_acc': 'Survival Accuracy'
}

checkpoints_dir = './checkpoints/TCGA_KIRC'
metrics = ['loss', 'cindex', 'surv_acc']
eval_folder = 'evaluation'
exp_name = 'surv_15'

# Initialize min and max values for loss and accuracy
global_min_loss = float('inf')
global_max_loss = float('-inf')
global_min_acc = float('inf')
global_max_acc = float('-inf')
global_min_cindex = float('inf')
global_max_cindex = float('-inf')

# Find global min and max loss and accuracy values
for model_name in model_names:
    results_dir = os.path.join(checkpoints_dir, exp_name, 'results')
    csv_filepath = os.path.join(results_dir, f'{model_name}_metrics_1.csv')
    df = pd.read_csv(csv_filepath)

    # Update global min and max loss values
    global_min_loss = min(global_min_loss, df['train_loss'].min(), df['test_loss'].min())
    global_max_loss = max(global_max_loss, df['train_loss'].max(), df['test_loss'].max())

    # Update global min and max accuracy values
    global_min_acc = min(global_min_acc, df['train_surv_acc'].min(), df['test_surv_acc'].min())
    global_max_acc = max(global_max_acc, df['train_surv_acc'].max(), df['test_surv_acc'].max())

    # Update global min and max accuracy values
    global_min_c = min(global_min_acc, df['train_cindex'].min(), df['test_cindex'].min())
    global_max_c = max(global_max_acc, df['train_cindex'].max(), df['test_cindex'].max())


# Calculate the number of rows and columns for the grid
num_cols = len(model_names)
num_rows = 3  # We have three rows of metrics

exp_name = 'surv_15'

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))

for metric_index, metric in enumerate(metrics):
    for i, model_name in enumerate(model_names):
        results_dir = os.path.join(checkpoints_dir, exp_name, 'results')
        csv_filepath = os.path.join(results_dir, f'{model_name}_metrics_1.csv')
        df = pd.read_csv(csv_filepath)

        train_metric = df[f'train_{metric}']
        test_metric = df[f'test_{metric}']

        row = metric_index
        col = i

        ax = axes[row, col]
        ax.plot(train_metric, label='train')
        ax.plot(test_metric, label='test', linestyle='--')
        if row == 0:
            ax.set_title(model_mappings.get(model_name, model_name))
        if row == num_rows - 1:
            ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_mapping.get(metric, metric))
        if col == 0:
            ax.legend()
        if metric == 'loss':
            # ax.set_ylim(0.5, 1.5)
            ax.set_ylim(global_min_loss, global_max_loss) # For KIRC
        elif metric == 'surv_acc':
            # ax.set_ylim(0.6, 1)
            ax.set_ylim(0, 1) # For KIRC
        else:
            # ax.set_ylim(0.5, 0.85) 
            ax.set_ylim(0, 1) # For KIRC

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Create the directory if it does not exist
os.makedirs(os.path.join(checkpoints_dir, eval_folder), exist_ok=True)

# Save the plot
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'{exp_name}_performance_plot_kirc.png')
plt.savefig(plot_filepath)

# %% 
#### Kaplan- Meier Curves for all models for overall survival only
model_names = ['path', 'clin', 'omic', 'pathomic_fusion', 'clinomic_fusion',  'pathclinomic_fusion']

# Define relabeling dictionaries
model_mappings = {
    'omic': 'Genomic SNN',
    'path': 'Histology CNN',
    'clin': 'Clinical SNN',
    'pathomic_fusion': 'Pathomic Fusion',
    'clinomic_fusion': 'Clinomic Fusion',
    'pathclinomic_fusion': 'Pathclinomic',
    'omicomic_fusion': 'Omicomic Fusion', 
    'pathpath_fusion': 'Pathpath Fusion',  
    'clinclin_fusion': 'Graphgraph Fusion'  
}

split = 'test'
use_patch = '_'
k=1
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
ckpt_name = './checkpoints/TCGA_KIRC/surv_15'
model = 'path'
with open(f'{ckpt_name}/{model}/{model}_{k}pred_{split}_append.pkl', 'rb') as file:
    pred_test, data_cv = pickle.load(file)
    print(pred_test, data_cv.keys())

for j, model in enumerate(['path', 'omic', 'clin', 'pathomic_fusion']):
    data = load_and_process_survival_kirc_data(model, 1, split)
    df = pd.read_csv('./data/TCGA_KIRC/kirc_tcga_pan_can_atlas_2018_clinical_data.tsv', sep='\t')
    data = data.merge(df, on='Patient ID', how='left') 
    print(data.shape)
    model_name = model_mappings.get(model, 'Unknown Model')
  
    # Grade status information
    mapping = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3}
    grade_status = data['Neoplasm Histologic Grade'].map(mapping)

    hazards = data['Hazards']
    censor_status = data['Censor Status']
    survival_times = data['Survival Time']
    survival_times = survival_times // 12

    # Kaplan-Meier curve by grade status
    kmf = KaplanMeierFitter()

    # Example hazard values and threshold probabilities
    thresholds = [25, 50, 75, 100]
    # Assign grades to hazard values
    percentiles_of_hazards = np.percentile(hazards, thresholds)
    # Assign grades to hazard values based on percentiles
    grade_status_predicted = np.array([hazard2grade(h, percentiles_of_hazards) for h in hazards])

    # Plot the Kaplan-Meier curves
    colours = ['green', 'blue', 'red', 'purple']
    colours2 = ['green', 'blue', 'red', 'purple']
    colours3 = ['green', 'blue', 'red', 'purple']

    for grade, color in zip([0, 1, 2, 3], colours):

        mask = (grade_status == grade)
        kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'True Grade {grade+2}')
        # kmf.plot_survival_function(ci_show=False, color=color, linestyle='-')
        kmf.survival_function_
        axs[j].plot(kmf.survival_function_, linestyle="-", color=color)
        if j == 0:
            axs[j].set_ylabel('Overall\nProportion Surviving')

    for grade, color in zip([0, 1, 2, 3], colours):
        if grade ==3:
            grade = 3 or 4
        mask = (grade_status_predicted == grade)
        kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'Predicted Grade {grade+2} from Hazards')
        # kmf.plot_survival_function(ci_show=False, color=color, linestyle='--')
        kmf.survival_function_
        axs[j].plot(kmf.survival_function_, linestyle="--", color=color)
        axs[j].set_xlim([0, 20])
        axs[j].set_title(f'{model_name}')
    axs[j].set_xlabel('Survival time (years)')

plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'kaplan_meier_curves.png')
plt.savefig(plot_filepath)
plt.show()
plt.close()


# %%
#### HISTOGRAM HAZARD PLOTS
k = 1
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the figsize to suit your display needs
data = load_and_process_survival_data('path', k, use_patch, split)

for j, model in enumerate(['graph', 'path', 'omic', 'pathgraphomic_fusion']):
    data = load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/')
    ax = axes[j]
    model_name = model_mappings.get(model, 'Unknown Model') 
    ## Z-scored hazards histograms
    # Calculate the z-scores of the 'Hazards'
    data['Hazards_z'] = zscore(data['Hazards'])

    # Split the data based on the survival time with low and high being thresholded by 5 years
    low = data[data['Survival Time'] <= 12*3.5]
    high = data[data['Survival Time'] > 12*3.5]

    # Histogram plotting with normalized density using plt directly
    sns.histplot(low['Hazards_z'], ax=ax,bins=10, kde=False, stat="density", color="red", alpha=0.5, edgecolor="black", label='<= 5 Years')
    sns.histplot(high['Hazards_z'], ax=ax, bins=10,kde=False, stat="density",color="blue", alpha=0.5, edgecolor="black", label='> 5 Years')

    # Adjusting the visual layout of the plot
    ax.set_xlabel('Hazard Values (Z-score)')
    ax.set_ylabel('Density')
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # ax.tick_params(axis='y', which='both', labelsize=10)
    # ax.tick_params(axis='x', which='both', labelsize=10)
    ax.set_xlim([-2, 2])
    ax.set_ylim([0, 2])

    # Title for each subplot
    ax.set_title(f'{model_name}')

    # Adding legend to each subplot
    ax.legend(title="Survival Time")

# Adjust layout to prevent overlapping
plt.tight_layout()   
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'hazard_histograms_kirc.png')
plt.show()
plt.savefig(plot_filepath)
plt.close()

# %%
