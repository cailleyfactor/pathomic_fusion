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
#### LOSS & METRIC PLOTS - GRADE

checkpoints_dir = './checkpoints/TCGA_GBMLGG'
metrics = ['loss', 'grad_acc']
model_names = ['path', 'graph', 'omic', 'pathomic_fusion', 'graphomic_fusion', 'pathgraph_fusion', 'pathgraphomic_fusion']
eval_folder = 'evaluation'

num_cols = len(model_names)
num_rows = 2  # One row for loss plots, one row for accuracy plots
exp_name = 'grad_15'

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
fig.suptitle('Model Performance', fontsize=16)

for i, model_name in enumerate(model_names):
    results_dir = os.path.join(checkpoints_dir, exp_name, 'results')
    csv_filepath = os.path.join(results_dir, f'{model_name}_metrics.csv')
    df = pd.read_csv(csv_filepath)

    # Plot loss
    train_loss = df['train_loss']
    test_loss = df['test_loss']
    ax = axes[0, i]
    ax.plot(train_loss, label='train')
    ax.plot(test_loss, label='test', linestyle='--')
    ax.set_title(model_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if i == 0:
        ax.legend()

    # Plot accuracy
    train_acc = df['train_grad_acc']
    test_acc = df['test_grad_acc']
    ax = axes[1, i]
    ax.plot(train_acc, label='train')
    ax.plot(test_acc, label='test', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Grade Accuracy')
    if i == 0:
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Create the directory if it does not exist
os.makedirs(os.path.join(checkpoints_dir, eval_folder), exist_ok=True)

# Save the plot
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'{exp_name}_performance_plot.png')
plt.savefig(plot_filepath)
plt.close()

# %%
#### LOSS & METRIC PLOTS - SURVIVAL ####

checkpoints_dir = './checkpoints/TCGA_GBMLGG'
metrics = ['loss', 'cindex', 'surv_acc']
eval_folder = 'evaluation'

num_cols = len(model_names)
num_rows = 3  # We have three rows of metrics

exp_name = 'surv_15_rnaseq'

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
fig.suptitle('Model Performance', fontsize=16)

for metric_index, metric in enumerate(metrics):
    for i, model_name in enumerate(model_names):
        results_dir = os.path.join(checkpoints_dir, exp_name, 'results')
        csv_filepath = os.path.join(results_dir, f'{model_name}_metrics.csv')
        df = pd.read_csv(csv_filepath)

        train_metric = df[f'train_{metric}']
        test_metric = df[f'test_{metric}']

        row = metric_index
        col = i

        ax = axes[row, col]
        ax.plot(train_metric, label='train')
        ax.plot(test_metric, label='test', linestyle='--')
        if row == 0:
            ax.set_title(model_name)
        if row == num_rows - 1:
            ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        if col == 0:
            ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Create the directory if it does not exist
os.makedirs(os.path.join(checkpoints_dir, eval_folder), exist_ok=True)

# Save the plot
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'{exp_name}_performance_plot.png')
plt.savefig(plot_filepath)
plt.close()

#### ####
# %%
## This code may no longer be needed
##

exp_name = 'surv_15_rnaseq'
metrics = ['loss', 'cindex', 'pvalue', 'surv_acc']
# Calculate the number of rows and columns for the gri

for metric in metrics:
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.suptitle(f'{metric} plot', fontsize=16)

    for i, model_name in enumerate(model_names):
        row = i // num_cols
        col = i % num_cols

        results_dir = os.path.join(checkpoints_dir, exp_name, 'results')
        csv_filepath = os.path.join(results_dir, f'{model_name}_metrics.csv')
        df = pd.read_csv(csv_filepath)
        train_metric = df[f'train_{metric}']
        test_metric = df[f'test_{metric}']

        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.plot(train_metric)
        ax.plot(test_metric)
        ax.set_xlabel(model_name)
        ax.set_ylabel(metric)
        ax.legend()

    # Remove any unused subplots
    for j in range(len(model_names), num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'{exp_name}_{metric}_plot.png')
    plt.savefig(plot_filepath)
    plt.close()


models = ['graph', 'path', 'omic', 'pathgraphomic_fusion']

# %%
#### ROC, AUC Plots - formatted

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for idx, model in enumerate(models):
    # Get predictions
    y_label, y_pred = getPredAggGrad_GBMLGG(model=model, agg_type='max') 
    y_label = np.squeeze(y_label)
    y_pred = np.squeeze(y_pred)
    
    # Initialize a figure
    ax = axes[idx]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_label.shape[1]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot micro-average ROC curve
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    
    # Plot macro-average ROC curve
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    
    # Plot ROC curves for each class
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve for class {i} (AUC = {roc_auc[i]:.2f})"
        )
    
    # Plot chance level
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Customize the plot
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curve for {model} model"
    )
    ax.legend(loc="lower right")

# Adjust the spacing between subplots
plt.tight_layout()
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'roc_curves.png')
plt.savefig(plot_filepath)
plt.close()

# Print micro-averaged and macro-averaged ROC AUC scores
print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.2f}")


# %% 
#### Kaplan- Meier Curves

model_mappings = {
    'graph': 'Graph GCN',
    'path': 'Histology CNN',
    'omic': 'Genomic SNN',
    'pathomic_fusion': 'Pathomic Fusion (CNN+SNN)',
    'graphomic_fusion': 'Graphomic Fusion (GCN+SNN)',
    'pathgraphomic_fusion': 'Pathgraphomic Fusion (CNN+GCN+SNN)',
    'pathgraph_fusion': 'Pathgraph Fusion (CNN+GCN)',
    'omicomic_fusion': 'Omicomic Fusion (SNN+SNN)', 
    'pathpath_fusion': 'Pathpath Fusion (CNN+CNN)',  
    'graphgraph_fusion': 'Graphgraph Fusion (GCN+GCN)'  
}

# Adding for idh mutation
df1 = pd.read_csv('./data/TCGA_GBMLGG/all_dataset.csv')
# Adding for histology
df2 = pd.read_csv('./data/TCGA_GBMLGG/grade_data.csv')
c=[(-1.5, -0.5), (1, 1.25), (1.25, 1.5)]
df = df1.merge(df2, on='TCGA ID', how='left') 

split = 'train'
use_patch = '_'
k=1
fig, axs = plt.subplots(4, 4, figsize=(20, 8))

data = load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/')
data['TCGA ID']=data['Patient ID']
data_merge = data.merge(df, on='TCGA ID', how='left')
data1 = data_merge[(data_merge['idh mutation'] == 0) & (data_merge['Histology'] != 'oligodendroglioma') & (data_merge['Grade Status'].isin([0, 1, 2]))]
data2 = data_merge[(data_merge['idh mutation'] == 1) & (data_merge['Histology'] != 'oligodendroglioma') & (data_merge['Grade Status'].isin([0, 1, 2]))]
data3 = data_merge[(data_merge['Histology'] == 'oligodendroglioma')]

for j, model in enumerate(['graph', 'path', 'omic', 'pathgraphomic_fusion']):
    model_name = model_mappings.get(model, 'Unknown Model') 
    data = load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/')
    data['TCGA ID']=data['Patient ID']
    data_merge = data.merge(df, on='TCGA ID', how='left')
    full_data = data_merge

    for idx, data_type in enumerate(['wt_astrocytoma', 'mut_astrocytoma', 'oligodendroglioma', 'all']):
        # This code serves to mean aggregate the hazards of the data
        if data_type == 'wt_astrocytoma':
            x_axis = 'IDHwt-astrocytoma'
            data = data_merge[(data_merge['idh mutation'] == 0) & (data_merge['Histology'] != 'oligodendroglioma') & (data_merge['Grade Status'].isin([0, 1, 2]))]
        if data_type == 'mut_astrocytoma':
            x_axis = 'IDH-mutant astrocytomas'
            data = data_merge[(data_merge['idh mutation'] == 1) & (data_merge['Histology'] != 'oligodendroglioma') & (data_merge['Grade Status'].isin([0, 1, 2]))]
        if data_type == 'oligodendroglioma':
            x_axis = 'oligodendroglioma'
            data = data_merge[(data_merge['Histology'] == 'oligodendroglioma')]
        else:
            x_axis = 'Overall'
            data = full_data

        # Extract grade_groups and hazards
        grade_status = data['Grade Status']
        hazards = data['Hazards']
        censor_status = data['Censor Status']
        survival_times = data['Survival Time']
        survival_times = survival_times // 365

        # Kaplan-Meier curve by grade status
        kmf = KaplanMeierFitter()

        # Example hazard values and threshold probabilities
        thresholds = [33, 66, 100]
        # Assign grades to hazard values
        percentiles_of_hazards = np.percentile(hazards, thresholds)
        # Assign grades to hazard values based on percentiles
        grade_status_predicted = np.array([hazard2grade(h, percentiles_of_hazards) for h in hazards])

        colours = ['green', 'blue', 'red']
        colours2 = ['green', 'red']
        colours3 = ['green', 'blue']

        if data_type=='wt_astrocytoma':
            for grade, color in zip([0,1,2], colours):
                mask = (grade_status == grade)
                kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'True Grade {grade+2}')
                # kmf.plot_survival_function(ci_show=False, color=color, linestyle='-')
                kmf.survival_function_
                axs[idx,j].plot(kmf.survival_function_, linestyle="-", color=color)
                if j == 0:
                    axs[idx,j].set_ylabel('Wt Astrocytoma\nProportion Surviving')
            for grade, color in zip([0,3], colours2):
                mask = (grade_status_predicted == grade)
                kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'Predicted Grade {grade+2} from Hazards')
                # kmf.plot_survival_function(ci_show=False, color=color, linestyle='--')
                kmf.survival_function_
                axs[idx,j].plot(kmf.survival_function_, linestyle="--", color=color)
                axs[idx,j].set_xlim([0, 20])
            axs[idx,j].set_xlabel('Time (years)')
        elif data_type=='oligodendroglioma':
            for grade, color in zip([0,1], colours3):
                mask = (grade_status == grade)
                kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'True Grade {grade+2}')
                # kmf.plot_survival_function(ci_show=False, color=color, linestyle='-')
                kmf.survival_function_
                axs[idx,j].plot(kmf.survival_function_, linestyle="-", color=color)
                if j == 0:
                    axs[idx,j].set_ylabel('Oligodendroglioma\nProportion Surviving')
            for grade, color in zip([0,1], colours3):
                mask = (grade_status_predicted == grade)
                kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'Predicted Grade {grade+2} from Hazards')
                # kmf.plot_survival_function(ci_show=False, color=color, linestyle='--')
                kmf.survival_function_
                axs[idx,j].plot(kmf.survival_function_, linestyle="--", color=color)
                axs[idx,j].set_xlim([0, 20])
            axs[idx,j].set_xlabel('Time (years)')
        else:
            for grade, color in zip([0,1,2], colours):
                mask = (grade_status == grade)
                kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'True Grade {grade+2}')
                # kmf.plot_survival_function(ci_show=False, color=color, linestyle='-')
                kmf.survival_function_
                axs[idx,j].plot(kmf.survival_function_, linestyle="-", color=color)
                if j == 0:
                    if data_type == 'all':
                        axs[idx,j].set_ylabel('Overall\nProportion Surviving')
                    else:
                        axs[idx,j].set_ylabel('IDH-mutant astrocytomas\nProportion Surviving')
            for grade, color in zip([0,1,2], colours):
                if grade ==2:
                    grade = 2 or 3
                mask = (grade_status_predicted == grade)
                kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'Predicted Grade {grade+2} from Hazards')
                # kmf.plot_survival_function(ci_show=False, color=color, linestyle='--')
                kmf.survival_function_
                axs[idx,j].plot(kmf.survival_function_, linestyle="--", color=color)
                axs[idx,j].set_xlim([0, 20])
        if idx == 0:
            axs[idx,j].set_title(f'{model_name}')
        print(f'{model_name} {data_type} {grade_status.value_counts()}done')

plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'kaplan_meier_curves.png')
plt.savefig(plot_filepath)
plt.close()

# %%
#### HISTOGRAM HAZARD PLOTS

fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the figsize to suit your display needs
for j, model in enumerate(['graph', 'path', 'omic', 'pathgraphomic_fusion']):
    data = load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/')
    ax = axes[j]
    model_name = model_mappings.get(model, 'Unknown Model') 
    ## Z-scored hazards histograms
    # Calculate the z-scores of the 'Hazards'
    data['Hazards_z'] = zscore(data['Hazards'])

    # Split the data based on the survival time with low and high being thresholded by 5 years
    low = data[data['Survival Time'] <= 365*5]
    high = data[data['Survival Time'] > 365*5]

    # Histogram plotting with normalized density using plt directly
    sns.histplot(low['Hazards_z'], ax=ax,bins=15, kde=False, stat="density", color="red", alpha=0.5, edgecolor="black", label='<= 5 Years')
    sns.histplot(high['Hazards_z'], ax=ax, bins=15, kde=False, stat="density",color="blue", alpha=0.5, edgecolor="black", label='> 5 Years')

    # Adjusting the visual layout of the plot
    ax.set_xlabel('Hazard Values (Z-score)')
    ax.set_ylabel('Density')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis='y', which='both', labelsize=10)
    ax.tick_params(axis='x', which='both', labelsize=10)
    ax.set_xticks(np.arange(-3, 4, 1))
    ax.set_xlim([-2, 2])

    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1])

    # Title for each subplot
    ax.set_title(f'{model} Model')

    # Adding legend to each subplot
    ax.legend(title="Survival Time")

# Adjust layout to prevent overlapping
plt.tight_layout()   
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'hazard_histograms.png')
plt.savefig(plot_filepath)
plt.close()
