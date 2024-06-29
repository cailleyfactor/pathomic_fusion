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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.special import softmax

# Making the loss curves nicer
# Load in the saved model information from the checkpoints
import math
import os
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

checkpoints_dir = './checkpoints/TCGA_GBMLGG'

# %%
#### LOSS & METRIC PLOTS - GRADE
metrics = ['loss', 'grad_acc']
model_names = ['path', 'graph', 'omic', 'pathomic_fusion', 'graphomic_fusion', 'pathgraph_fusion', 'pathgraphomic_fusion']
eval_folder = 'evaluation'

# Define relabeling dictionaries
model_mappings = {
    'graph': 'Graph GCN',
    'path': 'Histology CNN',
    'omic': 'Genomic SNN',
    'pathomic_fusion': 'Pathomic Fusion',
    'graphomic_fusion': 'Graphomic Fusion',
    'pathgraphomic_fusion': 'Pathgraphomic Fusion',
    'pathgraph_fusion': 'Pathgraph Fusion',
    'omicomic_fusion': 'Omicomic Fusion', 
    'pathpath_fusion': 'Pathpath Fusion',  
    'graphgraph_fusion': 'Graphgraph Fusion'  
}

metric_mapping = {
    'loss': 'Loss',
    'grad_acc': 'Grade Accuracy'
}

num_cols = len(model_names)
num_rows = 2  # One row for loss plots, one row for accuracy plots
exp_name = 'grad_15'
results = 'results_embeddings'
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))

train_color = 'blue'
test_color = 'red'

for metric_index, metric in enumerate(metrics):
    for i, model_name in enumerate(model_names):
        all_train_metrics = []
        all_test_metrics = []

        for k in range(1, 6):
            results_dir = os.path.join(checkpoints_dir, exp_name, results)
            csv_filepath = os.path.join(results_dir, f'{model_name}_metrics_{k}.csv')
            df = pd.read_csv(csv_filepath)

            train_metric = df[f'train_{metric}']
            test_metric = df[f'test_{metric}']

            all_train_metrics.append(train_metric)
            all_test_metrics.append(test_metric)

        # Calculate mean and std
        train_mean = np.mean(all_train_metrics, axis=0)
        train_std = np.std(all_train_metrics, axis=0)
        test_mean = np.mean(all_test_metrics, axis=0)
        test_std = np.std(all_test_metrics, axis=0)

        row = metric_index
        col = i

        ax = axes[row, col]
        epochs = range(len(train_mean))

        ax.plot(epochs, train_mean, label='train', color=train_color)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color=train_color, alpha=0.3)

        ax.plot(epochs, test_mean, label='test', linestyle='--', color=test_color)
        ax.fill_between(epochs, test_mean - test_std, test_mean + test_std, color=test_color, alpha=0.3)

        if row == 0:
            ax.set_title(model_mappings.get(model_name, model_name))
        if row == num_rows - 1:
            ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_mapping.get(metric, metric))
        if col == 0 and metric_index == 0:
            ax.legend()
        if metric == 'loss':
            ax.set_ylim(0, 2)  
        elif metric == 'surv_acc':
            ax.set_ylim(0, 1)  
        else:
            ax.set_ylim(0, 1)  

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Create the directory if it does not exist
os.makedirs(os.path.join(checkpoints_dir, eval_folder), exist_ok=True)

# Save the plot
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'{exp_name}_performance_plot_grade_gbmlgg_stdev.png')
plt.savefig(plot_filepath)
plt.show()


# %%
### Summary tables for TCGA_GBMLGG
# Define relabeling dictionaries
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
# List of model names
model_names = ['path','graph','omic','pathomic_fusion','graphomic_fusion','pathgraph_fusion', 'pathgraphomic_fusion']

# Initialize list to store metrics
metrics_list = []

for idx, model in enumerate(model_names):
    for k in range(1, 6):
        # Get predictions
        y_label, y_pred = getPredAggGrad_GBMLGG(k, model=model, agg_type='max') 
        y_label = np.squeeze(y_label)
        y_pred = np.squeeze(y_pred)

        # Compute ROC area for each class
        auc_val = roc_auc_score(y_label, y_pred)

        # Convert logits to probabilities using softmax
        y_pred_prob = softmax(y_pred, axis=1)

        # Binarize the predicted probabilities (threshold=0.5)
        y_pred_binary = (y_pred_prob > 0.5).astype(int)

        ap = average_precision_score(y_label, y_pred_prob)

        # F1 score calculation
        f1 = f1_score(y_label, y_pred_binary, average = 'micro')

        # F1 score calculation for grade IV by extracting the relevant column
        f1_grade_iv = f1_score(y_label[:,2], y_pred_binary[:,2], average = 'micro')

        # Append the metrics to the list
        metrics_list.append({
            'Model': model_mappings[model],
            'AUC': auc_val,
            'AP': ap,
            'F1-Score (Micro)': f1,
            'F1-Score (Grade IV)': f1_grade_iv
        })


# Create a DataFrame from the metrics list
metrics_df = pd.DataFrame(metrics_list)

# # Display the DataFrame
# print(metrics_df)


# Compute average and standard deviation
avg_metrics = metrics_df.groupby('Model').mean().reset_index()
std_metrics = metrics_df.groupby('Model').std().reset_index()

# Rename columns for clarity
avg_metrics.columns = ['Model', 'Mean AUC', 'Mean AP', 'Mean F1-Score (Micro)', 'Mean F1-Score (Grade IV)']
std_metrics.columns = ['Model', 'Stdev AUC', 'Stdev AP', 'Stdev F1-Score (Micro)', 'Stdev F1-Score (Grade IV)']

# Merge mean and std into a single DataFrame
metrics_summary = pd.merge(avg_metrics, std_metrics, on='Model')

# Format standard deviation values to more decimal places
metrics_summary = metrics_summary.round({'Stdev AUC': 4, 'Stdev AP': 4, 'Stdev F1-Score (Micro)': 4, 'Stdev F1-Score (Grade IV)': 4})

# Display the summarized metrics
print(metrics_summary)

# %%
#### LOSS & METRIC PLOTS - SURVIVAL ####
checkpoints_dir = './checkpoints/TCGA_GBMLGG'
metrics = ['loss', 'cindex', 'surv_acc']
eval_folder = 'evaluation'

# List of model names
model_names = ['path', 'graph', 'omic', 'pathomic_fusion', 'graphomic_fusion', 'pathgraph_fusion', 'pathgraphomic_fusion']

# Define relabeling dictionaries
model_mappings = {
    'graph': 'Graph GCN',
    'path': 'Histology CNN',
    'omic': 'Genomic SNN',
    'pathomic_fusion': 'Pathomic Fusion',
    'graphomic_fusion': 'Graphomic Fusion',
    'pathgraphomic_fusion': 'Pathgraphomic Fusion',
    'pathgraph_fusion': 'Pathgraph Fusion',
    'omicomic_fusion': 'Omicomic Fusion', 
    'pathpath_fusion': 'Pathpath Fusion',  
    'graphgraph_fusion': 'Graphgraph Fusion'  
}
metric_mapping = {
    'loss': 'Loss',
    'cindex': 'C-index',
    'surv_acc': 'Survival Accuracy'
}

checkpoints_dir = './checkpoints/TCGA_GBMLGG'
metrics = ['loss', 'cindex', 'surv_acc']
eval_folder = 'evaluation'
results = 'results_embeddings'

# Calculate the number of rows and columns for the grid
num_cols = len(model_names)
num_rows = 3  # We have three rows of metrics

exp_name = 'surv_15_rnaseq'

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))

train_color = 'blue'
test_color = 'red'

for metric_index, metric in enumerate(metrics):
    for i, model_name in enumerate(model_names):
        all_train_metrics = []
        all_test_metrics = []

        for k in range(1, 6):
            results_dir = os.path.join(checkpoints_dir, exp_name, results)
            csv_filepath = os.path.join(results_dir, f'{model_name}_metrics_{k}.csv')
            df = pd.read_csv(csv_filepath)

            train_metric = df[f'train_{metric}']
            test_metric = df[f'test_{metric}']

            all_train_metrics.append(train_metric)
            all_test_metrics.append(test_metric)

        # Calculate mean and std
        train_mean = np.mean(all_train_metrics, axis=0)
        train_std = np.std(all_train_metrics, axis=0)
        test_mean = np.mean(all_test_metrics, axis=0)
        test_std = np.std(all_test_metrics, axis=0)

        row = metric_index
        col = i

        ax = axes[row, col]
        epochs = range(len(train_mean))

        ax.plot(epochs, train_mean, label='train', color=train_color)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color=train_color, alpha=0.3)

        ax.plot(epochs, test_mean, label='test', linestyle='--', color=test_color)
        ax.fill_between(epochs, test_mean - test_std, test_mean + test_std, color=test_color, alpha=0.3)

        if row == 0:
            ax.set_title(model_mappings.get(model_name, model_name))
        if row == num_rows - 1:
            ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_mapping.get(metric, metric))
        if col == 0 and metric_index == 0:
            ax.legend()
        if metric == 'loss':
            ax.set_ylim(0, 2)
        elif metric == 'surv_acc':
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Create the directory if it does not exist
os.makedirs(os.path.join(checkpoints_dir, eval_folder), exist_ok=True)

# Save the plot
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'{exp_name}_performance_plot_surv_gbmlgg_stdev.png')
plt.savefig(plot_filepath)
plt.show()
# plt.close()

# %%
#### ROC, AUC Plots - for just path and pgo as in the paper
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

# Models and grades
model_names = ['path',  'pathgraphomic_fusion']
grades = [2, 3, 4]

# Initialize plot
fig, axes = plt.subplots(1, len(grades) + 1, figsize=(30, 8))
axes = axes.flatten()

for idx, grade in enumerate(grades + ['overall']):
    ax = axes[idx]
    
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    for model in model_names:
        for k in range(1, 6):
            # Get predictions
            y_label, y_pred = getPredAggGrad_GBMLGG(k=k, model=model, agg_type='max') 
            y_label = np.squeeze(y_label)
            y_pred = np.squeeze(y_pred)
            
            if grade != 'overall':
                y_label = y_label[:, grades.index(grade)]
                y_pred = y_pred[:, grades.index(grade)]
            
            fpr, tpr, _ = roc_curve(y_label.ravel(), y_pred.ravel())
            roc_auc = auc(fpr, tpr)
            
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        ax.plot(mean_fpr, mean_tpr, label=f'{model_mappings[model]} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2)
    
    ax.set_title(f'Grade {grade}' if grade != 'overall' else 'Overall')
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.legend(loc='lower right')

plt.tight_layout()
plt.show()

#%%
## Original ROC plots

# Define relabeling dictionaries
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

# List of model names
model_names = ['path',  'pathgraphomic_fusion']
fig, axes = plt.subplots(1, len(model_names), figsize=(10, 5))

for idx, model in enumerate(model_names):
    for k in range(1, 6):
        # Get predictions
        y_label, y_pred = getPredAggGrad_GBMLGG(k=k, model=model, agg_type='max') 
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
            label=f"ROC curve for grade {i+2} (AUC = {roc_auc[i]:.2f})"
        )
    
    # Plot chance level
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Customize the plot
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{model_mappings.get(model, model)}"
    )
    ax.legend(loc="lower right")

# Adjust the spacing between subplots
plt.tight_layout()
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'roc_curves.png')
plt.savefig(plot_filepath)
plt.show()
plt.close()

# Print micro-averaged and macro-averaged ROC AUC scores
print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.2f}")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Define relabeling dictionaries
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

# Load the data
df1 = pd.read_csv('./data/TCGA_GBMLGG/all_dataset.csv')
df2 = pd.read_csv('./data/TCGA_GBMLGG/grade_data.csv')
df = df1.merge(df2, on='TCGA ID', how='left') 

models = ['graph', 'path', 'omic', 'pathgraphomic_fusion']

fig, axs = plt.subplots(4, 4, figsize=(20, 20))

def apply_oligodendro_condition(grade_status_predicted):
    return np.where((grade_status_predicted == 2) | (grade_status_predicted == 3), 1, grade_status_predicted)

for j, model in enumerate(models):
    data_saver = []
    for k in range(1, 6):
        data = load_and_process_survival_data(model, k, '_', 'test', ckpt_name=f'./checkpoints/TCGA_GBMLGG/surv_15_rnaseq')
        data_saver.append(data)
    data = pd.concat(data_saver, ignore_index=True)
    data['TCGA ID'] = data['Patient ID']
    data_merge = data.merge(df, on='TCGA ID', how='left')
    full_data = data_merge
    model_name = model_mappings.get(model, 'Unknown Model')
    
    from utils import getCleanAllDataset
    metadata, all_dataset = getCleanAllDataset(use_rnaseq=True)
    astro_wt_ids = all_dataset.loc[all_dataset.iloc[:, 0] == 'idhwt_ATC', 'TCGA ID'].values
    astro_mut_ids = all_dataset.loc[all_dataset.iloc[:, 0] == 'idhmut_ATC', 'TCGA ID'].values
    oligodendro_ids = all_dataset.loc[all_dataset.iloc[:, 0] == 'ODG', 'TCGA ID'].values

    # Filter the full_data DataFrame by the extracted IDs
    datasets = {
        'Astro WT': full_data[full_data['TCGA ID'].isin(astro_wt_ids)],
        'Astro Mut': full_data[full_data['TCGA ID'].isin(astro_mut_ids)],
        'Oligodendro': full_data[full_data['TCGA ID'].isin(oligodendro_ids)],
        'Full': full_data
    }

    for i, (dataset_name, dataset) in enumerate(datasets.items()):
        grade_status = dataset['Grade Status']
        hazards = dataset['Hazards']
        censor_status = dataset['Censor Status']
        survival_times = dataset['Survival Time']
        survival_times = survival_times // 365

        if dataset_name == 'Oligodendro':
            grade_status_predicted = apply_oligodendro_condition(grade_status_predicted)

        kmf = KaplanMeierFitter()

        thresholds = [33, 66, 100]
        percentiles_of_hazards = np.percentile(hazards, thresholds)
        grade_status_predicted = np.array([hazard2grade(h, percentiles_of_hazards) for h in hazards])

        colours = ['green', 'blue', 'red']
        for grade, color in zip([0, 1, 2], colours):
            mask = (grade_status == grade)
            if mask.any():
                kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'True Grade {grade+2}')
                axs[i, j].plot(kmf.survival_function_, linestyle="--", color=color)
                if j == 0:
                    axs[i, j].set_ylabel(f'{dataset_name}\nOverall\nProportion Surviving')

        for grade, color in zip([0, 1, 2], colours):
            if grade == 2:
                mask = (grade_status_predicted == 2) | (grade_status_predicted == 3)
            else:
                mask = (grade_status_predicted == grade)


            mask = (grade_status_predicted == grade)

            if dataset_name == 'Oligodendro' and grade == 2:
                continue  # Skip plotting for grade 2 in Oligodendro
            
            if mask.any():
                kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'Predicted Grade {grade+2} from Hazards')
                axs[i, j].plot(kmf.survival_function_, linestyle="-", color=color)
                axs[i, j].set_xlim([0, 15])
                axs[i, j].set_title(f'{model_name}')
        axs[i, j].set_xlabel('Survival time (years)')
plt.tight_layout()
plt.show()


# %% 
#### Kaplan- Meier Curves for all models for overall survival only
# Define relabeling dictionaries
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

split = 'test'
use_patch = '_'
k=1
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for j, model in enumerate(['graph', 'path', 'omic', 'pathgraphomic_fusion']):
    data_saver= []
    for k in range(1,6):
        data = load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq')
        data_saver.append(data)
    data = pd.concat(data_saver, ignore_index=True)
    data['TCGA ID']=data['Patient ID']
    data_merge = data.merge(df, on='TCGA ID', how='left')
    full_data = data_merge
    model_name = model_mappings.get(model, 'Unknown Model')
    data = full_data
    print(data.shape)

    from utils import getCleanAllDataset
    metadata, all_dataset = getCleanAllDataset(use_rnaseq=True)
    astro_wt_ids = all_dataset.loc[all_dataset.iloc[:, 0] == 'idhwt_ATC', 'TCGA ID'].values
    astro_mut_ids = all_dataset.loc[all_dataset.iloc[:, 0] == 'idhmut_ATC', 'TCGA ID'].values
    oligodendro_ids = all_dataset.loc[all_dataset.iloc[:, 0] == 'ODG', 'TCGA ID'].values

    # Filter the full_data DataFrame by the extracted IDs
    astro_wt_data = full_data[full_data['TCGA ID'].isin(astro_wt_ids)]
    astro_mut_data = full_data[full_data['TCGA ID'].isin(astro_mut_ids)]
    oligodendro_data = full_data[full_data['TCGA ID'].isin(oligodendro_ids)]

    # Example of printing the shapes of filtered datasets
    print(f'Astro WT Data Shape: {astro_wt_data.shape}')
    print(f'Astro Mut Data Shape: {astro_mut_data.shape}')
    print(f'Oligodendro Data Shape: {oligodendro_data.shape}')

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

    for grade, color in zip([0,1,2], colours):
        mask = (grade_status == grade)
        kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'True Grade {grade+2}')
        # kmf.plot_survival_function(ci_show=False, color=color, linestyle='-')
        kmf.survival_function_
        axs[j].plot(kmf.survival_function_, linestyle="--", color=color)
        if j == 0:
            axs[j].set_ylabel('Overall\nProportion Surviving')

    for grade, color in zip([0,1,2], colours):
        if grade ==2:
            grade = 2 or 3
        mask = (grade_status_predicted == grade)
        kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'Predicted Grade {grade+2} from Hazards')
        # kmf.plot_survival_function(ci_show=False, color=color, linestyle='--')
        kmf.survival_function_
        axs[j].plot(kmf.survival_function_, linestyle="-", color=color)
        axs[j].set_xlim([0, 15])
        axs[j].set_title(f'{model_name}')
    axs[j].set_xlabel('Survival time (years)')

# plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'kaplan_meier_curves.png')
# plt.savefig(plot_filepath)
plt.show()
# plt.close()


# %%
#### HISTOGRAM HAZARD PLOTS - by long and short survival
fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # Adjust the figsize to suit your display needs
for j, model in enumerate(['graph', 'path', 'omic', 'pathomic_fusion', 'pathgraphomic_fusion']):
    data_saver= []
    for k in range(1,6):
        data = load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq')
        data_saver.append(data)
    data = pd.concat(data_saver, ignore_index=True)
    ax = axes[j]
    model_name = model_mappings.get(model, 'Unknown Model') 
    ## Z-scored hazards histograms
    # Calculate the z-scores of the 'Hazards'
    data['Hazards_z'] = zscore(data['Hazards'])

    # Split the data based on the survival time with low and high being thresholded by 5 years
    low = data[data['Survival Time'] <= 365*5]
    high = data[data['Survival Time'] > 365*5]

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
    ax.set_ylim([0, 4])

    # Title for each subplot
    ax.set_title(f'{model_name}')

    # Adding legend to each subplot
    ax.legend(title="Survival Time")

# Adjust layout to prevent overlapping
plt.tight_layout()   
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'hazard_histograms.png')

plt.show()
plt.savefig(plot_filepath)
plt.close()

# %%
#### HISTOGRAM HAZARD PLOTS BY PATIENT SUBTYPE
fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # Adjust the figsize to suit your display needs
for j, model in enumerate(['graph', 'path', 'omic', 'pathomic_fusion', 'pathgraphomic_fusion']):
    data_saver= []
    for k in range(1,5):
        data = load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq')
        data_saver.append(data)
    data = pd.concat(data_saver, ignore_index=True)
    ax = axes[j]
    model_name = model_mappings.get(model, 'Unknown Model') 
    ## Z-scored hazards histograms
    # Calculate the z-scores of the 'Hazards'
    data['Hazards_z'] = zscore(data['Hazards'])

    # Split the data based on the survival time with low and high being thresholded by 5 years
    low = data[data['Survival Time'] <= 365*5]
    high = data[data['Survival Time'] > 365*5]

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
    ax.set_ylim([0, 4])

    # Title for each subplot
    ax.set_title(f'{model_name}')

    # Adding legend to each subplot
    ax.legend(title="Survival Time")

# Adjust layout to prevent overlapping
plt.tight_layout()   
eval_folder = 'evaluation'
plot_filepath = os.path.join(checkpoints_dir, eval_folder, f'hazard_histograms.png')

plt.show()
plt.savefig(plot_filepath)
plt.close()

# %%
