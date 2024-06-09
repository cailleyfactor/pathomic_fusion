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

# Get predicted survivals and turn into hazards for KM - code modified from https://allendowney.github.io/SurvivalAnalysisPython/02_kaplan_meier.html
# ts = np.unique(predicted_survival_times)
# ts.sort()
# at_risk = pd.Series(0, index=ts)
# for t in ts:
#     k = (t <= predicted_survival_times)
#     at_risk[t] = k.sum()
# death = pd.Series(0, index=ts)
# for t in ts:
#     k = (censor_status == 1) & (t == predicted_survival_times)
#     death[t] = k.sum()
# d = dict(death=death, at_risk=at_risk)
# df = pd.DataFrame(d, index=ts)
# df['hazard'] = df['death'] / df['at_risk']

model_mappings = {
    'graph': 'Genomic SNN',
    'path': 'Histology GCN',
    'omic': 'Histology CNN',
    'pathomic_fusion': 'Pathomic Fusion (CNN+SNN)',
    'graphomic_fusion': 'Graphomic Fusion (GCN+SNN)',
    'pathgraphomic_fusion': 'Pathgraphomic Fusion (CNN+GCN+SNN)',
    'pathgraph_fusion': 'Pathgraph Fusion (CNN+GCN)',
    'omicomic_fusion': 'Omicomic Fusion (SNN+SNN)', 
    'pathpath_fusion': 'Pathpath Fusion (CNN+CNN)',  
    'graphgraph_fusion': 'Graphgraph Fusion (GCN+GCN)'  
}
k=1
use_patch = "_"
split = "test"
models = ['graph', 'path', 'omic', 'pathomic_fusion', 'graphomic_fusion', 'pathgraphomic_fusion', 'pathgraph_fusion']

# Adding for idh mutation
df1 = pd.read_csv('./data/TCGA_GBMLGG/all_dataset.csv')
# Adding for histology
df2 = pd.read_csv('./data/TCGA_GBMLGG/grade_data.csv')
c=[(-1.5, -0.5), (1, 1.25), (1.25, 1.5)]
df = df1.merge(df2, on='TCGA ID', how='left') 

for model in ['graph', 'path', 'omic', 'pathomic_fusion', 'graphomic_fusion', 'pathgraphomic_fusion', 'pathgraph_fusion', 'pathpath_fusion', 'graphgraph_fusion']:
    
    model_name = model_mappings.get(model, 'Unknown Model') 

    # Kaplan-Meier Curves
    ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/'
    split = "test"
    use_patch = "_"
    k=1

    # This code surves to mean aggregate the hazards of the duplicate data
    data = load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/')
    data['TCGA ID']=data['Patient ID']

    # Add on Histology and IDH mutation
    data = data.merge(df, on='TCGA ID', how='left')  

    # Option to filter by a grade status, e.g., astrocytoma (glioblastoma), oligodendroglioma, astrocytoma, oligastrocytoma
    # data = data[(data['idh mutation'] ==1) & (data['Histology'] != 'oligodendroglioma')]
    # data = data[(data['Histology'] == 'oligodendroglioma')]

    # Extract grade_groups and hazards
    grade_status = data['Grade Status']
    hazards = data['Hazards']
    censor_status = data['Censor Status']
    survival_times = data['Survival Time']
    # survival_time = survival_times // 365
    grade_groups = set(grade_status)

    # Kaplan-Meier curve by grade status
    kmf = KaplanMeierFitter()

    # Example hazard values and threshold probabilities
    thresholds = [33, 66, 100]

    # Assign grades to hazard values
    percentiles_of_hazards = np.percentile(hazards, thresholds)

    # Assign grades to hazard values based on percentiles
    grade_status_predicted = np.array([hazard2grade(h, percentiles_of_hazards) for h in hazards])

    plt.figure(figsize=(10, 8))

    colours = ['green', 'blue', 'red']

    for grade, color in zip([0,1,2], colours):
        mask = (grade_status == grade)
        kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'True Grade {grade+2}')
        kmf.plot_survival_function(ci_show=False, color=color, linestyle='-')

    for grade, color in zip([0,1,2], colours):
        mask = (grade_status_predicted == grade)
        kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'Predicted Grade {grade+2} from Hazards')
        kmf.plot_survival_function(ci_show=False, color=color, linestyle='--')

    plt.title(f'Kaplan-Meier Curve by Grade Status for {model_name}')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.show()
    plt.close()

    ## Z-scored hazards histograms
    # Calculate the z-scores of the 'Hazards'
    data['Hazards_z'] = zscore(data['Hazards'])

    # Split the data based on the survival time
    low = data[data['Survival Time'] <= 365*3]
    high = data[data['Survival Time'] > 365*3]
    
    # Setting the size of the entire plot
    plt.figure(figsize=(10, 6))

    # Histogram plotting with normalized density using plt directly
    sns.histplot(low['Hazards_z'], bins=15, kde=False, stat="density",
                color="red", alpha=0.5, edgecolor="black", label='<= 5 Years')
    sns.histplot(high['Hazards_z'], bins=15, kde=False, stat="density",
                color="blue", alpha=0.5, edgecolor="black", label='> 5 Years')

    # Adjusting the visual layout of the plot
    plt.xlabel('Hazard Values (Z-score)')
    plt.ylabel('Density')
    plt.gca().spines["right"].set_visible(True)
    plt.gca().spines["top"].set_visible(True)
    plt.tick_params(axis='y', which='both', labelsize=10)
    plt.tick_params(axis='x', which='both', labelsize=10)
    plt.xticks(np.arange(-3, 4, 1))  # Adjusted for a wider range of Z-scores
    plt.xlim([-10, 10])

    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim([0, 1])

    # Adding a title and legend
    plt.title(f'Histogram of Hazard Values by Survival Time for {model}')
    plt.legend(title="Survival Time")  # Adding a title to the legend for better clarity

    # Display the plot
    plt.show()

# # Code for AUC, ROC
# # Assuming getPredAggGrad_GBMLGG, models, and other necessary variables are defined
# for model in tqdm(models):
#     # Get predictions
#     y_label, y_pred = getPredAggGrad_GBMLGG(model=model, agg_type='max') 
#     y_label = np.squeeze(y_label)
#     y_pred = np.squeeze(y_pred)
    
#     # Initialize a figure
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     n_classes = y_label.shape[1]
    
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
    
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
#     # Compute macro-average ROC curve and ROC area
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
    
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
#     # Plot micro-average ROC curve
#     ax.plot(
#         fpr["micro"],
#         tpr["micro"],
#         label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
#         color="deeppink",
#         linestyle=":",
#         linewidth=4,
#     )
    
#     # Plot macro-average ROC curve
#     ax.plot(
#         fpr["macro"],
#         tpr["macro"],
#         label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
#         color="navy",
#         linestyle=":",
#         linewidth=4,
#     )
    
#     # Plot ROC curves for each class
#     colors = cycle(["aqua", "darkorange", "cornflowerblue"])
#     for i, color in zip(range(n_classes), colors):
#         ax.plot(
#             fpr[i],
#             tpr[i],
#             color=color,
#             lw=2,
#             label=f"ROC curve for class {i} (AUC = {roc_auc[i]:.2f})"
#         )
    
#     # Plot chance level
#     ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
#     # Customize the plot
#     ax.set(
#         xlabel="False Positive Rate",
#         ylabel="True Positive Rate",
#         title=f"ROC Curve for multiclass classification {model}"
#     )
#     ax.legend(loc="lower right")
    
#     # Show plot
#     plt.show()

#     # Print micro-averaged and macro-averaged ROC AUC scores
#     print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
#     print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.2f}")

