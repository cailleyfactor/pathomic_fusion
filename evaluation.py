from core.utils_data import getCleanGBMLGG
import pandas as pd
import numpy as np
from tqdm import tqdm
from core.utils_analysis import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
# from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
from lifelines import KaplanMeierFitter

models = ['graph'] #, 'path', 'pathomic_fusion', 'graphomic_fusion', 'pathgraphomic_fusion', 'pathgraph_fusion']
# model_names = ['Genomic SNN'], 'Histology GCN', 'Histology CNN', 
               #'Pathomic F. (CNN+SNN)', 'Pathomic F. (GCN+SNN)', 'Pathomic F. (CNN+GCN+SNN)', 'Pathomic F. (CNN+GCN)']

# Kaplan-Meier Curves
ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/'
split = "test"
use_patch = "_"
k=1
model = 'omic'

# Loading the data
pred_test, data_cv  = pickle.load(open(ckpt_name+'%s/%s_%d%spred_%s_data.pkl' % (model, model, k, use_patch, split), 'rb'))
data_cv_splits = data_cv['cv_splits']
data_cv_split_k = data_cv_splits[k]
patients = data_cv_split_k[split]['x_patname']
grade_status = data_cv_split_k[split]['g']
censor_status = data_cv_split_k[split]['e'] # censored
survival_times = data_cv_split_k[split]['t'] # survival time in day

# Get predicted survivals and turn into hazards for KM - code modified from https://allendowney.github.io/SurvivalAnalysisPython/02_kaplan_meier.html
hazards = pred_test[0]
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

# Kaplan-Meier curve by grade status
kmf = KaplanMeierFitter()
grade_groups = set(grade_status)

# Example hazard values and threshold probabilities
thresholds = [33, 66, 100]

# Assign grades to hazard values
percentiles_of_hazards = np.percentile(hazards, thresholds)

# Assign grades to hazard values based on percentiles
grade_status_predicted = np.array([hazard2grade(h, percentiles_of_hazards) for h in hazards])

plt.figure(figsize=(10, 6))

colours = ['blue', 'green', 'red']

for grade, color in zip([0,1,2], colours):
    mask = (grade_status == grade)
    kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'True Grade {grade+2}')
    kmf.plot_survival_function(ci_show=False, color=color, linestyle='-')

for grade, color in zip([0,1,2], colours):
    mask = (grade_status_predicted == grade)
    kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label=f'Predicted Grade {grade+2} from Hazards')
    kmf.plot_survival_function(ci_show=False, color=color, linestyle='--')

plt.title('Kaplan-Meier Curve by Grade Status')
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# for grade in grade_groups:
#     mask = (grade_status == grade)
#     kmf.fit(durations=survival_times[mask], event_observed=censor_status[mask], label='Grade %s' % grade)
#     kmf.plot_survival_function(ci_show=False)

# plt.title('Kaplan-Meier Curve by Grade Status')
# plt.xlabel('Time (days)')
# plt.ylabel('Survival Probability')
# plt.legend()
# plt.show()



# Assuming getPredAggGrad_GBMLGG, models, and other necessary variables are defined
for model in tqdm(models):
    # Get predictions
    y_label, y_pred = getPredAggGrad_GBMLGG(model=model, agg_type='max') 
    y_label = np.squeeze(y_label)
    y_pred = np.squeeze(y_pred)

    
    
    # # Initialize a figure
    # fig, ax = plt.subplots(figsize=(6, 6))

    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # n_classes = y_label.shape[1]
    
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # # Compute macro-average ROC curve and ROC area
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes
    
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # # Plot micro-average ROC curve
    # ax.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
    #     color="deeppink",
    #     linestyle=":",
    #     linewidth=4,
    # )
    
    # # Plot macro-average ROC curve
    # ax.plot(
    #     fpr["macro"],
    #     tpr["macro"],
    #     label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    #     color="navy",
    #     linestyle=":",
    #     linewidth=4,
    # )
    
    # # Plot ROC curves for each class
    # colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    # for i, color in zip(range(n_classes), colors):
    #     ax.plot(
    #         fpr[i],
    #         tpr[i],
    #         color=color,
    #         lw=2,
    #         label=f"ROC curve for class {i} (AUC = {roc_auc[i]:.2f})"
    #     )
    
    # # Plot chance level
    # ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # # Customize the plot
    # ax.set(
    #     xlabel="False Positive Rate",
    #     ylabel="True Positive Rate",
    #     title="One-vs-Rest ROC curves for all classes\nwith micro and macro average"
    # )
    # ax.legend(loc="lower right")
    
    # # Show plot
    # plt.show()

    # # Print micro-averaged and macro-averaged ROC AUC scores
    # print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
    # print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.2f}")

