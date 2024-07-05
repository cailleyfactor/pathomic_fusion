# Base / Native
import math
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Numerical / Array
import lifelines
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
from imblearn.over_sampling import RandomOverSampler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
from PIL import Image
import pylab
import scipy
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import average_precision_score, auc, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from scipy import interp
mpl.rcParams['axes.linewidth'] = 3 #set the value globally

def p(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'p%s' % n
    return percentile_

# Modified code to get the test patients
def getGradTestPats_GBMLGG(k, ckpt_name='./checkpoints/TCGA_GBMLGG/grad_15/', model='omic', split='test', use_rnaseq=False, agg_type='mean'):
    pats = {}
    # Set up flags appropriately
    ignore_missing_histype = 1 
    ignore_missing_moltype = 1 if "omic" in model else 0
    use_patch, roi_dir, use_vgg_features = ('_', 'all_st', 0)
    if "omic" in model:
        use_rnaseq = '_rnaseq' 
    else:
        use_rnaseq = ''
    # data_cv_path = './data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq)
    for k in range(k, k+1):
        # './checkpoints/TCGA_GBMLGG/grad_15/path/path_1_pred_test.pkl'
        pred_test, data_cv  = pickle.load(open(ckpt_name+'%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))
        #pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]
        # Transposes the grade predictions and converts them to a data frame
        grad_all = pred_test[3].T
        grad_all = pd.DataFrame(np.stack(grad_all)).T
        grad_all.columns = ['score_0', 'score_1', 'score_2']
        # data_cv = pickle.load(open(data_cv_path, 'rb'))
        data_cv_splits = data_cv['cv_splits']
        data_cv_split_k = data_cv_splits[k]
        assert np.all(data_cv_split_k[split]['g'] == pred_test[4]) # Data is correctly registered
        all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
        all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
        assert np.all(np.array(all_dataset_regstrd['Grade']) == pred_test[4])
        grad_all.index = data_cv_split_k[split]['x_patname']
        grad_all.index.name = 'TCGA ID'
        # Sets the aggregation function based on the agg type parameter 
        fun = p(0.75) if agg_type == 'p0.75' else agg_type
        # Groups the data by TCGA ID and aggregates the data for each group
        grad_all = grad_all.groupby('TCGA ID').agg({'score_0': [fun], 'score_1': [fun], 'score_2': [fun]})
        # Filters the aggregated predictions to only include those for the test patients.
        pats[k] = grad_all.index     
    # This is a dictionary returning the patient names
    return pats


# This code aggregated predictions based on patient name
def getPredAggGrad_GBMLGG(k, ckpt_name='./checkpoints/TCGA_GBMLGG/grad_15/', model='omic', split='test', use_rnaseq=False, 
                         agg_type='max',  label='all'):
    
    y_label, y_pred = [], []
    ignore_missing_moltype = 1 if 'omic' in model else 0
    ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
    use_patch, roi_dir, use_vgg_features = ('_', 'all_st', 0)
    if "omic" in model:
        use_rnaseq = '_rnaseq' 
    else:
        use_rnaseq = ''
    # data_cv_path = './data/TCGA_GBMLGG/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq)
    
    #print(data_cv_path)
    
    for k in range(k,k+1):
        ### Loads Prediction Pickle File. Registers predictions with TCGA IDs for the test split.
        pred_test, data_cv  = pickle.load(open(ckpt_name+'%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))
        
        # probabilities
        grad_pred = pred_test[3].T
        grad_pred = pd.DataFrame(np.stack(grad_pred)).T
        grad_pred.columns = ['score_0', 'score_1', 'score_2']

        # data_cv = pickle.load(open(data_cv_path, 'rb'))
        data_cv_splits = data_cv['cv_splits']
        data_cv_split_k = data_cv_splits[k]

        # # assert np.all(data_cv_split_k[split]['g'] == pred[4]) # Data is correctly registered
        all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
        # Link grad pred to the TCGA ID
        grad_pred.index = data_cv_split_k[split]['x_patname']
        grad_pred.index.name = 'TCGA ID'
        
        ### Amalgamates predictions together.
        # This computes a specified aggregation function probably returning percentile of 90%
        fun = p(0.90) if agg_type == 'p0.75' else agg_type

        # Filters the aggregated predictions to only include those for the test patients.
        grad_pred = grad_pred.groupby('TCGA ID').agg({'score_0': [fun], 'score_1': [fun], 'score_2': [fun]})

        test_pats = getGradTestPats_GBMLGG(k)
        test_pat = test_pats[k]
        # Include only the aggregated predictions for test patients
        grad_pred = grad_pred.loc[test_pat]
        # Retrieves the actual grades or labels for the test patients
        grad_gt = np.array(all_dataset.loc[test_pat]['Grade'])

        grad_pred = np.array(grad_pred)
        # Binarizes the labels for evaluations
        enc = LabelBinarizer()
        enc.fit(grad_gt)
        grad_gt = enc.transform(grad_gt)
        # Store the labels and predictions in a list
        y_label.append(grad_gt)
        y_pred.append(grad_pred)
    return y_label, y_pred

### Survival Outcome Prediction
def hazard2grade(hazard, p):
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)

def load_and_process_survival_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/'):
    # Load the data from a pickle file
    with open(f'{ckpt_name}/{model}/{model}_{k}{use_patch}pred_{split}.pkl', 'rb') as file:
        pred_test, data_cv = pickle.load(file)

    # Extract data from the cross-validation splits
    data_cv_splits = data_cv['cv_splits']
    data_cv_split_k = data_cv_splits[k]

    # Extract relevant data
    patients = data_cv_split_k[split]['x_patname']
    grade_status = data_cv_split_k[split]['g']
    censor_status = data_cv_split_k[split]['e']  # censored
    survival_times = data_cv_split_k[split]['t']  # survival time in days

    # Hazards are assumed to be the first element of predictions
    hazards = pred_test[0]

    # Construct a DataFrame from the loaded data
    data = pd.DataFrame({
        'Patient ID': patients,
        'Grade Status': grade_status,
        'Censor Status': censor_status,
        'Survival Time': survival_times,
        'Hazards': hazards
    })

    # Define a custom aggregation dictionary
    aggregation = {
        'Grade Status': 'first',
        'Censor Status': 'first',
        'Survival Time': 'first',
        'Hazards': 'mean'
    }

    # Group by 'Patient ID' and apply the custom aggregation
    grouped_data = data.groupby('Patient ID').agg(aggregation).reset_index()

    return grouped_data

def load_and_process_grade_data(model, k, use_patch, split, ckpt_name='./checkpoints/TCGA_GBMLGG/grad_15/'):
    # Load the data from a pickle file
    with open(f'{ckpt_name}/{model}/{model}_{k}{use_patch}pred_{split}_data.pkl', 'rb') as file:
        pred_test, data_cv = pickle.load(file)

    # Extract data from the cross-validation splits
    data_cv_splits = data_cv['cv_splits']
    data_cv_split_k = data_cv_splits[k]

    # Extract relevant data
    patients = data_cv_split_k[split]['x_patname']
    grade_status = data_cv_split_k[split]['g']
    censor_status = data_cv_split_k[split]['e']  # censored
    survival_times = data_cv_split_k[split]['t']  # survival time in days

    # Hazards are assumed to be the first element of predictions
    hazards = pred_test[0]

    # Construct a DataFrame from the loaded data
    data = pd.DataFrame({
        'Patient ID': patients,
        'Grade Status': grade_status,
        'Censor Status': censor_status,
        'Survival Time': survival_times,
        'Hazards': hazards
    })

    # Define a custom aggregation dictionary
    aggregation = {
        'Grade Status': 'first',
        'Censor Status': 'first',
        'Survival Time': 'first',
        'Hazards': 'mean'
    }

    # Group by 'Patient ID' and apply the custom aggregation
    grouped_data = data.groupby('Patient ID').agg(aggregation).reset_index()

    return grouped_data

def load_and_process_survival_kirc_data(model, k, split, ckpt_name='./checkpoints/TCGA_KIRC/surv_15'):
    # Load the data from a pickle file
    with open(f'{ckpt_name}/{model}/{model}_{k}pred_{split}.pkl', 'rb') as file:
        pred_test = pickle.load(file)
        
    with open(f'./data/TCGA_KIRC/splits/KIRC_st_0_clin.pkl', 'rb') as file:
        data_cv = pickle.load(file)

    data_cv_split_k = data_cv[k]
    # Extract relevant data
    patients = data_cv_split_k[split]['x_patname']
    grade_status = data_cv_split_k[split]['g']
    censor_status = data_cv_split_k[split]['e']  # censored
    survival_times = data_cv_split_k[split]['t']  # survival time in days

    # # Hazards are assumed to be the first element of predictions
    hazards = pred_test[0]

    # Construct a DataFrame from the loaded data
    data = pd.DataFrame({
        'Patient ID': patients,
        'Grade Status': grade_status,
        'Censor Status': censor_status,
        'Survival Time': survival_times,
        'Hazards': hazards
    })

    # Define a custom aggregation dictionary
    aggregation = {
        'Grade Status': 'first',
        'Censor Status': 'first',
        'Survival Time': 'first',
        'Hazards': 'mean'
    }

    # Group by 'Patient ID' and apply the custom aggregation
    grouped_data = data.groupby('Patient ID').agg(aggregation).reset_index()
    return grouped_data
# %%
