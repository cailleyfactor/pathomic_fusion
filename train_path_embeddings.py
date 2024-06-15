import os
import logging
import numpy as np
import pickle
import torch

# Env
from data_loaders import *
from options import parse_args
from train_test_embed import train, test

import torch_geometric
print(torch_geometric.__version__)

from result_plots import save_metric_logger, plots_train_vs_test
from filter_patients import filter_unique_patients
from option_file_converter import parse_opt_file

### 1. Initializes parser and device
# This also prints opt but this is before the changes
# opt = parse_args()
checkpoints_dir = ".\\checkpoints\\TCGA_KIRC"
results_folder = "results_2"

for k in range(1, 17):
    for mode in ["path"]:
        setting = "surv_15"
        file_path = os.path.join(checkpoints_dir, setting, mode)
        opt = parse_opt_file(os.path.join(file_path, "train_opt.txt"))

        # Adding in changes away from default opmodel options
        opt.dataroot = './data/TCGA_KIRC'
        opt.verbose = 1
        opt.print_every = 1
        opt.checkpoints_dir = checkpoints_dir
        opt.vgg_features = 0
        opt.use_vgg_features = 0
        opt.gpu_ids = []

        # Currently loading./data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_1_0_0.pkl
        #  Not using vgg_features or rnaseq
        data_cv_path = "%s/splits/KIRC_st_0.pkl" % (opt.dataroot)
        # data_cv_path = "./data/TCGA_KIRC/splits/KIRC_st_0.pkl
        
        print("Loading %s" % data_cv_path)

        # Grad settings bespoke
        if mode=='path':
            opt.batch_size = 128
            opt.mode = "path"

        # Added in code to print changed attributes
        for attr, value in vars(opt).items():
            print(f"{attr} = {value}")

        # Use device CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # Create directories
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)
        if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)):
            os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
        if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)):
            os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

        data_cv_splits = pickle.load(open(data_cv_path, "rb"))
        results = []

        ### 3. Sets-Up Main Loop
        # data = data_cv_splits[k]
        data = data_cv_splits['split'][k]

        ### 3.1 Trains Model
        model, optimizer, metric_logger = train(opt, data, device, k)

        # # In running this this, files are saved like this: %s_%d%s%d_pred_test.pkl" % (opt.model_name, k, use_patch, epoch)
        # df = save_metric_logger(metric_logger, opt, results_folder, k)
        # plots_train_vs_test(df, opt, results_folder, k)
    
