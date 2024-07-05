import os
import logging
import numpy as np
import pickle
import torch

# Env
from data_loaders import *
from options import parse_args
from train_test_kirc import test

import torch_geometric
print(torch_geometric.__version__)

from result_plots import save_metric_logger, plots_train_vs_test
from option_file_converter import parse_opt_file

### 1. Initializes parser and device
# This also prints opt but this is before the changes
# opt = parse_args()
checkpoints_dir = "./checkpoints/TCGA_KIRC"

for setting in ["surv_15"]: 
    for mode in ["omic", "clin", "path", "pathclinomic_fusion", "pathomic_fusion", "clinomic_fusion", "pathclin_fusion"]:
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
        data_cv_path = "%s/splits/KIRC_st_0_clin.pkl" % (opt.dataroot)
        # data_cv_path = "./data/TCGA_KIRC/splits/KIRC_st_0.pkl"
        print("Loading %s" % data_cv_path)

        if 'omic' in mode:
            opt.input_size_omic = 362

        # Grad settings bespoke
        if mode=='path':
            opt.batch_size = 32
            opt.mode = "path"

        if "clin" in mode:
            opt.mode = "clin"
            opt.model_name = "clin"
            opt.input_size_clin = 9
            opt.clin_dim = 3

        if "clinomic_fusion" in mode:
            opt.mode = "clinomic"
            opt.model_name = "clinomic_fusion"
            opt.input_size_clin = 9
            opt.clin_dim = 3
            opt.clin_gate = 1
            opt.clin_scale = 1
        
        if "pathclin_fusion" in mode:
            opt.mode = "pathclin"
            opt.model_name = "pathclin_fusion"
            opt.input_size_clin = 9
            opt.clin_dim = 3
            opt.clin_gate = 1
            opt.clin_scale = 1
            opt.reg_type = 'all'

        if "pathclinomic_fusion" in mode:
            opt.mode = "pathclinomic"
            opt.model_name = "pathclinomic_fusion"
            opt.input_size_clin = 9
            opt.clin_dim = 3
            opt.clin_gate = 1
            opt.clin_scale = 1
        
        if "clinclin_fusion" in mode:
            opt.mode = "clinclin"
            opt.model_name = "clinclin_fusion"
            opt.input_size_clin = 9
            opt.clin_dim = 3
            opt.clin_gate = 1
            opt.clin_scale = 1
            opt.reg_type = 'all'


        # Set device to MPS if GPU is available
        device = (
            torch.device("mps:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
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
        k = 1

        ### 3. Sets-Up Main Loop
        data = data_cv_splits[k]

        # Load in the trained model
        checkpoint = torch.load(
            os.path.join(
                opt.checkpoints_dir,
                opt.exp_name,
                opt.model_name,
                "%s_%d.pt" % (opt.model_name, k),
            )
        )

        # Define the model
        from networks_kirc import define_net
        model = define_net(opt, k)
        model.load_state_dict(checkpoint["model_state_dict"])     

        ### 3.2 Evalutes Train + Test Error, and Saves Model
        (
            loss_train,
            cindex_train,
            pvalue_train,
            surv_acc_train,
            grad_acc_train,
            pred_train,
        ) = test(opt, model, data, "train", device)

        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(
            opt, model, data, "test", device
        )

        data_train = (pred_train, data_cv_splits)
        data_test = (pred_test, data_cv_splits)

        pickle.dump(
            data_train,
            open(
                os.path.join(
                    opt.checkpoints_dir,
                    opt.exp_name,
                    opt.model_name,
                    "%s_%dpred_train_append.pkl" % (opt.model_name, k),
                ),
                "wb",
            ),
        )
        pickle.dump(
            data_test,
            open(
                os.path.join(
                    opt.checkpoints_dir,
                    opt.exp_name,
                    opt.model_name,
                    "%s_%dpred_test_append.pkl" % (opt.model_name, k),
                ),
                "wb",
            ),
        )
