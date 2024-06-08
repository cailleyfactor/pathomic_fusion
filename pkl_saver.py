import os
import logging
import numpy as np
import pickle
import torch

# Env
from data_loaders import *
from options import parse_args
from train_test import train, test

import torch_geometric
print(torch_geometric.__version__)

from result_plots import save_metric_logger, plots_train_vs_test
from option_file_converter import parse_opt_file

### 1. Initializes parser and device
# This also prints opt but this is before the changes
# opt = parse_args()
checkpoints_dir = "./checkpoints/TCGA_GBMLGG"


for setting in ["grad_15","surv_15_rnaseq"]: 
    for mode in ["omic", "graph", "omic", "pathgraphomic_fusion", "pathomic_fusion", "graphomic_fusion", "pathgraph_fusion", "path", "pathpath_fusion", "graphgraph_fusion", "omicomic_fusion"]:
        
        file_path = os.path.join(checkpoints_dir, setting, mode)
        opt = parse_opt_file(os.path.join(file_path, "train_opt.txt"))

        # Adding in changes away from default opmodel options
        opt.dataroot = './data/TCGA_GBMLGG'
        opt.verbose = 1
        opt.print_every = 1
        opt.checkpoints_dir = checkpoints_dir
        opt.vgg_features = 0
        opt.use_vgg_features = 0
        opt.gpu_ids = []

        # Grad settings bespoke
        if setting=="grad_15" and mode=="path":
            opt.model_name = opt.model
            opt.batch_size = 32
            opt.niter_decay = 25
        
        if mode=='pathgraph_fusion':
            opt.mode = "pathgraph"
            opt.fusion_type ='pofusion'
            opt.model_name = "pathgraph_fusion"
            opt.lambda_reg = 0.0
            opt.reg_type = 'none'


        
        if setting=="grad_15" and mode=='graphomic_fusion':
            opt.model_name = opt.mode_name

        if setting=="grad_15" and mode=='pathgraphomic_fusion':
            opt.model_name = opt.model
        
        # Surv settings bespoke
        if setting=="surv_15_rnaseq" and mode=="path":
            opt.beta1 = 0.9
            opt.beta2 = 0.999
            opt.batch_size = 32
            opt.niter_decay = 25

        if setting=="surv_15_rnaseq" and mode=="omic":
            opt.model_name = opt.model

        # General
        if setting=="grad_15" and mode=="pathpath_fusion":
            opt.model_name = opt.model

        if setting=="grad_15" and mode=="omicomic_fusion":
            opt.model_name = opt.model

        # RNASeq setting
        if "omic" in mode:
            opt.use_rnaseq = 1
            opt.input_size_omic = 320
        else:
            opt.use_rnaseq = 0
        #     opt.input_size_omic = 80

        # opt.fusion_type = "pofusion_A" # pofusion for bimodal 

        # # Options for surv
        # opt.task = "surv"
        # opt.exp_name = "surv"
        # opt.label_dim = 1 
        # opt.act_type = "Sigmoid" # surv

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

        # Create directories
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)
        if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)):
            os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
        if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)):
            os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

        ### 2. Initializes Data
        # 1 if the string grad is found in opt.task, 0 otherwise
        ignore_missing_histype = 1 if "grad" in opt.task else 0
        ignore_missing_moltype = 1 if "omic" in opt.mode else 0

        # Use_vgg_features defaults to 0 - need to make sure I'm doing this the right way around
        # Does 1 mean use pre-trained embeddings? Should this be set to 1?
        use_patch, roi_dir = (
            ("_patch_", "all_st_patches_512") if opt.use_vgg_features else ("_", "all_st")
        )

        # Use_rnaseq defaults to 0
        use_rnaseq = "_rnaseq" if opt.use_rnaseq else ""

        # Currently loading./data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_1_0_0.pkl
        #  Not using vgg_features or rnaseq
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
        results = []

        ### 3. Sets-Up Main Loop
        k = 1
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
        from networks import define_net
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

        data_train = (pred_train, data_cv)
        data_test = (pred_test, data_cv)

        pickle.dump(
            data_train,
            open(
                os.path.join(
                    opt.checkpoints_dir,
                    opt.exp_name,
                    opt.model_name,
                    "%s_%d%spred_train_data.pkl" % (opt.model_name, k, use_patch),
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
                    "%s_%d%spred_test_data.pkl" % (opt.model_name, k, use_patch),
                ),
                "wb",
            ),
        )


        # print("Split Results:", results)
        # print("Average:", np.array(results).mean())
        # pickle.dump(
        #     results,
        #     open(
        #         os.path.join(
        #             opt.checkpoints_dir,
        #             opt.exp_name,
        #             opt.model_name,
        #             "%s_results.pkl" % opt.model_name,
        #         ),
        #         "wb",
        #     ),
        # )
