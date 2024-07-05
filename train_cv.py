import os
import logging
import numpy as np
import pickle
import torch

# Env
from data_loaders import *
from options import parse_args
from train_test_use_emb import train, test

import torch_geometric
print(torch_geometric.__version__)

from result_plots import save_metric_logger, plots_train_vs_test
from filter_patients import filter_unique_patients
from option_file_converter import parse_opt_file

### 1. Initializes parser and device
# This also prints opt but this is before the changes
# opt = parse_args()

checkpoints_dir = "./checkpoints/TCGA_GBMLGG"
# Change for each run
results_folder = "fresh_results"

for setting in ["grad_15"]:
    for mode in ["omic", "graph", "pathomic_fusion", "pathgraph_fusion", "graphomic_fusion", "pathgraphomic_fusion"]:
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
            opt.reg_type = 'all'
        
        if mode=='pathgraph_fusion':
            opt.mode = "pathgraph"
            opt.fusion_type ='pofusion'
            opt.model_name = "pathgraph_fusion"
            opt.lambda_reg = 0.0
            # Changed this to reg_type all from last run
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
        if setting=="surv_15_rnaseq" and "omic" in mode:
            opt.use_rnaseq = 1
            opt.input_size_omic = 320
        else:
            opt.use_rnaseq = 0
            opt.input_size_omic = 80

        # Added in code to print changed attributes
        for attr, value in vars(opt).items():
            print(f"{attr} = {value}")

        # Set device to MPS if GPU is available
        # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        device = torch.device("cpu")
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
        for k in range(1,6):
            data = data_cv_splits[k]
            print("*******************************************")
            print(
                "************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items()))
            )
            print("*******************************************")
            # %%
            # # This is currently: ./checkpoints/TCGA_GBMLGG/grad_15/pathgraphomic_fusion/pathgraphomic_fusion_0_patch_pred_train.pkl
            # if os.path.exists(
            #     os.path.join(
            #         opt.checkpoints_dir,
            #         opt.exp_name,
            #         opt.model_name,
            #         "%s_%d_patch_pred_train.pkl" % (opt.model_name, k+1),
            #     )
            # ):
            #     print("Train-Test Split already made.")

            ### 3.1 Trains Model
            model, optimizer, metric_logger = train(opt, data, device, k)
            # In running this this, files are saved like this: %s_%d%s%d_pred_test.pkl" % (opt.model_name, k, use_patch, epoch)
            df = save_metric_logger(metric_logger, opt, results_folder, k)
            plots_train_vs_test(df, opt, results_folder, k)

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

            # These print at the end - 
            if opt.task == "surv":
                print(
                    "[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e"
                    % (cindex_train, pvalue_train)
                )
                logging.info(
                    "[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e"
                    % (cindex_train, pvalue_train)
                )
                print(
                    "[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e"
                    % (cindex_test, pvalue_test)
                )
                logging.info(
                    "[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e"
                    % (cindex_test, pvalue_test)
                )
                results.append(cindex_test)

            elif opt.task == "grad":
                print(
                    "[Final] Apply model to training set: Loss: %.10f, Acc: %.4f"
                    % (loss_train, grad_acc_train)
                )
                logging.info(
                    "[Final] Apply model to training set: Loss: %.10f, Acc: %.4f"
                    % (loss_train, grad_acc_train)
                )
                print(
                    "[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f"
                    % (loss_test, grad_acc_test)
                )
                logging.info(
                    "[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f"
                    % (loss_test, grad_acc_test)
                )
                results.append(grad_acc_test)

            ### 3.3 Saves Model
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                model_state_dict = model.module.cpu().state_dict()
            else:
                model_state_dict = model.cpu().state_dict()

            # Save the model in a pt file
            torch.save(
                {
                    "split": k,
                    "opt": opt,
                    "epoch": opt.niter + opt.niter_decay,
                    "data": data,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metric_logger,
                },
                os.path.join(
                    opt.checkpoints_dir,
                    opt.exp_name,
                    opt.model_name,
                    "%s_%d.pt" % (opt.model_name, k),
                ),
            )

            pickle.dump(
                (pred_train, data_cv),
                open(
                    os.path.join(
                        opt.checkpoints_dir,
                        opt.exp_name,
                        opt.model_name,
                        "%s_%d%spred_train.pkl" % (opt.model_name, k, use_patch),
                    ),
                    "wb",
                ),
            )
            pickle.dump(
                (pred_test, data_cv),
                open(
                    os.path.join(
                        opt.checkpoints_dir,
                        opt.exp_name,
                        opt.model_name,
                        "%s_%d%spred_test.pkl" % (opt.model_name, k, use_patch),
                    ),
                    "wb",
                ),
            )


        print("Split Results:", results)
        print("Average:", np.array(results).mean())
        pickle.dump(
            results,
            open(
                os.path.join(
                    opt.checkpoints_dir,
                    opt.exp_name,
                    opt.model_name,
                    "%s_%d_results.pkl" % (opt.model_name, k)
                ),
                "wb",
            ),
        )
