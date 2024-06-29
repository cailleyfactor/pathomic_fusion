import os
import logging
import numpy as np
import pickle
import torch

# Env
from data_loaders import *
from options import parse_args
from train_test_kirc import train, test

import torch_geometric
print(torch_geometric.__version__)

from result_plots import save_metric_logger, plots_train_vs_test
from filter_patients import filter_unique_patients
from option_file_converter import parse_opt_file

### 1. Initializes parser and device
# This also prints opt but this is before the changes
# opt = parse_args()
checkpoints_dir = "./checkpoints/TCGA_KIRC"
results_folder = "results_2"

# Decide number of folds for training
for k in range(1, 5):
    for mode in ["omic", "clin", "path", "clinomic_fusion", "pathclin_fusion", "pathomic_fusion","pathclinomic_fusion"]:
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
        data_cv_path = "%s/splits/KIRC_st_0_clin.pkl" % (opt.dataroot)
        # data_cv_path = "./data/TCGA_KIRC/splits/KIRC_st_0.pkl"
        print("Loading %s" % data_cv_path)

        if 'omic' in mode:
            opt.input_size_omic = 362

        #  settings bespoke
        if mode=='path':
            opt.batch_size = 32
            opt.mode = "path"

        if "clin" in mode:
            opt.mode = "clin"
            opt.model_name = "clin"
            opt.input_size_clin = 9
            opt.clin_dim = 32

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

        # Added in code to print changed attributes
        for attr, value in vars(opt).items():
            print(f"{attr} = {value}")

        # Set device to MPS if GPU is available
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        plots_train_vs_test(df, opt, results_folder,k)

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
            pred_train,
            open(
                os.path.join(
                    opt.checkpoints_dir,
                    opt.exp_name,
                    opt.model_name,
                    "%s_%dpred_train.pkl" % (opt.model_name, k),
                ),
                "wb",
            ),
        )
        pickle.dump(
            pred_test,
            open(
                os.path.join(
                    opt.checkpoints_dir,
                    opt.exp_name,
                    opt.model_name,
                    "%s_%dpred_test.pkl" % (opt.model_name, k),
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
                    "%s_results.pkl" % opt.model_name,
                ),
                "wb",
            ),
        )
