import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from data_loaders import KircClinDatasetLoader
from networks_kirc_use_emb import define_net, define_reg, define_optimizer, define_scheduler
from utils import (
    unfreeze_unimodal,
    CoxLoss,
    CIndex_lifeline,
    cox_log_rank,
    accuracy_cox,
    mixed_collate,
    count_parameters,
)

# from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os


def train(opt, data, device, k):
    # nn.deterministic = True
    # torch.manual_seed_all(2019)
    # torch.manual_seed(2019)
    # random.seed(2019)

    model = define_net(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    # use_patch, roi_dir = (
    #     ("_patch_", "all_st_patches_512") if opt.use_vgg_features else ("_", "all_st")
    # )

    # Augmented dataset
    # opt.mode is carried through
    custom_data_loader = (
        KircClinDatasetLoader(opt, data, split="train", mode=opt.mode) 
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=mixed_collate,
    )
    metric_logger = {
        "train": {
            "loss": [],
            "pvalue": [],
            "cindex": [],
            "surv_acc": [],
            "grad_acc": [],
        },
        "test": {
            "loss": [],
            "pvalue": [],
            "cindex": [],
            "surv_acc": [],
            "grad_acc": [],
        },
    }

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):

        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch)
            
        model.to(device)
        model.train()
        risk_pred_all, censor_all, survtime_all = (
            np.array([]),
            np.array([]),
            np.array([]),
        )  # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_path, x_clin, x_omic, censor, survtime, grade) in enumerate(
            train_loader
        ):
            draw = random.randint(2, 3)
            directory = './KIRC_embeddings'
            file_path = os.path.join(directory, f'embeddings_{draw}.npy')
            embeddings = np.load(file_path)
            emb_patnames = embeddings[:,0]
            emb_x_paths = embeddings[:,1:-1]
            
            x_path = x_path.to(device)
            indices = []
            for path_file_name in x_path:
                if path_file_name in emb_patnames:
                    index = emb_patnames.index(path_file_name)
                    indices.append(index)
                else:
                    raise ValueError(f"Missing pathology embedding in embeddings_{draw}.npy")

            # make sure that you got the dimensionality right
            x_path = emb_x_paths[indices,1:-1]

            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade

            # kwargs not working for path thus doing it this way
            if opt.mode == "path":
                __, pred = model(x_path.to(device))
            else:
                _, pred = model(
                    x_path=x_path.to(device),
                    x_omic=x_omic.to(device),
                    x_clin=x_clin.to(device),
                )

            # Loss calculations
            loss_cox = (
                CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            )
            loss_reg = define_reg(opt, model)
            loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
            loss = (
                opt.lambda_cox * loss_cox + opt.lambda_nll * loss_nll
            ) + opt.lambda_reg*loss_reg
            loss_epoch += loss.data.item()

            optimizer.zero_grad()  # Zero the gradients
            loss.backward() # Backpropagation
            optimizer.step() # Update the weights

            # For surv, concatenate the model prediction and censoring indicators and the survival times to respective arrays 
            if opt.task == "surv":
                risk_pred_all = np.concatenate(
                    (risk_pred_all, pred.detach().cpu().numpy().reshape(-1))
                )  # Logging Information
                censor_all = np.concatenate(
                    (censor_all, censor.detach().cpu().numpy().reshape(-1))
                )  # Logging Information
                survtime_all = np.concatenate(
                    (survtime_all, survtime.detach().cpu().numpy().reshape(-1))
                )  # Logging Information

            # For grad, find the class with the highest prediction score and compare to ground truth
            elif opt.task == "grad":

                # Handmade acc calculation for each batch
                pred = pred.argmax(dim=1, keepdim=True)
                acc = pred.eq(grade.view_as(pred)).sum().item()
                batch_size = pred.size(0)
                accuracy = acc / batch_size

                # Epoch acc calculation
                grad_acc_epoch += pred.eq(grade.view_as(pred)).sum().item()

            # Check if should print out training progress
            if (
                opt.verbose > 0
                and opt.print_every > 0
                and (
                    batch_idx % opt.print_every == 0
                    or batch_idx + 1 == len(train_loader)
                )
            ):
                if opt.task == "grad":
                    # Add an accuracy to print out here
                    print(
                        "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}, Accuracy {:9.4f}".format(
                            epoch,
                            opt.niter + opt.niter_decay,
                            batch_idx + 1,
                            len(train_loader),
                            loss.item(),
                            accuracy,
                        )
                    )
                else:
                    print(
                        "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                            epoch,
                            opt.niter + opt.niter_decay,
                            batch_idx + 1,
                            len(train_loader),
                            loss.item(),
                        )
                    )

        scheduler.step()
        # lr = optimizer.param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

        if opt.measure or epoch == (opt.niter + opt.niter_decay - 1):
            loss_epoch /= len(train_loader)

            cindex_epoch = (
                CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
                if opt.task == "surv"
                else None
            )
            pvalue_epoch = (
                cox_log_rank(risk_pred_all, censor_all, survtime_all)
                if opt.task == "surv"
                else None
            )
            surv_acc_epoch = (
                accuracy_cox(risk_pred_all, censor_all) if opt.task == "surv" else None
            )
            grad_acc_epoch = (
                grad_acc_epoch / len(train_loader.dataset)
                if opt.task == "grad"
                else None
            )

            (
                loss_test,
                cindex_test,
                pvalue_test,
                surv_acc_test,
                grad_acc_test,
                pred_test,
            ) = test(opt, model, data, "test", device)

            metric_logger["train"]["loss"].append(loss_epoch)
            metric_logger["train"]["cindex"].append(cindex_epoch)
            metric_logger["train"]["pvalue"].append(pvalue_epoch)
            metric_logger["train"]["surv_acc"].append(surv_acc_epoch)
            metric_logger["train"]["grad_acc"].append(grad_acc_epoch)

            metric_logger["test"]["loss"].append(loss_test)
            metric_logger["test"]["cindex"].append(cindex_test)
            metric_logger["test"]["pvalue"].append(pvalue_test)
            metric_logger["test"]["surv_acc"].append(surv_acc_test)
            metric_logger["test"]["grad_acc"].append(grad_acc_test)
            k = k if k is not None else 0  # Default k to 0 if None    
                    
            # Changed this - construct the directory path
            dir_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)

            # Check that the directory exists
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Folder created: {dir_path}")
            else:
                print(f"Folder already exists: {dir_path}")

            # Construct the file path
            file_path = os.path.join(dir_path, "%s_%d%d_pred_test.pkl" % (opt.model_name, k,  epoch))
 
            # Save the pickle file
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(pred_test, f)
                print(f"File saved: {file_path}")
            except Exception as e:
                print(f"Error saving file: {e}")

            # This code prints out the training and testing results for each epoch but only if the verbose option is > 0
            if opt.verbose > 0:
                if opt.task == "surv":
                    print(
                        "[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}".format(
                            "Train", loss_epoch, "C-Index", cindex_epoch
                        )
                    )
                    print(
                        "[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n".format(
                            "Test", loss_test, "C-Index", cindex_test
                        )
                    )
                elif opt.task == "grad":
                    print(
                        "[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}".format(
                            "Train", loss_epoch, "Accuracy", grad_acc_epoch
                        )
                    )
                    print(
                        "[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n".format(
                            "Test", loss_test, "Accuracy", grad_acc_test
                        )
                    )

            if opt.task == "grad" and loss_epoch < opt.patience:
                print("Early stopping at Epoch %d" % epoch)
                break

    return model, optimizer, metric_logger

#### Define the test function ###
def test(opt, model, data, split, device):

    model.to(device)
    model.eval()
    custom_data_loader = (
        KircClinDatasetLoader(opt, data, split=split, mode=opt.mode) 
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_size,
        shuffle=False,
        collate_fn=mixed_collate,
    )

    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (x_path, x_clin, x_omic, censor, survtime, grade) in enumerate(
        test_loader
    ):

        draw = random.randint(2, 3)
        directory = './KIRC_embeddings'
        file_path = os.path.join(directory, f'embeddings_{draw}.npy')
        embeddings = np.load(file_path)
        emb_patnames = embeddings[:,0]
        emb_x_paths = embeddings[:,1:-1]
        
        x_path = x_path.to(device)
        indices = []
        for path_file_name in x_path:
            if path_file_name in emb_patnames:
                index = emb_patnames.index(path_file_name)
                indices.append(index)
            else:
                raise ValueError(f"Missing pathology embedding in embeddings_{draw}.npy")

        # make sure that you got the dimensionality right
        x_path = emb_x_paths[indices,1:-1]


        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade

        if opt.mode == "path":
                __, pred = model(x_path.to(device))
        else:
            _, pred = model(
                x_path=x_path.to(device),
                x_omic=x_omic.to(device),
                x_clin=x_clin.to(device)
            )

        loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
        loss_reg = define_reg(opt, model)
        loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
        loss = (
            opt.lambda_cox * loss_cox
            + opt.lambda_nll * loss_nll
            + opt.lambda_reg * loss_reg
        )
        loss_test += loss.data.item()

        gt_all = np.concatenate(
            (gt_all, grade.detach().cpu().numpy().reshape(-1))
        )  # Logging Information

        if opt.task == "surv":
            risk_pred_all = np.concatenate(
                (risk_pred_all, pred.detach().cpu().numpy().reshape(-1))
            )  # Logging Information
            censor_all = np.concatenate(
                (censor_all, censor.detach().cpu().numpy().reshape(-1))
            )  # Logging Information
            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1))
            )  # Logging Information
        elif opt.task == "grad":
            # Finding the index of the maximum value along dimensin 1 of pred and storing it in grade_pred
            grade_pred = pred.argmax(dim=1, keepdim=True)
            # Increment a running total of correct predictions
            grad_acc_test += grade_pred.eq(grade.view_as(grade_pred)).sum().item()
            probs_np = pred.detach().cpu().numpy()
            # If probs_all is None it is set to probs_np, otherwise it is concatenated with probs_np
            probs_all = (
                probs_np
                if probs_all is None
                else np.concatenate((probs_all, probs_np), axis=0)
            )  # Logging Information

    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader)
    cindex_test = (
        CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        if opt.task == "surv"
        else None
    )
    pvalue_test = (
        cox_log_rank(risk_pred_all, censor_all, survtime_all)
        if opt.task == "surv"
        else None
    )
    surv_acc_test = (
        accuracy_cox(risk_pred_all, censor_all) if opt.task == "surv" else None
    )
    grad_acc_test = (
        grad_acc_test / len(test_loader.dataset) if opt.task == "grad" else None
    )
    # Risk_pred_all is hazards, probs all
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test
