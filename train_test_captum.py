import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from data_loaders import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from networks_captum import define_net, define_reg, define_optimizer, define_scheduler
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

from captum.attr import IntegratedGradients

def retrieve_captum_data(opt, data, device, k):
    # nn.deterministic = True
    # torch.manual_seed_all(2019)
    # torch.manual_seed(2019)
    # random.seed(2019)

    model = define_net(opt, k)
    opt.batch_size = len(data)
    # print(model)
    # print("Number of Trainable Parameters: %d" % count_parameters(model))
    # print("Activation Type:", opt.act_type)
    # print("Optimizer Type:", opt.optimizer_type)
    # print("Regularization Type:", opt.reg_type)

    # use_patch, roi_dir = (
    #     ("_patch_", "all_st_patches_512") if opt.use_vgg_features else ("_", "all_st")
    # )


    # Augmented dataset
    # opt.mode is carried through
    custom_data_loader = (
        PathgraphomicFastDatasetLoader(opt, data, split="train", mode=opt.mode)
        if opt.use_vgg_features
        else PathgraphomicDatasetLoader(opt, data, split="train", mode=opt.mode)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=mixed_collate,
    )

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):

        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch)

        model.train()
        risk_pred_all, censor_all, survtime_all = (
            np.array([]),
            np.array([]),
            np.array([]),
        )  # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(
            train_loader
        ):


            # censor = censor.to(device) if "surv" in opt.task else censor
            # grade = grade.to(device) if "grad" in opt.task else grade
            x_omic = x_omic.to(device)
            x_path = x_path.to(device)
            x_grph = x_grph.to(device)


            # # kwargs not working for path thus doing it this way
            # if opt.mode == "omic":
            #     pred = model(x_omic.to(device))
            # else:
            #     pred = model(x_omic.to(device))

    return x_omic, x_path, x_grph