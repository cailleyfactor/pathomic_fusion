import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from data_loaders import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from evaluation_utils.networks_captum import define_net, define_reg, define_optimizer, define_scheduler
from evaluation_utils.utils import (
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
    model = define_net(opt, k)
    opt.batch_size = len(data)

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

            x_omic = x_omic.to(device)
            x_path = x_path.to(device)
            x_grph = x_grph.to(device)

    return x_omic, x_path, x_grph