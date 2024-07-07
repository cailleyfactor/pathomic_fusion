import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from data_loaders import MakeEmbeddingsDatasetLoader
from networks_make_emb import define_net, define_reg, define_optimizer, define_scheduler
from evaluation_utils.utils import (
    unfreeze_unimodal,
    CoxLoss,
    CIndex_lifeline,
    cox_log_rank,
    accuracy_cox,
    mixed_collate,
    count_parameters,
)
import os 


def train(opt, data, device, k):
    model = define_net(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    # Augmented dataset
    # opt.mode is carried through
    custom_data_loader = (
        MakeEmbeddingsDatasetLoader(opt, data, split="train", mode=opt.mode)
        if opt.use_vgg_features
        else MakeEmbeddingsDatasetLoader(opt, data, split="train", mode=opt.mode)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=mixed_collate,
    )

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
        dir = './GBMLGG_embeddings'
        directory = os.path.join(dir, f'{epoch}')
        os.makedirs(directory, exist_ok=True)

        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch)

        model.train()
        for batch_idx, (x_path, x_pathfilename, x_omic, censor, survtime, grade) in enumerate(
            train_loader
        ):

            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade

            # kwargs not working for path thus doing it this way
            if opt.mode == "path":
                x_path, x_pathfilename = model(x_path.to(device), x_pathfilename)
                x_path = x_path.numpy()
                x_pathfilename = np.array(x_pathfilename)
                for i in range(len(x_pathfilename)):
                    x_pat_i = os.path.basename(x_pathfilename[i])
                    file_path = os.path.join(directory, f'{x_pat_i}.npy')
                    np.save(file_path, x_path[i])
                print(f'training {epoch}')

            else:
                _, pred = model(
                    x_path=x_path.to(device),
                    x_pathfilename=x_pathfilename.to(device),
                    x_omic=x_omic.to(device),
                )

        scheduler.step()

        # Adding in code to test the model
        test(opt, model, data, "test", device, epoch)

    return model, optimizer

#### Define the test function ###
def test(opt, model, data, split, device, epoch):
    model.eval()

    dir = './GBMLGG_embeddings'
    directory = os.path.join(dir, f'{epoch}')

    custom_data_loader = (
        MakeEmbeddingsDatasetLoader(opt, data, split, mode=opt.mode)
        if opt.use_vgg_features
        else MakeEmbeddingsDatasetLoader(opt, data, split=split, mode=opt.mode)
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_size,
        shuffle=False,
        collate_fn=mixed_collate,
    )
    for batch_idx, (x_path, x_patfilename, x_omic, censor, survtime, grade) in enumerate(
        test_loader
    ):
    
        if opt.mode == "path":
            x_path_test, x_pathfilename_test = model(x_path.to(device), x_patfilename)
             # Converting resulting patient names and embedding names into numpy arrays
            x_path_test = x_path_test.numpy()
            x_pathfilename_test = np.array(x_pathfilename_test)
            for i in range(len(x_pathfilename_test)):
                x_pat_it = os.path.basename(x_pathfilename_test[i])
                file_path = os.path.join(directory, f'{x_pat_it}.npy')
                np.save(file_path, x_path_test[i])
            print(f'testing {epoch}')
        else:
            _, pred = model(
                x_path=x_path.to(device),
                x_patfilename=x_patfilename.to(device),
                x_omic=x_omic.to(device),
            )
        
    return None
