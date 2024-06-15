import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from data_loaders import MakeEmbeddingsDatasetLoader
from networks_embed import define_net, define_reg, define_optimizer, define_scheduler
from utils import (
    unfreeze_unimodal,
    CoxLoss,
    CIndex_lifeline,
    cox_log_rank,
    accuracy_cox,
    mixed_collate,
    count_parameters,
)
import os 

directory = './KIRC_embeddings'

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
        batch_arrays = []
        pat_arrays = []
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
                batch_arrays.append(x_path)
                pat_arrays.append(x_pathfilename)
                print(f"training {batch_idx}")
            else:
                _, pred = model(
                    x_path=x_path.to(device),
                    x_pathfilename=x_pathfilename.to(device),
                    x_omic=x_omic.to(device),
                )

        scheduler.step()

        # Adding in code to test the model
        x_path_test, x_pathfilename_test = test(opt, model, data, "test", device)

        # Converting resulting patient names and embedding names into numpy arrays
        x_path_test = x_path_test.numpy()
        x_pathfilename_test = np.array(x_pathfilename_test)

        # Appending to lists
        batch_arrays.append(x_path_test)
        pat_arrays.append(x_pathfilename_test)

        # Converting the lists to one large numpy array for saving
        epoch_array = np.concatenate(batch_arrays, axis=0)
        epoch_pat_array = np.concatenate(pat_arrays, axis=0).reshape(-1,1)
        final_array = np.concatenate((epoch_pat_array, epoch_array), axis=1)
        
        # Save the array for the epoch
        # Check if the directory exists, and if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f'embeddings_{epoch}.npy')
        np.save(file_path, final_array)
        print(f'saving embeddings_{epoch}.npy')

    return model, optimizer

#### Define the test function ###
def test(opt, model, data, split, device):
    model.eval()

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
            x_path, x_patfilename = model(x_path.to(device), x_patfilename)
            print(f"testing {batch_idx}")
        else:
            _, pred = model(
                x_path=x_path.to(device),
                x_patfilename=x_patfilename.to(device),
                x_omic=x_omic.to(device),
            )


    return x_path, x_patfilename
