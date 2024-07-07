import os
import logging
import numpy as np
import pickle
import torch

# Env
from data_loaders import *
from evaluation_utils.options import parse_args
from train_test_make_emb import train, test

import torch_geometric
print(torch_geometric.__version__)

from evaluation_utils.result_plots import save_metric_logger, plots_train_vs_test
from evaluation_utils.filter_patients import filter_unique_patients
from evaluation_utils.option_file_converter import parse_opt_file

checkpoints_dir = "./checkpoints/TCGA_GBMLGG"

for mode in ["path"]:
    setting = "surv_15_rnaseq"
    file_path = os.path.join(checkpoints_dir, setting, mode)
    opt = parse_opt_file(os.path.join(file_path, "train_opt.txt"))
    device = torch.device("cpu")

	# Adding in changes away from default opmodel options
    opt.dataroot = './data/TCGA_GBMLGG'
    opt.verbose = 1
    opt.print_every = 1
    opt.checkpoints_dir = checkpoints_dir
    opt.vgg_features = 0
    opt.use_vgg_features = 0
    opt.gpu_ids = []

	# Surv settings bespoke
    if setting=="surv_15_rnaseq" and mode=="path":
        opt.beta1 = 0.9
        opt.beta2 = 0.999
        opt.niter_decay = 25

    opt.use_rnaseq = 0
    opt.input_size_omic = 80
    opt.batch_size = 128
    opt.mode = "path"

	### Initializes Data
    ignore_missing_histype = 1 if 'grad' in opt.task else 0
    ignore_missing_moltype = 1 if 'omic' in opt.mode else 0

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')
    use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''

    if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

    data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
    print("Loading %s" % data_cv_path)
    data_cv = pickle.load(open(data_cv_path, 'rb'))
    data_cv_splits = data_cv['cv_splits']
    results = []

    data_cv_splits = pickle.load(open(data_cv_path, "rb"))
    results = []

    k=1
    data = data_cv_splits['cv_splits'][k]

    model, optimizer, metric_logger = train(opt, data, device, k)

