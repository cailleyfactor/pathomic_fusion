import os
import logging
import numpy as np
import random
import pickle
import torch

# Env
from networks_use_emb import define_net
from data_loaders import *
from additional_core.options import parse_args
from train_test_use_emb import train, test
from additional_core.option_file_converter import parse_opt_file

checkpoints_dir = "./checkpoints/TCGA_GBMLGG"
setting = 'surv_15_rnaseq'

for mode in ["omic", "graph", "path", "pathomic_fusion", "graphomic_fusion", "pathgraphomic_fusion", "pathgraph_fusion"]:
	### 1. Initializes parser and device
	file_path = os.path.join(checkpoints_dir, setting, mode)
	opt = parse_opt_file(os.path.join(file_path, "train_opt.txt"))
 	# Set device to MPS if GPU is available
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
		opt.batch_size = 32
		opt.niter_decay = 25

	if setting=="surv_15_rnaseq" and mode=="omic":
		opt.model_name = opt.model

	if mode=='pathgraph_fusion':
		opt.mode = "pathgraph"
		opt.fusion_type ='pofusion'
		opt.model_name = "pathgraph_fusion"
		opt.lambda_reg = 0.0
		# Changed this to reg_type all from last run
		opt.reg_type = 'none'

	# RNASeq setting
	if setting=="surv_15_rnaseq" and "omic" in mode:
		opt.use_rnaseq = 1
		opt.input_size_omic = 320
	else:
		opt.use_rnaseq = 0
		opt.input_size_omic = 80

	### 2. Initializes Data
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

	### 3. Sets-Up Main Loop
	for k in range(1,6):
		data = data_cv_splits[k]
		### 3. Sets-Up Main Loop
		# for k, data in data_cv_splits.items():
		print("*******************************************")
		print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
		print("*******************************************")
		load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
		model_ckpt = torch.load(load_path, map_location=device)

		#### Loading Env
		model_state_dict = model_ckpt['model_state_dict']
		if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

		model = define_net(opt, None)
		if isinstance(model, torch.nn.DataParallel): model = model.module

		print('Loading the model from %s' % load_path)
		model.load_state_dict(model_state_dict)


		### 3.2 Evalutes Train + Test Error, and Saves Model
		loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device)

		if opt.task == 'surv':
			print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
			logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
			results.append(cindex_test)
		elif opt.task == 'grad':
			print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
			logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
			results.append(grad_acc_test)

			# ### 3.3 Saves Model
			# pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))

	print('Split Results:', results)
	print("Average:", np.array(results).mean())
	pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))