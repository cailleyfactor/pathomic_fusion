# ReadMe
## Description
This repository is based on a repository from Harvard University's Mahmood Lab available at https://github.com/mahmoodlab/PathomicFusion. The corresponding paper is: Richard J. Chen, Ming Y. Lu, Jingwen Wang, Drew F. K. Williamson, Scott J. Rodig, Neal I. Lindeman, Faisal Mahmood. "Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis." arXiv preprint arXiv:1912.08937 (2020). Available at: https://arxiv.org/abs/1912.08937. 

This project ammends this existing codebase as a substantial protion of their existing code could not be run in its available form or was lacking functionality mentioned in the paper. The evaluation methodology needed to be substantially modified to run and no code was available for running TCGA-KIRC directly, as the available repository was structured for TCGA-GBMLGG. New functionality was also added for utilising clinical data and clinical networks for TCGA-KIRC. The project report and an executive summary are available in the folder entitled report_and_summary in the root directory. The executive summary is 996 words and the dissertation/report is 6428 words.

## Repository structure
The code base structure is explained below: 
- **Evaluation_GBMLGG.ipynb**: This is a Jupyter notebook to run the evaluation methodology for TCGA_GBMLGG.
- **Evaluation_KIRC.ipynb**: This is a Jupyter notebook to run the evaluation methodology for TCGA_KIRC.
- **train_cv.py**: This file trains the models for TCGA_GBMLGG grade and survival analysis and evaluates the train and test splits. This script will save evaluation metrics and predictions on the train + test split for each epoch in **checkpoints**. 
- **train_cv_kirc.py**: This file trains the models for TCGA_KIRC survival analysis and evaluates the train and test splits. This script will save evaluation metrics and predictions on the train + test split for each epoch in **checkpoints**.
- **checkpoints**: Place the folder here from the onedrive link discussed below.
- **data**: Place the folder here from the dropbox link discussed below.
- **CellGraph**: contains the files needed for cell graph construction
    - However, the premade graph files in the data were utilised for training the models in the reproduction due to a lack of the code needed to segment the nuclei and difficulties in attempts to create segmentations that were similar.
- **data_utils**: Contains utilities for the data
    - **make_splits.py**: Script for generating a pkl file that saves + aligns the path for multimodal data for cross-validation.
    - **modify_splits_kirc.py**: Add clinical data to existing pkl files for KIRC
    - **option_file_converter.py**: Code for parsing the opt.txt files
    - **options.py**: Contains all the options for the argparser.
    - **utils_data.py**: Files to add histomolecular subtypes and clean the data.
    - **updates.py**: This code is used to get around the PyG error of utilising the graph data files in obsolete formats from the Mahmood repository.
- **evaluation_utils**: Contains additional utilies for evaluation
    - **captum_data_retriever.py**: Retrieve the data for the captum integrated gradients evaluation.
    - **integrated_gradients.py**: Code to run to generate integrated gradients for the TCGA-GBMLGG genomic model
    - **run_cox_baselines.py**: Script for running Cox baselines.
    - **networks_captum.py**: This script modifies networks in order to utilise integrated gradients for the genomic SNN
    - **utils_analysis_new.py**: Utilities for evaluation
- **training_utils**: Contains core utilities for training
    - **data_loaders_kirc.py**: Contains the PyTorch DatasetLoader definition for loading multimodal data for training on TCGA-KIRC.
    - **data_loaders.py**: Contains the PyTorch DatasetLoader definition for loading multimodal data for training on TCGA-GBMLGG.
    - **fusion.py**: Contains PyTorch model definitions for fusion.
    - **networks_kirc.py**: Contains PyTorch model definitions for all unimodal and multimodal network for TCGA-KIRC.
    - **networks.py**: Contains PyTorch model definitions for all unimodal and multimodal network for TCGA-GBMLGG.
    - **results_plots.py** Code to save metrics during training and plot training losses and other metrics per epoch.
    - **test_cv.py**: Code to test trained models, but this functionality is already integrated in train_cv.py
    - **train_test.py**: Contains the definitions for "train" and "test" for TCGA-GBMLGG. 
    - **train_test_kirc.py**: Contains the definitions for "train" and "test" for TCGA-GBMLGG.
    - **utils_models.py**: Code for losses, network initialisation, freezing the models, and regularization, etc.
    - **utils.py**: Contains definitions for collating, survival loss functions, data preprocessing, evaluation, figure plotting, etc...    
- **report_and_summary**: Contains the dissertation and executive summary in PDF format

## Usage
Jupyter notebooks for the evaluation code are available in the root directory. 

To run the code for training, first clone the repository from git. To make the environment for running the project a n environment.yml file is provided with the necessary packages needed to run the environment. Create this environment using: conda env create -f environment.yml

A folder entitled 'checkpoints' needs to be created in the root directory and the trained checkpoints from this reproduction can be downloaded here: https://1drv.ms/f/s!AorA9DLtGnxlhMNWgd0K-rRJSO13BQ?e=WwqAox
A folder entitled 'data' needs to be created also in the root directory and the relevant data is available from the Mahmood lab here in the 'data' directory: https://drive.google.com/drive/folders/1swiMrz84V3iuzk8x99vGIBd5FCVncOlf
The updates.py file in evaluation_utils needs to be run before training any graph networks.

To train the models, run the following for TCGA-GBMLGG and TCGA-KIRC respectively:
$python train_cv.py 
$python train_cv_kirc.py 

## Auto-generation tool citations
ChatGPT was used for tasks such as generating code with improved syntax for handling pandas dataframes and other similar tasks for the data engineering of new data. It was used to help write code in order to pull from the opt.txt files with the prompt explaining the structure of the opt.txt files. ChatGPT was also used to help improve plot appearance for matplotlib and seaborn plots in the evaluation methodology. It was also used in the report to help improve the flow of some of the sentences alongside small segments of drafts of the report. Github copilot was occasionally used for commenting the code.

## License
The Mahmood lab licenses this code under the GNU GPLv3 License - see the LICENSE.txt file for details.