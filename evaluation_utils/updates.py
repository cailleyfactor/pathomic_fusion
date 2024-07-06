import torch
import torch_geometric
import os
from tqdm import tqdm

# This code helps get around the PyG error


def reformat_graph_data(data):
    """
    ensure that the object structure is correct
    """
    # use `data.py` module within the `data` subpackage
    return torch_geometric.data.Data.from_dict(data.__dict__)

# Set up the directories
source_dir = "data/TCGA_KIRC/KIRC_st_cpc/pt_bi"
target_dir = "data/TCGA_KIRC/KIRC_st_cpc/pt_bi_new"

# Create target directory if it does not exist
os.makedirs(target_dir, exist_ok=True)

# Process files with a progress bar
files = os.listdir(source_dir)

for file in tqdm(files, desc="Processing Files"):
    try:
        # Load old data
        old_data = torch.load(os.path.join(source_dir, file))
        
        # Reformat and save the new data
        new_data = reformat_graph_data(old_data)
        torch.save(new_data, os.path.join(target_dir, file))
        
    except Exception as e:
        print(f"Failed to process {file}: {e}")
