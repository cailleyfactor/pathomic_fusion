import os 
import pickle
# with open('./data/TCGA_KIRC/splits/KIRC_st_0_clin.pkl', 'rb') as file:
#     data = pickle.load(file)
#     print(data['split']['train']['x_omic'].shape)

dir_path = 'data/TCGA_KIRC/splits'
file_path = 'KIRC_st_1.pkl'
full_path = os.path.join(dir_path,file_path)
with open(full_path, 'rb') as file:
    data = pickle.load(file)
    print(data.keys())
    print(data['split'][1]['train']['x_omic'].shape)