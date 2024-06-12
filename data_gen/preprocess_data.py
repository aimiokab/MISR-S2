import argparse
import os
import sys
##sys.path.insert(1, '/home/aimi/Documents/code/tests/SRDiff_test/')
#sys.path.insert(1, '/share/projects/sesure/aimi/SRDiff_test/')

import pandas as pd
import torch

from tqdm.auto import tqdm

from dataset import Dataset


def preprocess_and_save(path, phase="train", sen2_amount=1):
    dataset_name = f"dataset_{phase}.pkl"

    dict = {}
    if sen2_amount == 1:
        dict_keys = ['path_lr','path_lr_up']
        sr_type = "SISR"
        image_keys = ['img_lr', 'img_lr_up']
    else:
        dict_keys = ['path_lr']
        sr_type = "MISR"
        image_keys = ['img_lr']

    type_path = [path, "preprocessed_", sr_type, "dataset_"+phase]
    target_path = os.path.join('', *type_path)

    
    # Read DataFrame corresponding to dataset split
    dataset_df = pd.read_pickle(os.path.join(path, dataset_name))
    # Create data generator
    dataset= Dataset(path,phase=phase,sen2_amount=sen2_amount)

    # Create subfolders if needed
    for key in image_keys:
        subfolder = os.path.join(target_path, key)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)


    pbar = tqdm(dataset)
    alphas_dates = {'alphas': [], 'dates_encoding': []}

    indexes = dataset_pkl.index

    for e, data in enumerate(pbar):

        index = indexes[e]
       
        # In case of multi-image SR, save positional encoding for dates
        if sen2_amount > 1:
            alphas_dates['alphas'].append(data["alphas"])
            alphas_dates['dates_encoding'].append(data["dates_encoding"])
        for key in image_keys:
            save_path = os.path.join(target_path, f"{key}{index}.pt")
            torch.save(data[key], save_path)
    
    #temp = pd.DataFrame(alphas_dates)
    #temp.to_pickle(os.path.join(target_path, "alphas_dates_"+phase+".pkl"))
    
    """
    for i,key in enumerate(dict_keys):
        dict[key] = [os.path.join('',*(type_path[1:]+[imgs[i],str(x)+".pt"])) for x in indexes]
    dict["index"] = indexes
    df = pd.DataFrame(dict).set_index("index")

    df = dataset_pkl.join(df)

    if sen2_amount>1:
        df = df.assign(alphas = alphas_dates['alphas'])
        df = df.assign(dates_encoding = alphas_dates['dates_encoding'])
    df.to_pickle(os.path.join(target_path, "dataset_"+phase+".pkl"))"""
    return 

"""
def preprocess_and_save(path, phase="train", sen2_amount=1):
    dataset_name = "dataset_"+phase+".pkl"
    
    if sen2_amount==1:
        #dict = {}
        #dict_keys = ['path_lr', 'path_hr' ,'path_lr_up']
        type = "SISR"
        imgs = ['img_hr']
    else:
        #dict = {}
        #dict_keys = ['path_lr']
        type = "MISR"
        imgs = ['img_lr']
    type_path = [path, "preprocessed", type, "dataset_"+phase]
    target_path = os.path.join('', *type_path)

    
    for key in ['img_hr_brut']:
        sub_path = os.path.join(target_path, key)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

    dataset_pkl = pd.read_pickle(os.path.join(path, dataset_name))
    dataset= Dataset(path,phase=phase,sen2_amount=sen2_amount)

    
    pbar = tqdm(dataset)
    #alphas_dates = {'alphas': [], 'dates_encoding': []}

    indexes = dataset_pkl.index

    for e,data in enumerate(pbar):

        index = indexes[e]
        
        #if sen2_amount > 1:
        #    alphas_dates['alphas'].append(data["alphas"])
        #    alphas_dates['dates_encoding'].append(data["dates_encoding"])
        for key in imgs:
            p = [target_path, 'img_hr_brut', str(index)+".pt"]
            new_p = os.path.join('', *p)

            torch.save(data[key], new_p)

    
    #temp = pd.DataFrame(alphas_dates)
    #temp.to_pickle(os.path.join(target_path, "alphas_dates_"+phase+".pkl"))
    
    for i,key in enumerate(dict_keys):
        dict[key] = [os.path.join('',*(type_path[1:]+[imgs[i],str(x)+".pt"])) for x in indexes]
    dict["index"] = indexes
    df = pd.DataFrame(dict).set_index("index")

    df = dataset_pkl.join(df)

    if sen2_amount>1:
        df = df.assign(alphas = alphas_dates['alphas'])
        df = df.assign(dates_encoding = alphas_dates['dates_encoding'])
    df.to_pickle(os.path.join(target_path, "dataset_"+phase+".pkl"))
    return """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'preprocessing')

    parser.add_argument("--phase", help='phase: train or test')
    parser.add_argument("--sen2_amount", help='number of images')
    args = parser.parse_args()

    path = "/share/projects/sesure/aimi/data/" #"/home/aimi/Documents/code/tests/data/"

    preprocess_and_save(path, args.phase, int(args.sen2_amount))
