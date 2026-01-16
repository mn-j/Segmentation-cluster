# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:53:41 2024

@author: Z984222
"""

import os
import csv
import pandas as pd
from tqdm import tqdm
import random 

#items_non_class = [ str('Off_7_cropped_square'),  str('On_7_cropped_square'), str('Off_8_cropped_square'),  str('On_8_cropped_square'), str('Off_9_cropped_square'),  str('On_9_cropped_square'), str('Off_10_cropped_square'),  str('On_10_cropped_square'),str('Off_11_cropped_square'),  str('On_11_cropped_square'),  str('Off_12_cropped_square'),  str('On_12_cropped_square'), str('Off_13_cropped_square'),  str('On_13_cropped_square'), str('Off_14_cropped_square'),  str('On_14_cropped_square'), str('Off_15_cropped_square'),  str('On_15_cropped_square') , str('Off_16_cropped_square'),  str('On_16_cropped_square'), str('Off_17_cropped_square'),  str('On_17_cropped_square') , str('Off_18_cropped_square'),  str('On_18_cropped_square') , str('Off_19_cropped_square'),  str('On_19_cropped_square') , str('Off_20_cropped_square'),  str('On_20_cropped_square'),  str('Off_21_cropped_square'),  str('On_21_cropped_square') , str('Off_22_cropped_square'),  str('On_22_cropped_square')   ]

#items = [(str('Off_2R_cropped_square'),1), (str('On_2R_cropped_square'),1), (str('Off_2L_cropped_square'),1), (str('On_2L_cropped_square'),1), (str('Off_3R_cropped_square'),2), (str('On_3R_cropped_square'),2),(str('Off_3L_cropped_square'),2), (str('On_3L_cropped_square'),2), (str('Off_4R_cropped_square'),3), (str('On_4R_cropped_square'),3),(str('Off_4L_cropped_square'),3), (str('On_4L_cropped_square'),3),  (str('Off_5R_cropped_square'),4), (str('On_5R_cropped_square'),4), (str('Off_5L_cropped_square'),4), (str('On_5L_cropped_square'),4), (str('Off_6R_cropped_square'),5), (str('On_6R_cropped_square'),5), (str('Off_6L_cropped_square'),5), (str('On_6L_cropped_square'),5)]


#items_non_class = [ str('Off_7'),  str('On_7'), str('Off_8'),  str('On_8'), str('Off_9'),  str('On_9'), str('Off_10'),  str('On_10'),str('Off_11'),  str('On_11'),  str('Off_12'),  str('On_12'), str('Off_13'),  str('On_13'), str('Off_14'),  str('On_14'), str('Off_15'),  str('On_15') , str('Off_16'),  str('On_16'), str('Off_17'),  str('On_17') , str('Off_18'),  str('On_18') , str('Off_19'),  str('On_19') , str('Off_20'),  str('On_20'),  str('Off_21'),  str('On_21') , str('Off_22'),  str('On_22')   ]

items = [(str('Off_2R'),1), (str('On_2R'),1), (str('Off_2L'),1), (str('On_2L'),1), (str('Off_3R'),2), (str('On_3R'),2),(str('Off_3L'),2), (str('On_3L'),2), (str('Off_4R'),3), (str('On_4R'),3),(str('Off_4L'),3), (str('On_4L'),3),  (str('Off_5R'),4), (str('On_5R'),4), (str('Off_5L'),4), (str('On_5L'),4), (str('Off_6R'),5), (str('On_6R'),5), (str('Off_6L'),5), (str('On_6L'),5)]



root_dir = [r"//chansey.umcn.nl/diag/Tahereh/Video/Visit 1",
            r"//chansey.umcn.nl/diag/Tahereh/Video/Visit 2",
            r"//chansey.umcn.nl/diag/Tahereh/Video/Visit 3"]

#root_dir = [r"/data/neuro/tahereh/Video/Visit 1",
           # r"/data/neuro/tahereh/Video/Visit 2",
           # r"/data/neuro/tahereh/Video/Visit 3"]
         
# Function to load dictionary from a CSV file
def load_from_csv(filename):
    data = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key, value = row
            data[key] = value
    return data


# Load the dictionary back from the CSV file
vid2id = load_from_csv(r'//chansey.umcn.nl/diag/tahereh/new/src/datasets/dataset_preprocessing/vid2id.csv')
#vid2id = load_from_csv(r'/home/user/src/datasets/dataset_preprocessing/vid2id.csv')
#################################################################################################################
patient_id_train = pd.read_csv(r'//chansey.umcn.nl/diag/tahereh/new/src/datasets/dataset_preprocessing/patient_id_train.csv')
#patient_id_train = pd.read_csv(r'/home/user/src/datasets/dataset_preprocessing/patient_id_train.csv')
# Convert the DataFrames to lists
patients_id_train = patient_id_train["80% of patients"].tolist()

###############################################################################################################
patient_id_test = pd.read_csv(r'//chansey.umcn.nl/diag/tahereh/new/src/datasets/dataset_preprocessing/patient_id_test.csv')
#patient_id_test = pd.read_csv(r'/home/user/src/datasets/dataset_preprocessing/patient_id_test.csv')
# Convert the DataFrames to lists
patients_id_test = patient_id_test["20% of patients"].tolist()

####################################################################################################################
# Open the file in write mode
with open('single_video_labels_train_chanseylaptop_sorted_noncropped.csv', 'w', newline='') as file:   #### Attention #####
    writer = csv.writer(file)
    writer.writerow(["video_path", "label"])  # Write the header
    for selector in tqdm(items):
    
        for visit in root_dir:
    
            p_videos = os.listdir(visit)
            p_videos = sorted(p_videos)
            for folder in p_videos:
                patient_id = vid2id[folder] 
                
                if patient_id in patients_id_train:                                 #######   Attention  #########
                    sub_folder =  os.path.join(visit, str(folder))
                    file_exist1 = os.path.join(sub_folder, selector[0]+ ".MP4")
                    file_exist2 = os.path.join(sub_folder, selector[0]+ ".mp4")
                    if os.path.exists(file_exist1):
                                          
                            # Write the path and label to the CSV file
                            writer.writerow([file_exist1, selector[1]])
                    elif os.path.exists(file_exist2):
                        writer.writerow([file_exist2, selector[1]])
                       
    '''                    
    container_0 = []                      
    for selector in tqdm(items_non_class):
    
        for visit in root_dir:
    
            p_videos = os.listdir(visit)
            #print(p_videos)
            p_videos = sorted(p_videos)
            #print(p_videos)
            for folder in p_videos:
                patient_id = vid2id[folder] 
            
                if patient_id in patients_id_train:                          ############### attention ###############
                    sub_folder =  os.path.join(visit, str(folder))
                    file_exist = os.path.join(sub_folder, selector + ".MP4")
                    if os.path.exists(file_exist):
                            container_0.append(file_exist)   
                            
                            
                            
    container_0_len = len(container_0)
    #random_indices = random.sample(range(container_0_len), 97)             ############### attention!!!!!!!!!!!
    random_indices = random.sample(range(container_0_len), 497)             ############## attention!!!!!!!!!!
    for i in random_indices:
        
        # Write the path and label to the CSV file
        writer.writerow([container_0[i], 0])      
  '''
             
             
             
             