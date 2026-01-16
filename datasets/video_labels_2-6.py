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

root_dir = [r"//chansey.umcn.nl/diag/Tahereh/Video/Visit 1",
            r"//chansey.umcn.nl/diag/Tahereh/Video/Visit 2",
            r"//chansey.umcn.nl/diag/Tahereh/Video/Visit 3"]

#root_dir = [r"/data/neuro/tahereh/Video/Visit 1",
            #r"/data/neuro/tahereh/Video/Visit 2",
          #  r"/data/neuro/tahereh/Video/Visit 3"]




def load_from_csv(filename):
    data = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key, value = row
            if value!='':
                value = process(value)
            data[key] = value
    return data

def load_from_csv1(filename):
    data = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key, value = row

            data[key] = value
    return data


# Define a custom converter function to handle entries with two numbers
def process(value):
    if pd.isna(value):
      return value  # Return the value as-is if it's NaN
    # Split the value by comma and convert each part to float
    parts = value.split(',')
    numbers = [int(part.strip()) for part in parts]
    return numbers



csv_file_path =  'videoname2items.csv'

videoname2items = load_from_csv(csv_file_path)


# Load the dictionary back from the CSV file
vid2id = load_from_csv1(r'//chansey.umcn.nl/diag/Tahereh/new/src/datasets/dataset_preprocessing/vid2id.csv')
#vid2id = load_from_csv1(r'/home/user/src/datasets/dataset_preprocessing/vid2id.csv')
#################################################################################################################
patient_id_train = pd.read_csv(r'//chansey.umcn.nl/diag/Tahereh/new/src/datasets/dataset_preprocessing/patient_id_train.csv')
#patient_id_train = pd.read_csv(r'/home/user/src/datasets/dataset_preprocessing/patient_id_train.csv')
# Convert the DataFrames to lists
patients_id_train = patient_id_train["80% of patients"].tolist()

###############################################################################################################
patient_id_test = pd.read_csv(r'//chansey.umcn.nl/diag/Tahereh/new/src/datasets/dataset_preprocessing/patient_id_test.csv')
#patient_id_test = pd.read_csv(r'/home/user/src/datasets/dataset_preprocessing/patient_id_test.csv')
# Convert the DataFrames to lists
patients_id_test = patient_id_test["20% of patients"].tolist()

####################################################################################################################
# Open the file in write mode
with open('video_labels_chanseylaptop_train_sorted_2-6_.csv', 'w', newline='') as file:       #### Attention ######
    writer = csv.writer(file)
    writer.writerow(["video_path", "label"])  # Write the header

    for visit in tqdm(root_dir):

        p_videos = os.listdir(visit)
        p_videos = sorted(p_videos)
        for folder in p_videos:
            #print(folder)
            patient_id = vid2id[folder] 
            
            if patient_id in patients_id_train:     ####### Attention ##########
                sub_folder =  os.path.join(visit, str(folder))
                videos_per_folder = os.listdir(sub_folder)
                for vid in videos_per_folder:
                    if vid.endswith(('.mp4', '.avi', '.mkv', '.mov', '.MP4')):
                        if not (vid.endswith('cropped.MP4') or vid.endswith('cropped_square1.mp4')  or vid.endswith('cropped_a.mp4') or vid.endswith('cropped_square.mp4')) :
                            # Add more video extensions if needed
                            items = videoname2items[vid]
                            if items!='':
                                items_to_check = [2, 3, 4, 5, 6]
            
                                # Check if any element is present in the list
                                if  any(elem in items for elem in items_to_check):
                                        # Write the path and label to the CSV file
                                    vid_path =  os.path.join(sub_folder, vid)
                
                                    writer.writerow([vid_path, items])
                                   
                                    
        
          
                 
                 
                 
                 
