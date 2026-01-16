import pandas as pd
import ast

# Paths to your CSV files (ensure these paths are correct)
id2vid_path = r'//chansey.umcn.nl/diag/Tahereh/new/src/datasets/dataset_preprocessing/id2vid.csv'
ids_path = r'//chansey.umcn.nl/diag/Tahereh/new/src/datasets/dataset_preprocessing/patient_id_all+holdout.csv'

# Read the CSV files without headers, since both files have no headers
ids_df = pd.read_csv(ids_path, header=None)  # CSV containing patient IDs
id2vid_df = pd.read_csv(id2vid_path, header=None)  # CSV containing patient-video mappings

# Initialize an empty list to store the rows for the new CSV
output_rows = []

# Medication status and examination side options
medication_statuses = ['on', 'off']
examination_sides = ['left body', 'right body']

# Iterate through each patient ID using the index
for index, row in ids_df.iterrows():
    patient_id = row[0]  # Access the patient ID directly using index 0
    
    # Find the corresponding row for the current patient in id2vid_df
    matching_row = id2vid_df[id2vid_df[0] == patient_id]
    
    if not matching_row.empty:
        # Extract the video list as a string from the second column and convert it using ast.literal_eval
        video_list = ast.literal_eval(matching_row.iloc[0, 1])
        
        # Append rows for each video with the patient ID, medication status, examination side
        # Repeat the additional columns three times, leaving them blank
        for video in video_list:
            for status in medication_statuses:
                for side in examination_sides:
                    output_rows.append([
                        patient_id, video, status, side, '', '', '', '', '', '', '', '', '', ''])

# Create a DataFrame from the collected rows
output_df = pd.DataFrame(output_rows, columns=[
    'Patient_ID', 'Video', 'Medication_Status', 'Examination_Side', 
    'FT_Presence_of_Examiner?', 'FT_Full_Hand_Captured?',
    'HM_Presence_of_Examiner?', 'HM_Full_Hand_Captured?',
    'PS_Presence_of_Examiner?', 'PS_Full_Hand_Captured?',
    'TT_Presence_of_Examiner?', 'TT_Full_toe_Captured?',
    'LA_Presence_of_Examiner?', 'LA_Full_leg_Captured?',
    ])

# Save the DataFrame to a new CSV file
output_df.to_csv('annotation.csv', index=False)

print("The data has been saved to 'patient_videos_with_additional_info_repeated.csv'.")
