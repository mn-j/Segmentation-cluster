
import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import pickle
from tqdm import tqdm
import imageio
import time
# Model

class VideoDataset_sliding(Dataset):
    def __init__(self,  video_labels, window_duration, overlap_duration, transform, status):
        self.root_dir = r'/data/diag/tahereh/new/src/datasets/dataset_preprocessing'
        #self.root_dir = r'//chansey.umcn.nl/neuro/Tahereh/new/src/datasets/dataset_preprocessing'
        self.transform = transform
        self.window_duration = window_duration  # Window duration in seconds
        self.overlap_duration = overlap_duration  # Overlap duration in seconds
        
        if window_duration ==2:
            self.cache_file_train_chansey = os.path.join(self.root_dir,'train_dataset_chansey_2_balanced.pkl')  # Path to the cache file hospital
            self.cache_file_test_chansey = os.path.join(self.root_dir,'test_dataset_chansey_2_balanced.pkl')  # Path to the cache file hard
        elif window_duration==10: 
            self.cache_file_train_chansey = os.path.join(self.root_dir,'train_dataset_chansey_10.pkl')  # Path to the cache file hospital
            self.cache_file_test_chansey = os.path.join(self.root_dir,'test_dataset_chansey_10.pkl')  # Path to the cache file hard
        
        
        self.video_labels = video_labels
        self.samples = []  # Store tuples of (video_path, start_frame, label)

        if status=='train':         

            # Attempt to load precomputed samples from cache
            if os.path.exists(self.cache_file_train_chansey):
                # Load the cached samples if the file exists
                with open(self.cache_file_train_chansey, 'rb') as f:
                    self.samples = pickle.load(f)
            else:
                # Otherwise, compute the samples and save them
                self.compute_samples()
                with open(self.cache_file_train_chansey, 'wb') as f:
                    pickle.dump(self.samples, f)  # Save the computed samples to the cache file


        if status=='test':         

            # Attempt to load precomputed samples from cache
            if os.path.exists(self.cache_file_test_chansey):
                # Load the cached samples if the file exists
                with open(self.cache_file_test_chansey, 'rb') as f:
                    self.samples = pickle.load(f)
            else:
                # Otherwise, compute the samples and save them
                self.compute_samples()
                with open(self.cache_file_test_chansey, 'wb') as f:
                    pickle.dump(self.samples, f)  # Save the computed samples to the cache file
                        

                        
                        

    def compute_samples(self):
        for (path, label) in tqdm(self.video_labels):
            frame_count = self.__frame_counter__(path)
            fps = 25

            # Calculate the number of frames to skip at the start and end
            skip_frames = int(1 * fps)

            # Calculate the number of frames per window and overlap
            frames_per_window = int(fps * self.window_duration)
            overlap_frames = int(fps * self.overlap_duration)

            # Compute all valid windows for this video, skipping the first and last 1.5 seconds
            if frame_count >= (frames_per_window + 2 * skip_frames):
                step_size = frames_per_window - overlap_frames  # This determines how much we move the window each iteration
                for start_frame in range(skip_frames, frame_count - frames_per_window - skip_frames + 1, step_size):
                    self.samples.append((path, start_frame, label))  # Append the video path, the start frame, and the label

                                                    
    def __frame_counter__(self, path):
                                         
       cap = cv2.VideoCapture(path)
       frames = []
       try:
           while True:
               ret, frame = cap.read()
               if not ret:
                   break
               frame = cv2.resize(frame, (25, 25), interpolation=cv2.INTER_LINEAR)

               frames.append(frame)
       finally:
           cap.release()
           
       return len(frames)
    
    
    def __len__(self):
        return len(self.samples)

    def __load_video_cv__(self, path, start_frame):   ## opencv cap.set is not reliable!!!! only frame by frame
    
        cap = cv2.VideoCapture(path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_width = original_width // 4
        new_height = original_height // 4
        new_height = 224
        new_width = 224
        frames = []
        frame_count = 0
        # Skip frames until the start_frame
        while frame_count < start_frame:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1

        # Read the next frames for the duration of the window
        for _ in range(int(25 * self.window_duration)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

        cap.release()
        return frames
    
    def __getitem__(self, idx):
        
        #start_time = time.time()

        video_path, start_frame, label = self.samples[idx]
        frames = self.__load_video_cv__(video_path, start_frame)
        
           # Check if the frames are empty
        if not frames:
            print(f"Warning: No frames extracted from {video_path} at {start_frame}. Attempting next sample.")
            # Attempt to load another sample if the current one is empty
            if idx + 1 < len(self.samples):
                return self.__getitem__(idx + 1)
            else:
                raise RuntimeError(f"No valid frames found for the last sample in the dataset from {video_path}.")        


        frames = np.array(frames)    #   T, H, W, c
        frames = np.moveaxis(frames, 3, 1)    # T, C, H , W 
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = self.transform(frames_tensor)        # C, T, H, W
        
        #end_time = time.time()
        #print(f"Data loading (including storage read) took {end_time - start_time:.2f} seconds")

        return frames_tensor, torch.tensor(int(label))

###########################################################################################################

class OpticalFlowVideoDataset(VideoDataset_sliding):
    def __init__(self, video_labels, window_duration, overlap_duration, transform, status):
        super(OpticalFlowVideoDataset, self).__init__(video_labels, window_duration, overlap_duration, transform, status)

   
    def compute_optical_flow(self, frames):
        flows = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx, fy = flow[..., 0], flow[..., 1]
            magnitude, _= cv2.cartToPolar(fx, fy)
            # Combine fx, fy, and magnitude into a single NumPy array
            output_array = np.stack((fx, fy, magnitude), axis=-1)
            flows.append(output_array)
            prev_gray = curr_gray
        return np.array(flows)

    def __getitem__(self, idx):
        self.video_path, start_frame, label = self.samples[idx]
        frames = self.__load_video_cv__(self.video_path, start_frame)
        flows = self.compute_optical_flow(frames)

        # Optional: resize the optical flow to match the expected input size

        flows = np.array(flows)

        # Convert to torch tensor and permute to match (C, T, H, W) format
        flows_tensor = torch.from_numpy(flows).permute(3, 0, 1, 2)

        return flows_tensor, torch.tensor(int(label))

   
   
