

import os
import torch
from models.my_models import Improved3DCNN
from tqdm import tqdm
import cv2
import numpy as np
import csv
import random
import ast
import tracemalloc
import psutil
from collections import OrderedDict
from config_R3D import CONFIG
import torch.nn as nn

class TestMultiple():
    def __init__(self, num_classes, checkpoint_path, window_size):
        self.class_dict = {0: 'Non-task', 1: 'FT', 2: 'HM', 3: 'PS', 4: 'TT', 5: 'LA'}
        
        model = Improved3DCNN(num_classes)

        # Define the learning rate and other optimizer parameters
        learning_rate = 0.01  # Adjust based on your specific needs
        momentum = 0.9
        weight_decay = 1e-4

        # Create the optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
      
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wrap the model for DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        model.to(device)

        print(f"The checkpoint file is: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']

        # Add 'module.' prefix to all keys if using DataParallel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k
            if torch.cuda.device_count() > 1 and not k.startswith('module.'):
                new_key = 'module.' + k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.model = model
        self.device = device
        self.window_size = window_size
        self.model.eval()
    def print_memory_usage(self):
        # Print GPU memory usage
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

        # Print CPU memory usage
        process = psutil.Process(os.getpid())
        print(f"CPU Memory Usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def load_from_csv1(self, filename):
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)
        return data

    def get_label(self, segment):
        # Convert segment to tensor
        segment = np.array(segment)    #   T, H, W, c
        segment = np.moveaxis(segment, 3, 1)    # T, C, H, W
        segment = torch.from_numpy(segment)
        inputs = self.model.preprocess(segment)
        inputs = inputs.to(self.device)
        inputs = inputs.unsqueeze(0)    # 1, T, C, H, W
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()

    def process_and_save_segment(self, frames, output_writer, new_size):
        label = self.get_label(frames)
        
        class_name = self.class_dict.get(label, 'Unknown')
        for frame in frames:
            cv2.putText(frame, f'Class: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
            output_writer.write(resized_frame)

    def test(self):
        csv_file_path = r'/data/diag/tahereh/new/src/datasets/dataset_preprocessing/video_labels_diag_train_sorted_2-6.csv'
        csv_file_path = r'//chansey.umcn.nl//diag/tahereh/new/src/datasets/dataset_preprocessing/video_labels_diag_train_sorted_2-6.csv'

        videos = self.load_from_csv1(csv_file_path)
        random.shuffle(videos)

        tracemalloc.start()  # Start memory profiling

        for vid in videos:
            str_labels = vid[1]
            int_labels = ast.literal_eval(str_labels)
            cap = cv2.VideoCapture(vid[0])

            window_size = int(25 * self.window_size)  # 2 seconds window
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            new_size = (original_width // 4, original_height // 4)
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            output_path = os.path.join(r'/data/diag/tahereh/new/src/results/scratch', f"segmented_video_{os.path.basename(vid[0]).split('.')[0]}_{vid[0].split('/')[-2]}.mp4")
            output_path = os.path.join(r'//chansey.umcn.nl//diag/tahereh/new/src/results/scratch', f"segmented_video_{os.path.basename(vid[0]).split('.')[0]}_{vid[0].split('/')[-2]}.mp4")

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, new_size)

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
                frames.append(frame)

                if len(frames) == window_size:
                    self.process_and_save_segment(frames, out, new_size)
                    frames = []

            # Process any remaining frames
            if frames:
                self.process_and_save_segment(frames, out, new_size)

            cap.release()
            out.release()
            print(f"Saved segmented video to {output_path}")

            self.print_memory_usage()

        snapshot = tracemalloc.take_snapshot()  # Take a snapshot of memory usage
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 Memory Consumption ]")
        for stat in top_stats[:10]:
            print(stat)

if __name__ == "__main__":
    num_classes = 6
    window_size = 2
    checkpoint_path = '/data/diag/tahereh/new/src/checkpoint/R3D/cropped_2sec_scratch/checkpoint_epoch_45.pth'
    checkpoint_path = '//chansey.umcn.nl//diag/tahereh/new/src/checkpoint/R3D/cropped_2sec_scratch/checkpoint_epoch_45.pth'

    test = TestMultiple(num_classes, checkpoint_path, window_size)
    test.test()
