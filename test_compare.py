

import os
import torch
from models.my_models import R3D_finetune, swin3d_b_finetune, swin3d_t_finetune, Simple3DCNN
from tqdm import tqdm
import cv2
import numpy as np
import tracemalloc
import psutil
from collections import OrderedDict
import torch.nn as nn
from torch.optim import AdamW
import csv
import random
import ast
def load_from_csv1(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
        return data


class TestModel():
    def __init__(self, model_type, num_classes, checkpoint_path, window_size):
        self.model_type = model_type
        self.class_dict = {0: 'Non-task', 1: 'FT', 2: 'HM', 3: 'PS', 4: 'TT', 5: 'LA'}
        
        if model_type == 'R3D_2sec' or model_type == 'R3D_10sec':
            model = R3D_finetune(num_classes)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        if model_type == 'Simple3DCNN':
            model = Simple3DCNN(num_classes)
            learning_rate = 0.001
            weight_decay = 0.0001  # This is a common starting point, but it might need tuning
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if model_type == 'swin_big':
            model = swin3d_b_finetune(num_classes)
            backbone_params = list(model.base_model.parameters())
            head_params = list(model.new_fc.parameters())
            param_groups = [{'params': backbone_params, 'lr': 3e-5}, {'params': head_params, 'lr': 3e-4}]
            optimizer = AdamW(param_groups, weight_decay=0.02)


        if model_type == 'swin_big_skip':
            model = swin3d_b_finetune(num_classes)
            backbone_params = list(model.base_model.parameters())
            head_params = list(model.new_fc.parameters())
            param_groups = [{'params': backbone_params, 'lr': 3e-5}, {'params': head_params, 'lr': 3e-4}]
            optimizer = AdamW(param_groups, weight_decay=0.02)
        if model_type == 'swin_tiny':
            model = swin3d_t_finetune(num_classes)
            backbone_params = list(model.base_model.parameters())
            head_params = list(model.new_fc.parameters())
            param_groups = [{'params': backbone_params, 'lr': 3e-5}, {'params': head_params, 'lr': 3e-4}]
            optimizer = AdamW(param_groups, weight_decay=0.02)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for {model_type}")
            model = nn.DataParallel(model)
        model.to(device)

        print(f"The checkpoint file for {model_type} is: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']

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

    def get_label(self, segment):
        segment = np.array(segment)
        segment = np.moveaxis(segment, 3, 1)
        segment = torch.from_numpy(segment)
        inputs = self.model.preprocess(segment)
        inputs = inputs.to(self.device)
        inputs = inputs.unsqueeze(0)
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

    def test(self, video_path):
        
        cap = cv2.VideoCapture(video_path)       
        window_size = int(25 * self.window_size)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if self.model_type=='Simple3DCNN':
            new_size = (224, 224)
        else:
            new_size = (original_width // 4, original_height // 4)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        root_path = r'/data/diag/Tahereh/new/src/results'
        if not os.path.exists(root_path):
            print(f"Root path does not exist: {root_path}")
            return

        output_path = os.path.join(root_path, f"{os.path.basename(video_path).split('.')[0]}__{video_path.split('/')[-2]}_{self.model_type}.mp4")
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

        cap.release()
        out.release()
        print(f"Saved segmented video to {output_path}")

if __name__ == "__main__":
    csv_file_path = r'/data/diag/tahereh/new/src/datasets/dataset_preprocessing/video_labels_diag_train_sorted_cropped_2-6.csv'
    videos = load_from_csv1(csv_file_path)
    random.shuffle(videos)
    for vid in videos:
        str_labels = vid[1]
        int_labels = ast.literal_eval(str_labels)
        video_path = vid[0]
        #video_path = r'/data/diag/Tahereh/video/visit 1/POM1VD0701550/Off_1-7.MP4'  # Replace with the actual video path
        num_classes = 6
        checkpoint_path_R3D_2 = r'/data/diag/Tahereh/new/src/checkpoint/R3D/cropped_2sec/checkpoint_epoch_55.pth'
        checkpoint_path_R3D_10 = r'/data/diag/Tahereh/new/src/checkpoint/R3D/cropped_10sec/checkpoint_epoch_66.pth'
        checkpoint_path_simplecnn = r'/data/diag/Tahereh/new/src/checkpoint/R3D/cropped_2sec_simplecnn/checkpoint_epoch_27.pth'
        checkpoint_path_swin_tiny = r'/data/diag/Tahereh/new/src/checkpoint/swin/cropped_2sec_tiny/checkpoint_epoch_27.pth'
        checkpoint_path_swin_big = r'/data/diag/Tahereh/new/src/checkpoint/swin/cropped_2sec_big/checkpoint_epoch_41.pth'
        checkpoint_path_swin_big_skip = r'/data/diag/Tahereh/new/src/checkpoint/swin/cropped_2sec_big_skip1sec/checkpoint_epoch_5.pth'

        # Test R3D model (2 seconds)
        r3d_test_2 = TestModel('R3D_2sec', num_classes, checkpoint_path_R3D_2, window_size=2)
        r3d_test_2.test(video_path)
        
        # Test R3D model (10 seconds)
        r3d_test_10 = TestModel('R3D_10sec', num_classes, checkpoint_path_R3D_10, window_size=10)
        r3d_test_10.test(video_path)
        
        # Test Simple3DCNN
        simple = TestModel('Simple3DCNN', num_classes, checkpoint_path_simplecnn, window_size=2)
        simple.test(video_path)

        # Test Swin model (tiny)
        swin_test_tiny = TestModel('swin_tiny', num_classes, checkpoint_path_swin_tiny, window_size=2)
        swin_test_tiny.test(video_path)

        # Test Swin model (big)
        swin_test_big = TestModel('swin_big', num_classes, checkpoint_path_swin_big, window_size=2)
        swin_test_big.test(video_path)


        swin_test_big = TestModel('swin_big_skip', num_classes, checkpoint_path_swin_big_skip, window_size=2)
        swin_test_big.test(video_path)
