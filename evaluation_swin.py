import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.my_models import swin3d_b_finetune
from config_swin import CONFIG
from tqdm import tqdm
from datasets.rgb_flow_dataset import prepare_data
import os
from collections import OrderedDict
import tracemalloc
import psutil
import csv
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_checkpoints(model, checkpoint_paths, dataloader_test, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    
        checkpoint = torch.load(checkpoint_paths)
  

        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k
            if torch.cuda.device_count() > 1 and not k.startswith('module.'):
                new_key = 'module.' + k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
        epoch = checkpoint.get('epoch', None)
        epoch_loss = checkpoint.get('epoch_loss', None)
        epoch_accuracy = checkpoint.get('epoch_accuracy', None)

        epoch_test_loss, epoch_test_accuracy, all_preds, all_labels = evaluate_model(epoch, model, dataloader_test, criterion, device)

        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_preds, epoch)


def evaluate_model(epoch, model, dataloader_test, criterion, device):
    model.eval()
    test_loss, test_accuracy, total_test = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for test_inputs, test_labels in tqdm(dataloader_test):
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)
            loss = criterion(test_outputs, test_labels)
            test_loss += loss.item()
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy += (test_predicted == test_labels).sum().item()
            total_test += test_labels.size(0)

            all_preds.extend(test_predicted.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

            del test_inputs, test_labels, test_outputs, loss, test_predicted
            torch.cuda.empty_cache()

    epoch_test_loss = test_loss / total_test
    epoch_test_accuracy = 100 * test_accuracy / total_test
    print(f'Epoch {epoch+1} - Test Loss: {epoch_test_loss}, Test Accuracy: {epoch_test_accuracy}%')

    return epoch_test_loss, epoch_test_accuracy, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, epoch):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Task', 'Finger Tapping', 'Hand Movement', 'Pronation-Supination', 'Toe Tapping', 'Leg Agility'], yticklabels=['Non-Task', 'Finger Tapping', 'Hand Movement', 'Pronation-Supination', 'Toe Tapping', 'Leg Agility'])
    plt.title(f'Confusion Matrix for fine-tuned 3D ResNet for Epoch {epoch+1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')  # Rotate x labels to fit better
    plt.tight_layout()  # Adjust layout to fit all labels
    plt.show()
    # Save plot to a file

    # Find the last occurrence of '/'
    last_slash_index = checkpoint_paths.rfind('/')
    
    # Extract the directory path
    directory_path = checkpoint_paths[:last_slash_index]
    plot_path = os.path.join(directory_path, f'confusion_matrix_epoch_{epoch+1}.png')
    plt.savefig(plot_path)
    print(f"Plot saved successfully at {plot_path}.")

def write_results_to_csv(file_path, epoch, train_loss, train_accuracy, test_loss, test_accuracy, header=False):
    mode = 'a' if not header else 'w'
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        if epoch is not None:
            writer.writerow([epoch, train_loss, train_accuracy, test_loss, test_accuracy])

if __name__ == "__main__":
    video_labels_train = CONFIG['video_labels_train']
    video_labels_test = CONFIG['video_labels_test']
    batch_size = CONFIG['batch_size']
    num_classes = CONFIG['num_classes']
    num_epochs = CONFIG['num_epochs']
    window_size = CONFIG['window_size']
    overlap_size = CONFIG['overlap_size']
    input_type = CONFIG['input']


    # List specific checkpoint paths you want to evaluate
    checkpoint_paths =  r'/data/diag/Tahereh/new/src/checkpoint/swin/cropped_2sec_big/checkpoint_epoch_41.pth'
   
    
    model = swin3d_b_finetune(num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    backbone_params = list(model.base_model.parameters())
    head_params = list(model.new_fc.parameters())

    backbone_lr = 3e-5
    head_lr = 3e-4

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ]

    optimizer = AdamW(param_groups, weight_decay=0.02)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    num_workers = 1

    print(f"Testing with {num_workers} workers...")
    dataloader_train, dataloader_test = prepare_data(input_type, video_labels_train, video_labels_test, window_size, overlap_size, batch_size, model, num_workers=num_workers)

    evaluate_checkpoints(model, checkpoint_paths, dataloader_test, criterion)
