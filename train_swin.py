import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.my_models import R3D_finetune, swin3d_b_finetune, r2dplus1d_finetune
from config_swin import CONFIG
from tqdm import tqdm
from datasets.rgb_flow_dataset import prepare_data
import multiprocessing
import os
import re
from collections import OrderedDict
import tracemalloc
import psutil
from utilss.mytimer import measure_time, Timer
import time
from utilss.write_csv import write_results_to_csv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
torch.set_float32_matmul_precision("high")

# Helper function to print memory usage
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

    process = psutil.Process(os.getpid())
    print(f"CPU Memory Usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

# Helper function to find the latest checkpoint
def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print("Directory does not exist")
        return None

    checkpoint_pattern = re.compile(r'checkpoint_epoch_(\d+)\.pth')
    checkpoints = []

    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, filename))

    if not checkpoints:
        print("No checkpoint files found")
        return None

    latest_checkpoint = max(checkpoints, key=lambda x: x[0])
    return latest_checkpoint[1]

# Main training function
def train_model(model, checkpoint_path, dataloader_train, dataloader_test, criterion, optimizer, num_epochs, results_file_path, scheduler):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    latest_checkpoint = find_latest_checkpoint(checkpoint_path)
    start_epoch = 0

    if latest_checkpoint:
        print(f"The latest checkpoint file is: {latest_checkpoint}")
        latest_checkpoint = os.path.join(checkpoint_path, latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']

        new_state_dict = OrderedDict()
        is_multi_gpu = torch.cuda.device_count() > 1
        for k, v in state_dict.items():
            new_key = k
            if is_multi_gpu and not k.startswith('module.'):
                new_key = 'module.' + k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    tracemalloc.start()

    # Write header to CSV file
    write_results_to_csv(results_file_path, None, None, None, None, None, header=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, running_accuracy, total = 0.0, 0, 0

        progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            running_loss += loss.item()
            running_accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()

            progress_bar.set_postfix({'loss': running_loss / total, 'acc': 100 * running_accuracy / total})

        epoch_loss = running_loss / total
        epoch_accuracy = 100 * running_accuracy / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        epoch_test_loss, epoch_test_accuracy = evaluate_model(epoch, model, dataloader_test, criterion, device, epoch_loss, epoch_accuracy)

        write_results_to_csv(results_file_path, epoch + 1, epoch_loss, epoch_accuracy, epoch_test_loss, epoch_test_accuracy)

        save_model(epoch, model, optimizer, epoch_loss, epoch_accuracy, checkpoint_path)

        scheduler.step()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 Memory Consumption ]")
    for stat in top_stats[:10]:
        print(stat)

# Function to save the model checkpoint
def save_model(epoch, model, optimizer, epoch_loss, epoch_accuracy, checkpoint_path):
    checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pth')
    model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_loss': epoch_loss,
        'epoch_accuracy': epoch_accuracy
    }, checkpoint_file)
    print(f'Model saved at {checkpoint_file}')

# Function to evaluate the model
def evaluate_model(epoch, model, dataloader_test, criterion, device, train_loss, train_accuracy):
    model.eval()
    test_loss, test_accuracy, total_test = 0.0, 0, 0

    with torch.no_grad():
        for test_inputs, test_labels in dataloader_test:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)
            loss = criterion(test_outputs, test_labels)
            test_loss += loss.item()
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy += (test_predicted == test_labels).sum().item()
            total_test += test_labels.size(0)

            del test_inputs, test_labels, test_outputs, loss, test_predicted
            torch.cuda.empty_cache()

    epoch_test_loss = test_loss / total_test
    epoch_test_accuracy = 100 * test_accuracy / total_test
    print(f'Epoch Test Loss: {epoch_test_loss}, Epoch Test Accuracy: {epoch_test_accuracy}%')

    return epoch_test_loss, epoch_test_accuracy

if __name__ == "__main__":
    video_labels_train = CONFIG['video_labels_train']
    video_labels_test = CONFIG['video_labels_test']
    batch_size = CONFIG['batch_size']
    num_classes = CONFIG['num_classes']
    num_epochs = CONFIG['num_epochs']
    window_size = CONFIG['window_size']
    overlap_size = CONFIG['overlap_size']
    input_type = CONFIG['input']
    checkpoint_path = CONFIG['checkpoint_path']
    
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

    num_workers = 8

    print(f"Testing with {num_workers} workers...")
    dataloader_train, dataloader_test = prepare_data(input_type, video_labels_train, video_labels_test, window_size, overlap_size, batch_size, model, num_workers=num_workers)

    results_file_path =  os.path.join(checkpoint_path, 'training_results.csv')
    train_model(model, checkpoint_path, dataloader_train, dataloader_test, criterion, optimizer, num_epochs, results_file_path, scheduler)
