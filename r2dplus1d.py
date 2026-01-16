import torch
import torch.nn as nn
from models.my_models import R3D_finetune, swin3d_b_finetune, r2dplus1d_finetune
from config import CONFIG
from tqdm import tqdm
from datasets.rgb_flow_dataset import prepare_data
import multiprocessing
import os
import re
from collections import OrderedDict


checkpoint_path = '/data/diag/tahereh/new/src/checkpoint/r2dplus1d'

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print("Directory does not exist")
        return None

    checkpoint_pattern = re.compile(r'checkpoint_r_epoch_(\d+)\.pth')
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

def train_model(model, dataloader_train, dataloader_test, criterion, optimizer, checkpoint_path, num_epochs=25):
    

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wrap the model for DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # Find the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_path)
    start_epoch = 0

    if latest_checkpoint:
        print(f"The latest checkpoint file is: {latest_checkpoint}")      
        latest_checkpoint = os.path.join(checkpoint_path ,latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint)
        # Extract the model state dictionary from the checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Add 'module.' prefix to all keys
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = 'module.' + k  # add 'module.' prefix
            new_state_dict[new_key] = v
                
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['epoch_loss']
        accuracy = checkpoint['epoch_accuracy']
   
    epoch_progress = tqdm(range(start_epoch, num_epochs), desc='Training Progress', total=num_epochs - start_epoch)

    for epoch in epoch_progress:
        model.train()  # Set model to training mode
        running_loss, running_accuracy, total = 0.0, 0, 0

        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_accuracy += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = 100 * running_accuracy / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')

        
        # Save the model at the end of each epoch
        save_model(epoch, model, optimizer, epoch_loss, epoch_accuracy, checkpoint_path)
        #evaluate_model(epoch, model, dataloader_test, criterion, device, writer, epoch_loss, epoch_accuracy)


    #writer.close()


def save_model(epoch, model, optimizer, epoch_loss, epoch_accuracy, checkpoint_path):
    checkpoint_path = os.path.join('/data/diag/tahereh/new/src/checkpoint/r2dplus1d', f'checkpoint_r_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_loss': epoch_loss,
        'epoch_accuracy': epoch_accuracy
    }, checkpoint_path)
    print(f'Model saved at {checkpoint_path}')

#def evaluate_model(epoch, model, dataloader_test, criterion, device, writer, train_loss, train_accuracy):
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

    epoch_test_loss = test_loss / total_test
    epoch_test_accuracy = 100 * test_accuracy / total_test
    print(f'Epoch Test Loss: {epoch_test_loss}, Epoch Test Accuracy: {epoch_test_accuracy}%')

    #writer.add_scalar('Test Loss (epoch)', epoch_test_loss, epoch)
    #writer.add_scalar('Test Accuracy (epoch)', epoch_test_accuracy, epoch)
    #writer.add_scalar('Train Loss (epoch)', train_loss, epoch)
    #writer.add_scalar('Train Accuracy (epoch)', train_accuracy, epoch)
    #writer.add_scalars('Losses (epoch)', {'train_epoch': train_loss, 'test_epoch': epoch_test_loss}, epoch)
    #writer.add_scalars('Accuracies (epoch)', {'train_epoch': train_accuracy, 'test_epoch': epoch_test_accuracy}, epoch)

if __name__ == "__main__":
    video_labels_train = CONFIG['video_labels_train']
    video_labels_test = CONFIG['video_labels_test']
    batch_size = CONFIG['batch_size']
    num_classes = CONFIG['num_classes']
    num_epochs = CONFIG['num_epochs']
    learning_rate = CONFIG['learning_rate']
    window_size = CONFIG['window_size']
    overlap_size = CONFIG['overlap_size']
    input_type = CONFIG['input']

    model = R3D_finetune(num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use multiple workers for data loading
    num_workers = multiprocessing.cpu_count()
    num_workers = 4
    #dataloader_train, dataloader_test, writer = prepare_data(input_type, video_labels_train, video_labels_test, window_size, overlap_size, batch_size, model, num_workers=num_workers)
    dataloader_train, dataloader_test = prepare_data(input_type, video_labels_train, video_labels_test, window_size, overlap_size, batch_size, model, num_workers=num_workers)
    train_model(model, dataloader_train, dataloader_test, criterion, optimizer, checkpoint_path, num_epochs)
