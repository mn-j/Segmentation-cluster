

def print_memory_usage():
    # Print GPU memory usage
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

    # Print CPU memory usage
    process = psutil.Process(os.getpid())
    print(f"CPU Memory Usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

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

def train_model(model, checkpoint_path, dataloader_train, dataloader_test, criterion, optimizer, num_epochs, results_file_path, scheduler):

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
        latest_checkpoint = os.path.join(checkpoint_path, latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint)
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
        start_epoch = checkpoint['epoch']
        loss = checkpoint['epoch_loss']
        accuracy = checkpoint['epoch_accuracy']

    tracemalloc.start()  # Start memory profiling

    epoch_progress = tqdm(range(start_epoch, num_epochs), desc='Training Progress', total=num_epochs - start_epoch)

    # Write header to CSV file
    write_results_to_csv(results_file_path, None, None, None, None, None, header=True)

    for epoch in epoch_progress:
        model.train()  # Set model to training mode
        running_loss, running_accuracy, total = 0.0, 0, 0

        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch+1}/{num_epochs}")

        epoch_start_time = time.time()
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            #iteration_start_time = time.time()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #iteration_end_time = time.time()

            total += labels.size(0)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_accuracy += (predicted == labels).sum().item()

            # Clear variables to free up memory
            del inputs, labels, outputs, loss, predicted
            torch.cuda.empty_cache()

            #print(f"Batch {i+1}: Iteration took {iteration_end_time - iteration_start_time:.2f} seconds")

        epoch_end_time = time.time()

        epoch_loss = running_loss / total
        epoch_accuracy = 100 * running_accuracy / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')

        # Evaluate the model on the test set
        epoch_test_loss, epoch_test_accuracy = evaluate_model(epoch, model, dataloader_test, criterion, device, epoch_loss, epoch_accuracy)

        # Save the results to CSV
        write_results_to_csv(results_file_path, epoch + 1, epoch_loss, epoch_accuracy, epoch_test_loss, epoch_test_accuracy)

        # Print memory usage after each epoch
        print_memory_usage()

        # Save the model at the end of each epoch
        save_model(epoch, model, optimizer, epoch_loss, epoch_accuracy, checkpoint_path)

        # Check memory usage
        print(torch.cuda.memory_summary())
        
        # Step the scheduler
        scheduler.step()

    snapshot = tracemalloc.take_snapshot()  # Take a snapshot of memory usage
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 Memory Consumption ]")
    for stat in top_stats[:10]:
        print(stat)

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

            # Clear variables to free up memory
            del test_inputs, test_labels, test_outputs, loss, test_predicted
            torch.cuda.empty_cache()

    epoch_test_loss = test_loss / total_test
    epoch_test_accuracy = 100 * test_accuracy / total_test
    print(f'Epoch Test Loss: {epoch_test_loss}, Epoch Test Accuracy: {epoch_test_accuracy}%')

    return epoch_test_loss, epoch_test_accuracy

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from models.my_models import R3D_finetune, swin3d_b_finetune, r2dplus1d_finetune
    from config_R3D import CONFIG
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
    
    video_labels_train = CONFIG['video_labels_train']
    video_labels_test = CONFIG['video_labels_test']
    batch_size = CONFIG['batch_size']
    num_classes = CONFIG['num_classes']
    num_epochs = CONFIG['num_epochs']
    window_size = CONFIG['window_size']
    overlap_size = CONFIG['overlap_size']
    input_type = CONFIG['input']
    checkpoint_path = CONFIG['checkpoint_path']
    
    
    #model = swin3d_b_finetune(num_classes)
    model = R3D_finetune(num_classes)
    criterion = torch.nn.CrossEntropyLoss()

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

    # Create the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)  # T_max is the number of epochs
    
    # Set the number of workers
    num_workers = 14

    print(f"Testing with {num_workers} workers...")
    dataloader_train, dataloader_test = prepare_data(input_type, video_labels_train, video_labels_test, window_size, overlap_size, batch_size, model, num_workers=num_workers)

    # Path to the results file
    results_file_path =  os.path.join(checkpoint_path, 'training_results.csv')
    # Train the model and save results after each epoch
    train_model(model, checkpoint_path, dataloader_train, dataloader_test, criterion, optimizer, num_epochs, results_file_path, scheduler)

