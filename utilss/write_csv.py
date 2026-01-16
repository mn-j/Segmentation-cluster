import csv


def write_results_to_csv(file_path, epoch, train_loss, train_accuracy, test_loss, test_accuracy, header=False):
    mode = 'a' if not header else 'w'
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        if epoch is not None:
            writer.writerow([epoch, train_loss, train_accuracy, test_loss, test_accuracy])
