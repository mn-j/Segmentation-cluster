from datasets.video_dataset import VideoDataset_sliding, OpticalFlowVideoDataset
from torch.utils.data import DataLoader
import time
from utilss.mytimer import measure_time, Timer



#@measure_time("Data Preparation")
def prepare_data(input_type, video_labels_train, video_labels_test, window_size, overlap_size, batch_size, model, num_workers):
    #with Timer("Data Loading"):
    if input_type == 'rgb':
        dataset_test = VideoDataset_sliding(video_labels_test, window_size, overlap_size, transform=model.preprocess, status='test')
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        dataset_train = VideoDataset_sliding(video_labels_train, window_size, overlap_size, transform=model.preprocess, status='train')
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)

        #writer = SummaryWriter(log_dir='runs/rgb')

    elif input_type == 'flow':
        dataset_test = OpticalFlowVideoDataset(video_labels_test, window_size, overlap_size, transform=None, status='test')
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        dataset_train = OpticalFlowVideoDataset(video_labels_train, window_size, overlap_size, transform=None, status='train')
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)

        #writer = SummaryWriter(log_dir='runs/flow')

    #return dataloader_train, dataloader_test, writer
    return dataloader_train, dataloader_test
