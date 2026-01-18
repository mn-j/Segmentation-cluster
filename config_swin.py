
from utilss.load_csv import load_from_csv_as_list_of_tuples


# single_path_train = r'datasets/single_video_labels_train_chansey_sorted_cropped.csv'
# single_path_test = r'datasets/single_video_labels_test_chansey_sorted_cropped.csv'

single_path_train = r'train.csv'
single_path_test = r'test.csv'

# single_path_train = r'//chansey.umcn.nl/neuro/Tahereh/new/src/datasets/dataset_preprocessing/single_video_labels_train_chansey_sorted_cropped.csv'
# single_path_test = r'//chansey.umcn.nl/neuro/Tahereh/new/src/datasets/dataset_preprocessing/single_video_labels_test_chansey_sorted_cropped.csv'



video_labels_train = load_from_csv_as_list_of_tuples(single_path_train)
video_labels_test = load_from_csv_as_list_of_tuples(single_path_test)



CONFIG = {
    'video_labels_train': video_labels_train,
    'video_labels_test': video_labels_test,
    'num_classes': 5, #in reality 4, (0,2,3,4)
    'batch_size': 16,
    'num_epochs': 10,
    'window_size': 2,
    'overlap_size': 1,
    'input': 'rgb',
    'checkpoint_path': '/data/diag/mohamadaljalab/checkpoint/swin'

    
}



