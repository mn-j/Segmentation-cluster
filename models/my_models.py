import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights, r2plus1d_18, R2Plus1D_18_Weights, swin3d_b, Swin3D_B_Weights, swin3d_t, Swin3D_T_Weights,  mvit_v2_s, MViT_V2_S_Weights      
import numpy as np
class TSN(nn.Module):
    def __init__(self, num_segments, num_classes, num_frames_per_segment):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.num_frames_per_segment = num_frames_per_segment
        self.base_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.new_fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        segment_outputs = []
        for i in range(self.num_segments):
            segment_input = x[:, :, i * self.num_frames_per_segment:(i + 1) * self.num_frames_per_segment, :, :]
            segment_output = self.base_model(segment_input)
            segment_outputs.append(segment_output.unsqueeze(1))
        outputs = torch.cat(segment_outputs, dim=1)
        outputs = self.new_fc(outputs)
        final_output = outputs.mean(dim=1)
        return final_output





#########################################################################
class R3D_finetune(nn.Module):
    def __init__(self, num_classes):
        super(R3D_finetune, self).__init__()
        self.weights = R3D_18_Weights.DEFAULT
        self.base_model = r3d_18(weights=self.weights)

        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.new_fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        output = self.base_model(x)            # (B, C, T, H, W)
        output = self.new_fc(output)

        return output
    
    def preprocess(self, data):
        preprocessss = self.weights.transforms()
        return preprocessss(data)

######################################################################
class MC3_finetune(nn.Module):
    def __init__(self, num_classes):
        super(MC3_finetune, self).__init__()
        self.weights = MC3_18_Weights.DEFAULT
        self.base_model = mc3_18(weights=self.weights)
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.new_fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        output = self.base_model(x)            # (B, C, T, H, W)
        output = self.new_fc(output)

        return output


    def preprocess(self, data):
            
        preprocessss = self.weights.transforms()
        return preprocessss(data)


###################################################################
class r2dplus1d_finetune(nn.Module):
    def __init__(self, num_classes):
        super(r2plus1d_finetune, self).__init__()
        self.weights = R2Plus1D_18_Weights.DEFAULT
        self.base_model = r2plus1d_18(weights=self.weights)
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.new_fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        output = self.base_model(x)              # (B, C, T, H, W)
        output = self.new_fc(output)

        return output


    def preprocess(self, data):
            
        preprocessss = self.weights.transforms()
        return preprocessss(data)

###################################################################

###################################################################
class swin3d_b_finetune(nn.Module):
    def __init__(self, num_classes):
        super(swin3d_b_finetune, self).__init__()
        self.weights = Swin3D_B_Weights.DEFAULT
        self.base_model = swin3d_b(weights=self.weights)
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()
        self.new_fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        output = self.base_model(x)              # (B, C, T, H, W)
        output = self.new_fc(output)

        return output


    def preprocess(self, data):
            
        preprocessss = self.weights.transforms()
        return preprocessss(data)

###################################################################

class swin3d_t_finetune(nn.Module):
    def __init__(self, num_classes):
        super(swin3d_t_finetune, self).__init__()
        self.weights = Swin3D_T_Weights.DEFAULT
        self.base_model = swin3d_t(weights=self.weights)
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()
        self.new_fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        output = self.base_model(x)              # (B, C, T, H, W)
        output = self.new_fc(output)

        return output


    def preprocess(self, data):
            
        preprocessss = self.weights.transforms()
        return preprocessss(data)
####################################################################
class mvit_finetune(nn.Module):
    def __init__(self, num_classes):
        super(mvit_finetune, self).__init__()
        self.weights = MViT_V2_S_Weights.DEFAULT
        self.base_model = mvit_v2_s(weights=self.weights)
        #print(self.base_model)
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.head[1].in_features
        self.base_model.head = nn.Identity()
        self.new_fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
    
        output = self.base_model(x)              # (B, C, T, H, W)
        output = self.new_fc(output)
    
        return output
    
    
    def preprocess(self, data):
            
        preprocessss = self.weights.transforms()
        return preprocessss(data)
    

##################################################################
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc1 = nn.Linear(64 * 12 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)  
    def preprocess(self, video_tensor):
        #resized_frames = [cv2.resize(frame.numpy(), (224, 224), interpolation=cv2.INTER_LINEAR) for frame in video_tensor]
        # Convert list of frames to a single numpy array
        resized_array = np.array(video_tensor)
        # Convert the numpy array to a tensor and normalize to [0, 1]
        resized_tensor = torch.tensor(resized_array).float() / 255.0
        # Rearrange to [C, T, H, W]
        resized_tensor = resized_tensor.permute(1, 0, 2, 3)
        return resized_tensor
    def forward(self, x):
        #print(x.shape)
        x = self.pool1(torch.relu(self.conv1(x)))
        #print(x.shape)

        x = self.pool2(torch.relu(self.conv2(x)))
        #print(x.shape)

        x = self.pool3(torch.relu(self.conv3(x)))
        #print(x.shape)

        x = x.view(x.size(0), -1)
        #print(x.shape)

        x = torch.relu(self.fc1(x))
        #print(x.shape)

        x = torch.relu(self.fc2(x))
        #print(x.shape)

        x = self.fc3(x)
        #print(x.shape)

        return x

    

##################################################
class Improved3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Improved3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
       
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 12 * 28 * 28, 1024)  # img_size=224
        #self.fc1 = nn.Linear(128 * 12 * 37 * 37, 1024)  # img_size=224

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def preprocess(self, video_tensor):
        resized_array = np.array(video_tensor)
        resized_tensor = torch.tensor(resized_array).float() / 255.0
        resized_tensor = resized_tensor.permute(1, 0, 2, 3)
        return resized_tensor
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        #print(x.shape)

        x = x.view(x.size(0), -1)
        #print(x.shape)

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x







