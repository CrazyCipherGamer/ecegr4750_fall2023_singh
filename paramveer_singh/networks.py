import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, output_dim: int):
        super(CNNClassifier, self).__init__()
        assert output_dim > 0, "Output dimension must be a positive integer"
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 16,
            kernel_size = (5, 5), 
            stride = (1, 1),
            padding = (0, 0)
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (0,0)
        )
        self.conv2 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 64, 
            kernel_size = (3, 3), 
            stride = (2, 2), 
            padding = (0, 0)
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size = (5,5),
            stride = (2,2),
            padding = (0,0)
        )
        self.linear1 = nn.Linear(
            in_features=64,
            out_features=output_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # reshape for linear layer
        # note that the output of maxpool 2 is (*,64,1,1) so we just need to take the first column and row. 
        # If the output size is not 1,1, we have to flatten x before going into linear using torch.flatten
        x = x[:,:,0,0] 
        x = self.linear1(x)     
        x = torch.sigmoid(x)  
        return x