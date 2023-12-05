import torch
import torch.nn as nn


# the following code was produced by ChatGPT where minor modifications are made to kernel size and stride
class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=10, stride=10, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()  # Adding ReLU activation

        # Adjusting the fully connected layer input size based on the updated feature map dimensions
        self.fc1 = nn.Linear(32 * 50 * 50, 1)  # Assuming two pooling layers reducing to 50x50

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Applying ReLU after the first convolution and pooling
        x = self.pool(self.relu(self.conv2(x)))  # Applying ReLU after the second convolution and pooling
        x = x.view(-1, 32 * 50 * 50)  # Flatten the output for the fully connected layer
        x = self.fc1(x)
        return x
