import torch
import torch.nn as nn


# the following code was produced by ChatGPT where minor modifications are made to kernel size and stride
class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()  # Adding ReLU activation

        # Because of the way that the image shape changes over the course of the convolution and maxpooling
        # an initial image of size 64 x 64 is halved in size each time it passes through a convolution or maxpool layer
        # Since there are 4 of these halving layers, the final size of the image is 64 / 2^4 or 4 x 4
        # 32 is multiplied by 4 x 4 to get the appropriate size for the linear layer as there were 32 channels at the end.
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=1)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))  # Applying ReLU after the first convolution and pooling
        x = self.relu(self.pool(self.conv2(x)))  # Applying ReLU after the second convolution and pooling
        x = torch.flatten(x, start_dim=1)  # Flatten the output for the fully connected layer
        x = self.fc1(x)
        return x
