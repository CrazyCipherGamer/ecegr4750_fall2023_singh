import torch
import torch.nn as nn


# the following code was produced by ChatGPT where minor modifications are made to kernel size and stride
class CNNRegression(nn.Module):
    def __init__(self, hidden_channels: int=16, out_channels: int=32):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()  # Adding ReLU activation

        # Because of the way that the image shape changes over the course of the convolution and maxpooling
        # an initial image of size 64 x 64 is halved in size each time it passes through a convolution or maxpool layer
        # Since there are 4 of these halving layers, the final size of the image is 64 / 2^4 or 4 x 4
        # 32 is multiplied by 4 x 4 to get the appropriate size for the linear layer as there were 32 channels at the end.
        self.fc1 = nn.Linear(in_features=out_channels * 4 * 4, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) 
        x = torch.reshape(x, (x.size(0), -1))  # Reshape the convolutional output for the fully connected layer
        x = self.fc1(x)
        return x


class MultimodalNetwork(nn.Module):
    def __init__(self, hidden_channels: int=16, out_channels: int=32, hidden_features: int=16):
        super(MultimodalNetwork, self).__init__()
        # define layers for the CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()  # Adding ReLU activation

        # define layers for the CSV data as a linear network
        self.linear1 = nn.Linear(in_features=5, out_features=hidden_features)
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=hidden_features)

        # Because of the way that the image shape changes over the course of the convolution and maxpooling
        # an initial image of size 64 x 64 is halved in size each time it passes through a convolution or maxpool layer
        # Since there are 4 of these halving layers, the final size of the image is 64 / 2^4 or 4 x 4
        # 32 is multiplied by 4 x 4 to get the appropriate size for the linear layer as there were 32 channels at the end.
        self.fc1 = nn.Linear(in_features=out_channels * 4 * 4, out_features=30)
        self.fc2 = nn.Linear(in_features=30 + hidden_features, out_features=1)



    def forward(self, x_image, x):
        x_image = self.conv1(x_image)
        x_image = self.relu(x_image)
        x_image = self.pool(x_image)
        x_image = self.conv2(x_image)
        x_image = self.relu(x_image)
        x_image = self.pool(x_image) 
        x_image = torch.reshape(x_image, (x_image.size(0), -1))  # Reshape the convolutional output for the fully connected layer
        x_image = self.fc1(x_image)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        final = torch.cat((x_image, x), dim=1)
        final = self.fc2(final)
        
        return final