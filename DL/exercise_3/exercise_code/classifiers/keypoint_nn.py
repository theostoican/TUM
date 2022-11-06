import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        #input shape: (1, 96, 96)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        #input shape: (32, 94, 94)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        #input shape: (32, 47, 47)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)
        #input shape: (64, 44, 44)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        #input shape: (64, 22, 22)
        self.fc1 = nn.Linear(22 * 22 * 64, 1000)
        self.dropout1 = nn.Dropout(p=0.5)
        #input shape: (1000)
        self.fc2 = nn.Linear(1000, 30)
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        # Aggregate all the dimensions in a single one, except for
        # the batch size
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.fc1(x))
        x = self.fc2(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
