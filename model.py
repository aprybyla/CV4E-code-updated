'''
    Model implementation.
    We'll be using a "simple" ResNet-18 for image classification here.
    2022 Benjamin Kellenberger
'''

import torch.nn as nn
from torchvision.models import resnet #here is where we import any model architecture that we want; we could change this depedning on the architecture we want (like ResNet50, ResNet101). List on Pytorch website. 

# This is a resnet18 model
class BeeNet(nn.Module):

    def __init__(self, num_classes):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(BeeNet, self).__init__() #CustomResNet18 has to match the name of my class

        # This resnet must match the model that you have chosen
        self.feature_extractor = resnet.resnet18(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet
        self.avgpool = nn.AdaptiveAvgPool2d(1)                          # allow for arbitrary input sizes

############## THIS IS ALL MATH STUFF NEXT ################

        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()                       # discard last layer...

        self.classifier = nn.Linear(in_features, num_classes)           # ...and create a new one
    
############## THIS IS LAYER STUFF ##########

# This is passing our data through the feature extractor and then putting out a prediction before it gets sent back in Train.py
    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-18 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction