import torch
import torch.nn as nn
import torchvision.models

class AlexNetFeatureExtractor(nn.Module):
    
    def __init__(self, model):
        super(AlexNetFeatureExtractor, self).__init__()
        # assumes model is torchvision alexnet
        feature_children = list(model.features.children())
        classifier_children = list(model.classifier.children())
        
        self.setup_features(feature_children)
        self.avgpool = model.avgpool
        self.setup_fc(classifier_children)
        self.setup_prevs()

    def setup_prevs(self):
        self.prev = {
            'pool3': 'conv5',
            'conv5': 'conv4',
            'conv4': 'conv3',
            'conv4': 'conv3',
            'conv3': 'pool2',
            'pool2': 'conv2',
            'conv2': 'pool1',
            'pool1': 'conv1'
        }

    def forward(self, x):
        # same as original
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def setup_features(self, feature_children):
        self.conv1 = nn.Sequential(
            *feature_children[0:2]
        )
        self.pool1 = feature_children[2]
        self.conv2 = nn.Sequential(
            *feature_children[3:5]
        )
        self.pool2 = feature_children[5]
        self.conv3 = nn.Sequential(
            *feature_children[6:8]
        )
        self.conv4 = nn.Sequential(
            *feature_children[8:10]
        )
        self.conv5 = nn.Sequential(
            *feature_children[10:12]
        )
        self.pool3 = feature_children[12]
        self.features = nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.pool3
        )

    def setup_fc(self, classifier_children):
        self.fc1 = nn.Sequential(
            *classifier_children[1:3]
        )
        self.fc2 = nn.Sequential(
            *classifier_children[4:6]
        )
        self.output = classifier_children[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.fc1,
            nn.Dropout(),
            self.fc2,
            self.output
        )
