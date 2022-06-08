import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        model = models.vgg19(pretrained=True)
        self.features = model.features[:28]
        for param in self.features.parameters():
            param.requires_grad = False
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4608, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
            nn.ReLU()
        )

    def forward_once(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = x.view(-1, 4608)
        x = self.classifier(x)
        return x

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3


class MixedFeaturesSiameseNetwork(nn.Module):

    def __init__(self):
        super(MixedFeaturesSiameseNetwork, self).__init__()

        model = models.vgg19(pretrained=True)
        self.conv_block1 = model.features[:2]
        for param in self.conv_block1.parameters():
            param.requires_grad = False

        self.conv_block2 = model.features[:7]
        for param in self.conv_block2.parameters():
            param.requires_grad = False

        self.conv_block3 = model.features[:16]
        for param in self.conv_block3.parameters():
            param.requires_grad = False

        self.conv_block4 = model.features[:25]
        for param in self.conv_block4.parameters():
            param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=3000, out_features=2000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1000, bias=True),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(128, 1000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(256, 1000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(512, 1000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward_once(self, x):
        # x = self.conv_block1(x)
        fc_features1 = self.conv_block2(x)

        fc_features1 = self.conv1(fc_features1)
        fc_features1 = F.adaptive_avg_pool2d(fc_features1, (1, 1))

        fc_features2 = self.conv_block3(x)

        fc_features2 = self.conv2(fc_features2)
        fc_features2 = F.adaptive_avg_pool2d(fc_features2, (1, 1))

        fc_features3 = self.conv_block4(x)

        fc_features3 = self.conv3(fc_features3)
        fc_features3 = F.adaptive_avg_pool2d(fc_features3, (1, 1))

        features = torch.cat((fc_features1, fc_features2, fc_features3)).view(1, -1)
        output = self.classifier(features)
        return output

    def forward(self, input1, input2, input3):

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3
