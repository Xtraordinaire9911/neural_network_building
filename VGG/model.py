import torch
import torch.nn as nn

"""first define the config of each vgg network, according to each one s architecture figure; 
    number: #conv kernels
    string: type of pooling"""
cfgs = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']}



class VGG(nn.Module):
    def __init__(self, name_model: str, num_class=1000, if_init_weights=False):
        super(VGG, self).__init__()
        try:
            self.name_model = name_model
        except:
            print("name_nn not in cfgs !")
            exit(-1)
        self.num_class = num_class
        self.features = self.build_features()
        self.classifier = self.build_classifier()
        if if_init_weights:
            self._init_weights()

    def build_features(self):
        layers = []
        in_channels = 3
        for num in cfgs[self.name_model]:
            if num == 'M':
                layers += [nn.MaxPool2d(2, stride=1)]
            else:
                conv2d = nn.Conv2d(in_channels, num, kernel_size=3, padding=1)
                in_channels = num
                layers += [conv2d, nn.ReLU(True)]
        return nn.Sequential(*layers)

    def build_classifier(self):
        classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, self.num_class)
        )
        return classifier

    def forward(self, x):
        ""
        """before: N * 3 * 224 * 224; last two dims: kernel size"""
        x = self.features(x)
        """before: N * 512 * 7 * 7; N: #batches"""
        x = torch.flatten(x, start_dim=1)
        """before: 2048"""
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                """indicates whether the conv kernel uses bias"""
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)



vgg16 = VGG('vgg16', 1000)



