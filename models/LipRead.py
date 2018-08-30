import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.cuda as cuda

import re

from .ConvFrontend import ConvFrontend
from .ResNetBBC import ResNetBBC
from .LSTMBackend import LSTMBackend
from .ConvBackend import ConvBackend

class LipRead(nn.Module):
    def __init__(self, options):
        super(LipRead, self).__init__()
        self.frontend = ConvFrontend()
        self.resnet = ResNetBBC(options)
        self.backend = ConvBackend(options)
        self.lstm = LSTMBackend(options)

        self.type = options["model"]["type"]
        if self.type == "temp-conv":
            self.model = nn.Sequential(self.frontend, self.resnet, self.backend)
        elif self.type == "LSTM":
            self.model = nn.Sequential(self.frontend, self.resnet, self.lstm)

        if cuda.is_available():
            self.model = self.model.cuda()

        if cuda.device_count() > 1:
            print("{} gpus detected. running on multiple gpus".format(cuda.device_count()))
            self.model = nn.DataParallel(self.model)

        #function to initialize the weights and biases of each module. Matches the
        #classname with a regular expression to determine the type of the module, then
        #initializes the weights for it.
        def weights_init(m):
            
            classname = m.__class__.__name__
            if re.search("Conv[123]d", classname):
                if m.weight.requires_grad:
                    m.weight.data.normal_(0.0, 0.02)
            elif re.search("BatchNorm[123]d", classname):
                if m.weight.requires_grad:
                    m.weight.data.fill_(1.0)
                if m.bias.requires_grad:
                    m.bias.data.fill_(0)
            elif re.search("Linear", classname):
                if m.bias.requires_grad:
                    m.bias.data.fill_(0)

        #Apply weight initialization to every module in the model.
        self.apply(weights_init)

    def forward(self, input):
        return self.model.forward(input)

    def loss(self):
        if(self.type == "temp-conv"):
            return self.backend.loss

        if(self.type == "LSTM" or self.type == "LSTM-init"):
            return self.lstm.loss

    def validator_function(self):
        if(self.type == "temp-conv"):
            return self.backend.validator

        if(self.type == "LSTM" or self.type == "LSTM-init"):
            return self.lstm.validator
