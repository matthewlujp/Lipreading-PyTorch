import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Separable convolution, which consists of
    1) convolution along time dimension 
    2) point-wise convolution

    To keep length along temporal dimension, kernel size should be odd.
    """
    def __init__(self, channel_in: int, channel_out: int, k_size: int):
        super().__init__()
        if k_size % 2 == 0:
            print("WARN: provided kernel size {} is even. cannot keep frame length".format(k_size), file=sys.stderr)

        self.depthwise_conv = nn.Conv1d(
            channel_in, channel_in, kernel_size=k_size, stride=1, padding=(k_size - 1)//2, groups=channel_in)
        self.pointwise_conv = nn.Conv1d(channel_in, channel_out, 1, 1)
        
    def forward(self, x):
        """Input & ouput: (batch_size, channel_in, frames_len)
        """
        h = self.depthwise_conv(x)
        return self.pointwise_conv(h)
        

class FCBlock(nn.Module):
    """Apply depth-wise separable convolution, add short cut connection, batch normalization, and ReLU.
    Convert input of (batch_size, channel_in, frames_len) -> (batch_size, channel_out, frames_len)
    """
    def __init__(self, channel:int, k_size: int, dropout=0.8):
        super().__init__()

        self.dps_conv = DepthwiseSeparableConv(channel, channel, k_size)
        self.norm = nn.BatchNorm1d(channel)

        self.dropout_p = dropout

    def forward(self, x):
        # x.shape: (batch_size, channel_in, frames_len)
        h = self.dps_conv(x)  # (batch_size, channel_out, frames_len)
        h = self.norm(h + x)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        return F.relu(h)
        

class NLLSequenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = nn.NLLLoss()

    def forward(self, prediction, target):
        """prediction.shape is (batch_size, frames_len, class_num)
        target shape is (batch_size)
        """
        loss = 0.0
        transposed = prediction.transpose(0, 1).contiguous()  # (frames_len, batch_size, class_num)
        for i in range(prediction.shape[0]):
            # print("prediction", prediction.shape, "target", target.shape, file=sys.stderr)
            loss += self.criteria(transposed[i], target)
        return loss


def _validate(modelOutput, labels):
    """modelOutput.shape is (batch_size, frames_len, class_num)
    labels.shape is (batch_size, 1)
    """
    averageEnergies = torch.sum(modelOutput.data, 1)
    maxindices = torch.argmax(averageEnergies, dim=1)
    _labels = labels.squeeze(1)

    count = 0
    for i in range(len(_labels)):
        if maxindices[i] == _labels[i]:
            count += 1

    return count


class FCBackend(nn.Module):
    def __init__(self, options: dict):
        super().__init__()

        input_dim = options['model']['inputdim']
        class_num = options["model"]["numclasses"]
        kernel_size = options['model']['fc_back_kernel_size']
        stack_amount = options['model']['fc_back_stack']
        dropout_p = options['model']['dropout_prob']

        self.stack = nn.Sequential(*[FCBlock(input_dim, kernel_size, dropout_p) for _ in range(stack_amount)])
        self.fc = nn.Linear(input_dim, class_num)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.loss = NLLSequenceLoss()

        self.validator = _validate

    def forward(self, x):
        """x.shape is (batch_size, frames_len, 512).
        Outputs LogSoftmax over vocabulary whose shape is (batch_size, frames_len, vocabulary_num).
        """
        transposed = x.transpose(1, 2).contiguous()  # (batch_size, input_dim, frames_len)
        
        h = self.stack(transposed)  # (batch_size, input_dim, frames_len)
        transposed_h = h.transpose(1, 2).contiguous()  # (batch_size, frames_len, input_dim)

        h = self.fc(transposed_h)  # (batch_size, frames_len, class_num)
        return self.log_softmax(h)




