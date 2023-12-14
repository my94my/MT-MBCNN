from numpy import flip, concatenate
from scipy.io import loadmat
from torch import cat, from_numpy
from torch.nn import Module, AvgPool2d, Conv2d, Sequential, BatchNorm2d, Dropout, LogSoftmax, AdaptiveAvgPool2d
from torch.nn.init import xavier_uniform_, zeros_, ones_

from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.modules import Expression


class WaveletTransform(Module):

    def __init__(self, channel, params_path='./filter_parameter.mat'):

        super(WaveletTransform, self).__init__()
        self.conv = Conv2d(in_channels=channel, out_channels=channel * 2, kernel_size=(1, 8), stride=(1, 2),
                           padding=0, groups=channel, bias=False)
        for m in self.modules():
            if isinstance(m, Conv2d):
                f = loadmat(params_path)
                Lo_D = flip(f['Lo_D'], axis=1).astype('float32')
                Hi_D = flip(f['Hi_D'], axis=1).astype('float32')
                weight = from_numpy(concatenate((Lo_D, Hi_D), axis=0))
                m.weight.data = weight.unsqueeze(1).unsqueeze(1).repeat(channel, 1, 1, 1)
                m.weight.requires_grad = False

    def forward(self, x):
        out = self.conv(self.self_padding(x))
        return out[:, 0::2, :, :], out[:, 1::2, :, :]

    def self_padding(self, x):
        return cat((x[:, :, :, -3:], x, x[:, :, :, :3]), 3)


class MTMBCNN(Module):
    def __init__(self, n_chan):
        super(MTMBCNN, self).__init__()
        n_map = 5

        self.conv1_1 = Conv2d(1, n_map, kernel_size=(1, 11), stride=(1, 1))
        self.conv1_2 = Sequential(Conv2d(n_map, n_map, kernel_size=(n_chan, 1), stride=(1, 1), bias=False),
                                  BatchNorm2d(n_map, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  Expression(square))
        self.conv1_3 = Conv2d(n_map, n_map*5, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.conv2_1 = Conv2d(1, n_map, kernel_size=(1, 7), stride=(1, 1))
        self.conv2_2 = Sequential(Conv2d(n_map, n_map, kernel_size=(n_chan, 1), stride=(1, 1), bias=False),
                                  BatchNorm2d(n_map, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  Expression(square))
        self.conv2_3 = Conv2d(n_map, n_map*5, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.conv3_1 = Conv2d(1, n_map, kernel_size=(1, 20), stride=(1, 1))
        self.conv3_2 = Sequential(Conv2d(n_map, n_map, kernel_size=(n_chan, 1), stride=(1, 1), bias=False),
                                  BatchNorm2d(n_map, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  Expression(square))
        self.conv3_3 = Conv2d(n_map, n_map*5, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.pool1_1 = Sequential(AvgPool2d(kernel_size=(1, 85), stride=(1, 15), padding=0),
                                  Expression(safe_log))

        self.dropout = Dropout(p=0.5)

        self.classfier = Sequential(Conv2d(n_map*15, 2, kernel_size=(1, 69), stride=(1, 1)),
                                    LogSoftmax(dim=1))

        self.WaveletTransform = WaveletTransform(channel=n_map)
        self.reshape1 = AdaptiveAvgPool2d((1, 69))


        for m in self.modules():
            if isinstance(m, Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, BatchNorm2d):
                ones_(m.weight)
                zeros_(m.bias)

    def waveconvs(self, x):
        out, gamma = self.WaveletTransform(x)
        out, beta = self.WaveletTransform(out)
        out, alpha = self.WaveletTransform(out)
        delta, theta = self.WaveletTransform(out)
        return cat((delta, theta, self.reshape1(alpha), self.reshape1(beta), self.reshape1(gamma)), dim=1)

    def forward(self, x):
        x_raw = x.squeeze().unsqueeze(1)
        x1_1 = self.conv1_1(x_raw)
        x1_2 = self.conv1_2(x1_1)
        xpool1_1 = self.pool1_1(x1_2)
        xpool1_1 = self.conv1_3(xpool1_1)

        x2_1 = self.conv2_1(x_raw)
        x2_2 = self.conv1_2(x2_1)
        xpool2_1 = self.pool1_1(x2_2)
        xpool2_1 = self.conv2_3(xpool2_1)

        x3_1 = self.conv3_1(x_raw)
        x3_2 = self.conv1_2(x3_1)
        xpool3_1 = self.pool1_1(x3_2)
        xpool3_1 = self.conv3_3(xpool3_1)

        feature = cat((xpool1_1, xpool2_1, xpool3_1), dim=1)
        output = self.dropout(feature)

        x_guide1 = self.waveconvs(x1_2)
        x_guide2 = self.waveconvs(x2_2)
        x_guide3 = self.waveconvs(x3_2)

        output = self.classfier(output)

        return output.squeeze(), x_guide1, x_guide2, x_guide3, xpool1_1, xpool2_1, xpool3_1, feature
