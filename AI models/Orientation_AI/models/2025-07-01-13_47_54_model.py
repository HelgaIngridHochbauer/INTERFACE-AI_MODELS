import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        
        if (stride == 1):
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding='same')
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)
        
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)            
        out = self.conv(out)

        return out


class NetworkOld(nn.Module):
    def __init__(self, in_planes, n_kernel, n_out):
        super(NetworkOld, self).__init__()
        
        self.A01 = ConvBlock(in_planes, n_kernel, kernel_size=3)
        
        self.C01 = ConvBlock(n_kernel, 2*n_kernel, stride=2)
        self.C02 = ConvBlock(2*n_kernel, 2*n_kernel)
        self.C03 = ConvBlock(2*n_kernel, 2*n_kernel)
        self.C04 = ConvBlock(2*n_kernel, 2*n_kernel, kernel_size=1)

        self.C11 = ConvBlock(2*n_kernel, 4*n_kernel, stride=2)
        self.C12 = ConvBlock(4*n_kernel, 4*n_kernel)
        self.C13 = ConvBlock(4*n_kernel, 4*n_kernel)
        self.C14 = ConvBlock(4*n_kernel, 4*n_kernel, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(4*n_kernel, 256)
        self.fc2 = nn.Linear(256, n_out)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):        
        A01 = self.A01(x)

        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)
        C04 = self.C04(C03)
        C04 += C01

        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = self.C14(C13)
        C14 += C11

        out = self.pool(C14)

        nb, nf, nx, ny = out.shape
        out = out.view(nb, nf*nx*ny)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
            
        return out

class Network(nn.Module):
    def __init__(self, in_planes, n_kernel, n_out):
        super(Network, self).__init__()
        
        self.net = nn.ModuleList()

        self.net.append(nn.Conv2d(1, n_kernel, kernel_size=7, stride=1, padding=1))
        self.net.append(nn.BatchNorm2d(n_kernel))
        self.net.append(nn.ReLU(inplace=True))
        # self.net.append(nn.Dropout(p=0.5))
        self.net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.net.append(nn.Conv2d(n_kernel, 2*n_kernel, kernel_size=3, stride=1, padding=1))
        self.net.append(nn.BatchNorm2d(2*n_kernel))
        self.net.append(nn.ReLU(inplace=True))
        # self.net.append(nn.Dropout(p=0.5))
        self.net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.net.append(nn.Conv2d(2*n_kernel, 4*n_kernel, kernel_size=3, stride=1, padding=1))
        self.net.append(nn.BatchNorm2d(4*n_kernel))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.net.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.net2 = nn.ModuleList()

        self.net2.append(nn.Linear(4*n_kernel, 4*n_kernel))
        self.net2.append(nn.ReLU(inplace=True))
        self.net2.append(nn.Dropout(p=0.3))
        self.net2.append(nn.Linear(4*n_kernel, 4*n_kernel))
        self.net2.append(nn.ReLU(inplace=True))
        self.net2.append(nn.Dropout(p=0.3))
        self.net2.append(nn.Linear(4*n_kernel, n_out))
        
    def forward(self, x):        

        tmp = x.clone()

        for layer in self.net:
            tmp = layer(tmp)
            
        tmp = tmp.view(x.size(0), -1)

        for layer in self.net2:
            tmp = layer(tmp)
            
        return tmp
    