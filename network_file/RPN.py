import torch.nn.init

from train_utils.myDataSet import *
from torch.nn.modules import Conv2d , Sequential

class RPNHead(torch.nn.Module):
    """
    RPNHead receive a feature map from backbone network,
    and return a new feature map conv by a set of small network for cls and reg.
    """

    def __init__(self,in_channels,feature_map:torch.tensor,k):
        super(RPNHead,self).__init__()
        self.small_network = Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.cls_network = Conv2d(in_channels=in_channels,out_channels=k,kernel_size=1,stride=1)
        self.reg_network = Conv2d(in_channels=in_channels,out_channels=k,kernel_size=1,stride=1)
        self.feature_map = feature_map
        #init params see paper 3.1.3
        for layer in self.children():
            torch.nn.init.normal_(layer.weigth,mean=0,std=0.01)
            torch.nn.init.zeros_(layer.bias)


    def forward(self) -> dict:
        a = self.small_network(self.feature_map)
        cls = self.cls_network(a)
        reg = self.reg_network(a)
        dict = {'cls' : cls , 'reg' : reg}
        return dict

class AnchorGenerator(torch.nn.Module):
    def __init__(self):
        super(AnchorGenerator,self).__init__()
