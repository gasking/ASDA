import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self,
                 inc,
                 num_classes
                 ):
        super(LeNet, self).__init__()

        self.c1 = nn.Sequential(
                  nn.Conv2d(in_channels = inc,out_channels = 6,kernel_size = 5,stride = 1),
                  nn.BatchNorm2d(6),
                  nn.ReLU(inplace = True)

        )


        self.s2 = nn.AvgPool2d(kernel_size = 2,stride = 2)


        self.c3 = nn.Sequential(
                  nn.Conv2d(in_channels = 6,
                            out_channels = 16,
                            stride = 1,kernel_size = 5),
                  nn.BatchNorm2d(16),
                  nn.ReLU(inplace = True)
        )

        self.s4 = nn.AvgPool2d(kernel_size = 2,stride = 2)

        self.c5 = nn.Sequential(
                  nn.Conv2d(in_channels = 16,
                            out_channels = 16,
                            stride = 1,kernel_size = 5,padding = 2),
                  nn.BatchNorm2d(16),
                  nn.LeakyReLU(inplace = True)
        )

        self.f6 = nn.Sequential(nn.Linear(82944,84),
                                nn.Dropout(0.5))

        #------------------------------------------#
        #  增加box和obj2个分支
        #------------------------------------------#
        self.obj = nn.Linear(84,1)
        self.box = nn.Linear(84,4)

        self.out = nn.Linear(84,num_classes)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.eval()#冻结BN层
    def forward(self,x):


        x = x.float()
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.c5(x)  #三维向量

        b,c,h,w = x.shape


        x = x.view((b,-1))

        x = self.f6(x)

        obj = self.obj(x)
        box = self.box(x)
        cls = self.out(x)


        return   obj,box,cls



if  __name__ == '__main__':
    x = torch.randn((2,3,300,300))


    model = LeNet(inc = 3,num_classes = 10)


    obj,box,cls = model.forward(x)


    print(obj.shape,box.shape,cls.shape)
