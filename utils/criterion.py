import torch.nn as nn
import torch
import torch.nn.functional as F


class BCELogitsloss(nn.Module):
    def __init__(self):
        super(BCELogitsloss, self).__init__()
        self.domain = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        # target = target[...,0]
        return self.domain(input, target)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.objloss = BCELogitsloss()

        self.boxloss = BCELogitsloss()          # 0.1 0.5
        self.clsloss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        obj = input[ 0 ][...,0]
        box = input[ 1 ]
        cls = input[ 2 ]


        target_obj = target[ 0 ]
        target_box = target[ 1 ]
        target_cls = target[ 2 ]

        #---------------------------------------------------#
        #                  损失函数定义
        #            对于没有目标的不进行计算
        #---------------------------------------------------#
        mask = (target_obj == 1).float() #have obj
        no_mask = (target_obj ==0 ).float() # no obj

        objloss = 2.5 * self.objloss(obj* mask,target_obj* mask) + 0.85 * self.objloss(obj*no_mask,target_obj*no_mask)

        boxloss = self.boxloss(box* mask[:,None],target_box* mask[:,None])

        clsloss = self.clsloss(cls,target_cls)

        return {
            "obj loss": objloss,
            "box loss": boxloss,
            "cls loss": clsloss
        }

def dice_loss(input,target):
    assert (input.shape) == (target.shape)


    union = input * target
    B = input.shape[0]

    A = (input.sum() + target.sum() -  union.sum())

    loss = 1 - (union.sum()) / (A + 1e-5)

    return  loss / B


def CE(input,target):

    return nn.BCEWithLogitsLoss()(input,target)



def build_loss():
    loss = Loss()

    return loss


if __name__ == '__main__':
    inf = -float('inf')
    print(inf * 0)