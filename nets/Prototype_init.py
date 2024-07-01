import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from nets.unet import SegHead
from nets.Upsample import UPSampler

from nets.deeplabv2 import get_deeplab_v2
from datasets.dataloader import UnetDataset
from torch.utils.data import  DataLoader
from utils.tools import prob2entropy
from PIL import Image
import numpy as np
import cv2
from utils.tools import resize
from utils.config import Config

def _resize(image,input_shape):
      # image is Image
      new_image,neww, newh ,dx,dy,w,h = resize(image,input_shape)


      img = np.array(new_image,dtype = np.float32)

      img = img / 255

      return new_image,img,neww, newh,dx,dy,w,h


class Prototype(object):
    def __init__(self,
                 inc,
                 num_classes):
        super(Prototype, self).__init__()

        # 原型中心质点
        #------------------------------------------------------------------------#
        #                     建立原型，以类别的中心特征进行聚类
        #------------------------------------------------------------------------#
        self.Proto = torch.zeros(num_classes,inc)

        # 计数
        self.amount = torch.zeros(num_classes)

        #self.pfeatc = nn.Parameter(torch.tensor([1,1,1,1]),requires_grad = False)

    # 原型初始化
    def frist_init(self,feature,label):
        #print(feature.shape,label.shape)
        mask = label.float() #获取所属类别信息


        self.pfeatc = feature * mask #取正样本



        self.pfeatc = self.pfeatc / label.sum()

        #self.pfeatc = torch.div(f,mask + 1e-8)


        return  self.pfeatc

    #-----------------------------------------#
    #              对抗对比损失
    #-----------------------------------------#
    def contrast_adaptation(self,
                            target_feature = None,
                            tempature = 1):
        #-------------------------------------------#
        #                   产生伪标签
        #             target_feature经过像素判别器
        #-------------------------------------------#
        pseudo_label = target_feature #one-hot编码
        # 计算特征和原型之间的相似度
        # 1. 从目标域 到 源域
        # 2. 从源域到源域的距离计算
        t = self.Proto.matmul(pseudo_label.div(tempature).T)
        #pts = F.softmax(t,dim = 1)

        return t.permute((1,0))


    def update(self,
               monteum = 0.99,
               target_feature = None,
               label = None):






        # 当前原型评估
        # r_feat = torch.zeros_like(self.pfeatc)
        # p_feat = torch.zeros_like(self.pfeatc)
        #
        # b,c,h,w = target_feature.shape
        # if not isinstance(monteum,float):
        #   for cls in range(c):
        #       T = self.pfeatc[:,cls,:,:].sum()
        #       r_feat[:,cls,:,:] = T
        #       T1 = target_feature[:,cls,:,:].sum()
        #       p_feat[:,cls,:,:] = T1
        #
        #   self.pfeatc = r_feat.mul(self.pfeatc) + \
        #                 p_feat.mul(target_feature)
        #   T_sum = r_feat.add(p_feat)
        #
        #   self.pfeatc = self.pfeatc.div(T_sum + 1e-8)
        #
        #   torch.save(self.pfeatc,"../Protype/protype.pth")
        #
        #   return self.pfeatc

        # 特征聚类
        #else:
            self.Proto = self.Proto.cuda(non_blocking=True)
            self.amount = self.amount.cuda(non_blocking=True)
            inds = label.unique()
            for ind in inds:
                cls = ind.item() #取出类别
                mask_i = (label == cls)

                feature = target_feature[mask_i]
                feature = torch.mean(feature,dim = 0)
                # print(mask_i.sum())
                #
                # print(monteum * feature + self.Proto[ind,:] * (1 - monteum))

                self.amount[ind] += (mask_i.sum())

                self.Proto[ind,:] = monteum * feature + self.Proto[ind,:] * (1 - monteum)

            #self.pfeatc = monteum * self.pfeatc + (1 - monteum) * target_feature

    def save(self):
        # mv cpu
        torch.save({
            'Proto':self.Proto.cpu(),
        'Amount': self.amount.cpu()}, "../Protype/protype.pth")


class PixContrasrt(nn.Module):
    def __init__(self):
        super(PixContrasrt, self).__init__()


        self.dist = nn.Parameter(torch.tensor(1.,
                                              requires_grad = True))


    def forward(self,feature,label):

        # 前景激活层
        masked = feature * label

        # 背景激活层
        unmasked = feature * (~label)


        #以余弦距离来构建三元组损失

        undist = nn.CosineSimilarity()(feature,unmasked)

        maskdist = nn.CosineSimilarity()(feature,masked)

        return torch.where(
            undist > maskdist + self.dist,
            1.5 * undist + 0.8 * maskdist,
            undist + 0.5 * maskdist
        )

if __name__ == '__main__':
    # 原型初始化
    # protype = Prototype(inc = 2,
    #                     num_classes = 2)
    cost = PixContrasrt()


    device = torch.device('cuda')
    model_path = r'../fristlogs/18.pth'

    feature = get_deeplab_v2(num_classes=2,multi_level=True,backbone='resnet50')
    model = torch.load(model_path, map_location = device)
    model = model[ 'featuremodel' ]
    feature.load_state_dict(model)



    feature = feature.to(device)


    featureoptimizer = optimizer.Adam(
                        feature.parameters(),lr = 2.5e-3,
                        weight_decay = 0.95
    )



    #feature = feature.eval()
    # seg = seg.eval()

    feature = feature.train()


    sourcedata = UnetDataset(source_file = Config['source_train_txt'],
                             target_file = Config['source_train_txt'],
                             input_shape = (256,256))
    sourceloader = DataLoader(sourcedata,batch_size = 16,
                              num_workers = 4,shuffle = True)
    Init = False
    criterior = nn.CrossEntropyLoss()
    feature = feature.train()



    Max_epoch = 30

    interp = nn.Upsample(size=(Config['input_shape'][0], Config['input_shape'][1]), mode='bilinear', align_corners=True)

    for epoch in range(Max_epoch):
        total_ts = 0.
        total_ss = 0.
        for ind,batch in enumerate(sourceloader):

            featureoptimizer.zero_grad()


            # 建立原型
            with torch.no_grad():
                sourceimage,sourcelabel,sourcepng,targetimage,targetlabel,targetpng = batch

                sourceimage = sourceimage.to(device)
                targetimage = targetimage.to(device)
                sourcelabel = sourcelabel.to(device)
                sourcepng = sourcepng.to(device)

            aux,permainmain = feature(sourceimage)

            aux = interp(aux)
            permainmain = interp(permainmain)


            src_feat = aux.permute(0,2,3,1).contiguous().view(-1,2)



            #featureseg = (permainmain)


            #----------------------------------------------------#
            #             针对当前的batch数据进行原型
            #----------------------------------------------------#
            b,c,h,w = aux.shape
            aux = aux.view((-1,c))
            old_png = sourcepng
            sourcepng = torch.unsqueeze(sourcepng,dim = 1)
            sourcepng = F.interpolate(sourcepng,(h,w))
            sourcepng = sourcepng.view((-1,))
            #res = protype.frist_init(featureseg,sourcelabel)

            taraux, tarmain = feature(targetimage)
            taraux = taraux.view((-1,c))

            # 获取目标域的one-hot掩码
            #data,inds = torch.max(tarmain,dim = 1)

            # mask = (tarmain > 0.2).float()
            # tarmain = tarmain * mask


           # tarmain = F.interpolate(tarmain, (h, w))
            One_hotTarget = torch.argmax(taraux, dim=1)

            One_hotTarget = One_hotTarget.view((-1,))


            #temp = One_hotTarget.unsqueeze(dim=1)

            protype.update(monteum= 0.999, target_feature=aux.detach(),label = sourcepng)
            protype.update(monteum= 0.999, target_feature=taraux.detach(),label = One_hotTarget)

            pts = protype.contrast_adaptation((taraux))



            t_sloss = criterior(pts,One_hotTarget.long())


            pss = protype.contrast_adaptation((aux))

            s_sloss = criterior(pss,sourcepng.long())

            total_ts = t_sloss.item()
            total_ss = s_sloss.item()


            print(f"EPOCH: {epoch + 1}\{Max_epoch} ind:{ind}/{len(sourceloader)} target->source:{total_ts / (ind + 1)}  source->source: {total_ss / (ind + 1)}",flush=True,end='\n')


            #print(permainmain.shape,old_png.shape)
            Tloss = (t_sloss + s_sloss)


            Tloss.backward()


            featureoptimizer.step()




        feature.eval()
        
        torch.save({
                'featuremodel': feature.state_dict(),

            }, f'../Protype/weight_{epoch + 1}.pth')

       # protype.save()

                # One_hotMix = torch.argmax(tfeature,dim = 1)
                #
                # im1 = One_hotMix[ 0 ].cpu().numpy()
                #
                # im1 = (im1 * 255).astype(np.uint8)
                #
                # im0 = targetimage[ 0 ].cpu().numpy()
                # im0 = np.transpose(im0, (1, 2, 0)) * 255
                # im0 = im0.astype(np.uint8)
                #
                # cv2.imshow('im0', im0)
                # cv2.imshow('im1', im1)
                # cv2.waitKey(0)
                #
                # print(One_hotTarget.shape)

                #print(res.shape)






