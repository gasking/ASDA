import math
import random

from tqdm import tqdm
import torch
import numpy as np
from nets.criterion import Dice_loss, score
import torch.nn as nn
from utils.tools import get_lr
from utils.tools import prob2entropy
from utils.config import Config
import torch.nn.functional as F
import cv2
import time
import shutil
import os

path = r'pselabel'
if os.path.exists(path=path):
    shutil.rmtree(path)
os.mkdir(path)


#---------------------------------------#
#           保存原图和伪标签
#--------------------------------------#
def get(images,feature,old_feature,tag = None):

    score = 0.1
    b,c,h,w = feature.shape

    feature = F.sigmoid(feature)

    old_feature = F.softmax(old_feature,dim=1)

    #TTA
    old_One_Hot = torch.argmax(old_feature,dim=1).detach().cpu().numpy()

    # 引入负例

    Neg_One_Hot = torch.argmin(feature, dim=1)



    #--------------------------------------------------#
    #                   二分类伪标签
    #--------------------------------------------------#

    bg = (old_One_Hot==0) #背景
    fg = (old_One_Hot==1) #前景




    #unsub = (((bg - fg) > 0.)  & ((bg - fg) <= 0.35) )#认为是前景


    # for channel in range(c):
    #
    #     temp = (feature[:,channel,:,:])
    #
    #     high_socre = temp >= score  #取出大于的样本
    #
    #     low_socre = temp < score
    #
    #     temp[low_socre] = 0.
    #
    #
    #
    #     feature[:, channel, :, :] = temp





    device = feature.device
    #feature = F.softmax(feature,dim=1)

    # feature = prob2entropy(feature)
    # feature = F.softmax(feature,dim=1)
    feature = feature.detach().cpu().numpy()

    #设置一个阈值 用来优化伪标签的精度
    #伪标签
    OneHot_feature = np.argmax(feature,axis = 1)



    #----------------------------------------------------#
    #                   伪标签指导
    #----------------------------------------------------#
    cur_bg = (OneHot_feature == 0 )
    cur_fg = (OneHot_feature == 1)

    #----------------------------------------------------#
    #               遍历One_Hot_feature
    #----------------------------------------------------#
    OneHot_feature[bg] = 0
    OneHot_feature[fg] = 1



    #OneHot_feature[unsub.cpu().numpy()] = 1 #认为是前景


    if tag:

        mask = (OneHot_feature == 1)


        OneHot_feature[mask] = 0



    la = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())


    for i in range(OneHot_feature.shape[0]):

        image = images[i].detach().cpu().data.numpy().transpose((1,2,0)) * 255
        image = image.astype(np.uint8)

        label = la + '_' + str(i) + '.png'
        jpg = la + '_' + str(i) + '.jpg'

        savedirlabel = os.path.join(f'pselabel',label)
        savedirjpg = os.path.join(f'pselabel',jpg)

        cv2.imwrite(savedirlabel,OneHot_feature[i] * 255)
        cv2.imwrite(savedirjpg,image)
        #print(np.unique(np.reshape(OneHot_feature[i],[-1])))

    b,c,h,w = feature.shape

    T_mask = np.zeros((b,h,w,c))
    for cls in range(c):

        index = (OneHot_feature == cls)

        #print(index.shape)
        T_mask[index,cls] = 1

        #print(T_mask[index,cls].sum())



    # for i in range(b):
    #  print(T_mask[i].sum())
     # for ind in range(c):
     #     im0 = T_mask[i,:,:,ind].detach().cpu().numpy() * 255
     #     im0 = (im0 * 255).astype('uint8')
     #     cv2.imshow('im',im0)
     #     cv2.waitKey(0)


    T_mask = np.transpose(T_mask,(0,3,1,2))
    T_mask = torch.from_numpy(T_mask).to(device)

    OneHot_feature = torch.from_numpy(OneHot_feature).to(device)



    return T_mask,OneHot_feature,Neg_One_Hot


def epoch_fit(cur_epoch, max_epoch, total_epoch, save_step, model, optimizer, traindataloader, device, logger,Studentmodel,old_model,args,savepath):



    # TODO 当前的模型 分割 + 分类 源域 or 目标域
    featuremodel, domainclassifer = model
    featuremodel = featuremodel.to(device)


    # old model永远不进行梯度更新
    old_model = old_model.to(device)



    domainclassifer = domainclassifer.to(device)

    interp = nn.Upsample(size=(Config['input_shape'][0], Config['input_shape'][1]), mode='bilinear', align_corners=True)

    # 优化器
    featureoptimizer, domainclassiferoptimizer = optimizer



    # 梯度累加
    accumulate = args.accumulate






    with tqdm(desc=f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total=len(traindataloader), mininterval=0.3, postfix=dict, colour='#6DB9EF') as pb:

        featuremodel = featuremodel.train()

        domainclassifer = domainclassifer.train()

        # -------------------------------------#
        # 训练损失计算
        # -------------------------------------#
        total_loss = 0.
        total_cls_loss = 0.
        total_domain = 0.

        # 源域
        for ind, batch in enumerate(traindataloader):
            sourceimage, targetimage = batch
            with torch.no_grad():
                # 源域
                sourceimage = sourceimage.to(device)


                # 目标域
                targetimage = targetimage.to(device)


                shape = sourceimage.shape


                Tag = False


                # if (random.random() > 0.9) & (ind &1 == 0 ) & (cur_epoch in [19,20,21]):
                # #----------------------------------------------------#
                # #                   伪造假标签
                # #               采用随机噪声生成图像
                # #----------------------------------------------------#
                #     Tag = True
                #     sourceimage = torch.zeros(shape).to(device)
                #     targetimage = torch.zeros(shape).to(device)
                #     sourceimage.fill_(128)
                #     targetimage.fill_(128)



            # ------------------------------------------#
            # 模型预测
            # ------------------------------------------#
            #featureoptimizer.zero_grad()

            #domainclassiferoptimizer.zero_grad()


            for para in domainclassifer.parameters():
                para.requires_grad = False


            # ---------------------------------#
            #           特征提取
            #           分割模型
            # ---------------------------------#
            # 学生模型
            student_eval = Studentmodel.ema.to(device)



            # 参数更新
            segaux, seg = featuremodel(sourceimage)

            segaux = interp(segaux)

            seg = interp(seg)


            scale = (cur_epoch) / (max_epoch)
            scale = np.exp(scale) - 1
            scale = scale * 0.5 + 0.3



            # 用学生模型进行推理
            studentsegaux, studentseg = student_eval(sourceimage)

            studentsegaux = interp(studentsegaux)

            studentseg = interp(studentseg)


            # old model
            oldsegaux, oldseg = old_model(sourceimage)

            oldsegaux = interp(oldsegaux)

            oldseg = interp(oldseg)


            studentsourcelabel,studentsourcepng,Neg_One_Hot = get(sourceimage,(studentsegaux),oldsegaux,Tag)

            # lossauxseg =  nn.CrossEntropyLoss()(segaux, sourcepng)
            # #
            # lossseg =  nn.CrossEntropyLoss()(seg, sourcepng)

            #lossauxseg = Dice_loss(segaux, sourcelabel)


            with torch.no_grad():
             consists = nn.MSELoss()(segaux,seg)  #知识需要代价回传

            lossseg =  nn.CrossEntropyLoss()(segaux, studentsourcepng)
            lossseg1 = nn.CrossEntropyLoss()(seg,studentsourcepng)

            #lossseg = torch.where(lossseg < )
            #losssegaux =  nn.CrossEntropyLoss()(segaux, sourcepng)

            # 计算负例损失
            weight1 =  1 - (lossseg.div(math.log2(2)))
            weight2 =  1 - (lossseg1.div(math.log2(2)))

            #-------------------------------------------------------------#
            #                       正例的损失
            #-------------------------------------------------------------#
            lossseg = (weight1 * lossseg)
            #print(lossseg)

            lossseg1 = (weight2 * lossseg1)
            #print(lossseg1)

            #------------------------------------------------------------#
            #                      负例损失
            #------------------------------------------------------------#
            Neg_loss1 = weight1 *  nn.CrossEntropyLoss()(1. - segaux,Neg_One_Hot)
            #print(Neg_loss1)
            Neg_loss2 = weight2 * nn.CrossEntropyLoss()(1 - seg,Neg_One_Hot)
            #print(Neg_loss2)




            Ssegloss = 5 * ( (lossseg + lossseg1)  + (Neg_loss1 + Neg_loss2) )

            logger.info(f'source domain seg loss: {Ssegloss.item()}\n')

            #print(Ssegloss.item())




            for para in domainclassifer.parameters():
                para.requires_grad = True

            paux, pseg = featuremodel(targetimage)

            paux = interp(paux)
            pseg = interp(pseg)



            # 用学生模型进行推理
            studentpaux, studentpseg = student_eval(targetimage)

            studentpaux = interp(studentpaux)

            studentpseg = interp(studentpseg)

            with torch.no_grad():
             consists = nn.MSELoss()(paux,pseg)  #知识需要代价回传



            # old model
            oldpaux, oldpseg = old_model(targetimage)

            oldpaux = interp(oldpaux)

            oldpseg = interp(oldpseg)

            targetlabel,_,Neg_One_Hot = get(targetimage, studentpaux, oldpaux,Tag)



            Tlossseg = nn.CrossEntropyLoss()(paux, _)
            Tlossseg1 = nn.CrossEntropyLoss()(pseg,_)

            # 计算负例损失
            weight1 = 1 - (Tlossseg.div(math.log2(2)))
            weight2 = 1 - (Tlossseg1.div(math.log2(2)))

            # -------------------------------------------------------------#
            #                       正例的损失
            # -------------------------------------------------------------#
            targetlossseg = (weight1 * Tlossseg)
            # print(lossseg)

            targetlossseg1 = (weight2 * Tlossseg1)
            # print(lossseg1)

            # ------------------------------------------------------------#
            #                      负例损失
            # ------------------------------------------------------------#
            target_Neg_loss1 = weight1 * nn.CrossEntropyLoss()(1. - paux, Neg_One_Hot)
            # print(Neg_loss1)
            target_Neg_loss2 = weight2 * nn.CrossEntropyLoss()(1 - pseg, Neg_One_Hot)
            # print(Neg_loss2)

            #
            d_taraux = domainclassifer(prob2entropy(F.sigmoid(pseg.detach())))

            d_saux = domainclassifer(prob2entropy(F.sigmoid(seg.detach())))

            real = torch.zeros_like(d_saux, requires_grad=True)
            fake = torch.ones_like(d_saux, requires_grad=True)

            criteror = nn.BCEWithLogitsLoss()
            DS = criteror(d_saux, real)

            DT = criteror(d_taraux, fake)

            Dloss =  (DS + DT)

            logger.info(f'targte domain loss: {Dloss.item()}\n')

            total_domain += Dloss.item()

            targetLoss = 8 * (targetlossseg + targetlossseg1 + target_Neg_loss1 + target_Neg_loss2)
            loss = ( Ssegloss + targetLoss)

            logger.info(f'targte seg loss: {targetLoss.item()}\n')


            loss.backward()


            #print(domainclassifer.state_dict())

            Dloss.backward()

            #print(old_model.state_dict())






            total_cls_loss += (Ssegloss.item() + Tlossseg.item())
            #total_domain += Dloss.item()

            total_loss += (total_cls_loss + total_domain)

            if ((ind + 1) % accumulate) == 0:
                torch.nn.utils.clip_grad_norm(featuremodel.parameters(),
                                              1.)

                featureoptimizer.step()

                #print(student_eval.state_dict())

                Studentmodel.update(featuremodel)

                #print(student_eval.state_dict())

                featureoptimizer.zero_grad()
            #featureoptimizer.step()
                domainclassiferoptimizer.step()


                domainclassiferoptimizer.zero_grad()



            pb.set_postfix(**{
                'total_loss': total_loss / (ind + 1),
                'total_cls_loss': total_cls_loss / (ind + 1),
                'total_domain_loss': total_domain / (ind + 1),
                'scale':scale
            })

            pb.update(1)

    # if ((ind + 1 ) % accumulate) == 0:
    #     # torch.nn.utils.clip_grad_norm(featuremodel.parameters(),
    #     #                               1.)
    #
    #     featureoptimizer.step()
    #
    #     featureoptimizer.zero_grad()
    #
    #     Studentmodel.update(featuremodel)



    featuremodel = featuremodel.eval()

    domainclassifer = domainclassifer.eval()

    if ((cur_epoch + 1) % save_step == 0) | (cur_epoch == (max_epoch - 1)):
        torch.save({
            'featuremodel': student_eval.state_dict(),

            'domainmodel': domainclassifer.state_dict()
        }, f'{savepath}/{(cur_epoch + 1)}.pth')
    if (cur_epoch + 1) == total_epoch:
        torch.save({
            'featuremodel': student_eval.state_dict(),
            'domainmodel': domainclassifer.state_dict()
        }, f'{savepath}/last_epoch_{total_epoch}.pth')
