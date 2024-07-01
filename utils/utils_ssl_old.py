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
from nets.criterion import PixContrasrt
import cv2
import time
import shutil
import os

path = r'pselabel'
if os.path.exists(path=path):
    shutil.rmtree(path)
os.mkdir(path)


# ---------------------------------------#
#           保存原图和伪标签
# --------------------------------------#
def get(images, feature, old_feature, tag=None,labels = 'aux',show = True):
    score = 0.1
    b, c, h, w = feature.shape

    #-------------------------------------------------#
    #           5.8修改
    #-------------------------------------------------#
    feature = F.softmax(feature,dim=1)

    old_feature = F.softmax(old_feature, dim=1)

    # TTA
    old_One_Hot = torch.argmax(old_feature, dim=1).detach().cpu().numpy()

    # 引入负例

    Neg_One_Hot = torch.argmin(feature, dim=1)

    # --------------------------------------------------#
    #                   二分类伪标签
    # --------------------------------------------------#

    bg = (old_One_Hot == 0)  # 背景
    fg = (old_One_Hot == 1)  # 前景

    # unsub = (((bg - fg) > 0.)  & ((bg - fg) <= 0.35) )#认为是前景

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
    # feature = F.softmax(feature,dim=1)

    # feature = prob2entropy(feature)
    # feature = F.softmax(feature,dim=1)
    feature = feature.detach().cpu().numpy()

    # 设置一个阈值 用来优化伪标签的精度
    # 伪标签
    OneHot_feature = np.argmax(feature, axis=1)

    # ----------------------------------------------------#
    #                   伪标签指导
    # ----------------------------------------------------#
    cur_bg = (OneHot_feature == 0)
    cur_fg = (OneHot_feature == 1)

    # ----------------------------------------------------#
    #               遍历One_Hot_feature
    # ----------------------------------------------------#
    #OneHot_feature[(bg & cur_bg)] = 0
    OneHot_feature[(fg | cur_fg)] = 1

    # OneHot_feature[unsub.cpu().numpy()] = 1 #认为是前景

    if tag:
        mask = (OneHot_feature == 1)

        OneHot_feature[mask] = 0

    la = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    if show:
        for i in range(OneHot_feature.shape[0]):
            image = images[i].detach().cpu().data.numpy().transpose((1, 2, 0)) * 255
            image = image.astype(np.uint8)

            label = la + '_' + str(i) + labels+'.png'
            jpg = la + '_' + str(i) + '.jpg'

            savedirlabel = os.path.join(f'pselabel', label)
            savedirjpg = os.path.join(f'pselabel', jpg)

            cv2.imwrite(savedirlabel, OneHot_feature[i] * 255)
            cv2.imwrite(savedirjpg, image)
        # print(np.unique(np.reshape(OneHot_feature[i],[-1])))

    b, c, h, w = feature.shape

    T_mask = np.zeros((b, h, w, c))
    for cls in range(c):
        index = (OneHot_feature == cls)

        # print(index.shape)
        T_mask[index, cls] = 1

        # print(T_mask[index,cls].sum())

    # for i in range(b):
    #  print(T_mask[i].sum())
    # for ind in range(c):
    #     im0 = T_mask[i,:,:,ind].detach().cpu().numpy() * 255
    #     im0 = (im0 * 255).astype('uint8')
    #     cv2.imshow('im',im0)
    #     cv2.waitKey(0)

    T_mask = np.transpose(T_mask, (0, 3, 1, 2))
    T_mask = torch.from_numpy(T_mask).to(device)

    OneHot_feature = torch.from_numpy(OneHot_feature).to(device)

    return T_mask, OneHot_feature, Neg_One_Hot


def epoch_fit(cur_epoch, max_epoch, total_epoch,
              save_step, model, optimizer,
              traindataloader, valdataloader,
              device, logger, Studentmodel, old_model, args, savepath,
              eval):
    featuremodel, domainclassifer_aux, domainclassifer_main = model

    featuremodel = featuremodel.to(device)

    old_model = old_model.to(device)

    domainclassifer_aux = domainclassifer_aux.to(device)

    domainclassifer_main = domainclassifer_main.to(device)

    interp = nn.Upsample(size=(Config['input_shape'][0], Config['input_shape'][1]), mode='bilinear', align_corners=True)

    # 优化器
    featureoptimizer, domainclassiferoptimizer_aux, domainclassiferoptimizer_main = optimizer

    accumulate = args.accumulate
    if torch.cuda.device_count() > 1:
        # -----------------------------------------------------#
        #                  训练的BS要能被显卡数整除
        # -----------------------------------------------------#
        print(f'GPUS Numbers :{torch.cuda.device_count()}')

        featuremodel = nn.DataParallel(featuremodel,device_ids = [0,1])

        domainclassifer_aux = nn.DataParallel(domainclassifer_aux,device_ids = [0,1])

        domainclassifer_main = nn.DataParallel(domainclassifer_main,device_ids = [0,1])

    # 损失函数定义
    criteror = nn.BCEWithLogitsLoss()


    # 像素对比损失

    pixciteror = PixContrasrt(max_epoch = total_epoch)

    with tqdm(desc=f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total=len(traindataloader), mininterval=0.3, postfix=dict, colour='#6DB9EF') as pb:

        featuremodel = featuremodel.train()

        domainclassifer_aux = domainclassifer_aux.train()

        domainclassifer_main = domainclassifer_main.train()

        # -------------------------------------#
        # 训练损失计算
        # -------------------------------------#
        total_loss = 0.
        total_cls_loss = 0.
        total_domain = 0.

        # 源域
        for ind, batch in enumerate(traindataloader):
            sourceimage, sourcelabel, sourcepng, targetimage, targetlabel, targetpng = batch
            with torch.no_grad():
                # 源域
                sourceimage = sourceimage.to(device)
                sourcelabel = sourcelabel.to(device)
                sourcepng = sourcepng.to(device)

                # 目标域
                targetimage = targetimage.to(device)
                targetlabel = targetlabel.to(device)
                targetpng = targetpng.to(device)

                shape = sourceimage.shape

                Tag = False

                # if (random.random() > 0.9) & (ind &1 == 0 ) :
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
            # featureoptimizer.zero_grad()

            # domainclassiferoptimizer.zero_grad()
            #
            for para in domainclassifer_aux.parameters():
                para.requires_grad = False

            for para in domainclassifer_main.parameters():
                para.requires_grad = False

            # 首先对优化器中的梯度进行清空
            featureoptimizer.zero_grad()
            domainclassiferoptimizer_aux.zero_grad()
            domainclassiferoptimizer_main.zero_grad()
            

            
            # ---------------------------------#
            #           特征提取
            #           分割模型
            # ---------------------------------#
            # 学生模型
            student_eval = Studentmodel.ema.to(device)

            # 分割模型
            segaux, seg = featuremodel(sourceimage)

            segaux = interp(segaux)

            seg = interp(seg)

            scale = (cur_epoch) / (max_epoch)
            scale = np.exp(scale) - 1
            scale = scale * 0.5 + 0.3

            # 用学生模型进行推理  得到伪标签
            studentsegaux, studentseg = student_eval(sourceimage)

            studentsegaux = interp(studentsegaux)

            studentseg = interp(studentseg)

            # old model
            oldsegaux, oldseg = old_model(sourceimage)

            oldsegaux = interp(oldsegaux)

            oldseg = interp(oldseg)
            

            # whether use anchor network
            if args.anchor:

                studentsourcelabel_aux, studentsourcepng_aux, Neg_One_Hot_aux = get(sourceimage, (studentsegaux), oldsegaux, Tag,'aux',False)

                studentsourcelabel_main, studentsourcepng_main, Neg_One_Hot_main = get(sourceimage, (studentseg), oldseg,
                                                                                    Tag,'main',False)
            else:
                studentsourcelabel_aux, studentsourcepng_aux, Neg_One_Hot_aux = get(sourceimage, (studentsegaux), studentsegaux, Tag,'aux',False)

                studentsourcelabel_main, studentsourcepng_main, Neg_One_Hot_main = get(sourceimage, (studentseg), studentseg,
                                                                                    Tag,'main',False)
            # lossauxseg =  nn.CrossEntropyLoss()(segaux, sourcepng)
            # #
            # lossseg =  nn.CrossEntropyLoss()(seg, sourcepng)

            # lossauxseg = Dice_loss(segaux, sourcelabel)

            with torch.no_grad():
                consists = nn.MSELoss()(segaux, seg)  # 知识需要代价回传

            lossseg = nn.CrossEntropyLoss()(segaux, studentsourcepng_aux)
            lossseg1 = nn.CrossEntropyLoss()(seg, studentsourcepng_main)
            lossconsist = 0.1* nn.MSELoss()(segaux,seg)
            # lossseg = torch.where(lossseg < )
            # losssegaux =  nn.CrossEntropyLoss()(segaux, sourcepng)

            # 计算负例损失
            weight1 = 1 - (- (torch.mul(segaux, torch.log2(segaux + 1e-10))) / math.log2(2))

            weight2 = 1 - (- (torch.mul(seg, torch.log2(seg + 1e-10))) / math.log2(2))

            # -------------------------------------------------------------#
            #                       正例的损失
            # -------------------------------------------------------------#
            lossseg = ( lossseg)
            # print(lossseg)

            lossseg1 = ( lossseg1)
            # print(lossseg1)

            # ------------------------------------------------------------#
            #                      负例损失
            # ------------------------------------------------------------#
            Neg_loss1 = nn.CrossEntropyLoss()(segaux, Neg_One_Hot_aux)
            # print(Neg_loss1)
            Neg_loss2 =  nn.CrossEntropyLoss()(seg, Neg_One_Hot_main)
            # print(Neg_loss2)


            #------------------------------------------------------------------#
            #                           像素对比损失
            #-----------------------------------------------------------------#
            pix_aux = pixciteror(segaux,studentsourcepng_aux,cur_epoch)
            pix_main = pixciteror(seg,studentsourcepng_main,cur_epoch)

            if args.pixcontrast:
               contrast_loss = args.contrast_para * (pix_aux + pix_main)
            else:
                contrast_loss = 0
            
            if (not args.consist):
                lossconsist = 0

            Ssegloss = args.seg_para *  (lossseg + lossseg1) + lossconsist + contrast_loss # + Neg_loss1 + Neg_loss2

            #Ssegloss =  2.0 * (pix_aux + pix_main)
           # Ssegloss = 0.1 * (lossseg + Neg_loss1) + (lossseg1 + Neg_loss2)

            logger.info(f'source domain seg loss: {Ssegloss.item()}\n')

            Ssegloss.backward()

            total_cls_loss += Ssegloss.item()
            # print(Ssegloss.item())

            # ---------------------------------------------#
            # adversarial training to fool the discriminator
            # ---------------------------------------------#
            paux, pseg = featuremodel(targetimage)

            paux = interp(paux)
            pseg = interp(pseg)



            d_taraux = domainclassifer_aux(prob2entropy(F.softmax(paux)))

            d_tarmain = domainclassifer_main(prob2entropy(F.softmax(pseg)))

            real = torch.zeros_like(d_taraux, requires_grad=True)
            fake = torch.ones_like(d_taraux, requires_grad=True)

            d_domain_aux_loss = criteror(d_taraux, real)
            d_domain_main_loss = criteror(d_tarmain, real)

            Tdomain_loss = (0.0001 * d_domain_main_loss + 0.0002 * d_domain_aux_loss)

            #Tdomain_loss = (d_domain_main_loss +  d_domain_aux_loss)

            Tdomain_loss.backward()

            # --------------------------------------------#
            #
            # --------------------------------------------#
            for para in domainclassifer_aux.parameters():
                para.requires_grad = True
            for para in domainclassifer_main.parameters():
                para.requires_grad = True

            # ------------------------------------------------------------------------#
            #                           only source domain
            # ------------------------------------------------------------------------#
            d_source_aux = domainclassifer_aux(prob2entropy(F.softmax(segaux.detach())))
            DS_aux = criteror(d_source_aux, real) / 2.
            DS_aux.backward()

            d_source_main = domainclassifer_main(prob2entropy(F.softmax(seg.detach())))
            DS_main = criteror(d_source_main, real) / 2.
            DS_main.backward()

            # --------------------------------------------------------------------#
            #                        目标域
            # --------------------------------------------------------------------#
            d_target_aux = domainclassifer_aux(prob2entropy(F.softmax(paux.detach())))
            DT_aux = criteror(d_target_aux, fake) / 2.
            DT_aux.backward()

            d_target_main = domainclassifer_main(prob2entropy(F.softmax(pseg.detach())))
            DT_main = criteror(d_target_main, fake) / 2.
            DT_main.backward()

            logger.info(
                f'domain loss: {(DT_main.item() + DT_aux.item() + DS_main.item() + DS_aux.item() + Tdomain_loss.item())}\n')

            total_domain += (DT_main.item() + DT_aux.item() +
                             DS_main.item() + DS_aux.item() + Tdomain_loss.item()
                             )


            total_loss += (total_domain + total_cls_loss)

            pb.set_postfix(**{
                'total_loss': total_loss / (ind + 1),
                'total_cls_loss': total_cls_loss / (ind + 1),
                'total_domain_loss': total_domain / (ind + 1),

                'scale': scale
            })

            pb.update(1)

    featureoptimizer.step()

    domainclassiferoptimizer_aux.step()

    domainclassiferoptimizer_main.step()

    # ------------------------------------------#
    #         学生模型参数不可学习进行更新
    # ------------------------------------------#
    Studentmodel.update(featuremodel)

    #featuremodel = featuremodel.eval()

    domainclassifer_aux = domainclassifer_aux.eval()

    domainclassifer_main = domainclassifer_main.eval()

    # Evaluatation
    with tqdm(desc=f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total=len(valdataloader), mininterval=0.3, postfix=dict, colour='#7E89EF') as pb:

        with torch.no_grad():
            # 目标域
            for ind, batch in enumerate(valdataloader):
                _, _, _, Ttargetimage, _, Ttargetpng, Ttargetname = batch

                Ttargetimage = Ttargetimage.to(device)

                Ttargetpng = Ttargetpng.to(device).detach().cpu().numpy()

                # 使用学生模型进行推理
                pred_seg, pred = student_eval(Ttargetimage)

                pred_seg = interp(pred_seg)
                pred = interp(pred)

                targetOneHotlabel = F.softmax(pred, dim=1)

                targetOneHotlabel = torch.argmax((targetOneHotlabel), dim=1).detach().cpu().numpy()  # 目标域

                eval.init(Ttargetpng, targetOneHotlabel)

                pb.update(1)

    # 评估结果
    eval.show()

    if ((cur_epoch + 1) % save_step == 0) | (cur_epoch == (max_epoch - 1)):
        torch.save({
            'featuremodel': student_eval.state_dict() if args.anchor else featuremodel.module.state_dict(),

            'domainmodel_aux': domainclassifer_aux.module.state_dict(),
            'domainmodel_main': domainclassifer_main.module.state_dict()
        }, f'{savepath}/{(cur_epoch + 1)}.pth')
    if (cur_epoch + 1) == total_epoch:
        torch.save({
            'featuremodel': student_eval.state_dict() if args.anchor else featuremodel.module.state_dict(),
            'domainmodel_aux': domainclassifer_aux.module.state_dict(),
            'domainmodel_main': domainclassifer_main.module.state_dict()
        }, f'{savepath}/last.pth')
