from tqdm import tqdm
import torch
import numpy as np
from nets.criterion import Dice_loss,score
import torch.nn as nn
from utils.tools import get_lr
from utils.tools import prob2entropy
import torch.nn.functional as F
from utils.config import Config


def epoch_fit(cur_epoch,total_epoch,save_step,
              model,optimizer,traindataloader,valdataloader,
              device,logger,args,save_path,
              eval,local_rank): #评估函数是对测试集


    featuremodel, domainclassifer_aux,domainclassifer_main = model
    featuremodel = featuremodel.to(device)

    domainclassifer_aux = domainclassifer_aux.to(device)

    domainclassifer_main = domainclassifer_main.to(device)

    if torch.cuda.device_count() > 1:
        #-----------------------------------------------------#
        #                  训练的BS要能被显卡数整除
        #-----------------------------------------------------#
        print(f'GPUS Numbers :{torch.cuda.device_count()}')


        #-------------------------------------------------------#
        #                         设置多卡训练
        #-------------------------------------------------------#
        featuremodel = nn.DataParallel(featuremodel)
        
        domainclassifer_aux = nn.DataParallel(domainclassifer_aux)
        
        domainclassifer_main = nn.DataParallel(domainclassifer_main)

        # featuremodel = nn.parallel.DistributedDataParallel(
        #     featuremodel,device_ids = [local_rank],
        #     output_device = local_rank
        # )
        # domainclassifer_aux = nn.parallel.DistributedDataParallel(
        #     domainclassifer_aux, device_ids = [ local_rank ],
        #     output_device = local_rank
        # )
        # domainclassifer_main = nn.parallel.DistributedDataParallel(
        #     domainclassifer_main, device_ids = [ local_rank ],
        #     output_device = local_rank
        # )


    interp = nn.Upsample(size = (Config['input_shape'][0],Config['input_shape'][1]),mode = 'bilinear',align_corners = True)

    # 优化器
    featureoptimizer, domainclassiferoptimizer_aux,domainclassiferoptimizer_main = optimizer



    # 损失函数定义
    criteror = nn.BCEWithLogitsLoss()

    with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total = len(traindataloader),mininterval = 0.3,postfix = dict,colour = '#6DB9EF') as pb:

        featuremodel = featuremodel.train()

        domainclassifer_aux = domainclassifer_aux.train()

        domainclassifer_main = domainclassifer_main.train()

        #-------------------------------------#
        # 训练损失计算
        #-------------------------------------#
        total_loss = 0.
        total_cls_loss = 0.
        total_domain = 0.


        # 源域
        for ind,batch in enumerate(traindataloader):
          sourceimage,sourcelabel,sourcepng,targetimage,targetlabel,targetpng = batch
          with torch.no_grad():
              # 源域
              sourceimage = sourceimage.to(device)
              sourcelabel = sourcelabel.to(device)
              sourcepng = sourcepng.to(device)

              # 目标域
              targetimage = targetimage.to(device)
              targetlabel = targetlabel.to(device)
              targetpng = targetpng.to(device)

          #------------------------------------------#
          #                冻结判别器
          #------------------------------------------#
          for para in domainclassifer_aux.parameters():
              para.requires_grad = False

          for para in domainclassifer_main.parameters():
              para.requires_grad = False

          # 首先对优化器中的梯度进行清空
          featureoptimizer.zero_grad()
          domainclassiferoptimizer_aux.zero_grad()
          domainclassiferoptimizer_main.zero_grad()

          #---------------------------------#
          #        源域特征提取 + 分割模型
          #---------------------------------#
          segaux,seg  = featuremodel(sourceimage)
          segaux = interp(segaux)
          seg = interp(seg)


          #------------------------------------------------------------------------------------#
          #                                     TODO 源域辅助分支分割损失
          #------------------------------------------------------------------------------------#
          lossauxseg = Dice_loss(segaux,sourcelabel) + nn.CrossEntropyLoss()(segaux,sourcepng)

          # ------------------------------------------------------------------------------------#
          #                                     TODO 源域主分支分割损失
          # ------------------------------------------------------------------------------------#
          lossseg = Dice_loss(seg,sourcelabel) + nn.CrossEntropyLoss()(seg,sourcepng)

          Tsegloss = (0.1 * lossauxseg + 1. * lossseg)

          #logger.info(f'source_domain seg loss: {Tsegloss.item()}\n')

          Tsegloss.backward()

          total_cls_loss += Tsegloss.item()

          #------------------------------------------------------------------#
          #           adversarial training to fool the discriminator
          #------------------------------------------------------------------#
          paux, pseg = featuremodel(targetimage)

          paux = interp(paux)
          pseg = interp(pseg)

          d_taraux = domainclassifer_aux(prob2entropy(F.softmax(paux)))

          d_tarmain = domainclassifer_main(prob2entropy(F.softmax(pseg)))

          real = torch.zeros_like(d_taraux, requires_grad=True)
          fake = torch.ones_like(d_taraux, requires_grad=True)

          d_domain_aux_loss = criteror(d_taraux,real)
          d_domain_main_loss = criteror(d_tarmain,real)



          Tdomain_loss = (0.0001 * d_domain_main_loss + 0.0002 * d_domain_aux_loss)

          Tdomain_loss.backward()






          # 解冻领域判别器
          for para in domainclassifer_aux.parameters():
              para.requires_grad = True

          for para in domainclassifer_main.parameters():
              para.requires_grad = True



          #------------------------------------------------------------------------#
          #                           only source domain
          #------------------------------------------------------------------------#
          d_source_aux = domainclassifer_aux(prob2entropy(F.softmax(segaux.detach())))
          DS_aux = criteror(d_source_aux, real) / 2.
          DS_aux.backward()


          d_source_main = domainclassifer_main(prob2entropy(F.softmax(seg.detach())))
          DS_main = criteror(d_source_main,real) / 2.
          DS_main.backward()



          #--------------------------------------------------------------------#
          #                        目标域
          #--------------------------------------------------------------------#
          d_target_aux = domainclassifer_aux(prob2entropy(F.softmax(paux.detach())))
          DT_aux = criteror(d_target_aux, fake) / 2.
          DT_aux.backward()

          d_target_main = domainclassifer_main(prob2entropy(F.softmax(pseg.detach())))
          DT_main = criteror(d_target_main, fake) / 2.
          DT_main.backward()







          #logger.info(f'domain loss: {(DT_main.item() + DT_aux.item()+ DS_main.item() + DS_aux.item()+ Tdomain_loss.item())}\n')

          total_domain += (DT_main.item() + DT_aux.item()+
                           DS_main.item() + DS_aux.item()+ Tdomain_loss.item()
                           )




          # if ((ind + 1 ) % args.accumulate) == 0:
          #
          #     featureoptimizer.step()
          #
          #     featureoptimizer.zero_grad()
          #
          #     domainclassiferoptimizer.step()
          #
          #     domainclassiferoptimizer.zero_grad()

          featureoptimizer.step()

          domainclassiferoptimizer_aux.step()

          domainclassiferoptimizer_main.step()


          total_loss += (total_domain + total_cls_loss)

          pb.set_postfix(**{
              'total_loss': total_loss/(ind + 1),
              'total_cls_loss':total_cls_loss/(ind + 1),
              'total_domain_loss':total_domain/(ind + 1)
          })


          pb.update(1)


    #----------------------------------------------------------------------------#
    #                       BN
    #----------------------------------------------------------------------------#
    featuremodel = featuremodel.eval()
    domainclassifer_aux = domainclassifer_aux.eval()
    domainclassifer_main = domainclassifer_main.eval()


    #Evaluatation
    with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total = len(valdataloader),mininterval = 0.3,postfix = dict,colour = '#7E89EF') as pb:

     with torch.no_grad():
        # 目标域
        for ind,batch in enumerate(valdataloader):
            _, _, _, Ttargetimage, _, Ttargetpng, Ttargetname = batch

            Ttargetimage = Ttargetimage.to(device)

            Ttargetpng = Ttargetpng.to(device).detach().cpu().numpy()

            pred_seg, pred = featuremodel(Ttargetimage)

            pred_seg = interp(pred_seg)
            pred = interp(pred)

            targetOneHotlabel = F.softmax(pred, dim=1)

            targetOneHotlabel = torch.argmax((targetOneHotlabel), dim=1).detach().cpu().numpy()  # 目标域


            eval.init(Ttargetpng,targetOneHotlabel)

            pb.update(1)



    # 评估结果
    eval.show()




    if ((cur_epoch + 1)%save_step == 0) :
            torch.save({
                   'featuremodel':featuremodel.module.state_dict(),
                   'domainmodel_aux':domainclassifer_aux.module.state_dict(),
                   'domainmodel_main':domainclassifer_main.module.state_dict()
            },f'{save_path}/{(cur_epoch + 1)}.pth')
    if (cur_epoch + 1) == total_epoch:
     torch.save({
                   'featuremodel':featuremodel.module.state_dict(),
                   'domainmodel_aux': domainclassifer_aux.module.state_dict(),
                   'domainmodel_main': domainclassifer_main.module.state_dict()
            },f'{save_path}/last.pth')


    # if ((cur_epoch + 1)%save_step == 0) :
    #         torch.save({
    #                'featuremodel':featuremodel.state_dict(),
    #                'domainmodel_aux':domainclassifer_aux.state_dict(),
    #                'domainmodel_main':domainclassifer_main.state_dict()
    #         },f'{save_path}/{(cur_epoch + 1)}.pth')
    # if (cur_epoch + 1) == total_epoch:
    #  torch.save({
    #                'featuremodel':featuremodel.state_dict(),
    #                'domainmodel_aux': domainclassifer_aux.state_dict(),
    #                'domainmodel_main': domainclassifer_main.state_dict()
    #         },f'{save_path}/last.pth')
