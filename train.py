import torch
import numpy as np
import torch.optim as optim
import random
import shutil
import os
from torch.utils.data import DataLoader
from datasets.dataloader import ODdatasets
from utils.criterion import build_loss
from tqdm import tqdm
from nets.Lenet import LeNet


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


#---------------------------------------------------#
#           simlpe OD
#---------------------------------------------------#
if __name__ == '__main__':

    seed = 15
    Batch_size = 128
    Max_epoch = 200

    save_dir = r'logs'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.mkdir(save_dir)

    save = 100
    lr = 1e-4
    decay = 1e-4 #衰减因子
    input_shape = (300,300)
    num_class = 21 #分类
    train_file = r'train.txt'

    fix_seed(seed) #确保结果唯一

    train_data = ODdatasets(train_file,input_shape)
    train_data = DataLoader(train_data,batch_size = Batch_size,shuffle = True,num_workers = 4)
    #-------------------------------------------------#
    #  模型 不进行kaiming初始化
    #  训练进行初始化
    #-------------------------------------------------#
    model = LeNet(inc = 3,num_classes = num_class)

    #-------------------------------------------------#
    #  冻结BN层
    #------------------------------------------------#
    #model.freeze_bn()


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    #------------------------------------------------------#
    #           优化器
    #------------------------------------------------------#
    optimizer = optim.Adam(model.parameters(),lr,weight_decay = decay)
    model = model.to(device)
    #-------------------------------------------------------#
    #                   数据读取
    #-------------------------------------------------------#
    for epoch in range(Max_epoch):
     #--------------------------------------#
     # 开启训练模式
     #-------------------------------------#
     model = model.train()
     with tqdm(desc = f'Epoch {epoch + 1}/{Max_epoch}',colour = '#378CE7',total = len(train_data),
                           postfix = dict,mininterval = 0.3) as pb:
      for ind,batch in enumerate(train_data):
       with torch.no_grad():
        images, boxes, objs, clses = batch
        images = images.to(device)
        boxes = boxes.to(device)
        objs = objs.to(device)
        clses = clses.to(device)

       optimizer.zero_grad() #清空梯度
       obj, box, cls = model(images)
       loss = build_loss()([obj,box,cls],[objs,boxes,clses])

       objloss = loss['obj loss']
       boxloss = loss['box loss']
       clsloss = loss['cls loss']

       Tloss = (objloss + boxloss + clsloss)

       Tloss.backward()
       optimizer.step()



       pb.set_postfix(**{
           'objloss':objloss.item(),
           'boxloss':boxloss.item(),
           'clsloss':clsloss.item()
       })

       pb.update(1)

     if (epoch + 1)%save == 0:
         torch.save(model.state_dict(), os.path.join(save_dir,f'{epoch+1}.pth'))

    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir,f'last.pth'))











