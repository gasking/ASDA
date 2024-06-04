import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import cv2

def resize(image,input_shape,box):
#--------------------------------------#
#           根据逻辑去写
#--------------------------------------#
    w,h = image.size

    scalew = w / input_shape[1]
    scaleh = h / input_shape[0]

    new_image = image.resize(input_shape) #resize

    box = np.array(box,dtype = np.float32)


    box[[0,2]] = box[[0,2]] * scalew

    box[[1,3]] = box[[1,3]] * scaleh


    box[[0,2]] = box[[0,2]] / input_shape[1]
    box[[1,3]] = box[[1,3]] / input_shape[0]




    return new_image,box



#-----------------------------------------#
#           读取数据的框架
#-----------------------------------------#
class ODdatasets(Dataset):
    def __init__(self,
                 train_file,
                 input_shape):
        #-----------------------------------#
        #            训练文件
        #           输入图像大小
        #-----------------------------------#
        super(ODdatasets, self).__init__()


        self.input_shape = input_shape

        self.content = [ ]  # 存放训练信息


        self.f = open(train_file,'r').readlines()

        #--------------------------------------------#
        # path box obj cls
        #--------------------------------------------#
        for f in self.f:
            name = f.strip().split(' ')
            path = name[0]
            obj = name[1]
            boxes = name[2:6]
            cls = name[-1]

            self.content.append(
                [path,obj,boxes,cls]
            )

    def __len__(self):

        return len(self.content)

    def __getitem__(self, ind):
    #-------------------------------------#
    #           读取图像
    #-------------------------------------#
      path = self.content[ind][0]
      obj = int(self.content[ind][1])
      cls = int(self.content[ind][-1])
      boxes = np.array(list(map(lambda x:int(x),self.content[ind][2])))

      image = Image.open(path).convert('RGB') #转为RGB

      image,boxes = resize(image,self.input_shape,boxes)

      #-------------------------------------------------------#
      #               归一化
      #-------------------------------------------------------#
      image = np.array(image) / 255.
      image = np.transpose(image,(2,0,1))


      return image,boxes,obj,cls




def collate(batch):
    images,boxes,objs,clses = [],[],[],[]

    for img,box,obj,cls in batch:
        images.append(img)
        boxes.append(box)
        objs.append(obj)
        clses.append(cls)


    images = np.array(images,dtype = np.float32)
    images = torch.from_numpy(images).float()


    boxes = np.array(boxes,dtype = np.float32)
    boxes = torch.from_numpy(boxes).float()

    objs = np.array(objs,dtype = np.long)
    objs = torch.from_numpy(objs).long()

    clses = np.array(clses,dtype = np.long)
    clses = torch.from_numpy(clses).long()


    return images,boxes,objs,clses





if __name__ == '__main__':

    train_data = ODdatasets(train_file = '../train.txt',
                            input_shape = [300,300])

    train = DataLoader(train_data,batch_size = 1,num_workers = 4,shuffle = True)


    train = iter(train)


    for ind,batch in enumerate(train):
        images,boxes,objs,clses = batch

        print(images.shape,boxes.shape,objs.shape,clses.shape)

        #---------------------------------------------------#
        #                   可视化
        #---------------------------------------------------#

        img = images.contiguous().numpy()[0]
        img = np.transpose(img,(1,2,0)) * 255
        img = img[ ..., ::-1 ]
        box = boxes.numpy()[0] * 300 #解码

        box = box.astype(np.int32)
        print(box)

        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)




        cv2.rectangle(img,pt1 = (int(box[0]),int(box[1])),pt2 = (int(box[2]),int(box[3])),color = (255,120,0),thickness = 2,lineType = cv2.LINE_AA)

        cv2.imshow('im0', img)

        cv2.waitKey(0)


