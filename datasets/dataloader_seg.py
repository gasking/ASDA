import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import cv2


#-----------------------------------------#
#           读取数据的框架
#-----------------------------------------#
class ODdatasets(Dataset):
    def __init__(self,
                 train_file,
                 input_shape,
                 num_classes):
        #-----------------------------------#
        #            训练文件
        #           输入图像大小
        #-----------------------------------#
        super(ODdatasets, self).__init__()


        self.input_shape = input_shape

        self.content = [ ]  # 存放训练信息

        self.num_classes = num_classes #类别信息

        self.f = open(train_file,'r').readlines()

        #--------------------------------------------#
        # path seg_path
        #--------------------------------------------#
        for f in self.f:
            name = f.strip().split(' ')
            jpg = name[0]
            seg = name[1]

            self.content.append(
                [jpg,seg]
            )

    def __len__(self):

        return len(self.content)

    def __getitem__(self, ind):
    #-------------------------------------#
    #           读取图像
    #-------------------------------------#
      path = self.content[ind][0]
      seg = (self.content[ind][1])

      image = Image.open(path).convert('RGB') #转为RGB
      mask = Image.open(seg).convert('L')

      #--------------------------------------------#
      #                resize图像
      #--------------------------------------------#
      image = image.resize(self.input_shape)
      mask = mask.resize(self.input_shape)

      mask = np.array(mask,np.long)



      #-----------------------------------------#
      #                编码
      #             二分类任务
      #-----------------------------------------#
      T_mask = np.zeros((self.input_shape[0],self.input_shape[1],
                         self.num_classes),np.float32)

      T_mask[mask == 1] = 1 #前景





      #-------------------------------------------------------#
      #               归一化
      #-------------------------------------------------------#
      image = np.array(image) / 255.
      image = np.transpose(image,(2,0,1))
      T_mask = np.transpose(T_mask,(2,0,1))


      return image,mask,T_mask




def collate(batch):
    images,masks,T_masks = [],[],[]

    for img,mask,T in batch:
        images.append(img)
        masks.append(mask)
        T_masks.append(T)



    images = np.array(images,dtype = np.float32)
    images = torch.from_numpy(images).float()


    masks = np.array(masks,dtype = np.long)
    masks = torch.from_numpy(masks).long()

    T_masks = np.array(T_masks,dtype = np.float32)
    T_masks = torch.from_numpy(T_masks).float()




    return images,masks,T_masks





if __name__ == '__main__':

    train_data = ODdatasets(train_file = '../train_seg.txt',
                            input_shape = [300,300],num_classes = 1)

    train = DataLoader(train_data,batch_size = 1,num_workers = 4,shuffle = True)


    train = iter(train)


    for ind,batch in enumerate(train):
        images,masks,T_masks = batch

        print(images.shape,masks.shape,T_masks.shape)

        T_mask = T_masks[0].numpy()
        T_mask = np.transpose(T_mask,(1,2,0))

        #---------------------------------------------------#
        #                   可视化
        #---------------------------------------------------#

        img = images.contiguous().numpy()[0]
        img = np.transpose(img,(1,2,0)) * 255
        img = img[ ..., ::-1 ]


        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        cv2.imshow('mask',T_mask * 255)
        cv2.imshow('im0', img)

        cv2.waitKey(0)


