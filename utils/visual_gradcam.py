import os
import sys
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 获取当前工作空间
path = os.getcwd()
sys.path.insert(0,path)

from nets.criterion import Dice_loss
from nets.deeplabv2 import get_deeplab_v2
from utils.tools import resize,_norm



class Grad_CAM(object):
    def __init__(self, modules = None,
                 pth_path = None,
                 input_shape = (256,256)):

        assert pth_path != None, '请输入权重...'

        self.net = get_deeplab_v2(2,True,'resnet50')

        self.input_shape = input_shape
        self.interp = nn.Upsample(size=(self.input_shape[0], self.input_shape[1]), mode='bilinear',
                                  align_corners=True)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = torch.load(pth_path, map_location = self.device)['featuremodel']

        self.net.load_state_dict(model)

        self.net = self.net.train()

        getattr(self.net, modules).register_forward_hook(self.__register_forward_hook)
        getattr(self.net, modules).register_backward_hook(self.__register_backward_hook)

        self.modules = modules

        # 保存梯度信息
        self.input_grad = [ ]
        # 收集feature map
        self.output_grad = [ ]

    def __register_backward_hook(self,
                                 module,
                                 grad_in,

                                 grad_out):

        #print(len(grad_in), len(grad_out))

        self.input_grad.append(grad_out[ 0 ].data.cpu().numpy())

    def __register_forward_hook(self,
                                module,
                                grad_in,
                                grad_out):
        self.output_grad.append(grad_out)

    def _get_cam(self, feature_map, grads):
        # -------------------------------------------------------#
        #                  feature_map: [c,h,w]
        #                  grads: [c,h,w]
        #                  return [h,w]
        # -------------------------------------------------------#
        cam = np.zeros(feature_map.shape[ 2: ], dtype = np.float32)
        alpha = np.mean(grads, axis = (2,3))

        for ind, c in enumerate(alpha):
            cam += c[ind] * feature_map[0][ ind ].detach().numpy()

        heatmap = np.maximum(cam, 0)

        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-8)



        return heatmap

    def show_cam_to_image(self, image,
                          heatmap,
                          is_show = False,
                          is_write = False,
                          name = None):
        #heatmap = np.transpose(heatmap,(1,2,0))

        # resize回到原图
        heatmap = cv2.resize(heatmap, self.input_shape)


        # 用来控制
        #heatmap = heatmap[...,0]

        heatmap = np.array(heatmap * 255 , np.uint8)
        heatmap = cv2.applyColorMap(heatmap,
                           cv2.COLORMAP_JET)

        heatmap = np.float32(heatmap) / 255.
        image = np.transpose(image,(1,2,0))

        img =  0.4 * heatmap +  0.6 * np.array(image)

        # --------------------------------------------#
        #               clip pix value
        # --------------------------------------------#

        img = (img  - np.min(img)) / np.max(img)

        img = np.uint8(img * 255)

        print('*'*20+'\t'+'Grad Cam Finished\t'+'*'*20+'\t')

        if is_show:
            plt.imshow(img[ :, :, ::-1 ])
            plt.show()
        if is_write:
            cv2.imwrite(f'{name}_cam.jpg', img[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,100])


    def __process(self,image,lettex = True):
        #--------------------------------------------------------------#
        #                   是否黑边填充
        #--------------------------------------------------------------#
        if lettex:
            new_image, neww, newh, dx, dy, w, h = resize(image, self.input_shape)

            img = np.array(new_image, dtype=np.float32)

            img = _norm(img)
            img = np.transpose(img, (2, 0, 1))

            return img, neww, newh, dx, dy, w, h

        else:
            image = image.resize(self.input_shape)
            image = np.array(image, dtype=np.float32) / 255.
            image = np.transpose(image, (2, 0, 1))
            return image,



    def forward(self, image,
                is_show = False,
                is_write = False,
                name = None,
                lettex = True):


        image,*_ = self.__process(image,lettex)

        # 网络模型输入
        x = torch.from_numpy(image).float()
        x = x.unsqueeze(dim = 0)

        self.net.zero_grad() # 清空梯度
        output_aux,output_main = self.net(x)
        output_aux = self.interp(output_aux)
        output_main = self.interp(output_main)
        # ---------------------------------#
        #            损失函数定义
        # ---------------------------------#

        b,c,h,w = output_main.shape
        #label = torch.ones((b,c,h,w),requires_grad = True).float()
        #print(output_main.shape,label.shape)

        unmasked = torch.argmax(torch.softmax(output_main,dim=1),dim=1).long()
        #unmasked = torch.unsqueeze(unmasked,dim=1)

        label = unmasked.float()

        #----------------------------------------------------------#
        #                 损失函数和训练时保持一致
        #----------------------------------------------------------#
        loss = nn.CrossEntropyLoss()(output_main,unmasked)

        loss.backward()


        #generate CAM
        grad = self.input_grad[0]

        fmap = self.output_grad[0]

        cam = self._get_cam(fmap,grad)

        # show
        image = np.float32(image)


        self.show_cam_to_image(image,cam,is_show ,
                               is_write,name)

        self.input_grad.clear()
        self.output_grad.clear()





if __name__ == '__main__':
   #----------------------------------------------------------------------#
   #                           parameter
   #----------------------------------------------------------------------#
   path = r'D:\JinKuang\TDA\HRC_WHU-copy\images\urban_24_2.jpg'
   is_show = True
   is_write = False
   lettex = True
   model_path = r'D:\JinKuang\UDS\Pth\entropy\step3_Clouds26_WHU\last.pth'



   parse = argparse.ArgumentParser(description = 'Generate Grad_CAM Image....')
   parse.add_argument('--image_path','--i',type = str,
                      default = path,help = 'input image path...')
   parse.add_argument('--model_path','--m',type = str,
                      default = model_path,help = 'inference model...')

   parse.add_argument('--show','--s',action = 'store_true',
                      help = 'whether visual image...')

   parse.add_argument('--write','--w',action = 'store_true',default = is_write,
                      help = 'whether write to image')

   parse.add_argument('--lettex','--l',action = 'store_true',default = lettex,
                      help = 'whether use 0 to padding image...')

   args = parse.parse_args()



   path = args.image_path

   image = Image.open(path).convert('RGB')


   #----------------------------------------------------------------#
   #                        网络层
   #---------------------------------------------------------------#
   cam = Grad_CAM('layer6',args.model_path)

   cam.forward(image,is_show = args.show,name = path.split('.')[0],lettex=args.lettex,is_write=args.write)