from nets.deeplabv2 import get_deeplab_v2
#------------------------------------------#
#               特征提取器
#------------------------------------------#
import cv2
import os
import torch
import argparse
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F



class Grad_CAM(object):
    def __init__(self, modules = None,
                 pth_path = None,
                 num_classes = 2):

        assert pth_path != None, '请输入权重...'
        

        
        self.net = get_deeplab_v2(num_classes = 2,multi_level = True,
                                   backbone = 'resnet50')
      
        self.num_classes = num_classes

        self.input_shape = (256,256) #direct resize

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = torch.load(pth_path, map_location = self.device)

        self.net.load_state_dict(model[ 'featuremodel' ])

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

        self.input_grad.append(grad_out[ 0 ].detach().data.cpu().numpy())

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
        heatmap = cv2.resize(heatmap,self.input_shape)

        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-8)

        return heatmap

    def show_cam_to_image(self, image,
                          heatmap,
                          is_show = False,
                          is_write = False,
                          name = None):
        #heatmap = np.transpose(heatmap,(1,2,0))

        heatmap = np.array(heatmap * 255 , np.uint8)
       
        heatmap = cv2.applyColorMap(heatmap,
                           cv2.COLORMAP_JET)

        heatmap = np.float32(heatmap) / 255.
        image = np.transpose(image,(1,2,0))

        img =  0.4 * heatmap +  0.6 *np.array(image)

        # --------------------------------------------#
        #               clip pix value
        # --------------------------------------------#

        img = (img  - np.min(img)) / np.max(img)

        img = np.uint8(img * 255)

        if is_show:
            plt.imshow(img[ :, :, ::-1 ])
            plt.show()
        if is_write:
            cv2.imwrite(f'{args.save_path}\\{args.method}\\{name}_cam.jpg', img,[cv2.IMWRITE_JPEG_QUALITY,100])

    def forward(self, image,
                label,
                is_show = False,
                is_write = False,
                name = None):

        image = image.resize(self.input_shape)

        image.save(f'{args.save_path}\{args.method}\{name}.jpg')

       
        image = np.array(image, dtype = np.float32) / 255.
        image = np.transpose(image, (2, 0, 1))


        # 网络模型输入
        x = torch.from_numpy(image).float()
        x = x.unsqueeze(dim = 0)

        self.net.zero_grad() # 清空梯度
        output_aux,output_main = self.net(x)
        # ---------------------------------#
        #            损失函数定义
        # ---------------------------------#

        b,c,h,w = output_main.shape
        #label = torch.ones((b,c,h,w),requires_grad = True).float()
        #print(output_main.shape,label.shape)



        #--------------------------------------------#
        #                   取值
        #--------------------------------------------#
       

        # TODO
        pre_main  =  F.interpolate(output_main,self.input_shape,mode='bilinear',
                                align_corners=True)
        pre_aux = F.interpolate(output_aux,self.input_shape,mode='bilinear',
                                align_corners=True)
        
        output_main = F.softmax(output_main,dim = 1) 
        output_aux = F.softmax(output_aux,dim = 1)

        unmask_main = torch.argmax(pre_main,dim = 1)
        unmask_aux = torch.argmax(pre_main,dim = 1)
        unmask = (unmask_aux | unmask_main)

        label = torch.argmax(output_main,dim = 1)
       

        
        with torch.no_grad():
            self.vis(unmask.detach().cpu().numpy(),name)

            self.vis_entropy(pre_main.detach().cpu(),name)




        #----------------------------------------------------------#
        #                 损失函数和训练时保持一致
        #----------------------------------------------------------#
        loss = nn.CrossEntropyLoss()(output_main,label)

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

    def vis(self,mask,name):
    #------------------------------#
    #           可视化
    #------------------------------#
     
     mask = mask[0]
    
     platte = [176, 235, 180, 255, 255, 255, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153,
                              250,
                              170, 30,
                              220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
                              0,
                              0, 70,
                              0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

     for i in range(256 *3 - len(platte)):
         platte.append(0)
    
     platte = np.array(platte,dtype = np.uint8)

     mask = np.array(mask,dtype = np.uint8)

     mask = Image.fromarray(mask).convert('P')

     

     mask.putpalette(platte)

     

     #mask.show()

     mask.save(f'{args.save_path}\\{args.method}\\{name}_vis.png')

    def prob2entropy(self,prob):


        b,c,h,w = prob.shape

        x = -torch.mul(prob,torch.log2(prob + 1e-20)) / np.log2(c)

        return x
  
    def vis_entropy(self,
                    prob,
                    name):
    #----------------------------------------#
    #               熵值结果
    #----------------------------------------#
        entropy = self.prob2entropy(F.softmax(prob,dim=1)) #

        entropy = entropy[0].detach().cpu().numpy()
      

        

        entropy_background = np.array(entropy[0, ...] * 255, dtype=np.uint8)
        entropy_foreground = np.array(entropy[1, ...] * 255, dtype=np.uint8)

        heatmap = (0.5 * entropy_background + 0.5 * entropy_foreground)

        #heatmap = (heatmap - np.max(heatmap)) / (np.min(heatmap))

        heatmap = np.array(heatmap , np.uint8)
        entropy = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        

    
        #cv2.imshow('entropy',entropy)
      
        cv2.imwrite(f'{args.save_path}\{args.method}\{name}_entropy.png',entropy)
        #cv2.waitKey(0)






if __name__ == '__main__':


   args = argparse.ArgumentParser(description = 'inference....')

   args.add_argument('--image_path','--i',default = '',help = 'inference image path...')
   args.add_argument('--model_path','--m',default = '',help = 'inference model path...')
   args.add_argument('--vis_layer','--v',default = 'layer6',help = 'vis layer...')
   args.add_argument('--is_show',action = 'store_true',help = 'Using vis image...')
   args.add_argument('--is_write',action = 'store_true',help = 'Using save vis...')
   args.add_argument('--save_path',default='vis',type = str,help = 'save inferece result path(source domain ->  target domain)...',required = True)
   args.add_argument('--method',type = str,required = True)

   #------------------------------------------------------------------------------------------#
   #                                文件夹推理
   #------------------------------------------------------------------------------------------#
   args.add_argument('--dir',type = str,default = '')
   args = args.parse_args()

   path = args.image_path

   model_path = args.model_path
   

   #----------------------------------------------------------#
   #                    
   #----------------------------------------------------------#
   from glob import glob

   images = glob(f'{args.dir}\*.jpg')

   if not os.path.exists(args.save_path):
      os.mkdir(args.save_path)


   if not os.path.exists(f'{args.save_path}\\{args.method}'):
      os.mkdir(f'{args.save_path}\\{args.method}')

   for path in images:

    image = Image.open(path).convert('RGB')

    cam = Grad_CAM(args.vis_layer,model_path)

    

    cam.forward(image,None,is_show = args.is_show,is_write = args.is_write,name = os.path.basename(path).split('.')[0])