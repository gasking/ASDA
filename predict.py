import torch
import cv2
import torch.nn.functional as F
from PIL import Image
import numpy as np
from nets.Lenet import LeNet

class ODPredict():
    def __init__(self,
                 path,
                 inc,
                 num_classes,
                 input_shape):

        self.net = LeNet(inc,num_classes)

        self.input_shape = input_shape

        self.device = torch.device('cuda')  if torch.cuda.is_available() else torch.device('cpu')

        self.net.load_state_dict(torch.load(path,map_location=self.device))




    @torch.no_grad()
    def predict(self,image):

        image = np.array(image,dtype=np.float32)

        old_image =  np.ascontiguousarray(image.copy())
        old_image = old_image[...,::-1]
        old_image = np.array(old_image,dtype=np.uint8)



        image = np.transpose(image,(2,0,1))

        x = torch.from_numpy(image).float()
        x = x.unsqueeze(dim = 0)

        obj, box, cls = self.net(x)
        obj = F.sigmoid(obj)
        box = F.sigmoid(box)
        cls = F.softmax(cls,dim=1)

        obj = obj.contiguous().cpu().numpy()[0]
        box = box.contiguous().cpu().numpy()[0]
        #box = np.clip(box,a_min=0,a_max=self.input_shape[0])
        cls = cls.contiguous().cpu().numpy()[0]

        #print(obj,box,cls)

        if obj>0.5:#表示有目标
            box[[0,2]] *= self.input_shape[1]
            box[[1,3]] *= self.input_shape[0]
            c = np.argmax(cls,axis=-1)
            box = box.astype(np.int32)
            print(box)
            print(f"label:{c}")

            cv2.rectangle(old_image,(box[0],box[1]),
                          (box[2],box[3]),color=(100,80,60),thickness=2,lineType=cv2.LINE_AA)

            cv2.putText(old_image,str(c),(box[0],box[1]-15),fontFace=2,fontScale=1.,color=(255,60,56))
        cv2.imshow('im0',old_image)
        cv2.waitKey(0)



if __name__ == '__main__':
    OD = ODPredict(
        'logs_od/last.pth',3,21,(300,300)
    )
    while True:
        try:
            path = input("请输入图像：")
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print("error!")
        else:
            OD.predict(image)







