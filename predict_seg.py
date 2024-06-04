import torch
import cv2
import torch.nn.functional as F
from PIL import Image
import numpy as np
from nets.LeNetSeg import LeNet

class ODPredict():
    def __init__(self,
                 path,
                 inc,
                 num_classes,
                 input_shape,
                 thresh ):

        self.net = LeNet(inc,num_classes)

        self.input_shape = input_shape

        self.thresh = thresh

        self.device = torch.device('cuda')  if torch.cuda.is_available() else torch.device('cpu')

        self.net.load_state_dict(torch.load(path,map_location=self.device))

        self.net = self.net.eval()


    @torch.no_grad()
    def predict(self,image):
        image = image.resize(self.input_shape)

        image = np.array(image,dtype=np.float32)

        old_image =  np.ascontiguousarray(image.copy())
        old_image = old_image[...,::-1]
        old_image = np.array(old_image,dtype=np.uint8)



        image = np.transpose(image,(2,0,1))

        x = torch.from_numpy(image).float()
        x = x.unsqueeze(dim = 0)

        seg = self.net(x)
        seg = F.sigmoid(seg)[0]


        seg = seg.contiguous().cpu().numpy()[0]

        out_seg = np.ones(self.input_shape + (3,))
        out_seg.fill(255)


        mask = seg <self.thresh

        out_seg[mask] = np.array([50, 120, 90])

        out_seg = out_seg.astype(np.uint8)




        cv2.imshow('im0',old_image)
        cv2.imshow('mask',out_seg)

        cv2.waitKey(0)



if __name__ == '__main__':
    OD = ODPredict(
        'logs_seg/last.pth',3,1,(300,300),0.5
    )
    while True:
        try:
            path = input("请输入图像：")
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print("error!")
        else:
            OD.predict(image)







