from glob import glob

import os

path = r'D:\JinKuang\Cloud\HRC_WHU\images'

file = r'train_seg.txt'

w_file = open(file, 'w+')


def get_imgs(path):
    for f in glob(os.path.join(path, '*.jpg')):
        seg = f.replace('images', 'labels').replace('.jpg', '.png')

        w_file.write(f + ' ' + seg + '\n')


if __name__ == '__main__':
    get_imgs(path)