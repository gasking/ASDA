from glob import glob

import os




path = r'F:\JinKuang\CjdxOD\imgs'

file = r'train.txt'



w_file = open(file,'w+')

def get_imgs(path):
    cls = set()
    for f in glob(os.path.join(path,'*.jpg')):



        rindex = f.rindex('\\') + 1
        name = f[rindex:].split('.')[:-1][1:]


        w_file.write(f+' ')


        _cls = int(name[-1])
        cls.add(_cls)

        for ind in name:
            w_file.write(ind+' ')
        w_file.write('\n')
    print(cls)







if __name__ == '__main__':
    get_imgs(path)