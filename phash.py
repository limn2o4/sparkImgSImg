import cv2
import numpy as np
import phash
import os
import csv

def get_pHash(img):
    
    
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img_gray,(64,64),interpolation=cv2.INTER_CUBIC)
    
    
    h,w = img_resize.shape[:2]
    img_float = np.zeros((h,w),np.float32)
    img_float[:h,:w] = img_resize
    img_dct = cv2.dct(cv2.dct(img_float))
    
    img_dct = cv2.resize(img_dct,(32,32),interpolation=cv2.INTER_CUBIC)

    num_list = img_dct.flatten()
    
    num_avg = sum(num_list)/len(num_list)
    bin_list = ['0' if i < num_avg else '1' for i in num_list]
    #print(''.join(bin_list))
    return ''.join(['%x' % int(''.join(bin_list[x:x+4]),2) for x in range(0,32*32,4)])



if __name__ == '__main__':
    
    
    root_path = '/home/limn2o4/Documents/jpg/'

    path_list = os.listdir(root_path)

    with open('img_data.csv','w') as f:
        csv_writter = csv.writer(f)
        for img_name in path_list :
            print(img_name)    
            img = cv2.imread(root_path+img_name)
            if img.all() == None:
                raise ValueError("wrong img data")
            csv_writter.writerow([img_name,get_pHash(img)])

            

        