import cv2 as cv
import numpy as np
import glob
import shutil
import os
from time import sleep

def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        sleep(2)
    os.mkdir(dirname)

def dirResize(src, dst):
    myfiles = glob.glob(src + '/*.jpg')

    if not myfiles:
        print('在目錄', src, '中找不到任何 JPEG 圖片。')
        return
    
    emptydir(dst)
    print(src + '資料夾')
    print('開始轉換圖片尺寸')
    for i, f in enumerate(myfiles):
        img = cv.imread(f)
        img_new = cv.resize(img, (300, 225), interpolation=cv.INTER_LINEAR)

        outname = "resizejpg{:0>3d}.jpg".format(i+1)
        cv.imwrite(os.path.join(dst, outname), img_new)
    print('轉換圖片尺寸完成\n')

dirResize('pen_sr', 'penPlate')
dirResize('real_sr', 'realPlate')
