import glob as glob
import cv2 as cv
import numpy as np
import random
import os
from skimage import exposure



def create_files():
    labels = []
    images = []
    tupple = []

    ff = open("Test.csv", 'r')
    ff.readline()
    line = ff.readline()
    count = 0
    while(line):
        data = line.split(',')
        count+=1
        data[7] = data[7].rstrip()
        img = cv.imread(data[7], 1)
        img = cv.resize(img, (32, 32), interpolation = cv.INTER_AREA)
        img = exposure.equalize_adapthist(img, clip_limit=0.1)
        img = np.array(img)
        data[6] = data[6].rstrip()
        tupple.append((img,int(data[6])))
        print('{} {}'.format(data[6], data[7]))
        line = ff.readline()
     
    ff.close()
    print "Number of files: ", count
  
    random.shuffle(tupple)

    print "Data read and shuffled...."
    print "size: ", len(tupple)

    for obj in tupple:
        images.append(obj[0])
        labels.append(obj[1])

    print "images number: ", len(images)
    print "labels number: ", len(labels)
 
    np.savez('test_data.npz', train = images, train_labels = labels)


create_files()
