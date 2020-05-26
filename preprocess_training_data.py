import glob as glob
import cv2
import numpy as np
import random
import os
from skimage import exposure



def create_files():
    labels = []
    images = []
    tupple = []
    path = []
    for x in range(0, 43):
        path.append("Train/"+str(x) +"/")
    print len(path)
    ii = 0
    count = 0
    for single in range(len(path)):
        folder =  glob.glob(str(path[single] + '*.png'))
        for file in folder:
            if os.path.exists(file):
                count +=1
                img = cv2.imread(file, 1)
                img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
                img = exposure.equalize_adapthist(img, clip_limit=0.1)
                img = np.array(img)
                tupple.append((img, single))
                print('{}'.format(count))

            else:
                print "Error: ", path
        ii += 1

    print "Number of folders: ", ii
    random.shuffle(tupple)

    print "Data read and shuffled...."
    print "size: ", len(tupple)

    for obj in tupple:
        images.append(obj[0])
        labels.append(obj[1])

    print "images number: ", len(images)
    print "labels number: ", len(labels)

    np.savez('training_data.npz', train = images, train_labels = labels)



create_files()
