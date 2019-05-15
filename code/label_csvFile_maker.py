import numpy as np
import cv2
import csv

def catagoryCount(label):
    no_0 = 0
    no_1 = 0
    no_2 = 0
    no_3 = 0
    no_4 = 0
    no_5 = 0
    no_6 = 0
    no_7 = 0
    no_8 = 0
    no_9 = 0

    for i in range(len(label)):
        # print(i, ' 번째 이미지 : ')
        # print(int(label[i]))

        if (int(label[i]) == 0):
            no_0 += 1
        elif (int(label[i]) == 1):
            no_1 += 1
        elif (int(label[i]) == 2):
            no_2 += 1
        elif (int(label[i]) == 3):
            no_3 += 1
        elif (int(label[i]) == 4):
            no_4 += 1
        elif (int(label[i]) == 5):
            no_5 += 1
        elif (int(label[i]) == 6):
            no_6 += 1
        elif (int(label[i]) == 7):
            no_7 += 1
        elif (int(label[i]) == 8):
            no_8 += 1
        elif (int(label[i]) == 9):
            no_9 += 1

    print('=' * 15)
    print('0', no_0)
    print('1', no_1)
    print('2', no_2)
    print('3', no_3)
    print('4', no_4)
    print('5', no_5)
    print('6', no_6)
    print('7', no_7)
    print('8', no_8)
    print('9', no_9)
    print('=' * 15)


temp = []
b= []


def label_csvfil():

    for i in range(421):  # 207
        image = '../total_image_data/total_image_data_ ({}).jpg'

        # image colors
        img = cv2.imread(image.format(i+1), cv2.IMREAD_GRAYSCALE)  # gray image reader
    #    img = cv2.imread(image.format(i+1), cv2.IMREAD_COLOR)  # RGB image reader to CV

        np.ravel(temp.append(img))
 #       temp.append(img)

    all_images = np.stack(temp)  # length = 32
    print("all_",all_images)


    temp2 = []
    for i in range(len(all_images)):
        a = np.ravel(all_images[i])
        temp2.append(a)


    label = np.loadtxt('../total_image_label.csv', delimiter=',', dtype=np.int, skiprows=1, usecols=[2])  # dtype=np.int

  #  catagoryCount(label)


    with open('../datafile2.csv', 'a') as f:
        for i in range(len(label)):
            d = '{},'.format(label[i])
            f.write(d)
            for j in range(len(temp2[i])):
                d = '{},'.format(temp2[i][j])
                f.write(d)
            f.write('\n')
        print('finish..')

    catagoryCount(label)



label_csvfil()
