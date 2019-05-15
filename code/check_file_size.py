"""
csv file 크기 확인,
갯수 확인

"""
import csv
import numpy as np


def classification_count(label):
    totla = 0
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
       # print(i,' 번째 이미지 : ')
       # print(int(label[i]))
        totla += 1
        if( int(label[i]) == 0):
            no_0 += 1
        elif( int(label[i]) == 1):
            no_1 += 1
        elif (int(label[i]) == 2):
            no_2 += 1
        elif (int(label[i]) == 3):
            no_3 += 1
        elif( int(label[i]) == 4):
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


    print('총 : ', totla)
    print('0',no_0)
    print('1',no_1)
    print('2',no_2)
    print('3',no_3)
    print('4',no_4)
    print('5',no_5)
    print('6',no_6)
    print('7',no_7)
    print('8',no_8)
    print('9',no_9)


def shape(file):
    fa = csv.reader(file)
    # fb = csv.reader(file2)

    tem1 = []
    tem2 = []

    for i in fa:
        tem1.append(i)
    # for j in fb:
    #    tem2.append(j)

    print('rows : ', len(tem1))
    print('columns : ', len(tem1[0]))


def main(file):
    # label = np.loadtxt('../div_file/test_label_grayScale_38.csv', dtype=np.float32, delimiter=',', usecols=[0])

    print('file name : ', str(file))
#    label = np.loadtxt(file, dtype=np.float32, delimiter=',', usecols=[1], skiprows=1)  # skiprows=1,
    label = np.loadtxt(file, dtype=np.float32, delimiter=',', usecols=[1], skiprows=1)  # skiprows=1,
    data = open(file)

    shape(data)
    classification_count(label)


####################################################################

file1 = '../dataFiles/test_84_datafile2.csv'
file2 = '../dataFiles/train_337_datafile2.csv'


main(file1)
print()
main(file2)
