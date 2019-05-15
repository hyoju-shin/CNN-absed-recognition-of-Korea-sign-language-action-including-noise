"""
이미지 -> save the csv.files
image color is RGB and Gray Scale

https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html
"""

import numpy as np
import cv2
import csv

temp = []

for i in range(409):  # 207
    image = '../total_image_data/total_image_data_ ({}).jpg'

    # image colors
    img = cv2.imread(image.format(i+1), cv2.IMREAD_GRAYSCALE)  # gray image reader
 #   img = cv2.imread(image.format(i+1), cv2.IMREAD_COLOR)  # RGB image reader to CV

    np.ravel(temp.append(img))

all_images = np.stack(temp)  # length = 32


temp2 = []
for i in range(len(all_images)):
    a = np.ravel(all_images[i])
    temp2.append(a)



with open('../tttt.csv', 'a', encoding='utf-8', newline='') as f:
    for i in range(len(temp2)):
        print(i+1, '\t', temp2[i])
        wr = csv.writer(f)
        wr.writerow(temp2[i])
    print('save successful')

print('finish!')


'''
# 파일 저장부분 위에꺼로(csv. 사용) 바꿈

with open('../make_testfile_RGB_tttt.csv', 'a') as f:
    for i in range(len(temp2)):
        print(i+1, '\t', temp2[i])
        for j in range(len(temp2[i])):
#            np.savetxt(f, temp2[i], delimiter=',')
            d = '{},'.format(temp2[i][j])
            f.write(d)
        f.write('\n')
            # np.savetxt(f, all_images[i], delimiter=',')
       # f.write('\n')
        print('save successful')

'''


