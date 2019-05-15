"""
csv file 나누기

"""
import os

# 나눠질 파일 라인 수
nDivCnt = 338

# 파일 경로
file_path = '../'
# 변경할 파일 명
file_name = 'shuffling_datafile2'
# 확장자
fileExe = '.csv'


# 나눈 파일 만들 폴더
file_folder = 'dataFiles/'


dirname = file_path+file_folder
if not os.path.isdir(dirname):
    os.mkdir(dirname)

nLineCnt = 0
nFiledx = 0

f = open('%s' % (file_path+file_name+fileExe), 'r')
fDivName = open('%s%06d%s' % (file_path + file_folder +file_name, nFiledx, fileExe), 'w')

while True:
    line = f.readline()
    if not line: break

    if nLineCnt == nDivCnt:
        fDivName.close()
        nFiledx += 1
        nLineCnt = 0
        strPat = '%s%06d%s' % (file_path + file_folder + file_name, nFiledx, fileExe)
        fDivName = open(strPat, 'w')
        print('생성 완료 %s' % strPat)

    nLineCnt += 1
    fDivName.write(line)

    #print(line)

fDivName.close()
f.close()


