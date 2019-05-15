"""
비디오 정보, 이미지 자르기, 뽑기(3배수, 4배수)

video info
video frame save
# image crop(이미지 부분적 추출(분할)) ; test_1
image 크기 변경 ; test_3

수화에 대해 이미지 뽑음...(폴더 자동생성 추가)

first 파일복사..

"""

import cv2
import os
from PIL import Image, ImageFilter


def video_info(filename, cap):
    '''
    video 정보에 대한걸 보여주는 함수
    '''

    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lenth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    print("================"*5)
    print("[video file information]".format('centered'))
    print("video file name is {}".format(str(filename)))
    print("video time(length) {}(s)".format(lenth/fps))
    print("all frame is ", lenth)
    print('video height : ', height)
    print('video width : ', width)
    print('fps is {}(s)'.format(fps))
    print("================" * 5)


def image_info(image):
    # print("using image : str", str(image))
    # print("int image : ", int(image))  # error
    # print("image : ", image)

    # image information print

    print("================" * 5)
    print("[image file information]".format('centered'))
    print('image size(byte) : ', image.size)
    print('image shape : ', image.shape)
    print('image dtype : ', image.dtype)
    print("================" * 5)


def video2frame(cap, save_path, save_3path, save_4path):
    count = 0  # 프레임을 읽기위해 count 함수 넣음

    # 한 프레임씩 읽는 것
    while(1):
        success, image = cap.read()
        if not success:  # video가 안읽히는 경우 종료를 위한 줄
            break
        #print('Read a no.{} frame'.format(count), success)
        #if(count >= 58):
        ## 59 프레임(약 2초) 부터 읽기시작

        #crop_image = img_crop(image)  # 프레임 당 이미지 자름
        #resize_image = img_resize(crop_image)  # 프레임 당 이미지(자른이미지)를 압축

       # image = img_crop(image)  # 이미지 자르기
        image = img_resize(image)  # 이미지 압축( 이미지 자름( 이미지)))


        fname = '{}.jpg'.format("{0:05d}".format(count))
        cv2.imwrite(save_path+fname, image)  # save frame as JPEG file
        if(count%3==0):
            cv2.imwrite(save_3path+fname, image)
        if(count % 4 == 0):
            cv2.imwrite(save_4path + fname, image)

        count += 1

    print('{} images are extracted in {}'.format(count, save_path))


def video_fileset(video_data, base_path):
    # file name


    '''
    # test_video = '../video/2/3_love_1.mp4'
    test_video1 = '3_love_1'  # folder / file
    test_video2 = '3_love_2'  # folder / file
    test_video3 = '3_love_3'  # folder / file

    video_1 = 'ashamed'
    video_2 = 'happy'
    video_3 = 'hello'
    video_4 = 'late'
    video_5 = 'love'
    video_6 = 'meet'
    video_7 = 'nice'
    video_8 = 'not'
    video_9 = 'sorry'
    video_10 = 'thank'

    video_11 = 'thank_Trim (2)'
    video_12 = 'thank_Trim (3)'
    video_13 = 'thank_Trim (4)'
    video_14 = 'thank_Trim (5)'
    video_15 = 'thank_Trim (6)'
    video_16 = 'thank_Trim'

    '''



    # save video frame path
    save_path = '../raw_data_folders/1_image/image3_2/{}/'
    save_3path = '../raw_data_folders/1_image/image3_2/{}/3_frame_{}/'
    save_4path = '../raw_data_folders/1_image/image3_2/{}/4_frame_{}/'



    base_path = base_path

    # choice file
    choice_file = video_data



    # saved local(path)
    use_file = base_path.format(choice_file)
    save_path = save_path.format(choice_file)
    save_3path = save_3path.format(choice_file, choice_file)  # format {} {} 2개야!!
    save_4path = save_4path.format(choice_file, choice_file)
    directory_check(save_path, save_3path, save_4path)

    return use_file, save_path, save_3path, save_4path



def directory_check(save_path, save_3path, save_4path):
    '''
    directory 없으면 만들어 주는 부분

    :param save_path:
    :param save_3path:
    :param save_4path:
    :return:
    '''

    try:
        if not(os.path.isdir(save_path)):
            os.makedirs(os.path.join(save_path))
            if not(os.path.isdir(save_3path)):
                os.makedirs(os.path.join(save_3path))
            if not(os.path.isdir(save_4path)):
                os.makedirs(os.path.join(save_4path))

    except OSError as e:
        if e.errno != e.EEXIST:
            print('[!] Failed to create directory {}'.format(save_path))
            raise




def img_crop(image):
    #print('image.shape', image.shape)

    height, width, BGR = image.shape

    # 이미지 자릅시다.
    # crop_img = image.crop()

    # print('\n\n{} \n{} \n{} \n{}'.format((5 * height / 100), (95 * height / 100), (25 * width / 100), (75 * width / 100)))
 #   crop_img = image[int(5 * height / 100):int(95 * height / 100),int(25 * width / 100):int(75 * width / 100)]  # 좌, 상, 우, 하  <--?
    crop_img = image[int(5 * height / 100):int(95 * height / 100),int(15 * width / 100):int(75 * width / 100)]  # 상, 하, 좌, 우

    return crop_img


def img_resize(image):
    '''
    이미지 압축(resize는 누르는것)

    '''


    # image resize
    # cv2 image file
    resize_img = cv2.resize(image, (128, 128))  # [cv] height, width
    # PIL image file
    # img2 = image.resize((1000, 10000))  # [PIL] width, height

    '''
    # image showing
    while (1):
        cv2.imshow('original image', image)
        cv2.imshow('cutting image', crop_img)
        cv2.imshow('resize image', resize_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    '''

    return resize_img





if __name__ == '__main__':

    # sign language video file list
#    base_path = '../video/{}.mp4'  # 1-10


    base_path = '../raw_data_folders/0_video/video3/{}'  # 1-10
    video_data = 'video_ ({}).mp4'

    for i in range(65):

        use_file, save_path, save_3path, save_4path = video_fileset(video_data.format(i+1), base_path)


        cap = cv2.VideoCapture(use_file)
        if not cap.isOpened():
            print('[!] could not open ', str(use_file))
            exit(0)

        video_info(use_file, cap)  # 비디오 파일 정보를 보는 것
        video2frame(cap, save_path, save_3path, save_4path)

        print('=================================================')
        print('{:^15}'.format('organization'))
        print('use video file :', use_file)
        print('save_path in :', save_path)
        print('save_3path in :', save_3path)
        print('save_4path in :', save_4path)
        print('=================================================')





