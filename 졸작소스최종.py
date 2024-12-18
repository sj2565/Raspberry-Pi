# 객체탐지에 필요한 모듈
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

# 번호판 인식에 필요한 모듈 
#import cv2
#import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import pytesseract


import RPi.GPIO as GPIO
import time

triggerPin = 24
echoPin = 23
pinPiezo = 25

GPIO.setmode(GPIO.BCM)
GPIO.setup(triggerPin, GPIO.OUT)    # 출력
GPIO.setup(echoPin, GPIO.IN)        # 입력
GPIO.setup(pinPiezo, GPIO.OUT)

Buzz = GPIO.PWM(pinPiezo, 440)      # 부저센서 초기화





# 카메라 화소 조절 (640 x 480)
IM_WIDTH = 640
IM_HEIGHT = 480

# 카메라 선택 (picamera, usbcam) 
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# 객체탐지 폴더를 작업 폴더로 사용한다는 경로 설정 
sys.path.append('..')

# utils import
from utils import label_map_util
from utils import visualization_utils as vis_util

# 객체탐지 모델 이름 
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# 현재 작업 폴더의 경로 설정
CWD_PATH = os.getcwd()

# 객체감지에 사용되는 그래프, pb파일 경로 설정
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# label맵 파일 경로 설정
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# 객체탐지가 구별할 수 있는 클래스 수 => 90가지 학습
NUM_CLASSES = 90

## label맵 로드
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 모델을 메모리에 로드
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()   # 여기 수정 해줌
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:    # 여기 수정 해줌
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)         # 여기 수정 해줌


# 객체 감지 분류기에 대한 입력 및 출력 텐서(데이터) 정의

# 입력 텐서는 이미지
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# 출력 텐서는 정확도 점수, 감지박스, 클래스 
# 각 박스는 특정 물체가 감지된 이미지의 일부를 나타냄
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# 각 정확도 점수는 각 개체에 대한 탐지 정확도 수준을 나타냄
# 점수는 클래스 label과 함께 결과 이미지에 표시
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# 감지된 개채 수 표시 
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# 카메라 프레임 속도 계산 초기화 
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


try:
    while True:
        
        #구형파 발생
        GPIO.output(triggerPin, GPIO.LOW)  
        time.sleep(0.00001) 
        GPIO.output(triggerPin, GPIO.HIGH)

        #시간측정
        while GPIO.input(echoPin) == 0:  # 펄스 발생
            start = time.time()
        while GPIO.input(echoPin) == 1:  # 펄스 돌아옴
            stop = time.time()

        rtTotime = stop - start                   # 리턴 투 타임 = (end시간 - start시간)

        distance = rtTotime * (34000 / 2 )
        print("distance : %.2f cm" %distance)     # 거리 출력
        time.sleep(0.2)  

        if(distance <= 20 and distance > 15):     # 20 ~ 15 cm 일때
            Buzz.start(50)
            Buzz.ChangeFrequency(523)
            time.sleep(0.3)
            Buzz.stop()
            time.sleep(0.3)
        if(distance <= 14 and distance > 6):   # 14 ~ 6 cm 일때    
            Buzz.start(50)
            Buzz.ChangeFrequency(523)
            time.sleep(0.15)
            Buzz.stop()
            time.sleep(0.1)
        elif(distance <= 5):                     # 5cm 이하일때
            Buzz.start(99)
            Buzz.ChangeFrequency(523) 
            time.sleep(0.05)
            Buzz.stop()
            time.sleep(0.05)
        #else:                                     # 그 외(겁나 멀때)
        #    Buzz.stop()
        #    time.sleep(0.5)


            ### Picamera ###
            if camera_type == 'picamera':
                # Picamera 초기화 및 설정
                camera = PiCamera()
                camera.resolution = (IM_WIDTH,IM_HEIGHT)
                camera.framerate = 10
                rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
                rawCapture.truncate(0)

                for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

                    t1 = cv2.getTickCount()
                    
                    # 프레임을 획득하고 모양이 [1, 없음, 없음, 3]이 되도록 프레임 치수를 배열화
                    # 열의 각 항목에 픽셀 RGB 값이 있는 단일 열 배열(1 x 100)
                    frame = np.copy(frame1.array)
                    frame.setflags(write=1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_expanded = np.expand_dims(frame_rgb, axis=0)

                    # 이미지를 입력으로 모델을 실행하여 실제 감지 수행
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: frame_expanded})

                    # 감지 결과 그리기(결과 시각화)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.40)

                    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

                    # 모든 결과가 프레임에 그려졌으므로 카메라에 표시
                    cv2.imshow('Object detector', frame)

                    t2 = cv2.getTickCount()
                    time1 = (t2-t1)/freq
                    frame_rate_calc = 1/time1

                    # 정확도가 85% 이상일 때 번호판 인식 진행
                    if np.any(scores >= 0.85):
                        #sleep(2)
                        camera.capture('/home/pi/tensorflow1/models/research/object_detection/numimage.jpg')
                        
                        #plt.style.use('dark_background')
             
                        img_ori = cv2.imread('/home/pi/tensorflow1/models/research/object_detection/numimage.jpg')

                        height, width, channel = img_ori.shape

                        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

                        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

                        imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
                        imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

                        imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
                        gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

                        # 노이즈 줄이기 위해
                        # 이미지 구별하기 쉽게(0, 255)

                        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

                        img_thresh = cv2.adaptiveThreshold(
                            img_blurred, 
                            maxValue=255.0, 
                            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            thresholdType=cv2.THRESH_BINARY_INV, 
                            blockSize=19, 
                            C=9
                        )
                        # 
                        # plt.figure(figsize=(12, 10))
                        # plt.imshow(img_thresh, cmap='gray')
                        # plt.show()

                        #윤곽선

                        contours, _ = cv2.findContours(
                            img_thresh, 
                            mode=cv2.RETR_LIST, 
                            method=cv2.CHAIN_APPROX_SIMPLE
                        )

                        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                        cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

                        # 컨투어의 사각형 범위 찾기

                        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                        contours_dict = []

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
                            
                            # insert to dict
                            contours_dict.append({
                                'contour': contour,
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h,
                                'cx': x + (w / 2),
                                'cy': y + (h / 2)
                            })


                        # 어떤게 번호판처럼 생겼는지 확인

                        MIN_AREA = 80
                        MIN_WIDTH, MIN_HEIGHT = 2, 8
                        MIN_RATIO, MAX_RATIO = 0.25, 1.0

                        possible_contours = []

                        cnt = 0
                        for d in contours_dict:
                            area = d['w'] * d['h']
                            ratio = d['w'] / d['h']
                            
                            if area > MIN_AREA \
                            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                            and MIN_RATIO < ratio < MAX_RATIO:
                                d['idx'] = cnt
                                cnt += 1
                                possible_contours.append(d)
                                
                        # 확인 가능한 윤곽선 시각화 
                        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                        for d in possible_contours:
                        #   cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
                            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

                        # 번호판 추려내기
                        MAX_DIAG_MULTIPLYER = 5 # 5
                        MAX_ANGLE_DIFF = 12.0 # 12.0
                        MAX_AREA_DIFF = 0.5 # 0.5
                        MAX_WIDTH_DIFF = 0.8
                        MAX_HEIGHT_DIFF = 0.2
                        MIN_N_MATCHED = 3 # 3

                        def find_chars(contour_list):
                            matched_result_idx = []
                            
                            for d1 in contour_list:
                                matched_contours_idx = []
                                for d2 in contour_list:
                                    if d1['idx'] == d2['idx']:
                                        continue

                                    dx = abs(d1['cx'] - d2['cx'])
                                    dy = abs(d1['cy'] - d2['cy'])

                                    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                                    distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                                    if dx == 0:
                                        angle_diff = 90
                                    else:
                                        angle_diff = np.degrees(np.arctan(dy / dx))
                                    area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                                    width_diff = abs(d1['w'] - d2['w']) / d1['w']
                                    height_diff = abs(d1['h'] - d2['h']) / d1['h']

                                    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                                        matched_contours_idx.append(d2['idx'])

                                # 윤관석 추가 진행
                                matched_contours_idx.append(d1['idx'])

                                if len(matched_contours_idx) < MIN_N_MATCHED:
                                    continue

                                matched_result_idx.append(matched_contours_idx)

                                unmatched_contour_idx = []
                                for d4 in contour_list:
                                    if d4['idx'] not in matched_contours_idx:
                                        unmatched_contour_idx.append(d4['idx'])

                                unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
                                
                                # 윤곽선 다시 확인
                                recursive_contour_list = find_chars(unmatched_contour)
                                
                                for idx in recursive_contour_list:
                                    matched_result_idx.append(idx)

                                break

                            return matched_result_idx
                            
                        result_idx = find_chars(possible_contours)

                        matched_result = []
                        for idx_list in result_idx:
                            matched_result.append(np.take(possible_contours, idx_list))

                        # 확인 가능한 윤곽선을 시각화
                        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                        for r in matched_result:
                            for d in r:
                        #       cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
                                cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

                        # 똑바로 돌리기
                        PLATE_WIDTH_PADDING = 1.3 # 1.3
                        PLATE_HEIGHT_PADDING = 1.5 # 1.5
                        MIN_PLATE_RATIO = 3
                        MAX_PLATE_RATIO = 10

                        plate_imgs = []
                        plate_infos = []

                        for i, matched_chars in enumerate(matched_result):
                            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

                            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
                            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
                            
                            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
                            
                            sum_height = 0
                            for d in sorted_chars:
                                sum_height += d['h']

                            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
                            
                            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
                            triangle_hypotenus = np.linalg.norm(
                                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
                                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
                            )
                            
                            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
                            
                            rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
                            
                            img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
                            
                            img_cropped = cv2.getRectSubPix(
                                img_rotated, 
                                patchSize=(int(plate_width), int(plate_height)), 
                                center=(int(plate_cx), int(plate_cy))
                            )
                            
                            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                                continue
                            
                            plate_imgs.append(img_cropped)
                            plate_infos.append({
                                'x': int(plate_cx - plate_width / 2),
                                'y': int(plate_cy - plate_height / 2),
                                'w': int(plate_width),
                                'h': int(plate_height)
                            })
                            
                            
                        # 최종확인
                        longest_idx, longest_text = -1, 0
                        plate_chars = []

                        for i, plate_img in enumerate(plate_imgs):
                            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
                            _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                            
                            # 윤곽선 다시 찾기 
                            contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
                            
                            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
                            plate_max_x, plate_max_y = 0, 0

                            for contour in contours:
                                x, y, w, h = cv2.boundingRect(contour)
                                
                                area = w * h
                                ratio = w / h

                                if area > MIN_AREA \
                                and w > MIN_WIDTH and h > MIN_HEIGHT \
                                and MIN_RATIO < ratio < MAX_RATIO:
                                    if x < plate_min_x:
                                        plate_min_x = x
                                    if y < plate_min_y:
                                        plate_min_y = y
                                    if x + w > plate_max_x:
                                        plate_max_x = x + w
                                    if y + h > plate_max_y:
                                        plate_max_y = y + h
                                        
                            img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
                            
                            img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
                            _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                            img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
                            
                            # 경로 설정 제대로 해주기
                            #tessdata_dir_config = r'--tessdata-dir --psm 7 --oem 0 "</usr/share/tesseract-ocr/4.00/tessdata>"'
                            chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 8 --oem 0')   # 여기는 psm8

                            # 번호판 인식 결과 출력
                            result_chars = ''
                            has_digit = False
                            for c in chars:
                                if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                                    if c.isdigit():
                                        has_digit = True
                                    result_chars += c
                            
                            print(result_chars)
                            plate_chars.append(result_chars)

                            if has_digit and len(result_chars) > longest_text:
                                longest_idx = i

                            plt.subplot(len(plate_imgs), 1, i+1)
                            plt.imshow(img_result, cmap='gray')
                            #plt.show()   

                        # 데이터 이미지 및 텍스처 저장
                        import sys
                        import io
                        #sys.stdout = open('num.txt', 'a', encoding='UTF-8')
                        #text = sys.stdout.read()
                        #sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
                        #print(result_chars)
                        #sys.stdin.encoding
                        #sys.stdout.close()
                        plt.savefig('savenum.png')

                        # 인식된 번호판 DB로 데이터 넘기기
                        
                        #from pymongo import MongoClient
                        #import datetime

                        #host = '192.168.1.186'
                        #port = '27017'

                        #conn = MongoClient("localhost", 27017)
                        #print(conn.list_database_names())   # db 목록 나열

                        #conn = MongoClient('mongodb://{}:{}'.format(host, port))

                        #db = conn['mevn-secure']  # db접근
                        #collection = db['books']  # collection접근

                        #info = {
                        #    "Number" : (result_chars),
                        #    "Date" : datetime.datetime.now(),
                        #    "Location" : "대전광역시 서구"
                        #    }
                        #collection.insert_one(info)
                        
                    # q를 누르면 객체탐지 종료 
                    if cv2.waitKey(1) == ord('q'):
                        break

                    rawCapture.truncate(0)

                camera.close()

            cv2.destroyAllWindows()

except KeyboardInterrupt:
   GPIO.cleanup()
        

