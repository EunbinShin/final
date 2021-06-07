
import threading
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtCore, uic
import pyttsx3
import datetime
import webbrowser
#-------------------------------------------import 구역
import cv2
from numpy.core.fromnumeric import mean
import pafy
import numpy as np
import pandas as pd
import math
from math import atan2, degrees, acos
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import argparse 
import posenet
#-------------------------------------------데이터베이스 접근
import login

#-------------------------------------------youtube 주소
url = "https://www.youtube.com/watch?v=cMkZ6A7wngk"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

#------------------------------------------- code parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

#------------------------------------------ yeonhoon

color = (0,0,250)
test = ""
count_flag = False
squat_count = 0
knee_flag = False
squat_set = 0

def angle_between(x1,y1, x2, y2, x3,y3): #세 x,y로 각도를 구하는 방식
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    degree = 360 - (deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2))
    return degree

def triangle_points(points):
    A = points[1] - points[0]
    B = points[2] - points[1]
    C = points[0] - points[2]
    angles = []
    for e1, e2 in ((A, -B), (B, -C), (C, -A)):
        num = np.dot(e1, e2)
        denom = np.linalg.norm(e1) * np.linalg.norm(e2)
        angles.append(np.arccos(num/denom) * 180 / np.pi)
    
    return min(angles)

#---------------------스쿼트 관련 상수
CUSTOM_SQUAT_SET = 3
CUSTOM_SQUAT_COUNT = 12

video_player = 1 #비디오 재생 트리거
running = False
rest_flag = False
set_flag = False
do_flag = False
#--------------------- TTS 엔진 초기화
engine = pyttsx3.init() # 엔진 초기화
engine.setProperty('volume', 0.5) # 0.0 ~ 1.0
engine.setProperty('rate', 180)

def speak(script):
    engine.stop()
    engine.say(script)
    engine.runAndWait()

def check_knee_width(left_shoulder_x, right_shoulder_x, left_knee_x, right_knee_x):
    shoulder_width = abs(left_shoulder_x - right_shoulder_x)
    knee_width = abs(left_knee_x - right_knee_x)
    knee_gap = shoulder_width / knee_width
    return knee_gap

def check_feet_width(left_shoulder_x, right_shoulder_x, left_ankle_x, right_ankle_x):
    shoulder_width = abs(left_shoulder_x - right_shoulder_x)
    ankle_width = abs(left_ankle_x - right_ankle_x)
    feet_gap = shoulder_width / ankle_width
    return feet_gap

knee_list = []

def squat_down(angle, angles_arr, squat_knee_angle, left_knee_angle, right_knee_angle, left_hip_gap, right_hip_gap, knee_gap):
    global color
    global test
    global count_flag
    global squat_count
    global video_player#비디오 재생 트리거
    global squat_set
    global rest_flag
    global set_flag
    global knee_flag
    ###################
    global knee_list

    # 내려갔을 때
    if len(angles_arr)>10:
        if mean(angles_arr[-10:-5]) < mean(angles_arr[-5:]) and angle > angles_arr[0] + 5 :
            knee_list.append((right_knee_angle+left_knee_angle)/2)
            
            if left_knee_angle > squat_knee_angle * 1.2 and right_knee_angle > squat_knee_angle * 1.2: 
                test="Lower"
                color = (0,0,250)

            elif (squat_knee_angle <= left_knee_angle < squat_knee_angle * 1.2 or 
                    squat_knee_angle <= right_knee_angle < squat_knee_angle * 1.2) or \
                    (left_hip_gap < 20 or right_hip_gap < 20):
                test="Good"
                video_player=0 #다 앉았을때 video재생 준비 완료
                count_flag = True
                color = (255,0,0)

            # 무릎이 너무 모이는 경우
            if knee_gap >= 1.2:
                knee_flag = True

        # 올라올 때
        if mean(angles_arr[-10:-5]) > mean(angles_arr[-5:]):
            knee_list.append((right_knee_angle+left_knee_angle)/2)
            
            # 내려갔다 올라와서 멈출 때
            if np.median(angles_arr[:5])*0.95 <= angle <= np.median(angles_arr[:5]) * 1.05: #완전히 "섰다"의 인식이 쫌 여유가 있는 듯 
                
                if count_flag:
                    video_player=1#다 서있을 때 video 재생 준비 완료
                    count_flag = False
                    squat_count +=1

                    #threading.Thread(target=speak, args=(str(squat_count)+'번',)).start()

                    if squat_count == CUSTOM_SQUAT_COUNT:
                        squat_set += 1
                        squat_count = 0

                        rest_flag = True

                    if squat_set == CUSTOM_SQUAT_SET:
                        rest_flag = False
                        set_flag = True

                    print(f"rep: {squat_count}")

                # 무릎이 너무 모이는 경우
                if knee_flag:
                    test = "Knee Wider"
                    color = (0,0,255)
                    knee_flag = False
    
    
angle_list_dict={
    'R hip':[],
    'L hip':[],
    'R knee':[],
    'L knee':[], 
}

#함수 --------------------------- 이준영
def angle_flag(now_angle,now_mins,now_maxs,now_flag,errocounter,joint,wrongtext): #이준영 반복부분 해결하기위한 것
    global angle_list_dict
    new_angle=0
    delay=10
    if len(angle_list_dict[joint])<5:
        angle_list_dict[joint].append(now_angle)
    else:
        angle_list_dict[joint].append(now_angle)
        summ=0
        for i in angle_list_dict[joint]:
            summ+=i
        new_angle=summ/6
        del angle_list_dict[joint][0]
        if now_flag == 0:
            if new_angle <= now_mins[1]:
                now_flag=2
            if  new_angle > now_maxs[0]:
                wrongtext="MU" #다음번엔 더 내려가세요
                now_flag=3
                errocounter=delay
        elif now_flag == 1:
            if new_angle < now_mins[1]:
                wrongtext="MO" #다음번엔 더 올라가세요
                now_flag=2 
                errocounter=delay
            if new_angle > now_maxs[0]:
                now_flag=3
        elif now_flag == 2:
            if new_angle > now_mins[1]:  #최저 범위 상위 탈출
                now_flag=1 #올라감
            if new_angle < now_mins[0]:
                wrongtext="TU" #너무 내려갔어요
                errocounter=delay
        elif now_flag == 3:
            if new_angle<now_maxs[0]:  #최고 범위 하위 탈출 
                now_flag=0 #내려감
        
    return now_flag,errocounter,wrongtext


def angle_text(errocounter,wrongtexts,out_img):
    global do_flag

    okcount=0
    wrongpoint = ""
    for (x,y),v in wrongtexts.items():
        if v =="":
           okcount+=1
        else:
            if errocounter[(x,y)]>0:
                # out_img = cv2.putText(out_img, wrongtexts[(x,y)], (50+150*x,200+100*y), cv2.FONT_HERSHEY_DUPLEX, 1.8, (30,30,200),2)
                errocounter[(x,y)]-=1
            if errocounter[(x,y)]<=0:
                wrongtexts[(x,y)]=""
    if okcount>=1:
        out_img = cv2.putText(out_img, "OK", (100,200), cv2.FONT_HERSHEY_DUPLEX, 2, (0,200,100),2)
    else:
        wrongpoint = max(wrongtexts.values())
        out_img = cv2.putText(out_img, wrongpoint, (100,200), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255),2)

        """
        if wrongpoint == 'TU':
            threading.Thread(target=speak, args=('엉덩이가 너무 내려갔습니다.',)).start()
            print('TU')
            do_flag = False
            
        elif wrongpoint == 'MU':
            threading.Thread(target=speak, args=('엉덩이를 더 내리세요.',)).start()
            print('MU')
            do_flag = False
        """
    return errocounter, wrongtexts, out_img, wrongpoint

#-------------------------------------------준영 함수
parts = ['nose', 'L eye', 'R eye', 'L ear', 'R ear', 
    'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L wrist', 'R wrist',
    'L pelvis', 'R pelvis', 'L knee', 'R knee', 'L ankle', 'R ankle']

angle_dict = {#구하고 싶은 각도들의 인덱스를 저장
    'R hip':(12,6,14),  #오른쪽 옆구리
    'L hip':(11,5,13),  #왼쪽 옆구리
    'R knee':(14,16,12),    #오른쪽 무릎
    'L knee':(13,15,11),     #왼쪽 무릎

    'L pelvis' : (11,13,12),
    'R pelvis' : (12,14,11),
    'L knee' : (13,15,11),
    'R knee' : (14,16,12),
    'L shoulder' : (5,7,11),
    'R shoulder' : (6,8,12),
    'L elbow' : (7,5,9),
    'R elbow' : (8,6,10)
}

def angle_cal(A,B,C):#A에 있는 각도 구하기
    a=math.sqrt((B[0]-C[0])**2+(B[1]-C[1])**2)
    b=math.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)
    c=math.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)
    angle=math.acos((b*b+c*c-a*a)/(2*b*c))
    angle=angle*180/math.pi
    return angle
    
#=------------------------은빈 변수
R_shoulder = False
R_pelvis = False
R_knee = False
R_elbow = False
L_shoulder = False
L_pelvis = False
L_knee = False
L_elbow = False

running = False
real_start = False
cam_start = 0
#--------------------------준영 변수
L_knee_mins=(65,110) # 왼쪽 무릎각도 최저 정상범위 설정
L_knee_maxs=(170,180)
R_knee_mins=(65,110) # 오른쪽 무릎각도 최저 정상범위 설정
R_knee_maxs=(170,180)
L_hip_mins=(80,125) # 왼쪽 옆구리각도 최저 정상범위 설정
L_hip_maxs=(155,175) 
R_hip_mins=(80,125) # 오른쪽 옆구리각도 최저 정상범위 설정
R_hip_maxs=(155,175)

form_class = uic.loadUiType('ui/test_ui.ui')[0]
setting_form = uic.loadUiType('ui/setting_ui.ui')[0]
rest_form = uic.loadUiType('ui/rest_ui.ui')[0]
login_form = uic.loadUiType('ui/login_ui.ui')[0]
success_form = uic.loadUiType('ui/success_ui.ui')[0]

#####################SESSION####################################
user_id = ""
#########################################################

class SuccessWindow(QDialog, success_form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('SUCCESS!!!!')

        self.endButton.clicked.connect(self.end_program)

    def end_program(self):
        global set_flag
        global user_id
        global CUSTOM_SQUAT_SET
        global CUSTOM_SQUAT_COUNT

        print('프로그램 종료')

        if user_id != '': #로그인 되어있는 상태라면
            #DB에 데이터 보냄
            login.insert_set_and_count(user_id, CUSTOM_SQUAT_SET, CUSTOM_SQUAT_COUNT)

        set_flag = False
        return self.close()

    def show(self):
        return super().exec_()

class RestWindow(QDialog, rest_form):
    def __init__(self):
        global knee_list

        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Rest Time')
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowSystemMenuHint, False)

        self.duration = 60

        print('REST WIndow start')

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.timeout)
        self.timer.start()


        knee_list = []

    def timeout(self):
        global rest_flag
        global running
        global cam_start
        global real_start
        global R_shoulder
        global R_pelvis
        global R_knee
        global R_elbow
        global L_shoulder
        global L_pelvis
        global L_knee
        global L_elbow

        self.duration -= 1
        self.lcdNumber.display(self.duration)

        if self.duration == 0: #
            rest_flag = False
            real_start = False
            R_shoulder = R_pelvis = R_knee = R_elbow = L_shoulder = L_pelvis = L_knee = L_elbow = False
            running = True
            
            myWindow.start_camera()

            return self.close()

    def show(self):
        threading.Thread(target=speak, args=('쉬는 시간 입니다',)).start()
        return super().exec_()

class SettingWindow(QDialog, setting_form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Setting')

        self.countText.setValidator(QtGui.QIntValidator())
        self.setsText.setValidator(QtGui.QIntValidator())

        self.countText.setText(str(CUSTOM_SQUAT_COUNT))
        self.setsText.setText(str(CUSTOM_SQUAT_SET))

        self.accepted.connect(self.my_accept)

    def show(self):
        return super().exec_()

    def my_accept(self):
        global CUSTOM_SQUAT_COUNT
        global CUSTOM_SQUAT_SET

        CUSTOM_SQUAT_COUNT = int(self.countText.text())
        CUSTOM_SQUAT_SET = int(self.setsText.text()) 

class LoginWindow(QDialog, login_form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('LOGIN')
        self.login = False
        self.loginButton.clicked.connect(self.login_btn_clicked)

    def login_btn_clicked(self):
        self.id = self.idText.text()
        password = self.passwordText.text()

        result = login.get_user_info(self.id, password)

        if result is None:
            self.login = False
            self.warning.setText('로그인 실패!')
        else:
            self.login = True
            self.close()
            print('로그인 성공')

    def show(self):
        super().exec_()

        if self.login == True:
            return self.id

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.running = False
        self.stop = False
        self.setupUi(self)
        self.setWindowTitle('Do Home-T')
        self.setWindowIcon(QtGui.QIcon('img/icon.png'))

        #####################첫 화면 사진######################
        self.img_label = QLabel(self)
        x, y = self.size().width(), self.size().height()
        pixmap = QtGui.QPixmap('img/main.jpg')
        pixmap_rect = pixmap.rect()
        self.img_label.setGeometry(pixmap_rect)
        self.img_label.setPixmap(pixmap)
        self.img_label.move(int(x/3.5),int(y/4))
        self.img_label.setStyleSheet("bor")
        #####################첫 화면 사진######################

        self.startButton.clicked.connect(self.start_btn_clicked)
        self.stopButton.clicked.connect(self.stop_btn_clicked)
        self.loginButton.clicked.connect(self.login_btn_clicked)
        self.webButton.clicked.connect(self.web_btn_clicked)

        self.actiontest.triggered.connect(self.test)

        #-----------------------1초마다 rest 체크
        self.timer = QtCore.QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.rest)

        #-----------------------1초마다 success 체크
        self.timer2 = QtCore.QTimer(self)
        self.timer2.start(1000)
        self.timer2.timeout.connect(self.success)

        #-----------------------1초마다 count 체크
        self.timer2 = QtCore.QTimer(self)
        self.timer2.start()
        self.timer2.timeout.connect(self.count)

        self.first = True

        self.setBox.hide()

        threading.Thread(target=speak, args=('두 홈트를 찾아주셔서 감사합니다',)).start()

    def count(self):
        global squat_count
        global squat_set

        self.setLabel.setText(str(squat_set))
        self.repLabel.setText(str(squat_count))

    def success(self):
        global set_flag
        global running

        if set_flag:
            self.stop = True
            running = False
            win = SuccessWindow()
            win.show()
            self.stop_btn_clicked()

    def test(self):
        win = SettingWindow()
        win.show()

    def rest(self):
        global rest_flag
        global running

        if rest_flag:
            running = False
            win = RestWindow()
            win.show()

    def web_btn_clicked(self):
        webbrowser.open('http://stackoverflow.com')

    def login_btn_clicked(self):
        global user_id
        win = LoginWindow()
        result = win.show()

        if result is not None:
            self.loginStatus.setText(result+"님")
            user_id = result

    def start_btn_clicked(self):
        global running
        running = True
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.setBox.show()
        self.img_label.setVisible(False)

        label_ref = self.refLabel
        th_ref = threading.Thread(target=self.run_ref, args=(label_ref,))
        th_ref.start()

        self.start_camera()

    def start_camera(self):
        label = self.camLabel
        th = threading.Thread(target=self.run, args=(label,))
        th.start()
        print(self.running,"started..")

    def stop_btn_clicked(self):
        global running
        self.stop = True
        running = False
        engine.stop()

        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        print(running,"stoped..")
        QtCore.QCoreApplication.quit()

    def run_ref(self, myLabel):
        global real_start
        global video_player
        global cam_start
        global running

        res=False
        startcount = -1
        now_tape = "l"
        while True:
            if self.stop:
                break
            if cam_start==0:
                continue
            
            if res and running: #영상이 인식이 된다면
                img = cv2.resize(img, dsize=(0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
                img = img[20:, :].copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h,w,c = img.shape

                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                
                myLabel.setPixmap(pixmap)
                time.sleep(0.03)
            else: #영상이 끝났거나 실행되고 있지 않다면
                if real_start==False:#사람이 인식되지 않았을 때
                    if startcount<=0:#두번 반복을 위한
                        cap_test = cv2.VideoCapture("video/hedo_a.mp4")#두번째 예시 스쿼트 시작
                        startcount+=1
                    elif startcount>=1:
                        cap_test = cv2.VideoCapture("video/heready.mp4")#대기 영상
                        now_tape = "l"
                        cam_start = 2  #캠 인식 시작

                if real_start==True:#사람이 인식 될 때
                    if now_tape=="f" and video_player==0:#비디오 재생 트리거가 켜져있고 이전 비디오가 f일때
                        cap_test = cv2.VideoCapture("video/hedo_l.mp4")#비디오를 l로 재생
                        now_tape="l"#현재 비디오 l
                    elif now_tape=="l" and video_player==1:
                        cap_test = cv2.VideoCapture("video/hedo_f.mp4")
                        now_tape="f"

            res, img = cap_test.read() #영상을 처음 불러오기위해 뒤에서 실행@@@@
                    
    def run(self, myLabel):
        global running
        global real_start
        global R_shoulder
        global R_pelvis
        global R_knee
        global R_elbow
        global L_shoulder
        global L_pelvis
        global L_knee
        global L_elbow
        global cam_start

        #모델 열고 시작
        
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(args.model, sess)  #model_outputs는 텐서 객체들의 리스트
            output_stride = model_cfg['output_stride']

            cap = cv2.VideoCapture(0)
            if self.first: #두번째 실행됐을때는 실행되지 않도록
                cam_start = 1
                self.first = False

            peaple_count=1 #한명만 실행 지금 코드가 그대로 되어있음

            #이전프레임 저장 기능 복구-----------------------------준영
            bf_keyscores=np.zeros((peaple_count,17),float) #이전프레임 저장 기능은 복구
            bf_keycoords=np.zeros((peaple_count,17,2),float)

            #----------------------------------------------------준영

            min_pose_score=0.15 #자세 인식 최소값
            min_part_score=0.1 #관절 포인트 인식 최소값

            angle_save={}
            L_knee_flag = 3
            R_knee_flag = 3
            L_hip_flag = 3
            R_hip_flag = 3

            wrongtexts={
                (0,0):"", 
                (0,1):"",
                (1,0):"",
                (1,1):""    
            }
            errocounter={
                (0,0):0,
                (0,1):0,
                (1,0):0,
                (1,1):0    
            }
            #-----------------------------------------------------연훈
            angles_arr = []
            time_count = 0
            feet_arr = []
            feet_range_arr = []
            k_c = []
            
            wrong_ex = ""

            while running:
                start_time = time.time()
                if cam_start==2:
                    try:
                        input_image, display_image, output_scale = posenet.read_cap(cap,
                                                scale_factor=args.scale_factor,
                                                output_stride=output_stride) #영상입력
                    except:
                        break #영상을 못받으면 탈출

                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                        model_outputs,
                        feed_dict={'image:0': input_image}
                    )

                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                        heatmaps_result.squeeze(axis=0),
                        offsets_result.squeeze(axis=0),
                        displacement_fwd_result.squeeze(axis=0),
                        displacement_bwd_result.squeeze(axis=0),
                        output_stride=output_stride,
                        max_pose_detections=peaple_count,
                        min_pose_score=min_pose_score)
                    
                    #카메라가 사람을 못찾으면 초기화, 스쿼트 초기화
                    if pose_scores[0]<min_pose_score:
                        real_start = False
                        global squat_count
                        squat_count = 0             
                    
                    keypoint_coords *= output_scale

                    out_img = display_image
                    h,w,c = out_img.shape
                    cv_keypoints = []
                    cv_keypoints_else = [] #점수보다 낮은 이미지 처리(연훈이형 부분에 없었음)
                    adjacent_keypoints = []
                    adjacent_keypoints_else = []

                    #이전프레임 저장 기능 복구-----------------------------준영
                    errorlist=[]
                    for ii, score in enumerate(pose_scores): #이전 프레임 저장 부분 복구
                        for jj,(ks, kc) in enumerate(zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :])):
                            #새로인식한 신체 부분 값이 min_part_score보다 높다 or 이전 값보다 높거나 or -0.2하면 음수가 되어버린다
                            if ks > min_part_score or bf_keyscores[ii][jj]<ks or bf_keyscores[ii][jj]-0.2<0: 
                                bf_keyscores[ii][jj]=ks
                                bf_keycoords[ii][jj]=kc
                            else : #기존 값을 사용한다면 최대 5프레임이라는 유통기한을 사용해야할듯
                                bf_keyscores[ii][jj]-=0.2
                                errorlist.append(jj)
                    #----------------------------------------------------준영

                    for ii, score in enumerate(pose_scores):
                        if score < min_part_score:
                            continue
                        
                        results = []
                        results_else = []#신뢰도 낮은 값 처리를 위한 나머지 리스트

                        k_s= bf_keyscores[ii, :]
                        k_c= bf_keycoords[ii, :, :]
                        
                        for left, right in posenet.CONNECTED_PART_INDICES: #선찾기
                            if k_s[left] < min_part_score or k_s[right] < min_part_score:#값보다 낮으면 낮은 쪽에 넣고
                                results_else.append(np.array([k_c[left][::-1], k_c[right][::-1]]).astype(np.int32),)
                            else :#값보다 높으면 높은 쪽에 넣고
                                results.append(np.array([k_c[left][::-1], k_c[right][::-1]]).astype(np.int32),)
                    
                        adjacent_keypoints.extend(results)#값보다 높은 것들의 묶음은 높은 쪽에 넣고
                        adjacent_keypoints_else.extend(results_else)#값보다 낮은 것들의 묶음은 낮은 쪽에 넣고

                        for jj, (ks, kc) in enumerate(zip(k_s, k_c)):#점찾기
                            if ks < min_part_score:#값보다 낮으면 낮은 쪽에 넣고
                                cv_keypoints_else.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
                            else:#값보다 높으면 높은 쪽에 넣고
                                cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

                            out_img = cv2.putText(out_img, parts[jj], (int(kc[1]), int(kc[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)#그 점 좌표에 관절명을 적음
                            if parts[jj] in angle_dict:
                                temp_angle = angle_cal(k_c[angle_dict[parts[jj]][0]],k_c[angle_dict[parts[jj]][1]],k_c[angle_dict[parts[jj]][2]])
                                out_img = cv2.putText(out_img, str(int(temp_angle)), (int(kc[1]), int(kc[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)#그 점 좌표에 관절명을 적음
                        
                        for i,v in angle_dict.items():
                            angle_save[i]=int(angle_cal(k_c[v[0]],k_c[v[1]],k_c[v[2]]))
                        

                    #================================================================
                    #정상 포즈 판별
                    end_time = time.time()

                    if real_start == False:
                        if len(k_c)>10:
                            feet_gap = check_feet_width(k_c[5][1], k_c[6][1], k_c[15][1],k_c[16][1])
                        if 'R shoulder' in angle_save:
                            if angle_save['R shoulder'] >=0 and angle_save['R shoulder']<=35:
                                R_shoulder = True
                            else:
                                time_count = 0
                                #print(str(angle_save['R shoulder']),'R shoulder False')
                                R_shoulder = False

                        if 'L shoulder' in angle_save:
                            if angle_save['L shoulder'] >=0 and angle_save['L shoulder']<=35:
                                L_shoulder = True
                            else:
                                time_count = 0
                                #print(str(angle_save['L shoulder']),'L shoulder False')
                                L_shoulder = False

                        if 'R elbow' in angle_save:
                            if angle_save['R elbow'] >=145 and angle_save['R elbow']<=205:
                                R_elbow = True
                            else:
                                time_count = 0
                                #print(str(angle_save['R elbow']),'R elbow F')
                                R_elbow = False

                        if 'L elbow' in angle_save:
                            if angle_save['L elbow'] >=145 and angle_save['L elbow']<=205:
                                L_elbow = True
                            else:
                                time_count = 0
                                #print(str(angle_save['L elbow']),'L elbow F')
                                L_elbow = False

                        if 'R pelvis' in angle_save:
                            if angle_save['R pelvis'] >=65 and angle_save['R pelvis']<=140:
                                R_pelvis = True
                            else:
                                time_count = 0
                                #print(str(angle_save['R pelvis']),'R pelvis F')
                                R_pelvis = False

                        if 'L pelvis' in angle_save:
                            if angle_save['L pelvis'] >=65 and angle_save['L pelvis']<=140:
                                L_pelvis = True
                            else:
                                time_count = 0
                                #print(str(angle_save['L pelvis']),'L pelvis F')
                                L_pelvis = False

                        if 'R knee' in angle_save:
                            if angle_save['R knee'] >=140 and angle_save['R knee']<=200:
                                R_knee = True
                            else:
                                time_count = 0
                                #print(str(angle_save['R knee']),'R knee F')
                                R_knee = False

                        if 'L knee' in angle_save:
                            if angle_save['L knee'] >=140 and angle_save['L knee']<=200:
                                L_knee = True
                            else:
                                time_count = 0
                                #print(str(angle_save['L knee']),'L knee F')
                                L_knee = False

                        if R_shoulder & R_elbow & R_pelvis & R_knee & L_shoulder & L_elbow & L_pelvis & L_knee:
                            time_count += end_time - start_time
                            #print(int(time_count))

                            out_img = cv2.putText(out_img,
                                    str(int(time_count)),
                                    (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
                            
                            if time_count > 2:
                                real_start = True

                        if len(k_c)>10:
                            if feet_gap < 0.4:
                                time_count = 0
                                out_img = cv2.putText(out_img,
                                        'Legs Too Wide',
                                        (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,255), 1, cv2.LINE_AA) 
                        
                            if feet_gap > 1.5:
                                time_count = 0
                                out_img = cv2.putText(out_img,
                                        'Legs Wider',
                                        (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,255), 1, cv2.LINE_AA) 
                        else:
                            time_count = 0
                            out_img = cv2.putText(out_img,
                                    'Please Re-pose',
                                    (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)

                    #----------------------------------------------------좌표 위치 그대로 그려주는 코드

                    out_img = cv2.drawKeypoints(out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    out_img = cv2.drawKeypoints(out_img, cv_keypoints_else, outImage=np.array([]), color=(0, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
                    out_img = cv2.polylines(out_img, adjacent_keypoints_else, isClosed=False, color=(0, 0, 255))
                    
                    #================================================================
                    
                    if real_start:
                        points = np.array([k_c[0],k_c[15],k_c[16]])

                        feet_range = abs(k_c[15][1] - k_c[16][1])
                        feet_range_arr.append(feet_range)

                        angle = triangle_points(points)
                        angles_arr.append(angle)
                        
                        feet_gap = check_feet_width(k_c[5][1], k_c[6][1], k_c[15][1],k_c[16][1])
                        feet_arr.append(feet_gap)
                        knee_gap = check_knee_width(k_c[5][1], k_c[6][1], k_c[13][1],k_c[14][1])    

                        # 자세 판별
                        squat_knee_angle = 90
                        left_knee_hip_gap = abs(k_c[11][0] - k_c[13][0])
                        right_knee_hip_gap = abs(k_c[12][0] - k_c[14][0])

                        global test
                        global color
                        if feet_range > 1.5 * feet_range_arr[0] or feet_range < 0.5 * feet_range_arr[0]:
                            angles_arr = []
                            out_img = cv2.putText(out_img, "Adjust Legs", (350,50), cv2.FONT_HERSHEY_PLAIN, 4, color = (0,0,0), thickness=4)
                        else:
                            squat_down(angle, angles_arr, 
                                    squat_knee_angle, angle_save["L knee"],  angle_save["R knee"], 
                                    left_knee_hip_gap, right_knee_hip_gap, knee_gap)
                            out_img=cv2.putText(out_img, test, (50,100), cv2.FONT_HERSHEY_DUPLEX, 2, color=color, thickness=2)
                            squat_rep = f"Rep: {squat_count}"
                            out_img = cv2.putText(out_img, squat_rep, (350,50), cv2.FONT_HERSHEY_PLAIN, 4, color = (0,0,0), thickness=4)

                        #<운동체커 0:내려가는중 1:올라가는중 2:최저 정상범위 3:최고 정상범위 4:운동 오류>
                        if angle_save:
                            # wrongtexts={} 유지를 위해 더 상위에 선언
                            L_knee_flag,errocounter[(1,1)],wrongtexts[(1,1)]=angle_flag(angle_save["L knee"],L_knee_mins,L_knee_maxs,L_knee_flag,errocounter[(1,1)],"L knee",wrongtexts[(1,1)])
                            # L_knee_flag,L_knee_errocounter,out_img=angle_text(0,0,L_knee_flag,L_knee_errocounter,out_img)

                            #------------------------------------------오른쪽 무릎 코드
                            R_knee_flag,errocounter[(0,1)],wrongtexts[(0,1)]=angle_flag(angle_save["R knee"],R_knee_mins,R_knee_maxs,R_knee_flag,errocounter[(0,1)],"R knee",wrongtexts[(0,1)])
                            # R_knee_flag,R_knee_errocounter,out_img=angle_text(1,0,R_knee_flag,R_knee_errocounter,out_img)

                            #------------------------------------------왼쪽 옆구리 코드
                            L_hip_flag,errocounter[(1,0)],wrongtexts[(1,0)]=angle_flag(angle_save["L hip"],L_hip_mins,L_hip_maxs,L_hip_flag,errocounter[(1,0)],"L hip",wrongtexts[(1,0)])
                            # L_hip_flag,L_hip_errocounter,out_img=angle_text(0,1,L_hip_flag,L_hip_errocounter,out_img)
                        
                            #------------------------------------------오른쪽 옆구리 코드
                            R_hip_flag,errocounter[(0,0)],wrongtexts[(0,0)]=angle_flag(angle_save["R hip"],R_hip_mins,R_hip_maxs,R_hip_flag,errocounter[(0,0)],'R hip',wrongtexts[(0,0)])
                            # R_hip_flag,R_hip_errocounter,out_img=angle_text(1,1,R_hip_flag,R_hip_errocounter,out_img)

                            
                            errocounter, wrongtexts, out_img, wrongpoint = angle_text(errocounter, wrongtexts, out_img)

                            ########################################################################################
                            if wrong_ex != wrongpoint:
                                if wrongpoint == 'MU':    
                                    threading.Thread(target=speak, args=('엉덩이를 더 내려주세요',)).start()
                                elif wrongpoint == 'TU':
                                    threading.Thread(target=speak, args=('엉덩이가 너무 내려갔어요',)).start()

                                print('wrong point :',wrongpoint)
                            ########################################################################################

                            wrong_ex = wrongpoint
                            #print('wrong point :', wrongpoint)
                        #print("LN:{},{:.1f}\tRN:{},{:.1f}\tLH:{},{:.1f}\tRT:{},{:.1f}".format(L_knee_flag,angle_save['L knee'],R_knee_flag,angle_save['R knee'],L_hip_flag,angle_save['L hip'],R_hip_flag,angle_save['R hip']))
                else:
                    _,out_img=cap.read()
                    cv2.waitKey(1)
                    h,w,c = out_img.shape
                    if cam_start==0:
                        cam_start=1
                out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                qImg = QtGui.QImage(out_img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                myLabel.setPixmap(pixmap)
                QApplication.processEvents()

            cap.release()
            print("Thread end.")

app = QApplication(sys.argv)
myWindow = MyWindow()
myWindow.show() 

app.exec_()