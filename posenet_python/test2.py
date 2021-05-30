import cv2
import threading
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtCore, uic
import time

#####################################
#-------------------------------------------import 구역
import cv2
from numpy.core.fromnumeric import mean
import pafy
import numpy as np
import pandas as pd
import math
from math import atan2, degrees
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import argparse 
import posenet

#-------------------------------------------youtube 주소
url = "https://www.youtube.com/watch?v=cMkZ6A7wngk"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

#-------------------------------------------코드 파싱파트
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

#-------------------------------------------연훈쓰 함수
"""
data에서 각 좌표를 뽑아내는 함수들
leg_points
shoulder_points
body_points
get_nose
"""
def angle_between(x1,y1, x2, y2, x3,y3): #세 x,y로 각도를 구하는 방식
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    degree = 360 - (deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2))
    return degree

#----------장연훈 함수 (수정)
def squat_down(out_img, nose_y, nose_arr, grad, squat_knee_angle, ready_knee_angle, left_knee_angle, right_knee_angle, left_side_angle, right_side_angle):
    global color
    global test
    squat_side_angle = 90
    
    # 정적인 단계
    if nose_arr[-10] - nose_arr[-1] < 0.5 and -0.5<= grad <=0.5 : # 코의 움직임이 거의 없을 때
        if  -0.5<= nose_y - min(nose_arr) <=0.5:
            test = ""
            color = (0,255,0)
        else: # 스쿼트 단계
            if nose_y > min(nose_arr) + 100: # 코가 충분히 아래로 갔을 때
                if abs(grad) <= 0.5 : 
                    if left_knee_angle > squat_knee_angle * 1.2 and right_knee_angle > squat_knee_angle * 1.2: 
                        
                        test="Lower"
                        color = (0,0,255)

                    elif squat_knee_angle * 0.98 <= left_knee_angle < squat_knee_angle * 1.1 and \
                        squat_knee_angle * 0.98 <= right_knee_angle < squat_knee_angle * 1.1 or \
                        squat_side_angle * 0.9 < left_side_angle < squat_side_angle * 1.1 or squat_side_angle * 0.9 <right_side_angle < squat_side_angle * 1.1:
                        test="Good"
                        color = (255,0,0)
                    
                    elif left_knee_angle < 60 and right_knee_angle <60:
                        test = "Higher"
                        color = (0,0,255)
                else:
                    test=""
                    color = (0,0,255)    
            else:
                test=""
                color = (0,0,255) 

    # 올라갈 때 ----------------------------------------------------------------------
    else:
        test=""
        color = (0,0,255)
    return test, color

def to_df(point_x, point_y, x_arr, y_arr):
    x_temp, y_temp = np.array(point_x), np.array(point_y)
    x_arr=np.vstack([x_arr, x_temp])
    y_arr=np.vstack([y_arr, y_temp])
    x_df = pd.DataFrame(x_arr, columns=parts) #의미상 중복되는 col_name 리스트 는 삭제함
    y_df = pd.DataFrame(y_arr, columns=parts)
    x_df['frame_num']=range(x_df.shape[0])
    y_df['frame_num']=range(y_df.shape[0])
    mg_df = pd.merge(x_df, y_df, on=['frame_num']).iloc[1:,:]

    return mg_df


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
#####################################
form_class = uic.loadUiType('ui/test_ui.ui')[0]
 
class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.running = False
        self.setupUi(self)
        self.setWindowTitle('Do홈트')

        self.imgLabel.setPixmap(QtGui.QPixmap('posenet/cat.jpg'))
        self.startButton.clicked.connect(self.start_btn_clicked)
        self.stopButton.clicked.connect(self.stop_btn_clicked)

    def start_btn_clicked(self):
        self.running = True
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

        label = self.imgLabel
        th = threading.Thread(target=self.run, args=(label,))
        th.start()
        print(self.running,"started..")

    def stop_btn_clicked(self):
        self.running = False
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        print(self.running,"stoped..")

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
        #모델 열고 시작
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(args.model, sess)  #model_outputs는 텐서 객체들의 리스트
            output_stride = model_cfg['output_stride']
            cap = cv2.VideoCapture(0)
            start = time.time()
            frame_count = 0
            
            peaple_count=1 #한명만 실행 지금 코드가 그대로 되어있음

            #이전프레임 저장 기능 복구-----------------------------준영
            bf_keyscores=np.zeros((peaple_count,17),float) #이전프레임 저장 기능은 복구
            bf_keycoords=np.zeros((peaple_count,17,2),float)
            #----------------------------------------------------준영

            min_pose_score=0.15 #자세 인식 최소값
            min_part_score=0.1 #관절 포인트 인식 최소값
            
            #-----------------------------------------------------준영
            angle_save={}
            knee_flag=3
            errocounter=0#에러표시 유지값
            #-----------------------------------------------------연훈
            nose_arr = []
            grad_check = []

            while self.running:
                start_time = time.time()
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
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
                            angle_save[i]=angle_cal(k_c[v[0]],k_c[v[1]],k_c[v[2]])
                        
                        #------------------------------신은빈 stading pose 검출------------------------------
                        angle_save['R shoulder'] = int(angle_cal(k_c[angle_dict['R shoulder'][0]],k_c[angle_dict['R shoulder'][1]],k_c[angle_dict['R shoulder'][2]]))
                        angle_save['R elbow'] = int(angle_cal(k_c[angle_dict['R elbow'][0]],k_c[angle_dict['R elbow'][1]],k_c[angle_dict['R elbow'][2]]))
                        angle_save['R pelvis'] = int(angle_cal(k_c[angle_dict['R pelvis'][0]],k_c[angle_dict['R pelvis'][1]],k_c[angle_dict['R elbow'][2]]))
                        angle_save['R knee'] = int(angle_cal(k_c[angle_dict['R knee'][0]],k_c[angle_dict['R knee'][1]],k_c[angle_dict['R elbow'][2]]))
                        angle_save['L shoulder'] = int(angle_cal(k_c[angle_dict['L shoulder'][0]],k_c[angle_dict['L shoulder'][1]],k_c[angle_dict['L shoulder'][2]]))
                        angle_save['L elbow'] = int(angle_cal(k_c[angle_dict['L elbow'][0]],k_c[angle_dict['L elbow'][1]],k_c[angle_dict['L elbow'][2]]))
                        angle_save['L pelvis'] = int(angle_cal(k_c[angle_dict['L pelvis'][0]],k_c[angle_dict['L pelvis'][1]],k_c[angle_dict['L elbow'][2]]))
                        angle_save['L knee'] = int(angle_cal(k_c[angle_dict['L knee'][0]],k_c[angle_dict['L knee'][1]],k_c[angle_dict['L elbow'][2]]))

                        #----------------------------------------------------연훈이형 코드(:코의 속도가 음수일때, 현재 무릎 각도를 측정하여/높이를 낮추고 있다면, 아직 부족하다를 보여줌)
                        nose_arr.append(k_c[0][0])#코의 y좌표 저장

                    #================================================================
                    #정상 포즈 판별
                    end_time = time.time()

                    if real_start == False:
                        if 'R shoulder' in angle_save:
                            if angle_save['R shoulder'] >=0 and angle_save['R shoulder']<=35:
                                R_shoulder = True
                            else:
                                time_count = 0
                                print(str(angle_save['R shoulder']),'R shoulder False')
                                R_shoulder = False

                        if 'L shoulder' in angle_save:
                            if angle_save['L shoulder'] >=0 and angle_save['L shoulder']<=35:
                                L_shoulder = True
                            else:
                                time_count = 0
                                print(str(angle_save['L shoulder']),'L shoulder False')
                                L_shoulder = False

                        if 'R elbow' in angle_save:
                            if angle_save['R elbow'] >=145 and angle_save['R elbow']<=205:
                                R_elbow = True
                            else:
                                time_count = 0
                                print(str(angle_save['R elbow']),'R elbow F')
                                R_elbow = False

                        if 'L elbow' in angle_save:
                            if angle_save['L elbow'] >=145 and angle_save['L elbow']<=205:
                                L_elbow = True
                            else:
                                time_count = 0
                                print(str(angle_save['L elbow']),'L elbow F')
                                L_elbow = False

                        if 'R pelvis' in angle_save:
                            if angle_save['R pelvis'] >=65 and angle_save['R pelvis']<=140:
                                R_pelvis = True
                            else:
                                time_count = 0
                                print(str(angle_save['R pelvis']),'R pelvis F')
                                R_pelvis = False

                        if 'L pelvis' in angle_save:
                            if angle_save['L pelvis'] >=65 and angle_save['L pelvis']<=140:
                                L_pelvis = True
                            else:
                                time_count = 0
                                print(str(angle_save['L pelvis']),'L pelvis F')
                                L_pelvis = False

                        if 'R knee' in angle_save:
                            if angle_save['R knee'] >=140 and angle_save['R knee']<=200:
                                R_knee = True
                            else:
                                time_count = 0
                                print(str(angle_save['R knee']),'R knee F')
                                R_knee = False

                        if 'L knee' in angle_save:
                            if angle_save['L knee'] >=140 and angle_save['L knee']<=200:
                                L_knee = True
                            else:
                                time_count = 0
                                print(str(angle_save['L knee']),'L knee F')
                                L_knee = False

                        if R_shoulder & R_elbow & R_pelvis & R_knee & L_shoulder & L_elbow & L_pelvis & L_knee:
                            time_count += end_time - start_time
                            print(int(time_count))

                            out_img = cv2.putText(out_img,
                                    str(int(time_count)),
                                    (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
                            
                            if time_count > 3:
                                real_start = True

                        else:
                            time_count = 0
                            out_img = cv2.putText(out_img,
                                    'Please Re-pose',
                                    (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 1, cv2.LINE_AA)
                    else:
                        if len(nose_arr)>10:    
                            grad = (mean(nose_arr[-10:-5]) - mean(nose_arr[-5:-1]))/10
                            print(nose_arr[-1])
                            #grad_check.append(grad)

                            # 자세 판별
                            ready_knee_angle = 160
                            squat_knee_angle = 90

                        #if len(grad_check)>10:
                            global test  
                            test, color = squat_down(out_img, nose_arr[-1], nose_arr, grad, squat_knee_angle, ready_knee_angle, angle_save["L knee"],  angle_save["R knee"], angle_save["L hip"], angle_save["R hip"])
                            out_img=cv2.putText(out_img, test, (75,100), cv2.FONT_HERSHEY_DUPLEX, 3, color=color, thickness=3)
                    
                        
                        #------------------------------------------왼쪽 무릎 코드( 수정됨 : 준영 )
                        
                        L_knee_flag=3
                        R_knee_flag=3
                        errocounter=0#에러표시 유지값
                        L_knee_mins=(82,110) # 무릎각도 최저 정상범위 설정
                        L_knee_maxs=(170,180)
                        R_knee_mins=(82,129) # 무릎각도 최저 정상범위 설정
                        R_knee_maxs=(170,180)

                        #운동체커 0:내려가는중 1:올라가는중 2:최저 정상범위 3:최고 정상범위 4:운동 오류
                        if angle_save and (set(errorlist)==set(errorlist)-set(angle_dict["L knee"])):#errorlist에 해당하는 각을 구성하는 성분이 없으면
                            if angle_save["L knee"] <= L_knee_mins[1]:#무릎각도 최저범위 내 일때
                                if L_knee_flag == 1: #올라가는 중이라면
                                    L_knee_flag=4 #에러
                                    errocounter=10
                                elif L_knee_flag == 0: #내려가는 중이라면
                                    L_knee_flag=2 #최저 정상범위
                            elif L_knee_flag==2:#이미 최저 범위였고
                                if angle_save["L knee"]>L_knee_mins[1]+10:  #최저 범위 상위 탈출
                                    L_knee_flag=1 #올라감
                                elif angle_save["L knee"]<L_knee_mins[0]:
                                    L_knee_flag=4 #에러
                                    errocounter=10
                            if angle_save["L knee"] >= L_knee_maxs[0]:#무릎각도 최고범위 내 일때
                                if L_knee_flag == 0: #내려가는 중이라면
                                    L_knee_flag=4 #에러
                                    errocounter=10  
                                elif L_knee_flag == 1: #올라가는 중이라면
                                    L_knee_flag=3 #최고 정상범위
                            elif L_knee_flag==3:#이미 최고 범위였고
                                if angle_save["L knee"]<L_knee_maxs[0]-10:  #최고 범위 하위 탈출
                                    L_knee_flag=0 #내려감
                            if L_knee_flag==4:
                                if errocounter>0:
                                    out_img = cv2.putText(out_img, "False", (100,300), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,255),3)
                                    errocounter-=1
                                    
                                else:
                                    L_knee_flag=3
                            else:
                                out_img = cv2.putText(out_img, "True", (100,300), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0),2)
                        else:
                            real_start = False
                           
                                
                        
                        #------------------------------------------오른쪽 무릎 코드( 수정됨 : 준영 )
                        if angle_save and (set(errorlist)==set(errorlist)-set(angle_dict["R knee"])):#
                            if angle_save["R knee"] <= R_knee_mins[1]:#무릎각도 최저범위 내 일때
                                if R_knee_flag == 1: #올라가는 중이라면
                                    R_knee_flag=4 #에러
                                    errocounter=10
                                elif R_knee_flag == 0: #내려가는 중이라면
                                    R_knee_flag=2 #최저 정상범위
                            elif R_knee_flag==2:#이미 최저 범위였고
                                if angle_save["R knee"]>R_knee_mins[1]+10:  #최저 범위 상위 탈출
                                    R_knee_flag=1 #올라감
                                elif angle_save["R knee"]<R_knee_mins[0]:
                                    R_knee_flag=4 #에러
                                    errocounter=10
                            if angle_save["R knee"] >= R_knee_maxs[0]:#무릎각도 최고범위 내 일때
                                if R_knee_flag == 0: #내려가는 중이라면
                                    R_knee_flag=4 #에러
                                    errocounter=10  
                                elif R_knee_flag == 1: #올라가는 중이라면
                                    R_knee_flag=3 #최고 정상범위
                            elif R_knee_flag==3:#이미 최고 범위였고
                                if angle_save["R knee"]<R_knee_maxs[0]-10:  #최고 범위 하위 탈출 
                                    R_knee_flag=0 #내려감
                            if R_knee_flag==4:
                                if errocounter>0:
                                    out_img = cv2.putText(out_img, "False", (50,300), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255),2)
                                    errocounter-=1
                                else:
                                    R_knee_flag=3
                            else :
                                out_img = cv2.putText(out_img, "True", (50,300), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0),2)
                        else:
                            real_start = False    
                        
                    #----------------------------------------------------좌표 위치 그대로 그려주는 코드

                    out_img = cv2.drawKeypoints(out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    out_img = cv2.drawKeypoints(out_img, cv_keypoints_else, outImage=np.array([]), color=(0, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
                    out_img = cv2.polylines(out_img, adjacent_keypoints_else, isClosed=False, color=(0, 0, 255))
                    
                    #================================================================
                    
                     
                    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                    
                    qImg = QtGui.QImage(out_img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    myLabel.setPixmap(pixmap)
                
            cap.release()
            print("Thread end.")

app = QApplication(sys.argv)
myWindow = MyWindow()
myWindow.show() 

app.exec_()