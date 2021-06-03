#할일 : 다양한 부위를 추가시켜 복합적인 측정하기, 기존 이전 값을 살리는 코드 활용하기

#-------------------------------------------import 구역
import cv2
from numpy.core.fromnumeric import mean
import pafy
import numpy as np
import pandas as pd
import math
from math import atan2, degrees, e
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

def squat_down(out_img, grad, squat_knee_angle, ready_knee_angle, left_knee_angle, right_knee_angle, left_side_angle, right_side_angle):
    global color
    global test
    squat_side_angle = 90

    if grad <= -2: # 내려갈 때
        # 시작 위치
        # if left_knee_angle > ready_knee_angle:
        #     test = "Ready"
        #     color = (255,255,255)
        # 스쿼트 판단 하기
        if grad <= -5:
            if left_knee_angle > squat_knee_angle * 1.2 and right_knee_angle > squat_knee_angle * 1.2: 
                test="Lower"
                color = (0,250,250)

            elif left_knee_angle < squat_knee_angle < 0.95 and right_knee_angle < squat_knee_angle < 0.95:
                test="Higher"
                color = (0,0,255)
            elif squat_knee_angle * 0.95 <= left_knee_angle < squat_knee_angle * 1.2 and squat_knee_angle * 0.95 <= right_knee_angle < squat_knee_angle * 1.2 and \
                squat_side_angle * 0.9 <left_side_angle < squat_side_angle * 1.1 or squat_side_angle * 0.9 <right_side_angle < squat_side_angle * 1.1:
                test="Good"
                color = (0,200,50)
        else:
            test= ""
            color = (255,0,0)
    else:
        test = ""
        color = ""
    return test, color

#-------------------------------------------준영 함수
parts = ['nose', 'L eye', 'R eye', 'L ear', 'R ear', 
    'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L wrist', 'R wrist',
    'L pelvis', 'R pelvis', 'L knee', 'R knee', 'L ankle', 'R ankle']

angle_dict = {#구하고 싶은 각도들의 인덱스를 저장
    'R hip' : (12,6,14),  #오른쪽 옆구리
    'L hip' : (11,5,13),  #왼쪽 옆구리
    'R knee' : (14,16,12),    #오른쪽 무릎
    'L knee' : (13,15,11),     #왼쪽 무릎
    'L pelvis' : (11,13,12),
    'R pelvis' : (12,14,11),
    'L knee' : (13,15,11),
    'R knee' : (14,16,12),
    'R shoulder' : (6,8,12),
    'R elbow' : (8,6,10)
}

def angle_cal(A,B,C):#A에 있는 각도 구하기
    a=math.sqrt((B[0]-C[0])**2+(B[1]-C[1])**2)
    b=math.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)
    c=math.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)
    angle=math.acos((b*b+c*c-a*a)/(2*b*c))
    angle=angle*180/math.pi
    return angle

L_knee_mins=(70,100) # 왼쪽 무릎각도 최저 정상범위 설정
L_knee_maxs=(170,180)
R_knee_mins=(70,100) # 오른쪽 무릎각도 최저 정상범위 설정
R_knee_maxs=(170,180)
L_hip_mins=(79,125) # 왼쪽 옆구리각도 최저 정상범위 설정
L_hip_maxs=(158,166) 
R_hip_mins=(82,125) # 오른쪽 옆구리각도 최저 정상범위 설정
R_hip_maxs=(169,174)

def angle_flag(now_angle,now_mins,now_maxs,now_flag,errocounter): #이준영 반복부분 해결하기위한 것
    if now_angle <= now_mins[1]:#무릎각도 최저범위 내 일때
        if now_flag == 1: #올라가는 중이라면
            now_flag=4 #에러
            errocounter=20
        elif now_flag == 0: #내려가는 중이라면
            now_flag=2 #최저 정상범위
    elif now_flag==2:#이미 최저 범위였고
        if now_angle>now_mins[1]+10:  #최저 범위 상위 탈출
            now_flag=1 #올라감
        elif now_angle<now_mins[0]:
            now_flag=4 #에러
            errocounter=20
    if now_angle >= now_maxs[0]:#무릎각도 최고범위 내 일때
        if now_flag == 0: #내려가는 중이라면
            now_flag=4 #에러
            errocounter=20 
        elif now_flag == 1: #올라가는 중이라면
            now_flag=3 #최고 정상범위
    elif now_flag==3:#이미 최고 범위였고
        if now_angle<now_maxs[0]-10:  #최고 범위 하위 탈출
            now_flag=0 #내려감
    return now_flag,errocounter

def angle_text(x,y,now_flag,errocounter,out_img):
    if now_flag==4:
        if errocounter>0:
            out_img = cv2.putText(out_img, "False", (50+150*x,200+100*y), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0,0,255),2)
            errocounter-=1
        else:
            now_flag=3
    else :
        out_img = cv2.putText(out_img, "True", (50+150*x,200+100*y), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0,255,0),2)
    return now_flag,errocounter,out_img


#-----------------------------------------메인실행 함수
def posenet_search():
    with tf.Session() as sess: #텐서플로우의 세션을 변수에 정의
        final_result=np.zeros(shape=(0,34))
        final_angles=pd.DataFrame(columns=angle_dict.keys())
        model_cfg, model_outputs = posenet.load_model(args.model, sess)  #model_outputs는 텐서 객체들의 리스트
        output_stride = model_cfg['output_stride']
        

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(best.url)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        
        peaple_count=1 #한명만 실행 지금 코드가 그대로 되어있음
        
        
        bf_keyscores=np.zeros((peaple_count,17),float) #이전프레임 저장 기능은 복구
        bf_keycoords=np.zeros((peaple_count,17,2),float)
        

        min_pose_score=0.15 #자세 인식 최소값
        min_part_score=0.1 #관절 포인트 인식 최소값
        
        #-----------------------------------------------------준영
        angle_save={}
        L_knee_flag=3
        R_knee_flag=3
        L_hip_flag=3
        R_hip_flag=3
        L_knee_errocounter=0#에러표시 유지값
        R_knee_errocounter=0
        L_hip_errocounter=0
        R_hip_errocounter=0

        #-----------------------------------------------------연훈
        x_arr = np.zeros((17))
        y_arr = np.zeros((17))
        nose_arr = []
        grad_check = []

        #----------------------------------------------------일치
        #프레임별 읽기
        while True:
            try:
                input_image, display_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride) #영상입력
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
            cv_keypoints = []
            cv_keypoints_else = []#점수보다 낮은 이미지 처리(연훈이형 부분에 없었음)
            adjacent_keypoints = []
            adjacent_keypoints_else = []
            
            
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
            
            
            for ii, score in enumerate(pose_scores):
                if score < min_part_score:
                    # overlay_image=out_img #왜있었는지 모르겠다
                    # bf_keyscores=np.zeros((peaple_count,17),float)
                    # bf_keycoords=np.zeros((peaple_count,17,2),float)
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
                
                angle_save={}
                for i,v in angle_dict.items():
                    angle_save[i]=angle_cal(k_c[v[0]],k_c[v[1]],k_c[v[2]])
                
                final_angles=final_angles.append(angle_save,ignore_index=True)
                final_result=np.vstack((final_result,np.ravel(k_c,order='C')))
            

            #----------------------------------------------------연훈이형 코드(:코의 속도가 음수일때, 현재 무릎 각도를 측정하여/높이를 낮추고 있다면, 아직 부족하다를 보여줌)
                nose_arr.append(k_c[0][0])#코의 y좌표 저장

            if len(nose_arr)>10: 
                
                grad = (mean(nose_arr[-10:-5]) - mean(nose_arr[-5:-1]))/10
                
                grad_check.append(grad)

                # 자세 판별
                ready_knee_angle = 160
                squat_knee_angle = 90
                # left_knee_angle_arr.append(left_knee_angle)

            if len(grad_check)>10:
                global test
                test, color = squat_down(out_img, grad, squat_knee_angle, ready_knee_angle, angle_save["L knee"],  angle_save["R knee"], angle_save["L hip"], angle_save["R hip"])
                out_img=cv2.putText(out_img, test, (50,100), cv2.FONT_HERSHEY_DUPLEX, 2, color=color, thickness=2)

            #----------------------------------------------------좌표 위치 그대로 그려주는 코드

            out_img = cv2.drawKeypoints(out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            out_img = cv2.drawKeypoints(out_img, cv_keypoints_else, outImage=np.array([]), color=(0, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
            out_img = cv2.polylines(out_img, adjacent_keypoints_else, isClosed=False, color=(0, 0, 255))
            
            #<운동체커 0:내려가는중 1:올라가는중 2:최저 정상범위 3:최고 정상범위 4:운동 오류>
            #------------------------------------------왼쪽 무릎 코드
            if angle_save and (set(errorlist)==set(errorlist)-set(angle_dict["L knee"])):#errorlist에 해당하는 각을 구성하는 성분이 없으면
                L_knee_flag,L_knee_errocounter=angle_flag(angle_save["L knee"],L_knee_mins,L_knee_maxs,L_knee_flag,L_knee_errocounter)
                L_knee_flag,L_knee_errocounter,out_img=angle_text(0,0,L_knee_flag,L_knee_errocounter,out_img)

            #------------------------------------------오른쪽 무릎 코드
            if angle_save and (set(errorlist)==set(errorlist)-set(angle_dict["R knee"])):#
                R_knee_flag,R_knee_errocounter=angle_flag(angle_save["R knee"],R_knee_mins,R_knee_maxs,R_knee_flag,R_knee_errocounter)
                R_knee_flag,R_knee_errocounter,out_img=angle_text(0,1,R_knee_flag,R_knee_errocounter,out_img)

            #------------------------------------------왼쪽 옆구리 코드
            if angle_save and (set(errorlist)==set(errorlist)-set(angle_dict['L hip'])):#
                L_hip_flag,L_hip_errocounter=angle_flag(angle_save["L hip"],L_hip_mins,L_hip_maxs,L_hip_flag,L_hip_errocounter)
                L_hip_flag,L_hip_errocounter,out_img=angle_text(1,0,L_hip_flag,L_hip_errocounter,out_img)
            
            #------------------------------------------오른쪽 옆구리 코드
            if angle_save and (set(errorlist)==set(errorlist)-set(angle_dict['R hip'])):#
                R_hip_flag,R_hip_errocounter=angle_flag(angle_save["R hip"],R_hip_mins,R_hip_maxs,R_hip_flag,R_hip_errocounter)
                R_hip_flag,R_hip_errocounter,out_img=angle_text(1,1,R_hip_flag,R_hip_errocounter,out_img)
            print("LN:{},{:.1f}\tRN:{},{:.1f}\tLH:{},{:.1f}\tRT:{},{:.1f}".format(L_knee_flag,angle_save['L knee'],R_knee_flag,angle_save['R knee'],L_hip_flag,angle_save['L hip'],R_hip_flag,angle_save['R hip']))

        #------------------------------------------완성된 이미지 추력    

            cv2.imshow('posenet', out_img)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
        #------------------------------------------데이터 프레임 저장
        col=['nose x','nose y','L eye x','L eyey','R eye x','R eye y','L ear x','L ear y','R ear x','R ear y','L shoulder x','L shoulder y',
            'R shoulder x','R shoulder y','L elbow x','L elbow y','R elbow x','R elbow y','L wrist x','L wrist y','R wrist x','R wrist y',
            'L pelvis x','L pelvis y','R pelvis x','R pelvis y','L knee x','L knee y','R knee x','R knee y','L ankle x','L ankle y','R ankle x','R ankle y']
        sam=pd.DataFrame(data=final_result, columns=col)
        sam.to_csv("JJ.csv",index=False,encoding="utf-8-sig")
        final_angles.to_csv("kk.csv",index=False,encoding="utf-8-sig")
        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    posenet_search()



