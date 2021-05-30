import cv2
import pafy
import pandas as pd
import math

url = "https://www.youtube.com/watch?v=cMkZ6A7wngk"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import argparse #사용 안함
import posenet

# import easydict
# args = easydict.EasyDict({
#     "model":101,
#     "cam_id":best.url,
#     "cam_width":1280,
#     "cam_height":720,
#     "scale_factor":0.712,
#     "file":None
# })

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

parts = ['nose', 'L eye', 'R eye', 'L ear', 'R ear', 
    'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L wrist', 'R wrist',
    'L pelvis', 'R pelvis', 'L knee', 'R knee', 'L ankle', 'R ankle']

angle_dict = {
    'L pelvis':(12,14,13),
    'R pelvis':(13,15,12),
    'L knee':(14,16,12),
    'R knee':(15,17,13)
}

def angle_cal(A,B,C):#A에 있는 각도 구하기
    a=math.sqrt((B[0]-C[0])**2+(B[1]-C[1])**2)
    b=math.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)
    c=math.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)
    angle=math.acos((b*b+c*c-a*a)/(2*b*c))
    angle=angle*180/math.pi
    return angle
    

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
        
        peaple_count=1
        
        # bf_keyscores=np.zeros((peaple_count,17),float)
        # bf_keycoords=np.zeros((peaple_count,17,2),float)
        min_pose_score=0.15
        min_part_score=0.1
        
        angle_save={}
        knee_flag=3
        errocounter=0
        while True:
            try:
                input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=args.scale_factor, output_stride=output_stride) #영상
            except:
                break

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

            #---------------------------------------------------------------------------------------------------
            out_img = display_image
            
            cv_keypoints = []
            cv_keypoints_else = []

            adjacent_keypoints = []
            adjacent_keypoints_else = []
            
            # for ii, score in enumerate(pose_scores): #이전 프레임 저장 부분 제거
            #     for jj,(ks, kc) in enumerate(zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :])):
            #         if ks > min_part_score:
            #             bf_keyscores[ii][jj]=ks
            #             bf_keycoords[ii][jj]=kc
            
            for ii, score in enumerate(pose_scores):
                if score < min_part_score:
                    # overlay_image=out_img #왜있었는지 모르겠다
                    # bf_keyscores=np.zeros((peaple_count,17),float)
                    # bf_keycoords=np.zeros((peaple_count,17,2),float)
                    continue
                
                results = []
                results_else = []

                k_s= keypoint_scores[ii, :] #bf_keyscores[ii, :]
                k_c= keypoint_coords[ii, :, :] #bf_keycoords[ii, :, :]
                
                for left, right in posenet.CONNECTED_PART_INDICES: #선찾기
                    if k_s[left] < min_part_score or k_s[right] < min_part_score:
                        results_else.append(np.array([k_c[left][::-1], k_c[right][::-1]]).astype(np.int32),)
                    else :
                        results.append(np.array([k_c[left][::-1], k_c[right][::-1]]).astype(np.int32),)
            
                adjacent_keypoints.extend(results)
                adjacent_keypoints_else.extend(results_else)

                for jj, (ks, kc) in enumerate(zip(k_s, k_c)):#점찾기
                    if ks < min_part_score:
                        cv_keypoints_else.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
                    else:
                        cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

                    out_img = cv2.putText(out_img, parts[jj], (int(kc[1]), int(kc[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)
                
                #------------------------------------------일시저장 저장
                angle_save={}
                for i,v in angle_dict.items():
                    angle_save[i]=angle_cal(k_c[v[0]-1],k_c[v[1]-1],k_c[v[2]-1])
                
                final_angles=final_angles.append(angle_save,ignore_index=True)
                final_result=np.vstack((final_result,np.ravel(k_c,order='C')))

                
            out_img = cv2.drawKeypoints(out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            out_img = cv2.drawKeypoints(out_img, cv_keypoints_else, outImage=np.array([]), color=(0, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
            out_img = cv2.polylines(out_img, adjacent_keypoints_else, isClosed=False, color=(0, 0, 255))
            #------------------------------------------
            knee_mins=(82,110) # 무릎각도 최저 정상범위 설정
            knee_maxs=(170,181)
            
            if angle_save:#운동체커 0:내려가는중 1:올라가는중 2:최저 정상범위 3:최고 정상범위 4:운동 오류
                if angle_save["L knee"] <= knee_mins[1]:#무릎각도 최저범위 내 일때
                    if knee_flag == 1: #올라가는 중이라면
                        knee_flag=4 #에러
                        errocounter=10
                    elif knee_flag == 0: #내려가는 중이라면
                        knee_flag=2 #최저 정상범위
                elif knee_flag==2:#이미 최저 범위였고
                    if angle_save["L knee"]>knee_mins[1]+10:  #최저 범위 상위 탈출
                        knee_flag=1 #올라감
                    elif angle_save["L knee"]<knee_mins[0]:
                        knee_flag=4 #에러
                        errocounter=10
                if angle_save["L knee"] >= knee_maxs[0]:#무릎각도 최고범위 내 일때
                    if knee_flag == 0: #내려가는 중이라면
                        knee_flag=4 #에러
                        errocounter=10  
                    elif knee_flag == 1: #올라가는 중이라면
                        knee_flag=3 #최고 정상범위
                elif knee_flag==3:#이미 최고 범위였고
                    if angle_save["L knee"]<knee_maxs[0]-10:  #최고 범위 하위 탈출
                        knee_flag=0 #내려감
                
                if knee_flag==4:
                    if errocounter>0:
                        out_img = cv2.putText(out_img, "False", (100,100), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,255))
                        errocounter-=1
                    else:
                        knee_flag=3
                else :
                    out_img = cv2.putText(out_img, "True", (100,110), cv2.FONT_HERSHEY_DUPLEX, 4, (0,255,0))
                
                print(str(knee_flag)+" "+str(angle_save["L knee"]))
            
            cv2.imshow('posenet', out_img)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        col=['nose x','nose y','L eye x','L eyey','R eye x','R eye y','L ear x','L ear y','R ear x','R ear y','L shoulder x','L shoulder y',
            'R shoulder x','R shoulder y','L elbow x','L elbow y','R elbow x','R elbow y','L wrist x','L wrist y','R wrist x','R wrist y',
            'L pelvis x','L pelvis y','R pelvis x','R pelvis y','L knee x','L knee y','R knee x','R knee y','L ankle x','L ankle y','R ankle x','R ankle y']

        #------------------------------------------데이터 프레임 저장
        sam=pd.DataFrame(data=final_result, columns=col)
        sam.to_csv("JJ.csv",index=False,encoding="utf-8-sig")
        final_angles.to_csv("kk.csv",index=False,encoding="utf-8-sig")
        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    posenet_search()



