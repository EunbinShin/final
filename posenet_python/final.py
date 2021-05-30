import cv2
from numpy.core.fromnumeric import mean
import pafy
import pandas as pd
# url = "https://www.youtube.com/watch?v=FksYBwUjJZc"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")

video_file = "video/LUNGE_1.mp4"

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import argparse
import posenet
from math import atan2, degrees
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


def leg_points(data):
    # data2 = data.iloc[:100,:]
    # max_point = data[data['nose_y']==max(data['nose_y'])].index
    left_ankle_x, left_ankle_y, left_knee_x, left_knee_y ,left_hip_x, left_hip_y = data.loc[:,['leftAnkle_x','leftAnkle_y','leftKnee_x','leftKnee_y', 'leftHip_x','leftHip_y']].iloc[-1]
    right_ankle_x, right_ankle_y, right_knee_x, right_knee_y ,right_hip_x, right_hip_y = data.loc[:,['leftAnkle_x','leftAnkle_y','leftKnee_x','leftKnee_y', 'leftHip_x','leftHip_y']].iloc[-1]
    
    return left_ankle_x, left_ankle_y, left_knee_x, left_knee_y ,left_hip_x, left_hip_y, right_ankle_x, right_ankle_y, right_knee_x, right_knee_y ,right_hip_x, right_hip_y

def shoulder_points(data):
    left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y = data.loc[:,['leftShoulder_x','leftShoulder_y','rightShoulder_x','rightShoulder_y']].iloc[-1]
    return left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y

def body_points(data):
    left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y = data.loc[:,['leftShoulder_x','leftShoulder_y','rightShoulder_x','rightShoulder_y']].iloc[-1]
    return left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y

def get_nose(data):
    nose_x, nose_y = data.loc[:,['nose_x','nose_y']].iloc[-1]
    return nose_x, nose_y

def angle_between(x1,y1, x2, y2, x3,y3):
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    degree = 360 - (deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2))
    return degree

def torso_area(points):  
    """Return the area of the polygon whose vertices are given by the
    sequence points.
    """
    area = 0
    q = points[-1]
    for p in points:  
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2

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
                color = (0,0,255)

            elif left_knee_angle < squat_knee_angle < 0.95 and right_knee_angle < squat_knee_angle < 0.95:
                test="Higher"
                color = (0,0,255)
            elif squat_knee_angle * 0.95 <= left_knee_angle < squat_knee_angle * 1.2 and squat_knee_angle * 0.95 <= right_knee_angle < squat_knee_angle * 1.2 and \
                squat_side_angle * 0.9 <left_side_angle < squat_side_angle * 1.1 or squat_side_angle * 0.9 <right_side_angle < squat_side_angle * 1.1:
                test="Good"
                color = (255,0,0)
        else:
            test= ""
            color = (255,0,0)
    else:
        test = ""
        color = ""
    return test, color

def squat_up(out_img, grad, test, squat_knee_angle, ready_knee_angle, left_knee_angle, right_knee_angle):
 # 올라올 때
    #     tqqqqest=""
    #     color = (255,255,255)

    # else:
    #     test= ""
    #     color = (255,255,255)
    
    return test

def to_df(point_x, point_y, x_arr, y_arr):
    x_temp, y_temp = np.array(point_x), np.array(point_y)
    x_arr=np.vstack([x_arr, x_temp])
    y_arr=np.vstack([y_arr, y_temp])
    x_df = pd.DataFrame(x_arr, columns=col_name)
    y_df = pd.DataFrame(y_arr, columns=col_name)
    x_df['frame_num']=range(x_df.shape[0])
    y_df['frame_num']=range(y_df.shape[0])
    mg_df = pd.merge(x_df, y_df, on=['frame_num']).iloc[1:,:]

    return mg_df

parts = ['nose', 'L eye', 'R eye', 'L ear', 'R ear', 
    'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L wrist', 'R wrist',
    'L pelvis', 'R pelvis', 'L knee', 'R knee', 'L ankle', 'R ankle']

col_name = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", 
    "leftShoulder","rightShoulder", "leftElbow", "rightElbow", "leftWrist", 
    "rightWrist","leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]

def posenet_search():
    with tf.Session() as sess: #텐서플로우의 세션을 변수에 정의
        model_cfg, model_outputs = posenet.load_model(args.model, sess)  #model_outputs는 텐서 객체들의 리스트
        output_stride = model_cfg['output_stride']
        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(0) # 
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(300/fps)
        start = time.time()
        frame_count = 0
        peaple_count=1
        bf_keyscores=np.zeros((peaple_count,17),float)
        bf_keycoords=np.zeros((peaple_count,17,2),float)
        min_pose_score=0.3
        min_part_score=0.1
        x_arr = np.zeros((17))
        y_arr = np.zeros((17))
        # left_knee_angle_arr = []
        # right_knee_angle_arr = []
        # left_side_angle_arr = []
        # right_side_angle_arr = []
        nose_arr = []
        grad_check = []
        # 프레임별 읽기
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride) #영상

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
            adjacent_keypoints = []
            cv_keypoints = []
            
            # 프레임별 ks, kc 계산
            for ii, score in enumerate(pose_scores):
                for jj,(ks, kc) in enumerate(zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :])):
                    if ks > min_part_score:
                        bf_keyscores[ii][jj]=ks
                        bf_keycoords[ii][jj]=kc
            
            for ii, score in enumerate(pose_scores):
                if score < min_part_score:
                    overlay_image=out_img 
                    bf_keyscores=np.zeros((peaple_count,17),float)
                    bf_keycoords=np.zeros((peaple_count,17,2),float)
                    continue
                results = []
                k_s= bf_keyscores[ii, :]
                k_c= bf_keycoords[ii, :, :]
                
                for left, right in posenet.CONNECTED_PART_INDICES: #선찾기
                    if k_c[left][0] == 0 or k_c[right][1] == 0:
                        continue
                    results.append(np.array([k_c[left][::-1], k_c[right][::-1]]).astype(np.int32),)
                new_keypoints = results
                adjacent_keypoints.extend(new_keypoints)

                # 좌표 받기
                point_x = []
                point_y = []

                # 각 부위별로 점을 찍는 곳
                for jj, (ks, kc) in enumerate(zip(k_s, k_c)):#점찾기
                    if kc[0]==0 and kc[0]==1:
                        continue
                    cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
                    point_x.append(kc[1])
                    point_y.append(kc[0])
                    # points = np.c_[points,pos]
                    out_img = cv2.putText(out_img, parts[jj], (int(kc[1]), int(kc[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)
                
                mg_df = to_df(point_x, point_y, x_arr, y_arr)
                left_ankle_x, left_ankle_y, left_knee_x, left_knee_y ,left_hip_x, left_hip_y, right_ankle_x, right_ankle_y, right_knee_x, right_knee_y, right_hip_x, right_hip_y = leg_points(mg_df)
                left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y = shoulder_points(mg_df)
                left_knee_angle = angle_between(left_ankle_x, left_ankle_y, left_knee_x, left_knee_y, left_hip_x, left_hip_y) # 왼쪽 무릎 각도
                right_knee_angle = angle_between(right_ankle_x,right_ankle_y, right_knee_x,right_knee_y, right_hip_x, right_hip_y) # 왼쪽 무릎 각도
                left_side_angle = 360 - angle_between(left_knee_x,left_knee_y,left_hip_x, left_hip_y, left_shoulder_x, left_shoulder_y) # 왼쪽 옆구리 각도
                right_side_angle = 360 - angle_between(right_knee_x,right_knee_y,right_hip_x,right_hip_y, right_shoulder_x, right_shoulder_y) # 오른쪽 옆구리 각도
                nose_x, nose_y = get_nose(mg_df) # 코 좌표 구하기
                nose_arr.append(nose_y) # 코 좌표 집어넣기

            if len(nose_arr)>10: 
                
                grad = (mean(nose_arr[-10:-5]) - mean(nose_arr[-5:-1]))/10
                grad_check.append(grad)

                # 자세 판별
                ready_knee_angle = 160
                squat_knee_angle = 90
                # left_knee_angle_arr.append(left_knee_angle)

            if len(grad_check)>10:
                # Down ------------------------------------------------------------------------------------------------------------------------
                global test
                test, color = squat_down(out_img, grad, squat_knee_angle, ready_knee_angle, left_knee_angle, right_knee_angle, left_side_angle, right_side_angle)
                cv2.putText(out_img, test, (75,100), cv2.FONT_HERSHEY_DUPLEX, 4, color=color, thickness=3)

            out_img = cv2.drawKeypoints(
                out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            overlay_image = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
            
            cv2.imshow('posenet', overlay_image)
            cv2.waitKey(delay)
            frame_count += 1
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

        # print(x_arr)
        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    posenet_search()
