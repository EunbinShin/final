import cv2
import pafy
import pandas as pd
url = "https://www.youtube.com/watch?v=FksYBwUjJZc"
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

def posenet_search():
    with tf.Session() as sess: #텐서플로우의 세션을 변수에 정의
        final_result=np.zeros(shape=(0,34))
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
        
        bf_keyscores=np.zeros((peaple_count,17),float)
        bf_keycoords=np.zeros((peaple_count,17,2),float)
        min_pose_score=0.15
        min_part_score=0.1
        
        
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
            
            adjacent_keypoints = []
            cv_keypoints = []
            
            
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

                for jj, (ks, kc) in enumerate(zip(k_s, k_c)):#점찾기
                    if kc[0]==0 and kc[0]==1:
                        continue
                    cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

                    out_img = cv2.putText(out_img, parts[jj], (int(kc[1]), int(kc[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)

                out_img = cv2.drawKeypoints(
                    out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                overlay_image = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))

                #------------------------------------------데이터 프레임 저장                
                final_result=np.vstack((final_result,np.ravel(k_c,order='C')))
                
                

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        col=['nose x','nose y','L eye x','L eyey','R eye x','R eye y','L ear x','L ear y','R ear x','R ear y','L shoulder x','L shoulder y',
            'R shoulder x','R shoulder y','L elbow x','L elbow y','R elbow x','R elbow y','L wrist x','L wrist y','R wrist x','R wrist y',
            'L pelvis x','L pelvis y','R pelvis x','R pelvis y','L knee x','L knee y','R knee x','R knee y','L ankle x','L ankle y','R ankle x','R ankle y']
        sam=pd.DataFrame(data=final_result, columns=col)
        sam.to_csv("JJ.csv",index=False,encoding="utf-8-sig")
        print('Average FPS: ', frame_count / (time.time() - start))




if __name__ == "__main__":
    posenet_search()



