#!/usr/bin/env python

# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix

from previous_state import PeopleState





script_path = os.path.dirname(os.path.realpath(__file__))


def load_camera_calibration(calibration_file):
    import yaml
    with open(calibration_file, 'r') as f:
        cal = yaml.safe_load(f)

    dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
    camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

    return dist_coefficients, camera_matrix


def extract_eye_image_patches(subjects):
    for subject in subjects:
        le_c, re_c, leftcenter_coor, rightcenter_coor, _, _ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c
        subject.leftcenter_coor = leftcenter_coor
        subject.rightcenter_coor = rightcenter_coor




def init_previous(num,bbox_l_list):
    people_list.clear()

    for i in range(num):
        people_list.append(PeopleState(bbox_l_list[i]))

def append_people(faceboxes):
    for facebox in faceboxes:
        append_flag =True
        for people in people_list:
            if people.isTheSame(facebox[0]):
                append_flag = False
                break
        if append_flag:
            people_list.append(PeopleState(facebox[0]))


def del_people(faceboxes):
    i=0
    while i < len(people_list):
        del_flag = True
        for facebox in faceboxes:
            if people_list[i].isTheSame(facebox[0]):
               del_flag =False
               break
        if del_flag:
            del people_list[i]
        else:
            i+=1

def  check_people(faceboxes):
    append_people(faceboxes)
    del_people(faceboxes)

def get_people(facebox_l):
    for idx in range(len(people_list)):
        if people_list[idx].isTheSame(facebox_l):
            return idx


# head_theta =[]
# head_phi =[]
# gaze_theta = []
# gaze_phi =[]
people_list =[]
FPS = "0"
gaze_error =[]
headpose_error =[]

EVAL =False
SHIFT =False
def estimate_gaze(base_name, color_img, dist_coefficients, camera_matrix ,label=False):
    global FPS


    # cv2.putText(color_img, "FPS : "+FPS, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    # start = time.time()

    #face box의 위치를 반환.(모든 대상 list로 반환) -[left_x, top_y, right_x, bottom_y]
    # faceboxes = landmark_estimator.get_face_bb(color_img)

    # if len(faceboxes) == 0:
    faceboxes = [[0,0,120,36]]
        # tqdm.write('Could not find faces in the image')
        # # if EVAL and not SHIFT:
        #     # head_phi.append(-100)
        #     # head_theta.append(-100)
        #     # gaze_phi.append(-100)
        #     # gaze_theta.append(-100)
        # return

    check_people(faceboxes)

    subjects = landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
    extract_eye_image_patches(subjects)

    input_r_list = []
    input_l_list = []
    input_head_list = []
    valid_subject_list = []

    people_count = 1;
    frame_img =color_img

    for idx, subject in enumerate(subjects):
        # people_idx = get_people(faceboxes[idx][0])
        # people_list[people_idx].set_bbox_l(faceboxes[idx][0])

        if subject.left_eye_color is None or subject.right_eye_color is None:
            tqdm.write('Failed to extract eye image patches')
            continue

        success, rotation_vector, _ = cv2.solvePnP(landmark_estimator.model_points,
                                                   subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                   cameraMatrix=camera_matrix,
                                                   distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_DLS)

        if not success:
            tqdm.write('Not able to extract head pose for subject {}'.format(idx))
            continue

        _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
        _m = np.zeros((4, 4))
        _m[:3, :3] = _rotation_matrix
        _m[3, 3] = 1
        # Go from camera space to ROS space
        _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]
        roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
        roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

        phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)
        # if EVAL:
            # head_phi.append(phi_head)
            # head_theta.append(theta_head)
        # face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        # head_pose_image = landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))

        #color_image의 facebox에 headpose vector를 그림.
        if EVAL:
            head_pose_image, headpose_err = landmark_estimator.visualize_headpose_result(frame_img,faceboxes[idx], (phi_head, theta_head), people_list[people_idx],label)
            print(headpose_err)
        else:
            head_pose_image, headpose_err = landmark_estimator.visualize_headpose_result(frame_img,faceboxes[idx], (phi_head, theta_head), people_list[people_idx])

        frame_img = head_pose_image
        if EVAL:
            headpose_error.append(headpose_err)
            print("head pose error:",headpose_err)

        if args.mode =='image':
            #show headpose
            # if args.vis_headpose:
            #     plt.axis("off")
            #     plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
            #     plt.show()

            if args.save_headpose:
                cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0]+str(people_count) + '_headpose.jpg'), head_pose_image)
        people_count +=1
        #size 등 format 변경.
        input_r_list.append(gaze_estimator.input_from_image(subject.right_eye_color))
        input_l_list.append(gaze_estimator.input_from_image(subject.left_eye_color))
        input_head_list.append([theta_head, phi_head])
        valid_subject_list.append(idx)


    # if args.mode =='video':
    #     # plt.axis("off")
    #     # plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
    #     # plt.show()
    #     headpose_out_video.write(frame_img)

    if len(valid_subject_list) == 0:
        return

    # returns [subject : [gaze_pose]]
    gaze_est = gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                    inference_input_right_list=input_r_list,

                                                    inference_headpose_list=input_head_list)
    people_count = 1
    for subject_id, gaze, headpose in zip(valid_subject_list, gaze_est.tolist(), input_head_list):
        subject = subjects[subject_id]
        facebox = faceboxes[subject_id]
        people_idx = get_people(facebox[0])
        # Build visualizations
        # r_gaze_img = gaze_estimator.visualize_eye_result(subject.right_eye_color, gaze)
        # l_gaze_img = gaze_estimator.visualize_eye_result(subject.left_eye_color, gaze)
        # if EVAL:
        #     gaze_theta.append(gaze[0])
        #     gaze_phi.append(gaze[1])
        if EVAL:
            r_gaze_img, r_gaze_err = gaze_estimator.visualize_eye_result(frame_img, gaze, subject.leftcenter_coor, facebox,people_list[people_idx], "gaze_r", label)
            l_gaze_img, l_gaze_err = gaze_estimator.visualize_eye_result(r_gaze_img, gaze, subject.rightcenter_coor, facebox,people_list[people_idx], "gaze_l", label)
        else:
            r_gaze_img, r_gaze_err = gaze_estimator.visualize_eye_result(frame_img, gaze, subject.leftcenter_coor, facebox,people_list[people_idx], "gaze_r")
            l_gaze_img, l_gaze_err = gaze_estimator.visualize_eye_result(r_gaze_img, gaze, subject.rightcenter_coor, facebox,people_list[people_idx], "gaze_l")

        frame_img = l_gaze_img
        if EVAL:
            print("right gaze error:",r_gaze_err)
            print("left gaze error:",l_gaze_err)
            gaze_error.append(r_gaze_err)
            gaze_error.append(l_gaze_err)

        #show gaze image
        # if args.vis_gaze:
        #     plt.axis("off")
        #     plt.imshow(cv2.cvtColor(s_gaze_img, cv2.COLOR_BGR2RGB))
        #     plt.show()
        if args.mode =='image':
            if args.save_gaze:
                cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0]+str(people_count) + '_gaze.jpg'), frame_img)
                # cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_left.jpg'), subject.left_eye_color)
                # cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_right.jpg'), subject.right_eye_color)


        if args.save_estimate:
            with open(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_output.txt'), 'w+') as f:
                f.write(os.path.splitext(base_name)[0] + ', [' + str(headpose[1]) + ', ' + str(headpose[0]) + ']' +

                        ', [' + str(gaze[1]) + ', ' + str(gaze[0]) + ']' + '\n')
        people_count +=1
    if args.mode =='video':
        out_video.write(frame_img)
    # end = time.time()
    # delay_time = end-start
    # FPS = str(int(1/delay_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('im_path', type=str, default=os.path.abspath(os.path.join(script_path, './samples_gaze/')),
                        nargs='?', help='Path to an image or a directory containing images')
    parser.add_argument('video_path', type=str, default=os.path.abspath(os.path.join(script_path, './samples_video/')),
                        nargs='?', help='Path to an video or a directory containing videos')
    parser.add_argument('--calib-file', type=str, dest='calib_file', default=None, help='Camera calibration file')
    parser.add_argument('--vis-headpose', dest='vis_headpose', action='store_true', help='Display the head pose images')
    parser.add_argument('--no-vis-headpose', dest='vis_headpose', action='store_false', help='Do not display the head pose images')
    parser.add_argument('--save-headpose', dest='save_headpose', action='store_true', help='Save the head pose images')
    parser.add_argument('--no-save-headpose', dest='save_headpose', action='store_false', help='Do not save the head pose images')
    parser.add_argument('--vis-gaze', dest='vis_gaze', action='store_true', help='Display the gaze images')
    parser.add_argument('--no-vis-gaze', dest='vis_gaze', action='store_false', help='Do not display the gaze images')
    parser.add_argument('--save-gaze', dest='save_gaze', action='store_true', help='Save the gaze images')
    parser.add_argument('--save-estimate', dest='save_estimate', action='store_true', help='Save the predictions in a text file')
    parser.add_argument('--no-save-gaze', dest='save_gaze', action='store_false', help='Do not save the gaze images')
    parser.add_argument('--gaze_backend', choices=['tensorflow', 'pytorch'], default='pytorch')
    parser.add_argument('--mode', choices=['video', 'image'], default='image')
    parser.add_argument('--output_path', type=str, default=os.path.abspath(os.path.join(script_path, './samples_gaze/out')),
                        help='Output directory for head pose and gaze images')
    parser.add_argument('--models', nargs='+', type=str, default=[os.path.abspath(os.path.join(script_path, '../model_nets/Model_allsubjects1.h5'))],
                        help='List of gaze estimators')
    parser.add_argument('--device-id-facedetection', dest="device_id_facedetection", type=str, default='cuda:0', help='Pytorch device id. Set to "cpu:0" to disable cuda')

    parser.set_defaults(vis_gaze=True)
    parser.set_defaults(save_gaze=True)
    parser.set_defaults(vis_headpose=False)
    parser.set_defaults(save_headpose=True)
    parser.set_defaults(save_estimate=False)

    args = parser.parse_args()


    image_path_list = []
    video_path_list = []


    if args.mode == 'image':
        if os.path.isfile(args.im_path):
            image_path_list.append(os.path.split(args.im_path)[1])
            args.im_path = os.path.split(args.im_path)[0]
        elif os.path.isdir(args.im_path):
            for image_file_name in sorted(os.listdir(args.im_path)):
                if image_file_name.endswith('.jpg') or image_file_name.endswith('.png'):
                    if '_gaze' not in image_file_name and '_headpose' not in image_file_name:
                        image_path_list.append(image_file_name)
        else:
            tqdm.write('Provide either a path to an image or a path to a directory containing images')
            sys.exit(1)
    else:
        args.output_path = os.path.abspath(os.path.join(script_path, './samples_video/out'))
        if os.path.isfile(args.video_path):
            video_path_list.append(os.path.split(args.video_path)[1])
            args.video_path_list = os.path.split(video_path_list)[0]
        elif os.path.isdir(args.video_path):
            for video_file_name in sorted(os.listdir(args.video_path)):
                if video_file_name.endswith('.mp4') or video_file_name.endswith('.avi'):
                    if '_gaze' not in video_path_list and '_headpose' not in video_path_list:
                        video_path_list.append(video_file_name)
        else:
            tqdm.write('Provide either a path to an video or a path to a directory containing videos')
            sys.exit(1)
        print("========================video list==================")
        print(video_path_list)
    tqdm.write('Loading networks')
    landmark_estimator = LandmarkMethodBase(device_id_facedetection=args.device_id_facedetection,
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "rt_gene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "rt_gene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "rt_gene/model_nets/face_model_68.txt")))

    if args.gaze_backend == "tensorflow":
        from rt_gene.estimate_gaze_tensorflow import GazeEstimator

        gaze_estimator = GazeEstimator("/gpu:0", args.models)
    elif args.gaze_backend == "pytorch":
        from rt_gene.estimate_gaze_pytorch import GazeEstimator

        gaze_estimator = GazeEstimator("cuda:0", args.models)
    else:
        raise ValueError("Incorrect gaze_base backend, choices are: tensorflow or pytorch")

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    if args.mode == 'image':
        for image_file_name in tqdm(image_path_list):
            tqdm.write('Estimate gaze on ' + image_file_name)
            image = cv2.imread(os.path.join(args.im_path, image_file_name))
            if image is None:
                tqdm.write('Could not load ' + image_file_name + ', skipping this image.')
                continue

            if args.calib_file is not None:
                _dist_coefficients, _camera_matrix = load_camera_calibration(args.calib_file)
            else:
                im_width, im_height = image.shape[1], image.shape[0]
                # tqdm.write('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
                _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                    [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

            estimate_gaze(image_file_name, image, _dist_coefficients, _camera_matrix)
    else:
        print("=-------------------------video path list--------------------")
        print(video_path_list)

        allVideo_total_error = 0
        for video_file_name in tqdm(video_path_list):
            tqdm.write('Estimate gaze on ' + video_file_name)

            video = cv2.VideoCapture(os.path.join(args.video_path, video_file_name))
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = video.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 코덱 정의

            #head pose + gaze
            out_path =os.path.join(args.output_path, video_file_name)
            out_video = cv2.VideoWriter(out_path, fourcc, fps, (int(width), int(height)))  # VideoWriter 객체 정의

            #head pose와 gaze를 각각 출력하고 싶을 때.
            # gaze_out_path = os.path.join(args.output_path, 'gaze_'+video_file_name)
            # headpose_out_path = os.path.join(args.output_path, 'headpose'+video_file_name)
            # gaze_out_video = cv2.VideoWriter(gaze_out_path, fourcc, fps, (int(width), int(height)))  # VideoWriter 객체 정의
            # headpose_out_video = cv2.VideoWriter(headpose_out_path, fourcc, fps, (int(width), int(height)))  # VideoWriter 객체 정의

            label = []
            if EVAL:
                label_path = "s000_label_combined.txt"
                if SHIFT:
                    # Shift video name :Shift_{num}.mp4
                    # 해당하는 라벨을 고정으로 넘김.
                    image_name = video_file_name.split("_")[1]
                    image_name = image_name.split(".")[0]
                    with open(label_path, "r") as f:
                        for line in f:
                            line = line.split(",")
                            if line[0] != image_name:
                                continue
                            else:
                                label = line
                                print(f"======= label found {line[0]} ======")
                                break

                else:
                    f = open(label_path, "r")
                    #header 제거
                    f.readline()

            count=0
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                #RT_GENE DataSet -> 한줄씩 읽어와 넘김.
                if EVAL:
                    if not SHIFT:
                        label = f.readline()
                        label = label.split(",")

                    for i in range(1,5):
                        label[i] = float(label[i])


                count+=1
                print("frame",count)

                # 대상의 수에 맞게 people_list 초기화
                if count == 1:
                    num, bboxes_l = landmark_estimator.get_init_value(frame)
                    init_previous(num, bboxes_l)
                if not ret:
                    print("Error:: Frame Road Fail")
                    break


                if args.calib_file is not None:
                    _dist_coefficients, _camera_matrix = load_camera_calibration(args.calib_file)

                else:
                    im_width, im_height = frame.shape[1], frame.shape[0]
                    # tqdm.write(
                    #     'WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
                    _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                        [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])
                print(label)
                estimate_gaze(video_file_name, frame, _dist_coefficients, _camera_matrix,label)

            if EVAL:
                average_headpose_err = 0
                average_gaze_err =0
                average_total_err = 0
                frame_num = len(headpose_error)

                for error in headpose_error:
                    average_headpose_err+= error
                for error in gaze_error:
                    average_gaze_err+= error

                average_headpose_err /=frame_num
                average_gaze_err/= (frame_num*2)
                total_error = (average_headpose_err +average_gaze_err) /2
                allVideo_total_error+= total_error

                print("==================Average Error=================")

                print("frame :",frame_num)
                print("Average Headpose Error :",average_headpose_err)
                print("Average Gaze Error :",average_gaze_err)
                print("Average Error(Total) :", total_error)

            video.release()
            out_video.release()

            if EVAL:
                f.close()
                headpose_error.clear()
                gaze_error.clear()
        if EVAL:
             print("all Video Average Error :",allVideo_total_error/len(video_path_list))





            # if EVAL:
            #     label_path = "s000_label_combined.txt"
            #     average_head_phi=0
            #     average_head_theta =0
            #     average_gaze_phi=0
            #     average_gaze_theta=0
            #
            #     frame_num = len(head_theta)
            #     skip_frame =0
            #
            #     image_name = video_file_name.split("_")[1]
            #     image_name = image_name.split(".")[0]

                # with open(label_path,"r") as f:
                #     label=[]
                #
                #     for line in f:
                #         line = line.split(",")
                #         if line[0] != image_name:
                #             continue
                #         else:
                #             label = np.copy(line)
                #             print("======= label found ======")
                #             break
                #     for i in range(frame_num):
                #         if (head_phi[i] == -100):
                #             skip_frame += 1
                #             continue
                #         average_head_phi += abs(head_phi[i] - float(label[1].strip()))
                #         average_head_theta += abs(head_phi[i] - float(label[2].strip()))
                #         average_gaze_phi += abs(head_phi[i] - float(label[3].strip()))
                #         average_gaze_theta += abs(head_phi[i] - float(label[4].strip()))



                # with open(label_path,"r") as f:
                #     #header 제거
                #     line = f.readline()
                #     for i in range(frame_num):
                #         line = f.readline()
                #         if(head_phi[i] == -100):
                #             skip_frame+=1
                #             continue
                #         line = line.split(",")
                #
                #         average_head_phi += abs(head_phi[i] - float(line[1].strip()))
                #         average_head_theta += abs(head_phi[i] - float(line[2].strip()))
                #         average_gaze_phi += abs(head_phi[i] - float(line[3].strip()))
                #         average_gaze_theta += abs(head_phi[i] - float(line[4].strip()))
                #     average_head_phi /=frame_num
                #     average_head_theta /=frame_num
                #     average_gaze_phi /=frame_num
                #     average_gaze_theta /=frame_num
                #
                #     total_error = (average_head_theta + average_head_phi +average_gaze_theta +average_gaze_phi) /4
                #
                #     print("============================= Average Error ==========================")
                #     print("evaluate frame :",frame_num-skip_frame)
                #     print("Head_phi Error =",average_head_phi)
                #     print("Head_theta Error =",average_head_theta)
                #     print("Gaze_phi Error =",average_gaze_phi)
                #     print("Gaze_theta Error =",average_gaze_theta)
                #     print("Total Error =",total_error)





