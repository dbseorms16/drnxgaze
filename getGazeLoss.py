from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelVGG
import torch
from functools import partial
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
import os
import sys
import cv2
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import eye_detection


_loss_fn = {
            "mse": partial(torch.nn.MSELoss, reduction="sum")
        }
_param_num = {
    "mse": 2
}

# RT-GENE weight
# ckpt = "rt_gene/model_nets/Alldata_1px_all_epoch=5-val_loss=0.551.model"

# NPII weight
# ckpt = "rt_gene/model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model"

# for name, param in _model.named_parameters():
    # param.requires_grad = False

# image_root_path = "./train_SRImage"
# left_path = os.path.join(image_root_path, "left","l")
# right_path = os.path.join(image_root_path, "right","r")
# face_path =os.path.join(image_root_path, "face")

# os.makedirs(left_path, exist_ok = True)
# os.makedirs(right_path, exist_ok = True)
# os.makedirs(face_path, exist_ok = True)

# def clearDir():
#         face_list = os.listdir(face_path)
#         left_list = os.listdir(left_path)
#         right_list = os.listdir(right_path)

#         for face in face_list:
#             _face = os.path.join(face_path,face)
#             os.remove(_face)
#         for i in range(len(left_list)):
#             _left = os.path.join(left_path,left_list[i])
#             _right = os.path.join(right_path,right_list[i])
#             os.remove(_left)
#             os.remove(_right)

def generateEyePatches(sr_batch_size):
    le_c_list = None
    re_c_list = None
    detected_list = []
    for i in range(len(sr_batch_size)):
        # faceboxes = [[0,0,224,224]]
        # MPII 데이터셋 변경으로 인한 좌표설정
        faceboxes = [[0,0,120,36]]

        # minim_l, maxim_l, minim_r, maxim_r = eye_detection.eye_detector(sr_batch_size[i])
        # MPII 데이터셋 변경으로 인한 좌표설정
        minim_l, maxim_l, minim_r, maxim_r = [0,0], [59,36], [60, 0], [119,36]
        if(minim_l ==None or minim_r ==None):
            # eye_detector가 양쪽 눈을 검출하지 못한다면, 해당 데이터셋을 skip 한다.
            continue

        # r,g,b 채널에 대하여 각각 crop한 후에, 병합한다.
        left_r = sr_batch_size[i][0][minim_l[1]:maxim_l[1] + 1, minim_l[0]:maxim_l[0] + 1]
        left_g = sr_batch_size[i][1][minim_l[1]:maxim_l[1] + 1, minim_l[0]:maxim_l[0] + 1]
        left_b = sr_batch_size[i][2][minim_l[1]:maxim_l[1] + 1, minim_l[0]:maxim_l[0] + 1]

        right_r = sr_batch_size[i][0][minim_r[1]:maxim_r[1] + 1, minim_r[0]:maxim_r[0] + 1]
        right_g = sr_batch_size[i][1][minim_r[1]:maxim_r[1] + 1, minim_r[0]:maxim_r[0] + 1]
        right_b = sr_batch_size[i][2][minim_r[1]:maxim_r[1] + 1, minim_r[0]:maxim_r[0] + 1]

        le_c = torch.stack([left_r,left_g,left_b])
        re_c = torch.stack([right_r,right_g,right_b])
        # Tensor.cat은 해당 dimention을 증가시키며 tensor를 합한다.
        # Tensor.stack은 새로운 dimention을 생성하여 tensor를 합한다.
        if le_c.shape != (3,36,60) or re_c.shape != (3,36,60):
            continue

        if i ==0 or le_c_list == None:
            le_c_list = le_c
            le_c_list = torch.reshape(le_c_list, (1, 3, 36, 60))
            re_c_list = re_c
            re_c_list = torch.reshape(re_c_list, (1, 3, 36, 60))
        else:
            le_c = torch.reshape(le_c, (1, 3, 36, 60))
            re_c = torch.reshape(re_c, (1, 3, 36, 60))
            le_c_list = torch.cat([le_c_list, le_c], dim=0)
            re_c_list = torch.cat([re_c_list, re_c], dim=0)

        detected_list.append(i)
        if len(detected_list) == 0:
            return None,None,None

    return le_c_list, re_c_list, detected_list

def computeGazeLoss(angular_out, gaze_batch_label):
    _criterion = _loss_fn.get("mse")()
    #------------------------------load Label------------------------------
    gaze_loss = _criterion(angular_out, gaze_batch_label).cuda()

    return gaze_loss

def Gaze_model_save(opt, gaze_model, is_best=False):
    path = opt.gaze_model_save_path
    torch.save(
        gaze_model.state_dict(), 
        os.path.join(path, 'gaze_model_latest.pt')
    )
    if is_best:
            torch.save(
                gaze_model.state_dict(),
                os.path.join(path, 'gaze_model_best.pt')
            )

def loadLabel(labels,names):
    gaze_batch_label = []
    head_batch_label = []
    flag = False

    #load label
    for name in names:
        #image name  : ex) s014_8465

        subj = name.split('_')[0]
        file_num = name.split('_')[1]
        file_num = f'{file_num:0>6}'
        image_name = subj+'_'+file_num
        
        #이진 탐색
        #label : idx, head1, head2, gaze1, gaze2, time
        start = 0
        end = len(labels) -1
        while start <= end:
            mid = (start + end) // 2
            # row
            label = labels[mid].split(",")
            # txt -> readlines
            name_label = label[0].split("_")
            # name split
            curr_name = f'{name_label[0]}_{name_label[1]:0>6}'
            # print(curr_name)
            if curr_name == image_name:
                head_batch_label.append([float(label[1]),float(label[2])])
                gaze_batch_label.append([float(label[3]),float(label[4])])

                flag =True
                break
            elif curr_name < image_name:
                start = mid+1
            else:
                end = mid-1

        if not flag:
            print("ERROR:: label not found")
            sys.exit()
        flag =False

    head_batch_label= torch.FloatTensor(head_batch_label)
    gaze_batch_label = torch.FloatTensor(gaze_batch_label)

    return head_batch_label, gaze_batch_label

def loadLabel_gazetest(labels,names):
    gaze_batch_label = []
    head_batch_label = []
    flag = False

    #load label
    for name in names:
        #image name  : ex) s014_8465
        
        #이진 탐색
        #label : idx, head1, head2, gaze1, gaze2, time
        start = 0
        end = len(labels) -1
        while start <= end:
            # row
            label = labels[start].split(",")
            label_name = label[0]
            if label_name == name:
                head_batch_label.append([float(label[1]),float(label[2])])
                gaze_batch_label.append([float(label[3]),float(label[4])])

                flag =True
                break
            else:
                start = start+1

        if not flag:
            print("ERROR:: label not found")
            sys.exit()
        flag =False

    head_batch_label= torch.FloatTensor(head_batch_label)
    gaze_batch_label = torch.FloatTensor(gaze_batch_label)

    return head_batch_label, gaze_batch_label


# def generateH5Dataset():
    # script_path = os.path.dirname(os.path.realpath(__file__))
    # _required_size = (224, 224)
    # subject_path ="./SRImage_to_h5/SR_Image"
    # h5_root = "./SRImage_to_h5/h5"

    # hdf_file = h5py.File(os.path.join(h5_root, 'batch_SR.hdf5')), mode='w')


def computeGazeLoss_val(labels, le_c_list, re_c_list, detected_list,image_names):
    _criterion = _loss_fn.get("mse")()

    #------------------------------load Label------------------------------

    head_batch_label, gaze_batch_label =loadLabel_val(labels,image_names)
    
    head_batch_label = head_batch_label.cuda()
    gaze_batch_label = gaze_batch_label.cuda()
    # print(le_c_list.shape)
    # print(re_c_list.shape)
    # print(head_batch_label.shape)

    angular_out = _model(le_c_list, re_c_list, head_batch_label)
    # print("angular_out :",angular_out)
    # diff =0
    # for i in range(len(angular_out)):
    #     for j in range(len(angular_out[i])):
    #         diff += pow((angular_out[i][j]-gaze_batch_label[i][j]),2)

    # print("sse  : ",diff)
    gaze_loss = _criterion(angular_out, gaze_batch_label).cuda()

    return gaze_loss


def loadLabel_val(labels,names):
    gaze_batch_label = []
    head_batch_label = []
    flag = False

    #load label
    #image name  : ex) s014_8465
    '''
    subj = name.split('_')[0]
    file_num = name.split('_')[1]
    file_num = f'{file_num:0>6}'
    image_name = subj+'_'+file_num
    '''
    # 지울 내용 ````````````````````
    image_name = names
    subj = names
    img_label = image_name.split('_')
    img_idx = f'{img_label[0][1:]}{img_label[1]:0>6}'
    #이진 탐색
    #label : idx, head1, head2, gaze1, gaze2, time
    start = 0
    end = len(labels) -1

    while start <= end:
        mid = (start + end) // 2
        # row
        label = labels[mid].split(",")
        # txt -> readlines
        name_label = label[0].split("_")
        # name split
        curr_name = f'{name_label[0]}_{name_label[1]}'
        curr_idx = f'{name_label[0][1:]}{name_label[1]:0>6}'

        # print(int(curr_idx))
        # print(int(img_idx))
        # print('curr_name',curr_name)
        # print('image_name',image_name)

        if curr_name == image_name:
            head_batch_label.append([float(label[1]),float(label[2])])
            gaze_batch_label.append([float(label[3]),float(label[4])])
            flag =True
            break

        elif int(curr_idx) < int(img_idx):
            start = mid+1
        else:
            end  = mid-1
    if not flag:
        print("ERROR:: label not found")
        sys.exit()
    flag =False

    head_batch_label= torch.FloatTensor(head_batch_label)
    gaze_batch_label = torch.FloatTensor(gaze_batch_label)

    return head_batch_label, gaze_batch_label