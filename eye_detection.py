import cv2
import numpy as np
import dlib
import torch
import torchvision.transforms as transforms

# Left, Right eyes -> (60,38) size bounding box

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_detector(img):
    img = torch.transpose(img,1,2)
    img = torch.transpose(img,2,0)
    img = img.to(torch.uint8)
    # print(img)

    img = img.detach().cpu().numpy()
    # print(img.shape)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    rects = detector(img, 1) # rects contains all the faces detected
    
    lx = []
    ly = []
    rx = []
    ry = []

    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        shape = shape_to_np(shape)

        for (x, y) in shape[36:42]:
            lx.append(x)
            ly.append(y)


        for (x, y) in shape[42:48]:
            rx.append(x)
            ry.append(y)

    if len(lx) !=0:
        center_l = (int((max(lx)+min(lx))/2), int((max(ly)+min(ly))/2))
        minim_l = (center_l[0]-29, center_l[1]-17)
        maxim_l = (center_l[0]+30, center_l[1]+18)

        center_r = (int((max(rx)+min(rx))/2), int((max(ry)+min(ry))/2))
        minim_r = (center_r[0]-29, center_r[1]-17)
        maxim_r = (center_r[0]+30, center_r[1]+18)

        return minim_l, maxim_l, minim_r, maxim_r

    return None,None,None,None

# cv2.line(img, center_l, center_l, (0,0,255), 10)

# print(minim_l)
# print(maxim_l)
# cv2.rectangle(img, minim_l, maxim_l, (0,0,255), 1)

# cv2.rectangle(img, minim_r, maxim_r, (0,255,0), 1)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()