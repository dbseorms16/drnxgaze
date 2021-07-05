"""
MIT License

Copyright (c) 2018 Jianzhu Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
import torch
from math import sqrt
from .ddfa import reconstruct_vertex
import cv2


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]

# crop_img의 반환 결과를 tensor 타입으로 변경하여야 한다.
def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    # print(sx, sy, ex, ey)

    dh, dw = ey - sy, ex - sx
    # print("dh"+str(dh))
    # print("dw"+str(dw))

    if len(img.shape) == 3:
        res =  torch.zeros([3, dh, dw ], dtype=torch.int32)
    else:
        res =  torch.zeros([1, dh, dw ], dtype=torch.int32)

    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh


    # res[sy:ey, sx:ex] = img[sy:ey, sx:ex]


    return img

# def crop_img(img, roi_box):
#
#     h, w = img.shape[:2] # shape = [3, 160, 160]
#
#     sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
#     dh, dw = ey - sy, ex - sx
#     print("dh" + str(dh))
#     print("dw"+str(dw))
#     if len(img.shape) == 3:
#         res = np.zeros((dh, dw, 3), dtype=np.uint8)
#     else:
#         res = np.zeros((dh, dw), dtype=np.uint8)
#     if sx < 0:
#         sx, dsx = 0, -sx
#     else:
#         dsx = 0
#
#     if ex > w:
#         ex, dex = w, dw - (ex - w)
#     else:
#         dex = dw
#
#     if sy < 0:
#         sy, dsy = 0, -sy
#     else:
#         dsy = 0
#
#     if ey > h:
#         ey, dey = h, dh - (ey - h)
#     else:
#         dey = dh
#     print("왼",img[sy:ey, sx:ex].shape)
#     print("오",res[dsy:dey, dsx:dex].shape)
#     res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
#     return res


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0.0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size
    return roi_box


def _predict_vertices(param, roi_bbox, dense):
    from .params import std_size
    vertex = reconstruct_vertex(param, dense=dense)
    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / std_size
    scale_y = (ey - sy) / std_size
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex


def predict_68pts(param, roi_box):
    return _predict_vertices(param, roi_box, dense=False)


def predict_dense(param, roi_box):
    return _predict_vertices(param, roi_box, dense=True)
