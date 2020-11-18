import cv2
import numpy as np
from PIL import Image
import os
import pickle
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from utils import center_to_corner

def map_lidar_to_imgidx(pts, info):
    # Param: pts=(N, 3)
    # Return: pts_img=(N, 2); (col, row)
    pts = pts.copy()

    pts = Quaternion(info['lidar2ego_rotation']).rotation_matrix @ pts
    pts = pts + np.array(info['lidar2ego_translation'])[:, np.newaxis]
    pts = Quaternion(info['ego2global_rotation_lidar']).rotation_matrix @ pts
    pts = pts + np.array(info['ego2global_translation_lidar'])[:, np.newaxis]
    pts = pts - np.array(info['ego2global_translation_cam'])[:, np.newaxis]
    pts = Quaternion(info['ego2global_rotation_cam']).rotation_matrix.T @ pts
    pts = pts - np.array(info['cam2ego_translation'])[:, np.newaxis]
    pts = Quaternion(info['cam2ego_rotation']).rotation_matrix.T @ pts

    pts_img = view_points(pts, np.array(info['cam_intrinsic']), normalize=True)
    pts_img = pts_img.T[:, :2].astype(np.int32)
    return pts_img


def draw_box3d_image(img_orig, corner_indices, color=(0, 0, 255)):
    # Param: corner_indices = (N, 8, 3) or (N, 3, 8);
    # Return: img with 3d-box; np.array = (H, W, 3)
    linewidth = 1
    img = img_orig.copy()  # (H, W, 3)

    def draw_rect(selected_indices):
        prev = selected_indices[-1]
        for cur in selected_indices:
            cv2.line(img,
                     (prev[0], prev[1]),
                     (cur[0], cur[1]), color, linewidth)
            prev = cur

    for pts_img in corner_indices:
        for i in range(4):
            cv2.line(img,
                     (pts_img[i][0], pts_img[i][1]),
                     (pts_img[i + 4][0], pts_img[i + 4][1]),
                     color, linewidth)
        # front and back
        draw_rect(pts_img[:4])
        draw_rect(pts_img[4:])
        
        # draw front direction
        cbf = np.mean(pts_img[2:4], axis=0).astype(np.int32)
        cb = np.mean(pts_img[[2, 3, 7, 6]], axis=0).astype(np.int32)
        cv2.line(img,
                 (cb[0], cb[1]),
                 (cbf[0], cbf[1]),
                 color, linewidth)
    return img
