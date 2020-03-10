from __future__ import absolute_import, division

import numpy as np
import torch

from .camera import world_to_camera, normalize_screen_coordinates


def compute_scale_numpy(poses_2d, poses_3d):
    assert poses_2d.shape[-1] == 2
    assert poses_3d.shape[-1] == 3
    assert poses_2d.shape[0] == poses_3d.shape[0]
    poses_3d_projected = poses_3d[:,:,:2].reshape(poses_3d.shape[0], -1)
    poses_2d = poses_2d.reshape(poses_2d.shape[0], -1)
    lamda = np.sum(poses_3d_projected*poses_2d, axis=1)/(np.linalg.norm(poses_2d, axis=1)*np.linalg.norm(poses_2d, axis=1))
    lamda = lamda / 2
    return lamda

def compute_scale_pytorch(poses_2d, poses_3d):
    assert poses_2d.size(-1) == 2
    assert poses_3d.size(-1) == 3
    assert poses_2d.size(0) == poses_3d.size(0)
    poses_3d_projected = torch.flatten(poses_3d[:,:,:2], start_dim=1)
    poses_2d_ = torch.flatten(poses_2d, start_dim=1)
    lamda = torch.sum(poses_3d_projected*poses_2d_, dim=1)/(torch.norm(poses_2d_, dim=1)*torch.norm(poses_2d_, dim=1))
    lamda = lamda / 2
    return lamda

def prepare_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()
    
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length] 
            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])


    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
               cam = dataset.cameras()[subject][cam_idx]
               kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
               keypoints[subject][action][cam_idx] = kps
    
    TRAIN_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8']
    for subject in TRAIN_SUBJECTS:
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions_3d = []
            for cam_idx, cam in enumerate(anim['cameras']):
                pos_3d = anim['positions_3d'][cam_idx]
                pos_2d = keypoints[subject][action][cam_idx]
                pos_2d = pos_2d[:,:,:] - pos_2d[:, :1, :]
                scale = compute_scale_numpy(pos_2d, pos_3d)
                scale = np.expand_dims(np.repeat(np.expand_dims(scale, axis = 1), 17, axis=1), axis =2)
                pos_3d = pos_3d/scale
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d
    return keypoints, dataset




def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    ###
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue
                
            for cam_idx in range(len(keypoints[subject][action])):
                
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
    ###

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset  ###
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions
