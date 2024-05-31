import numpy as np
import torch.nn as nn

def joint_angle_loss(angle_pred, angle_gt): 
    mse_loss = nn.MSELoss()
    loss = mse_loss(angle_pred, angle_gt)
    
    return loss

def d3_keypoint_loss(obj_pred, obj_gt):
    L1_loss = nn.L1Loss()
    loss = 0

    keypoint = find_keypoint(obj_gt)

    for i in range(len(keypoint)): 
        loss = loss
    return loss

def surface_vertices_loss(): 
    loss = 0
    return loss

def auxiliary_obj_IUV_loss(): 
    loss = 0
    return loss

def find_keypoint(obj):
    #for i in range(len(obj.vertices)):
        
    
    keypoint = []
    return keypoint