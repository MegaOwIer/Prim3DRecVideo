import torch
import numpy as np
import torch.nn as nn
import trimesh
from networks.baseline_network import ObjectNetwork_pts
from math import cos, sin
from myutils.losses import chamfer_loss

def compute_loss(pred_dict, data_dict, init_template_data_dict, batch_size) -> torch.Tensor:
    loss = 0

    object_num_bones = 2
    
    vs = Reform(data_dict, init_template_data_dict, batch_size, object_num_bones)

    loss = loss + part_segment_masks_loss() + \
             d3_keypoint_loss(pred_dict, [data_dict['vs'], data_dict['fs']], init_template_data_dict) + \
             (1 - torch.cos(pred_dict['object_pred_angle_leaf'].reshape(batch_size).cuda() - data_dict['joint_state'].reshape(batch_size).cuda()).mean()) + \
             surface_vertices_loss(pred_dict['deformed_object'], vs) + \
             auxiliary_obj_IUV_loss()
    #pred_angle = pred_dict['object_pred_angle_leaf'].reshape(1, batch_size).cuda()
    #gt_angle = data_dict['joint_state'].reshape(1, batch_size).cuda()
    # print('pred: ', pred_angle)
    # print('gt: ', gt_angle)
    #loss = 1 + cos_sim(pred_angle, gt_angle)
    return loss

def part_segment_masks_loss():
    loss = 0
    return loss

def d3_keypoint_loss(pred_dict, v_gt, init_template_data_dict):
    L1_loss = nn.L1Loss()
    loss = 0
    
    #v_gt = pred_dict['deformed_object']

    #v0_gt = v_gt[0]
    #v1_gt = v_gt[1]

    #object_joint_tree = init_template_data_dict['joint_tree']
    #object_primitive_align = init_template_data_dict['primitive_align']
    #object_joint_parameter_leaf = init_template_data_dict['joint_parameter_leaf']
    
    #gt_part_scale = torch.ones(batch_size, object_num_bones, 3).cuda()
    #gt_total_scale = 0
    #gt_total_trans = 0

    #ObjectNetwork_pts.deform(v1_gt, old_center, 2, object_joint_tree, object_primitive_align, 
    #                         object_joint_parameter_leaf, pred_dict['object_pred_rotmat_root'], 
    #                         pred_dict['object_pred_angle_leaf'], gt_part_scale, gt_total_scale,
     #                        gt_total_trans)

    return loss

def surface_vertices_loss(v_pred, v_gt):
    loss = 0
    loss = loss + chamfer_loss(v_pred[0], v_gt[0])
    loss = loss + chamfer_loss(v_pred[1], v_gt[1])

    return loss

def auxiliary_obj_IUV_loss(): 
    loss = 0
    return loss

def compute_rotmat(yaw, pitch, roll):
    yaw = float(yaw)
    pitch = float(pitch)
    roll = float(roll)
    
    Rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
    Ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
    Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])

    rotmat = np.dot(np.dot(Rz, Ry), Rx)
    rotmat = torch.from_numpy(rotmat)
    rotmat = rotmat.float()

    return torch.Tensor(rotmat)

def compute_trans(x_offset, y_offset, z_offset): 
    x_offset = list(map(float, x_offset))
    y_offset = list(map(float, y_offset))
    z_offset = list(map(float, z_offset))

    ret = torch.Tensor([x_offset, y_offset, z_offset])
    ret = ret.transpose(0, 1)

    return ret

def Reform(data_dict, init_template_data_dict, batch_size, object_num_bones): 
    object_center = data_dict['part_centers'][0, :, :]
    #print(data_dict['part_centers'])
    #input()
    object_joint_tree = init_template_data_dict['joint_tree']
    #print(object_joint_tree)
    object_primitive_align = init_template_data_dict['primitive_align'].reshape(2)
    object_joint_parameter_leaf = init_template_data_dict['joint_parameter_leaf'] 
    object_angle_leaf = data_dict['joint_state']
    object_part_scale = torch.ones(batch_size, object_num_bones, 3).cuda()
    
    object_total_trans = compute_trans(data_dict['3d_info']['x_offset'], 
                                        data_dict['3d_info']['y_offset'], 
                                        data_dict['3d_info']['z_offset'])
    
    object_rotmat_root = [compute_rotmat(data_dict['3d_info']['yaw'][i], 
                                        data_dict['3d_info']['pitch'][i], 
                                        data_dict['3d_info']['roll'][i]) for i in range(batch_size)]
    object_rotmat_root = torch.stack(object_rotmat_root)

    vs, _, _ = ObjectNetwork_pts.deform(data_dict['vs'], object_center.cuda(), 2, object_joint_tree.reshape(2).cuda(), 
                                    object_primitive_align.cuda(), object_joint_parameter_leaf.cuda(), 
                                    object_rotmat_root.cuda(), object_angle_leaf.cuda(), 
                                    object_part_scale.cuda(), 0, object_total_trans.cuda())
    
    return vs