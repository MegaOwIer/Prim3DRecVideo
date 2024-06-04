import os
import trimesh
import torch
import numpy as np
import cv2

import matplotlib
matplotlib.use("agg")
import seaborn as sns
sns.set()

from myutils.visualization_utils import save_prediction_as_ply_v4, _from_primitive_parms_to_mesh_v2
from myutils.visualization_utils import save_prediction_as_ply_v3, save_prediction_as_ply_v5, save_prediction_as_ply_v6


def get_colors(M):
    return sns.color_palette("Paired")


def visualize_mesh(img_name, vertices, faces, out_path, is_ori=False, is_obejct=False):
    '''
    Args:
        vertices: bs x bone_num x num_points_every x 3
        faces: bs x bone_num x num_faces x 3
    '''

    if is_ori:
        if is_obejct:
            batch_size = len(vertices)
            for bsi in range(batch_size):
                filename = os.path.join(out_path, img_name[bsi] + '-' + str(bsi) + '.obj')
                vi = vertices[bsi].cpu().detach().data.numpy()
                fi = faces[bsi].cpu().detach().data.numpy()
                with open(filename, 'w') as f:
                    f.write('# %s\n' % os.path.basename(filename))
                    f.write('#\n')
                    f.write('\n')

                    for vertex in vi:
                        f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
                    f.write('\n')
                    for face in fi:
                        f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))
                f.close()
                return

        batch_size = vertices.shape[0]
        vertices = vertices.cpu().detach().data.numpy()
        faces = faces.cpu().detach().data.numpy()
        for bsi in range(batch_size):
            filename = os.path.join(out_path, img_name[bsi] + '-' + str(bsi) + '.obj')
            vi = vertices[bsi]
            fi = faces[bsi]
            with open(filename, 'w') as f:
                f.write('# %s\n' % os.path.basename(filename))
                f.write('#\n')
                f.write('\n')

                for vertex in vi:
                    f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
                f.write('\n')
                for face in fi:
                    f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))
            f.close()
        return

    # print ('In utils.visualize.visualize_mesh:')
    # print ('len(vertices) is', len(vertices), ', vertices[0].size is', vertices[0].size())
    # os._exit(0)
    bone_num = len(vertices)
    colors = get_colors(bone_num)
    batch_size = vertices[0].size(0)
    images = [[] for i in range(batch_size)]

    for bone_idx in range(bone_num):
        vs = vertices[bone_idx] # (bs, num_points, 3)
        for bsi in range(batch_size):
            vi = vs[bsi] # (num_points, 3)
            images[bsi].append(vi)

    for bsi in range(batch_size):
        image_paths = img_name[bsi].split('/')
        out_file_name = image_paths[-3] + '-' + image_paths[-1][:-4]

        the_image = images[bsi]
        filename = os.path.join(out_path, out_file_name + '.ply')
        m = None
        for bone_idx in range(bone_num):
            # out_path = os.path.join(filename, 'bone-' + str(bone_idx) + '.ply')
            vi = the_image[bone_idx].cpu().detach().data.numpy()

            _m = _from_primitive_parms_to_mesh_v2(vi, (colors[bone_idx % len(colors)]) + (1.0,))
            m = trimesh.util.concatenate(_m, m)
        m.export(filename, file_type="ply")


def visualize_sq(sq_surface_points, output_path):
    '''
    Args:
        sq_surface_points: bs x bone_num x num_sampled_points x 3
    '''
    print ('In utils.visualize.visualize_sq:')
    print ('sq_surface_points.size is', sq_surface_points.size())
    exit(0)

    batch_size = sq_surface_points.size(0)
    n_primitives = sq_surface_points.size(1)
    colors = get_colors(n_primitives)
    # colors = np.array(colors)

    for bsi in range(batch_size):
        output_directory = os.path.join(output_path, 'image-' + str(bsi))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # save_prediction_as_ply_v2(
        #     sq_surface_points[bsi].cpu().detach().data.numpy(),
        #     colors,
        #     os.path.join(output_directory, "primitives.ply")
        # )
        save_prediction_as_ply_v3(
            sq_surface_points[bsi].cpu().detach().data.numpy(),
            colors,
            output_directory
        )


def visualize_sq_pts(sq_surface_points, output_path):
    '''
    Args:
        sq_surface_points: bs x bone_num x num_sampled_points x 3
    '''
    # print ('In utils.visualize.visualize_sq:')
    # print ('sq_surface_points.size is', sq_surface_points.size())
    # os._exit(0)

    batch_size = sq_surface_points.size(0)
    n_primitives = sq_surface_points.size(1)
    colors = get_colors(n_primitives)
    # colors = np.array(colors)

    for bsi in range(batch_size):
        output_directory = os.path.join(output_path, 'image-' + str(bsi))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        save_prediction_as_ply_v4(
            sq_surface_points[bsi].cpu().detach().data.numpy(),
            colors,
            output_directory
        )


def visualize_sq_list(verts_list, faces_list, output_path):
    batch_size = 1
    n_primitives = len(verts_list)
    colors = get_colors(n_primitives)
    # colors = np.array(colors)

    for bsi in range(batch_size):
        output_directory = os.path.join(output_path, 'image-' + str(bsi))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # save_prediction_as_ply_v5(
        #     verts_list, faces_list,
        #     colors,
        #     output_directory
        # )
        save_prediction_as_ply_v6(
            verts_list, faces_list,
            colors,
            output_directory
        )


def visualize_predictions(
        h_ori_vertices, h_ori_faces,
        o_ori_vertices, o_ori_faces,
        h_mesh_preds, o_mesh_preds,
        h_sq_preds, o_sq_preds,
        out_path
):
    # visualize gt human mesh
    the_out_path = os.path.join(out_path, 'gt_human_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(h_ori_vertices, h_ori_faces, the_out_path, is_ori=True)
    # visualize gt object mesh
    the_out_path = os.path.join(out_path, 'gt_object_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(o_ori_vertices, o_ori_faces, the_out_path, is_ori=True, is_obejct=True)

    # visualize deformed human mesh
    the_out_path = os.path.join(out_path, 'pred_deformed_human')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(h_mesh_preds, h_ori_faces, the_out_path)
    # print ('human deformed mesh finished')
    # visualize deformed object mesh
    the_out_path = os.path.join(out_path, 'pred_deformed_object')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(o_mesh_preds, o_ori_faces, the_out_path)
    # print ('object deformed mesh finished')

    # visualize human superquadric
    the_out_path = os.path.join(out_path, 'pred_sq_human')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_sq(h_sq_preds, the_out_path)
    # visualize object superquadric
    the_out_path = os.path.join(out_path, 'pred_sq_object')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_sq(o_sq_preds, the_out_path)

def visualize_predictions_onlySQ(
        h_ori_vertices, h_ori_faces,
        h_sq_preds,
        out_path
):
    # visualize gt human mesh
    the_out_path = os.path.join(out_path, 'gt_human_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(h_ori_vertices, h_ori_faces, the_out_path, is_ori=True)
    # # visualize gt object mesh
    # the_out_path = os.path.join(out_path, 'gt_object_mesh')
    # if not os.path.exists(the_out_path):
    #     os.makedirs(the_out_path)
    # visualize_mesh(o_ori_vertices, o_ori_faces, the_out_path, is_ori=True, is_obejct=True)

    # visualize human superquadric
    the_out_path = os.path.join(out_path, 'pred_sq_human')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_sq(h_sq_preds, the_out_path)
    # # visualize object superquadric
    # the_out_path = os.path.join(out_path, 'pred_sq_object')
    # if not os.path.exists(the_out_path):
    #     os.makedirs(the_out_path)
    # visualize_sq(o_sq_preds, the_out_path)


def visualize_predictions_RigidPos(
    h_ori_vertices, h_ori_faces,
    o_ori_vertices, o_ori_faces,
    o_pred_vertices, template_o_fs,
    out_path, img_name,
    renderer, pred_camera_trans, camera_RT, FOV,
    H, W
):
    # visualize gt human mesh
    the_out_path = os.path.join(out_path, 'gt_human_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(img_name, h_ori_vertices, h_ori_faces, the_out_path, is_ori=True)

    # visualize gt object mesh
    the_out_path = os.path.join(out_path, 'gt_object_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(img_name, o_ori_vertices, o_ori_faces, the_out_path, is_ori=True)

    # visualize pred object mesh
    the_out_path = os.path.join(out_path, 'pred_object_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(img_name, o_pred_vertices, template_o_fs, the_out_path, is_ori=False)

    # save rendered masks
    # print ('o_pred_vertices.size is', o_pred_vertices.size(), ', o_ori_faces.size is', o_ori_faces.size())
    # os._exit(0)

    the_out_path = os.path.join(out_path, 'masks')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    # pred_object_image, pred_object_mask = renderer(o_pred_vertices.clone(), o_ori_faces)
    R, t = camera_RT
    fov = FOV
    batch_size = (o_pred_vertices[0]).size(0)
    pred_camera = [R, t, fov]
    pred_object_mask = torch.zeros(batch_size, 224, 224).cuda()
    for bone_i in range(len(template_o_fs)):
        the_verts = o_pred_vertices[bone_i]
        the_verts_trans = the_verts + pred_camera_trans[:, None, :].repeat(1, the_verts.size(1), 1)
        the_faces = template_o_fs[bone_i]
        the_faces = the_faces[None, :, :].repeat(batch_size, 1, 1)
        the_pred_image, the_pred_mask = renderer(the_verts_trans.clone(), the_faces, H, W, cam_params=pred_camera)
        pred_object_mask = torch.clamp(pred_object_mask + the_pred_mask, min=0., max=1.)

    gt_human_image, gt_human_mask = renderer((h_ori_vertices+pred_camera_trans[:, None, :].repeat(1, h_ori_vertices.size(1), 1)).clone(), h_ori_faces, H, W, cam_params=pred_camera)
    gt_object_image, gt_object_mask = renderer((o_ori_vertices+pred_camera_trans[:, None, :].repeat(1, o_ori_vertices.size(1), 1)).clone(), o_ori_faces, H, W, cam_params=pred_camera)
    pred_whole_mask = torch.clamp(pred_object_mask + gt_human_mask, min=0., max=1.)
    gt_whole_mask = torch.clamp(gt_object_mask + gt_human_mask, min=0., max=1.)

    pred_object_mask = pred_object_mask.cpu().detach().data.numpy()
    pred_whole_mask = pred_whole_mask.cpu().detach().data.numpy()
    gt_whole_mask = gt_whole_mask.cpu().detach().data.numpy()
    gt_human_mask = gt_human_mask.cpu().detach().data.numpy()
    gt_object_mask = gt_object_mask.cpu().detach().data.numpy()
    # print ('pred_object_mask.shape is', pred_object_mask.shape)
    # print ('pred_whole_mask.shape is', pred_whole_mask.shape)
    # print ('gt_whole_mask.shape is', gt_whole_mask.shape)
    # os._exit(0)
    pred_object_mask = np.transpose(pred_object_mask, (1, 2, 0))
    pred_whole_mask = np.transpose(pred_whole_mask, (1, 2, 0))
    gt_whole_mask = np.transpose(gt_whole_mask, (1, 2, 0))
    gt_human_mask = np.transpose(gt_human_mask, (1, 2, 0))
    gt_object_mask = np.transpose(gt_object_mask, (1, 2, 0))

    cv2.imwrite(os.path.join(the_out_path, 'pred_object_mask.png'), pred_object_mask * 255.)
    cv2.imwrite(os.path.join(the_out_path, 'pred_whole_mask.png'), pred_whole_mask * 255.)
    cv2.imwrite(os.path.join(the_out_path, 'gt_whole_mask.png'), gt_whole_mask * 255.)
    cv2.imwrite(os.path.join(the_out_path, 'gt_human_mask.png'), gt_human_mask * 255.)
    cv2.imwrite(os.path.join(the_out_path, 'gt_object_mask.png'), gt_object_mask * 255.)

def reformMeshData(vs): 
    vs0 = vs[0]
    vs1 = vs[1]
    vs_new = []

    for i in range(vs0.shape[0]): 
        
        vs_new.append([vs0[i], vs1[i]])
    
    return vs_new