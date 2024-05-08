import os
import trimesh
import torch
import numpy as np
import cv2

import matplotlib
matplotlib.use("agg")
import seaborn as sns
sns.set()


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
        the_image = images[bsi]
        filename = os.path.join(out_path, 'image-' + str(bsi) + '.ply')
        m = None
        for bone_idx in range(bone_num):
            vi = the_image[bone_idx].cpu().detach().data.numpy()

            _m = _from_primitive_parms_to_mesh(vi, (colors[bone_idx % len(colors)]) + (1.0,))
            m = trimesh.util.concatenate(_m, m)
        m.export(filename, file_type="ply")


def visualize_sq(sq_surface_points, output_path):
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

    for bsi in range(batch_size):
        output_directory = os.path.join(output_path, 'image-' + str(bsi))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        save_prediction_as_ply(
            sq_surface_points[bsi].cpu().detach().data.numpy(),
            colors,
            output_directory
        )


def visualize_sq_pts(sq_surface_points, output_path):
    '''
    Args:
        sq_surface_points: bs x bone_num x num_sampled_points x 3
    '''

    batch_size = sq_surface_points.size(0)
    n_primitives = sq_surface_points.size(1)
    colors = get_colors(n_primitives)

    for bsi in range(batch_size):
        output_directory = os.path.join(output_path, 'image-' + str(bsi))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        save_prediction_as_ply(
            sq_surface_points[bsi].cpu().detach().data.numpy(),
            colors,
            output_directory
        )


def visualize_sq_list(verts_list, faces_list, output_path):
    batch_size = 1
    n_primitives = len(verts_list)
    colors = get_colors(n_primitives)

    for bsi in range(batch_size):
        output_directory = os.path.join(output_path, 'image-' + str(bsi))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        save_prediction_as_ply(
            verts_list, faces_list,
            colors,
            output_directory
        )


def visualize_predictions(
        o_ori_vertices, o_ori_faces, o_mesh_preds, o_sq_preds, out_path
):
    # visualize gt object mesh
    the_out_path = os.path.join(out_path, 'gt_object_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(o_ori_vertices, o_ori_faces, the_out_path, is_ori=True, is_obejct=True)

    # print ('human deformed mesh finished')
    # visualize deformed object mesh
    the_out_path = os.path.join(out_path, 'pred_deformed_object')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_mesh(o_mesh_preds, o_ori_faces, the_out_path)
    # print ('object deformed mesh finished')

    # visualize object superquadric
    the_out_path = os.path.join(out_path, 'pred_sq_object')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_sq(o_sq_preds, the_out_path)

