import os
import trimesh
import torch
import numpy as np
import cv2
from PIL import Image
import kornia
import torchvision

import matplotlib
matplotlib.use("agg")
import seaborn as sns
sns.set()

from myutils.visualization_utils import save_prediction_as_ply_v3, save_prediction_as_ply_v5
from myutils.renderer_romp import Py3DR_mask, Py3DR_SQMesh, Py3DR_mask_syn, Py3DR_mask_test_align
from myutils.romp_vis_utils import rotate_view_perspective_my


def get_colors(M):
    return sns.color_palette("Paired")


def visualize_sq(sq_surface_points, image_name, output_path):
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

    # for img_n in image_name:
    for bsi in range(batch_size):
        output_directory = os.path.join(output_path, image_name[bsi])
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


def visualize_sq_new(sq_surface_points, output_path, human_or_object):

    # print ('In utils.visualize.visualize_sq_new:')
    # print ('sq_surface_points.size is', sq_surface_points.size())
    # [num_bone, num_points, 3]
    # os._exit(0)

    n_primitives = sq_surface_points.size(0)
    colors = get_colors(n_primitives)

    output_directory = os.path.join(output_path, human_or_object)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    save_prediction_as_ply_v3(
        sq_surface_points.cpu().detach().data.numpy(),
        colors,
        output_directory
    )


def visualize_sq_new_pts(sq_surface_points, faces, output_path, human_or_object):

    n_primitives = len(sq_surface_points)
    colors = get_colors(n_primitives)

    output_directory = os.path.join(output_path, human_or_object)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    save_prediction_as_ply_v5(
        sq_surface_points, faces,
        colors,
        output_directory
    )


def visualize_predictions(
        h_pred_vs, o_pred_vs,
        h_gt_vs, o_gt_vs,
        h_faces, o_faces,
        image_name, out_path
):
    # out_path = os.path.join(out_path, image_name)

    # visualize gt human mesh
    the_out_path = os.path.join(out_path, 'gt_human_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_sq(h_gt_vs, image_name, the_out_path)
    # visualize gt object mesh
    the_out_path = os.path.join(out_path, 'gt_object_mesh')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_sq(o_gt_vs, image_name, the_out_path)

    # visualize deformed human mesh
    the_out_path = os.path.join(out_path, 'pred_deformed_human')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_sq(h_pred_vs, image_name, the_out_path)
    # print ('human deformed mesh finished')
    # visualize deformed object mesh
    the_out_path = os.path.join(out_path, 'pred_deformed_object')
    if not os.path.exists(the_out_path):
        os.makedirs(the_out_path)
    visualize_sq(o_pred_vs, image_name, the_out_path)
    # print ('object deformed mesh finished')


def visualize_predictions_new(
    pred_human_vs, pred_object_vs,
    cam_trans, image_pad_info,
    image_names, out_path
):
    for i, img_name in enumerate(image_names):
        img_id = img_name.split('.')[0]
        outdir = os.path.join(out_path, img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # save 3D shapes in meshes
        pred_human_vi = pred_human_vs[i]
        pred_object_vi = pred_object_vs[i]
        visualize_sq_new(pred_human_vi, outdir, 'human')
        visualize_sq_new(pred_object_vi, outdir, 'object')

        # render_points(pred_human_vs, pred_object_vs)


def render_seg_mask_new_pts(vs_list, fs_list, image_path, output_path, human_or_object):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)  # (H, W, 3)
    height = img.shape[0]
    width = img.shape[1]

    renderer = Py3DR_mask(height=height, width=width)
    # renderer = Py3DR_SQMesh(height=height, width=width)

    result_image = []
    result_image.append(img)

    output_directory = os.path.join(output_path, human_or_object + '.jpg')

    mesh_colors = np.array([[.9, .9, .8] for _ in range(2)])
    vs_list_numpy = []
    for vs in vs_list:
        vs_list_numpy.append(vs.cpu().detach().data.numpy())
    fs_list_numpy = []
    for fs in fs_list:
        fs_list_numpy.append(fs.cpu().detach().data.numpy())
    rendered_mask = renderer(vs_list_numpy, fs_list_numpy, img, mesh_colors=mesh_colors)

    result_image.append(rendered_mask)
    result_image = np.concatenate(result_image, axis=0)

    cv2.imwrite(output_directory, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    # return result_image


def render_hoi_mesh_new_pts(vs_list, fs_list, image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)  # (H, W, 3)
    height = img.shape[0]
    width = img.shape[1]

    renderer = Py3DR_SQMesh(height=height, width=width)

    result_image = []
    result_image.append(img)

    output_directory = os.path.join(output_path,  'total.jpg')

    mesh_colors = np.array([[.9, .9, .8] for _ in range(2)])
    vs_list_numpy = []
    for vs in vs_list:
        vs_list_numpy.append(vs.cpu().detach().data.numpy())
    fs_list_numpy = []
    for fs in fs_list:
        fs_list_numpy.append(fs.cpu().detach().data.numpy())
    rendered_rgb = renderer(vs_list_numpy, fs_list_numpy, img, mesh_colors=mesh_colors)

    result_image.append(rendered_rgb)
    result_image = np.concatenate(result_image, axis=0)

    cv2.imwrite(output_directory, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    return result_image


def visualize_predictions_new_pts(
    pred_human_vs, pred_object_vs,
    human_fs, object_fs,
    deformed_human_verts, seg_mask_image_path,
    image_names, out_path
):
    img_id = image_names[0]
    img_id = img_id.split('.')[0]
    outdir = os.path.join(out_path, img_id)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save pred meshes
    pred_human_vi = pred_human_vs
    pred_object_vi = pred_object_vs
    human_fs_i = human_fs
    object_fs_i = object_fs
    visualize_sq_new_pts(pred_human_vi, human_fs_i, outdir, 'human')
    visualize_sq_new_pts(pred_object_vi, object_fs_i, outdir, 'object')

    # render segmasks and save them
    dhv_i = deformed_human_verts
    render_seg_mask_new_pts(dhv_i, human_fs_i, seg_mask_image_path[0], outdir, 'human')
    render_seg_mask_new_pts(pred_object_vi, object_fs_i, seg_mask_image_path[0], outdir, 'object')


def render_from_three_views(image_path, vs_list, fs_list, output_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)  # (H, W, 3)
    height = img.shape[0]
    width = img.shape[1]

    background = np.ones([height, width, 3], dtype=np.uint8) * 255

    renderer = Py3DR_SQMesh(height=height, width=width)

    result_image = []
    result_image.append(img)

    output_directory = os.path.join(output_path, 'total.jpg')

    mesh_colors = np.array([[.9, .9, .8] for _ in range(2)])
    vs_list_numpy = []
    for vs in vs_list:
        vs_list_numpy.append(vs.cpu().detach().data.numpy())
    fs_list_numpy = []
    for fs in fs_list:
        fs_list_numpy.append(fs.cpu().detach().data.numpy())
    rendered_rgb = renderer(vs_list_numpy, fs_list_numpy, img, mesh_colors=mesh_colors)
    # print ('rendered_rgb.shape is', rendered_rgb.shape)
    renderer.delete()
    cv2.imwrite(output_directory, cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))

    result_image.append(rendered_rgb)
    # result_image = np.concatenate(result_image, axis=0)

    # renderer = Py3DR_SQMesh(height=max(height, width), width=max(height, width))
    # # bird view
    # vs_bird_view_list = rotate_view_perspective_my(vs_list_numpy, rx=90, ry=0)
    # rendered_bv_image = renderer(vs_bird_view_list, fs_list_numpy,
    #                                          background,
    #                                          mesh_colors=mesh_colors)
    # # print (rendered_bv_image.shape)
    # result_image.append(rendered_bv_image)
    # # result_image = np.concatenate(result_image, axis=0)
    # renderer.delete()
    #
    # # side view
    # renderer = Py3DR_SQMesh(height=max(height, width), width=max(height, width))
    # vs_side_view_list = rotate_view_perspective_my(vs_list_numpy, rx=0, ry=90)
    # rendered_sv_image = renderer(vs_side_view_list, fs_list_numpy, background,
    #                                          mesh_colors=mesh_colors)
    # # result_image.append(cv2.resize(rendered_sv_image, (image_height, image_width)))
    # result_image.append(rendered_sv_image)
    # result_image = np.concatenate(result_image, axis=1)
    # renderer.delete()

    # cv2.imwrite(output_directory, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    return result_image


def visualize_predictions_mesh(
    pred_human_vs, pred_object_vs,
    human_fs, object_fs,
    deformed_human_verts, rgb_image_path,
    image_names, out_path
):
    img_id = image_names[0]
    img_id = img_id.split('.')[0]
    outdir = os.path.join(out_path, img_id)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save pred meshes
    pred_human_vi = pred_human_vs
    pred_object_vi = pred_object_vs
    human_fs_i = human_fs
    object_fs_i = object_fs

    total_pred_vs, total_fs = [], []
    for hv in deformed_human_verts:
        total_pred_vs.append(hv)
    for ov in pred_object_vi:
        total_pred_vs.append(ov)

    for hf in human_fs_i:
        total_fs.append(hf)
    for o_f in object_fs_i:
        total_fs.append(o_f)

    # render_hoi_mesh_new_pts(total_pred_vs, total_fs, rgb_image_path, outdir)
    rendered_results = render_from_three_views(
        rgb_image_path,
        total_pred_vs, total_fs,
        outdir
    )


def visualize_predictions_mesh_only_human(
    pred_human_vs,
    human_fs,
    cam_intri,
    cam_extri,
    out_path
):
    renderer = Py3DR_mask_test_align()

    vs_list_numpy = []
    for vs in pred_human_vs:
        vs_list_numpy.append(vs.cpu().detach().data.numpy())
    fs_list_numpy = []
    for fs in human_fs:
        fs_list_numpy.append(fs.cpu().detach().data.numpy())

    rendered_rgb = renderer(vs_list_numpy, fs_list_numpy, cam_intri, cam_extri)
    cv2.imwrite(out_path, cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))


def visualize_predictions_mesh_training_data(
    pred_human_vs, pred_object_vs,
    human_fs, object_fs,
    deformed_human_verts, rgb_image_path,
    image_names, out_path
):
    batch_size = pred_object_vs[0].size(0)

    # print (rgb_image_path)
    # os._exit(0)

    pred_object_vs_group, object_fs_group = regroup_parts(pred_object_vs, object_fs)
    pred_human_vs_group, human_fs_group = regroup_parts(pred_human_vs, human_fs)
    deformed_human_vs_group, _ = regroup_parts(deformed_human_verts, human_fs)

    for bsi in range(batch_size):
        img_id = image_names[bsi]
        img_id = img_id.split('.')[0]
        outdir = os.path.join(out_path, img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # save pred meshes
        pred_human_vi = pred_human_vs_group[bsi]
        pred_object_vi = pred_object_vs_group[bsi]
        human_fs_i = human_fs_group[bsi]
        object_fs_i = object_fs_group[bsi]

        total_pred_vs, total_fs = [], []
        for hv in deformed_human_vs_group[bsi]:
            # print ('hv.size is', hv.size())
            total_pred_vs.append(hv)
        for ov in pred_object_vi:
            total_pred_vs.append(ov)

        for hf in human_fs_i:
            total_fs.append(hf)
        for o_f in object_fs_i:
            total_fs.append(o_f)

        # render_hoi_mesh_new_pts(total_pred_vs, total_fs, rgb_image_path, outdir)
        rendered_results = render_from_three_views(
            rgb_image_path[bsi],
            total_pred_vs, total_fs,
            outdir
        )


def regroup_parts(verts_list, faces_list):
    bone_num = len(verts_list)
    batch_size = verts_list[0].size(0)

    regrouped_vs_list = [[] for i in range(batch_size)]
    regrouped_fs_list = [[] for i in range(batch_size)]
    for i in range(bone_num):
        for j in range(batch_size):
            part_i_vs = verts_list[i][j]
            part_i_fs = faces_list[i][j].cuda()
            regrouped_vs_list[j].append(part_i_vs)
            regrouped_fs_list[j].append(part_i_fs)

    return regrouped_vs_list, regrouped_fs_list


def visualize_predictions_training_data(
    pred_object_vs, object_fs,
    seg_mask_image_path,
    image_names, out_path
):
    batch_size = pred_object_vs[0].size(0)

    pred_object_vs_group, object_fs_group = regroup_parts(pred_object_vs, object_fs)

    rendered_imgs = []
    for bsi in range(batch_size):
        img_id = image_names[bsi]
        img_id = img_id.split('.')[0]
        print ('img_id =', img_id)
        outdir = os.path.join(out_path, img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # save pred meshes
        pred_object_vi = pred_object_vs_group[bsi]
        object_fs_i = object_fs_group[bsi]
        visualize_sq_new_pts(pred_object_vi, object_fs_i, outdir, 'object')

        # render segmasks and save them
        rendered = render_seg_mask_new_pts(pred_object_vi, object_fs_i, seg_mask_image_path[bsi], outdir, 'object')
        rendered_imgs.append(rendered)

    return rendered_imgs


def visualize_predictions_new_pts_training_data(
    pred_human_vs, pred_object_vs,
    human_fs, object_fs,
    deformed_human_verts, seg_mask_image_path,
    image_names, out_path
):
    batch_size = pred_object_vs[0].size(0)

    pred_object_vs_group, object_fs_group = regroup_parts(pred_object_vs, object_fs)
    pred_human_vs_group, human_fs_group = regroup_parts(pred_human_vs, human_fs)
    deformed_human_vs_group, _ = regroup_parts(deformed_human_verts, human_fs)

    for bsi in range(batch_size):
        img_id = image_names[bsi]
        img_id = img_id.split('.')[0]
        print ('img_id =', img_id)
        outdir = os.path.join(out_path, img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # save pred meshes
        pred_human_vi = pred_human_vs_group[bsi]
        pred_object_vi = pred_object_vs_group[bsi]
        human_fs_i = human_fs_group[bsi]
        object_fs_i = object_fs_group[bsi]
        visualize_sq_new_pts(pred_human_vi, human_fs_i, outdir, 'human')
        visualize_sq_new_pts(pred_object_vi, object_fs_i, outdir, 'object')

        # render segmasks and save them
        dhv_i = deformed_human_vs_group[bsi]
        render_seg_mask_new_pts(dhv_i, human_fs_i, seg_mask_image_path[bsi], outdir, 'human')
        render_seg_mask_new_pts(pred_object_vi, object_fs_i, seg_mask_image_path[bsi], outdir, 'object')


def kornia_projection(kp3d, img_h, img_w):
    FOV = 60
    focal_length = 1 / (np.tan(np.radians(FOV / 2)))
    focal_length = focal_length * max(img_h, img_w) / 2
    K = torch.Tensor(
        [
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ]
    ).cuda()

    # print ('kp3d.size is', kp3d.size())
    # kp3d.size is torch.Size([24, 3])
    # os._exit(0)
    # kp3d = kp3d.unsqueeze(0)
    kp2d = kornia.geometry.camera.perspective.project_points(kp3d, K[None, :, :].repeat(kp3d.size(0), 1, 1))
    # print ('kp2d.size is', kp2d.size())
    # kp2d.size is torch.Size([24, 2])
    # os._exit(0)
    kp2d = kp2d.cpu().detach().data.numpy()

    return kp2d


def visualize_hoi_kp_vectors(
    pred_human_kp3d, gt_human_kp3d, cam_trans,
    pred_object_kp3d, gt_object_kp3d,
    rendered_mask_list,
    seg_mask_image_path,
    image_names, out_path
):
    batch_size = pred_human_kp3d.size(0)
    human_kp_num = pred_human_kp3d.size(1)
    object_kp_num = pred_object_kp3d.size(1)

    pred_human_kp3d_trans = torch.add(pred_human_kp3d, cam_trans[:, None, :].repeat(1, pred_human_kp3d.size(1), 1))
    gt_human_kp3d_trans = torch.add(gt_human_kp3d, cam_trans[:, None, :].repeat(1, gt_human_kp3d.size(1), 1))[:, :24]

    gt_object_kp3d_only_centers = gt_object_kp3d[:, :2]

    for bsi in range(batch_size):
        img_id = image_names[bsi]
        img_id = img_id.split('.')[0]
        print ('img_id =', img_id)
        outdir = os.path.join(out_path, img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        seg_mask_i = cv2.imread(seg_mask_image_path[bsi])
        rendered_mask_i = rendered_mask_list[bsi]
        # print ('seg_mask_i.shape is', seg_mask_i.shape)
        # seg_mask_i.shape is (339, 510, 3)
        # os._exit(0)
        img_h, img_w = seg_mask_i.shape[:2]

        pred_human_kp2d = kornia_projection(pred_human_kp3d_trans[bsi], img_h, img_w)
        gt_human_kp2d = kornia_projection(gt_human_kp3d_trans[bsi], img_h, img_w)
        pred_object_kp2d = kornia_projection(pred_object_kp3d[bsi], img_h, img_w)
        gt_object_kp2d = kornia_projection(gt_object_kp3d_only_centers[bsi], img_h, img_w)

        for i in range(human_kp_num - 12):
            for j in range(object_kp_num):
                pred_h_kp = pred_human_kp2d[i]
                rendered_mask_i = cv2.circle(rendered_mask_i, (int(pred_h_kp[0]), int(pred_h_kp[1])), radius=5, color=(255, 255, 255), thickness=-1)

                gt_h_kp = gt_human_kp2d[i]
                seg_mask_i = cv2.circle(seg_mask_i, (int(gt_h_kp[0]), int(gt_h_kp[1])), radius=5, color=(139, 0, 0), thickness=-1)

                pred_o_kp = pred_object_kp2d[j]
                rendered_mask_i = cv2.circle(rendered_mask_i, (int(pred_o_kp[0]), int(pred_o_kp[1])), radius=5, color=(255, 255, 255), thickness=-1)

                gt_o_kp = gt_object_kp2d[j]
                seg_mask_i = cv2.circle(seg_mask_i, (int(gt_o_kp[0]), int(gt_o_kp[1])), radius=5, color=(139, 0, 0), thickness=-1)

                seg_mask_i = cv2.line(seg_mask_i, (int(gt_h_kp[0]), int(gt_h_kp[1])), (int(gt_o_kp[0]), int(gt_o_kp[1])), color=(255, 255, 255))
                rendered_mask_i = cv2.line(rendered_mask_i, (int(pred_h_kp[0]), int(pred_h_kp[1])), (int(pred_o_kp[0]), int(pred_o_kp[1])), color=(139, 0, 0))

        cv2.imwrite(os.path.join(outdir, 'gt_mask_vectors.png'), seg_mask_i)
        cv2.imwrite(os.path.join(outdir, 'pred_mask_vectors.png'), rendered_mask_i)


def visualize_iuvfeature_map(iuv_list, image_path, out_path):
    # print ('iuv_mid.size is', iuv_mid.size(), ', iuv_out.size is', iuv_out.size())
    # iuv_mid.size is torch.Size([1, 9, 224, 224]) , iuv_out.size is torch.Size([1, 9, 224, 224])
    # os._exit(0)
    feat_num = len(iuv_list)
    print ('image_path is', image_path)
    image_id = image_path[0].split('.')[0]
    image_id = image_id.split('_')[-1]

    out_path = '../../tmptest_iuv_visual/laptop_addiuv'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    resize_img = torchvision.transforms.Resize((224, 224))
    feat_img_list_i, feat_img_list_u, feat_img_list_v = [], [], []
    for i in range(feat_num):
        the_feat = iuv_list[i]
        the_feat = the_feat.squeeze(0)
        the_feat_i = the_feat[:3]
        the_feat_u = the_feat[3:6]
        the_feat_v = the_feat[6:9]
        # print ('i.size is', the_feat_i.size(), ', u.size is', the_feat_u.size(), ', v.size is', the_feat_v.size())
        # os._exit(0)
        # the_feat_i = torch.mean(the_feat_i, dim=0).unsqueeze(0)
        the_feat_i = resize_img(the_feat_i)#.squeeze(0)
        feat_img_list_i.append(the_feat_i.cpu().detach().data.numpy() * 255)
        # the_feat_u = torch.mean(the_feat_u, dim=0).unsqueeze(0)
        the_feat_u = resize_img(the_feat_u)#.squeeze(0)
        feat_img_list_u.append(the_feat_u.cpu().detach().data.numpy() * 255)
        # the_feat_v = torch.mean(the_feat_v, dim=0).unsqueeze(0)
        the_feat_v = resize_img(the_feat_v)#.squeeze(0)
        feat_img_list_v.append(the_feat_v.cpu().detach().data.numpy() * 255)
        # feat_img_list.append(the_feat.cpu().detach().data.numpy() * 255)
    # print (np.concatenate(feat_img_list_i, axis=1).shape)
    # os._exit(0)
    cv2.imwrite(os.path.join(out_path, image_id + '_i.png'), np.concatenate(feat_img_list_i, axis=1).transpose((1, 2, 0)))
    cv2.imwrite(os.path.join(out_path, image_id + '_u.png'), np.concatenate(feat_img_list_u, axis=1).transpose((1, 2, 0)))
    cv2.imwrite(os.path.join(out_path, image_id + '_v.png'), np.concatenate(feat_img_list_v, axis=1).transpose((1, 2, 0)))

    # os._exit(0)
