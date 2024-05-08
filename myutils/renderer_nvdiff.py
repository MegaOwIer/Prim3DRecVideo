import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import nvdiffrast.torch as dr
import trimesh
import torch
import numpy as np
import pytorch3d.transforms


class Nvdiffrast(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

        self.yfov = np.radians(FOV)

        self.mv = torch.eye(4).cuda()
        # print ('?')
        self.campos = torch.linalg.inv(self.mv)[:3, 3]
        # print ('??')
        # self.glctx = dr.RasterizeGLContext()
        # print ('???')

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def intrinsics_v2(self, yfov, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        aspect_ratio = float(width) / float(height)
        a = aspect_ratio
        t = np.tan(yfov / 2.0)
        n = znear
        f = zfar

        P[0][0] = 1.0 / (a * t)
        P[1][1] = -1.0 / t
        P[3][2] = -1.0

        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_seg_map(self, v_pos_clip, faces, colors, render_reso):
        # print ('prepare raster.')
        glctx = dr.RasterizeGLContext()
        # print ('raster prepared.')
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                seg_map, _ = self.interpolate(colors[None, ...], rast, faces)
                # print('*** seg_map: ', seg_map.shape, seg_map.max(), seg_map.mean())
                # print ('seg_map.requires_grad =', seg_map.requires_grad, ', seg_map.is_cuda =', seg_map.is_cuda)
                # *** seg_map:  torch.Size([1, 867, 1300, 3]) tensor(1., device='cuda:0', grad_fn=<MaxBackward1>) tensor(0.1686, device='cuda:0', grad_fn=<MeanBackward0>)
                # seg_map.requires_grad = True , seg_map.is_cuda = True
                # os._exit(0)

        return seg_map

    def render_weight_map(self, v_pos_clip, faces, weights, render_reso):
        # print ('prepare raster.')
        glctx = dr.RasterizeGLContext()
        # print ('raster prepared.')
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                weight_map, _ = self.interpolate(weights[None, ...], rast, faces)
                # print('*** seg_map: ', seg_map.shape, seg_map.max(), seg_map.mean())
                # print ('seg_map.requires_grad =', seg_map.requires_grad, ', seg_map.is_cuda =', seg_map.is_cuda)
                # *** seg_map:  torch.Size([1, 867, 1300, 3]) tensor(1., device='cuda:0', grad_fn=<MaxBackward1>) tensor(0.1686, device='cuda:0', grad_fn=<MeanBackward0>)
                # seg_map.requires_grad = True , seg_map.is_cuda = True
                # os._exit(0)

        return weight_map

    def render_prim_index(self, v_pos_clip, faces, colors, render_reso):
        # print ('prepare raster.')
        glctx = dr.RasterizeGLContext()
        # print ('raster prepared.')
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                prim_idx, _ = self.interpolate(mesh.primitives[None, ...].float(), rast, mesh.faces.int())
                print('*** prim_idx: ', prim_idx.shape, prim_idx.max(), prim_idx.min())

        return seg_map

    def __call__(self, mesh, image_pad_info, focal_length):
        # print ('In renderer_nvdiff:')
        # print ('mesh.vertices.size is', mesh.vertices.size(), ', mesh.faces.size is', mesh.faces.size())
        # mesh.vertices.size is torch.Size([38418, 3]) , mesh.faces.size is torch.Size([76756, 3])
        # os._exit(0)

        verts = mesh.vertices
        verts = torch.matmul(self.rot, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))
        faces = mesh.faces

        top, bottom, left, right, height, width = image_pad_info
        # print ('In renderer_nvidiff:')
        # print ('focal_length:', focal_length)
        if focal_length == 0:
            the_focal_length = self.focal_length * max(height, width) / 2
        else:
            the_focal_length = focal_length
        render_reso = (height, width)

        # proj = self.intrinsics(fx=the_focal_length, fy=the_focal_length, cx=width / 2, cy=height / 2)
        proj = self.intrinsics_v2(yfov=self.yfov, cx=width / 2, cy=height / 2)
        mvp = proj @ self.mv
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])
        # print ('v_pos_clip.size is', v_pos_clip.size())
        # v_pos_clip.size is torch.Size([1, 38418, 4])
        # os._exit(0)

        seg_map = self.render_seg_map(v_pos_clip, faces, mesh.colors, render_reso)  # torch.Size([1, ori_h, ori_w, 3])
        weight_map = self.render_weight_map(v_pos_clip, faces, mesh.weights, render_reso)

        return seg_map, weight_map


class NvdiffrastPartIdx(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

        self.mv = torch.eye(4).cuda()
        # print ('?')
        self.campos = torch.linalg.inv(self.mv)[:3, 3]
        # print ('??')
        # self.glctx = dr.RasterizeGLContext()
        # print ('???')

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_prim_index(self, v_pos_clip, faces, primitives, uvs, render_reso):
        # print ('prepare raster.')
        glctx = dr.RasterizeGLContext()
        # print ('raster prepared.')
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                prim_idx, _ = self.interpolate(primitives[None, ...].float(), rast, faces.int())
                prim_uvs, _ = self.interpolate(uvs[None, ...].float(), rast, faces.int())
                # print ('prim_uvs.size is', prim_uvs.size())
                # prim_uvs.size is torch.Size([1, 720, 1280, 2])
                # os._exit(0)
                # print('*** prim_idx: ', prim_idx.shape, prim_idx.max(), prim_idx.min())
        #         *** prim_idx:  torch.Size([1, 720, 1280, 1]) tensor(1., device='cuda:0') tensor(0., device='cuda:0')
        # os._exit(0)

        prim_idx = prim_idx.squeeze(0)
        prim_uvs = prim_uvs.squeeze(0)
        # print ('prim_idx.size is', prim_idx.size())
        # prim_idx.size is torch.Size([720, 1280, 1])
        # os._exit(0)
        return prim_idx, prim_uvs

    def __call__(self, mesh, height, width, focal_length):
        verts = mesh.vertices
        verts = torch.matmul(self.rot, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))
        faces = mesh.faces

        if focal_length == 0:
            the_focal_length = self.focal_length * max(height, width) / 2
        else:
            the_focal_length = focal_length
        render_reso = (height, width)

        proj = self.intrinsics(fx=the_focal_length, fy=the_focal_length, cx=width / 2, cy=height / 2)
        mvp = proj @ self.mv
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        seg_map, uv_map = self.render_prim_index(v_pos_clip, faces, mesh.primitives, mesh.uvs, render_reso)

        return seg_map, uv_map


class NvdiffrastPartIdx_SYN(object):
    def __init__(self):
        self.mv = torch.eye(4)

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def intrinsics_v2(self, yfov, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        aspect_ratio = float(width) / float(height)
        a = aspect_ratio
        t = np.tan(yfov / 2.0)
        n = znear
        f = zfar

        P[0][0] = 1.0 / (a * t)
        P[1][1] = -1.0 / t
        P[3][2] = -1.0

        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_prim_index(self, v_pos_clip, faces, primitives, uvs, render_reso):
        # print ('prepare raster.')
        glctx = dr.RasterizeGLContext()
        # print ('raster prepared.')
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                prim_idx, _ = self.interpolate(primitives[None, ...].float(), rast, faces.int())
                prim_uvs, _ = self.interpolate(uvs[None, ...].float(), rast, faces.int())

        prim_idx = prim_idx.squeeze(0)
        prim_uvs = prim_uvs.squeeze(0)

        return prim_idx, prim_uvs

    def ccw_to_cw_rotation_matrix(self, rotation_matrix_ccw):
        # Extract the rotation angle from the given rotation matrix
        theta = np.arccos(rotation_matrix_ccw[0, 0])

        # Convert the counterclockwise rotation angle to clockwise rotation angle
        theta_cw = -theta

        # Compute the clockwise rotation matrix using the new angle
        cos_theta_cw = np.cos(theta_cw)
        # sin_theta_cw = -np.sin(theta_cw)  # Flip the sign of sine component
        sin_theta_cw = np.sin(theta_cw)

        rotation_matrix_cw = rotation_matrix_ccw.copy()
        rotation_matrix_cw[0][0] = cos_theta_cw
        rotation_matrix_cw[0][2] = sin_theta_cw
        rotation_matrix_cw[2][0] = -sin_theta_cw
        rotation_matrix_cw[2][2] = cos_theta_cw

        return rotation_matrix_cw

    def preprocess_campose_v2(self, campose):
        new_campose = campose.copy()

        old_rot = campose[:3, :3]
        old_trans = campose[:3, 3]
        mat = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        # try_new_rot = old_rot
        try_new_rot = np.transpose(old_rot, (1, 0))
        try_new_rot = np.matmul(mat, try_new_rot)
        try_new_rot = np.matmul(try_new_rot, np.linalg.inv(mat))

        rotmat_x = trimesh.transformations.rotation_matrix(
            np.radians(90), [1, 0, 0])
        new_rot = np.matmul(rotmat_x[:3, :3], try_new_rot)

        new_rot_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.Tensor(new_rot), 'XYZ')
        # print ('new_rot_euler =', torch.rad2deg(new_rot_euler))


        new_x = old_trans[0] #* (-1)
        new_y = old_trans[2]
        new_z = old_trans[1] * (-1)
        new_trans = np.array([new_x, new_y, new_z])

        new_campose[:3, :3] = new_rot
        new_campose[:3, 3] = new_trans

        return new_campose

    def preprocess_campose(self, campose):
        new_campose = campose.copy()

        old_rot = campose[:3, :3]
        old_trans = campose[:3, 3]

        mat = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        try_new_rot = np.transpose(old_rot, (1, 0))
        try_new_rot = np.matmul(mat, try_new_rot)
        try_new_rot = np.matmul(try_new_rot, np.linalg.inv(mat))

        rotmat_x = trimesh.transformations.rotation_matrix(
            np.radians(90), [1, 0, 0])
        new_rot = np.matmul(rotmat_x[:3, :3], try_new_rot)

        new_rot = self.ccw_to_cw_rotation_matrix(new_rot)

        new_x = old_trans[0] * (-1)
        new_y = old_trans[2]
        new_z = old_trans[1] * (-1)
        new_trans = np.array([new_x, new_y, new_z])

        new_campose[:3, :3] = new_rot
        new_campose[:3, 3] = new_trans

        return new_campose

    def focal_length_to_fovy(self, focal_length, sensor_height):
        return 2 * np.arctan(0.5 * sensor_height / focal_length)

    def __call__(self, mesh, cam_intri, cam_extri):
        verts = mesh.vertices
        faces = mesh.faces

        # print ('verts.size is', verts.size(), ', faces.size is', faces.size())
        # os._exit(0)

        fx, fy = cam_intri[0][0], cam_intri[1][1]
        cx, cy = cam_intri[0][2], cam_intri[1][2]

        the_focal_length = fx
        height = int(cy * 2)
        width = int(cx * 2)

        # cam_pose = self.preprocess_campose_v2(cam_extri)
        # cam_pose = torch.Tensor(cam_pose).cuda()
        cam_pose = cam_extri

        render_reso = (height, width)

        yfov = self.focal_length_to_fovy(fx, 2 * cx)

        # proj = self.intrinsics(fx=the_focal_length, fy=the_focal_length, cx=width / 2, cy=height / 2)
        proj = self.intrinsics_v2(yfov=yfov, cx=cx, cy=cy)
        mvp = proj @ cam_pose
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        seg_map, uv_map = self.render_prim_index(v_pos_clip, faces, mesh.primitives, mesh.uvs, render_reso)

        return seg_map, uv_map


class NvdiffrastPartIdx_SYN_test(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        if focal_length is None:
            self.focal_length = 1 / (np.tan(np.radians(FOV / 2)))
        else:
            self.focal_length = focal_length / max(height, width) * 2
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

        self.mv = torch.eye(4).cuda()
        # print ('?')
        self.campos = torch.linalg.inv(self.mv)[:3, 3]
        self.yfov = np.radians(FOV)
        # print ('??')
        # self.glctx = dr.RasterizeGLContext()
        # print ('???')

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        # print ('attr.data_type is', type(attr), ', rast.data_type is', type(rast))
        # os._exit(0)
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def intrinsics_v2(self, yfov, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        aspect_ratio = float(width) / float(height)
        a = aspect_ratio
        t = np.tan(yfov / 2.0)
        n = znear
        f = zfar

        P[0][0] = 1.0 / (a * t)
        P[1][1] = -1.0 / t
        P[3][2] = -1.0

        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_seg_map(self, v_pos_clip, faces, colors, render_reso):
        # print ('prepare raster.')
        glctx = dr.RasterizeGLContext()
        # print ('raster prepared.')
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                seg_map, _ = self.interpolate(colors[None, ...], rast, faces)
                # print('*** seg_map: ', seg_map.shape, seg_map.max(), seg_map.mean())
                # print ('seg_map.requires_grad =', seg_map.requires_grad, ', seg_map.is_cuda =', seg_map.is_cuda)
                # *** seg_map:  torch.Size([1, 867, 1300, 3]) tensor(1., device='cuda:0', grad_fn=<MaxBackward1>) tensor(0.1686, device='cuda:0', grad_fn=<MeanBackward0>)
                # seg_map.requires_grad = True , seg_map.is_cuda = True
                # os._exit(0)

        return seg_map

    def __call__(self, mesh, cam_intri, cam_extri, focal_length=0):
        verts = mesh.vertices
        verts = torch.matmul(self.rot, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))
        faces = mesh.faces

        # print ('verts.size is', verts.size(), ', faces.size is', faces.size())
        # os._exit(0)

        # fx, fy = cam_intri[0][0], cam_intri[1][1]
        cx, cy = cam_intri[0][2], cam_intri[1][2]

        # the_focal_length = fx
        height = int(cy * 2)
        width = int(cx * 2)

        if focal_length == 0:
            the_focal_length = self.focal_length * max(height, width) / 2
        else:
            the_focal_length = focal_length

        render_reso = (height, width)

        # proj = self.intrinsics(fx=the_focal_length, fy=the_focal_length, cx=width / 2, cy=height / 2)
        proj = self.intrinsics_v2(yfov=self.yfov, cx=width / 2, cy=height / 2)
        mvp = proj @ self.mv
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        seg_map = self.render_seg_map(v_pos_clip, faces, mesh.colors, render_reso)
        # print ('max.seg_map is', torch.max(seg_map), ', min.seg_map is', torch.min(seg_map))
        # print (seg_map.shape)
        # os._exit(0)

        return seg_map.squeeze(0) * 255.


class NvdiffrastColorANDIdx(object):
    def __init__(self, FOV=39.6, height=512, width=512, focal_length=None):
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

        self.mv = torch.eye(4).cuda()
        self.mv[:3, -1] = torch.Tensor([0, 0, -5]).cuda()
        self.campos = torch.linalg.inv(self.mv)[:3, 3]

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_idx_and_color(self, v_pos_clip, faces, primitives, colors, render_reso):
        glctx = dr.RasterizeGLContext()

        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                prim_idx, _ = self.interpolate(primitives[None, ...].float(), rast, faces.int())
                seg_map, _ = self.interpolate(colors[None, ...], rast, faces.int())

        prim_idx = prim_idx.squeeze(0)
        seg_map = seg_map.squeeze(0)

        return prim_idx, seg_map

    def __call__(self, mesh, rotmat, height=1000, width=800, focal_length=0):
        verts = mesh.vertices
        verts = torch.matmul(rotmat, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))
        faces = mesh.faces

        verts_mean_x = torch.mean(verts[:, 0])
        verts_mean_y = torch.mean(verts[:, 1])
        verts_mean_z = torch.mean(verts[:, 2])

        # verts = torch.sub(verts, torch.Tensor([verts_mean_x, verts_mean_y, verts_mean_z])[None, :].repeat(verts.size(0), 1).cuda())

        if focal_length == 0:
            the_focal_length = self.focal_length * max(height, width) / 2
        else:
            the_focal_length = focal_length
        render_reso = (height, width)

        proj = self.intrinsics(fx=the_focal_length, fy=the_focal_length, cx=width / 2, cy=height / 2)
        mvp = proj @ self.mv
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        idx_map, color_map = self.render_idx_and_color(v_pos_clip, faces, mesh.primitives, mesh.colors, render_reso)

        return idx_map, color_map


class NvdiffrastPartIdxANDColor_SYN(object):
    def __init__(self):
        self.mv = torch.eye(4)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def intrinsics_v2(self, yfov, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        aspect_ratio = float(width) / float(height)
        a = aspect_ratio
        t = np.tan(yfov / 2.0)
        n = znear
        f = zfar

        P[0][0] = 1.0 / (a * t)
        P[1][1] = -1.0 / t
        P[3][2] = -1.0

        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_index_color(self, v_pos_clip, faces, primitives, colors, render_reso):
        glctx = dr.RasterizeGLContext()
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                prim_idx, _ = self.interpolate(primitives[None, ...].float(), rast, faces.int())
                prim_colors, _ = self.interpolate(colors[None, ...].float(), rast, faces.int())

        prim_idx = prim_idx.squeeze(0)
        prim_colors = prim_colors.squeeze(0)

        return prim_idx, prim_colors

    def preprocess_campose_v2(self, campose):
        new_campose = campose.copy()

        old_rot = campose[:3, :3]
        old_trans = campose[:3, 3]
        mat = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        # try_new_rot = old_rot
        try_new_rot = np.transpose(old_rot, (1, 0))
        try_new_rot = np.matmul(mat, try_new_rot)
        try_new_rot = np.matmul(try_new_rot, np.linalg.inv(mat))

        rotmat_x = trimesh.transformations.rotation_matrix(
            np.radians(90), [1, 0, 0])
        new_rot = np.matmul(rotmat_x[:3, :3], try_new_rot)

        new_x = old_trans[0]
        new_y = old_trans[2]
        new_z = old_trans[1] * (-1)
        new_trans = np.array([new_x, new_y, new_z])

        new_campose[:3, :3] = new_rot
        new_campose[:3, 3] = new_trans

        return new_campose

    def focal_length_to_fovy(self, focal_length, sensor_height):
        return 2 * np.arctan(0.5 * sensor_height / focal_length)

    def __call__(self, mesh, cam_intri, cam_extri):
        verts = mesh.vertices
        faces = mesh.faces

        # print ('verts.size is', verts.size(), ', faces.size is', faces.size())
        # os._exit(0)

        fx, fy = cam_intri[0][0], cam_intri[1][1]
        cx, cy = cam_intri[0][2], cam_intri[1][2]

        the_focal_length = fx
        height = int(cy * 2)
        width = int(cx * 2)

        # cam_pose = self.preprocess_campose_v2(cam_extri)
        # cam_pose = torch.Tensor(cam_pose).cuda()
        cam_pose = cam_extri.cuda()

        render_reso = (height, width)

        yfov = self.focal_length_to_fovy(fx, 2 * cx)

        # proj = self.intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
        proj = self.intrinsics_v2(yfov=yfov, cx=cx, cy=cy)
        mvp = proj @ cam_pose
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        idx_map, color_map = self.render_index_color(v_pos_clip, faces, mesh.primitives, mesh.colors, render_reso)

        return idx_map, color_map


class NvdiffrastPartIdxANDColor_SYN_rotx(object):
    def __init__(self):
        self.mv = torch.eye(4)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def intrinsics_v2(self, yfov, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        aspect_ratio = float(width) / float(height)
        a = aspect_ratio
        t = np.tan(yfov / 2.0)
        n = znear
        f = zfar

        P[0][0] = 1.0 / (a * t)
        P[1][1] = -1.0 / t
        P[3][2] = -1.0

        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_index_color(self, v_pos_clip, faces, primitives, colors, render_reso):
        glctx = dr.RasterizeGLContext()
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                prim_idx, _ = self.interpolate(primitives[None, ...].float(), rast, faces.int())
                prim_colors, _ = self.interpolate(colors[None, ...].float(), rast, faces.int())

        prim_idx = prim_idx.squeeze(0)
        prim_colors = prim_colors.squeeze(0)

        return prim_idx, prim_colors

    def preprocess_campose_v2(self, campose):
        new_campose = campose.copy()

        old_rot = campose[:3, :3]
        old_trans = campose[:3, 3]
        mat = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        # try_new_rot = old_rot
        try_new_rot = np.transpose(old_rot, (1, 0))
        try_new_rot = np.matmul(mat, try_new_rot)
        try_new_rot = np.matmul(try_new_rot, np.linalg.inv(mat))

        rotmat_x = trimesh.transformations.rotation_matrix(
            np.radians(90), [1, 0, 0])
        new_rot = np.matmul(rotmat_x[:3, :3], try_new_rot)

        new_x = old_trans[0]
        new_y = old_trans[2]
        new_z = old_trans[1] * (-1)
        new_trans = np.array([new_x, new_y, new_z])

        new_campose[:3, :3] = new_rot
        new_campose[:3, 3] = new_trans

        return new_campose

    def focal_length_to_fovy(self, focal_length, sensor_height):
        return 2 * np.arctan(0.5 * sensor_height / focal_length)

    def __call__(self, mesh, cam_intri, cam_extri):
        verts = mesh.vertices
        verts = torch.matmul(self.rot, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))
        faces = mesh.faces

        # print ('verts.size is', verts.size(), ', faces.size is', faces.size())
        # os._exit(0)

        fx, fy = cam_intri[0][0], cam_intri[1][1]
        cx, cy = cam_intri[0][2], cam_intri[1][2]

        the_focal_length = fx
        height = int(cy * 2)
        width = int(cx * 2)

        # cam_pose = self.preprocess_campose_v2(cam_extri)
        # cam_pose = torch.Tensor(cam_pose).cuda()
        cam_pose = cam_extri.cuda()

        render_reso = (height, width)

        yfov = self.focal_length_to_fovy(fx, 2 * cx)

        # proj = self.intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
        proj = self.intrinsics_v2(yfov=yfov, cx=cx, cy=cy)
        mvp = proj @ cam_pose
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        idx_map, color_map = self.render_index_color(v_pos_clip, faces, mesh.primitives, mesh.colors, render_reso)

        return idx_map, color_map