# -*- coding: utf-8 -*-
# brought from https://github.com/mkocabas/VIBE/blob/master/lib/utils/renderer.py
import sys, os
import json
import torch
from torch import nn
import pickle
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,
    SoftSilhouetteShader
)
import numpy as np
import torchvision
import torchvision.transforms

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}


class Renderer(nn.Module):
    def __init__(self, resolution=(224, 224), perps=True, R=None, T=None, use_gpu=True):
        super(Renderer, self).__init__()
        self.perps = perps
        if use_gpu:
            self.device = torch.device('cuda')
            print('visualize in gpu mode')
        else:
            self.device = torch.device('cpu')
            print('visualize in cpu mode')

        # self.transform = torchvision.transforms.Resize((224, 224))

        if R is None:
            R = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        if T is None:
            T = torch.Tensor([[0., 0., 0.]])

        if self.perps:
            self.cameras = FoVPerspectiveCameras(R=R, T=T, fov=60, device=self.device)
            self.lights = PointLights(ambient_color=((0.56, 0.56, 0.56),), location=torch.Tensor([[0., 0., 0.]]),
                                      device=self.device)
        else:
            self.cameras = FoVOrthographicCameras(R=R, T=T, znear=0., zfar=100.0, max_y=1.0, min_y=-1.0, max_x=1.0,
                                                  min_x=-1.0, device=self.device)
            self.lights = DirectionalLights(direction=torch.Tensor([[0., 1., 0.]]), device=self.device)

    def __call__(self, verts, faces, res = (224,224), colors=np.array(colors['neutral']), merge_meshes=False, cam_params=None,
                 **kwargs):
        # assert len(verts.shape) == 3, print('The input verts of visualizer is bounded to be 3-dims (Nx6890 x3) tensor')

        # print ('resolution[0] =', resolution[0])
        # os._exit(0)
        
        # print (res[0].item(), res[1].item())
        # os._exit(0)
        resolution = (224,224)

        # setup resolution for each image
        raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=216
        )

        # renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(
        #         cameras=self.cameras,
        #         raster_settings=raster_settings),
        #     shader=SoftPhongShader(
        #         device=self.device,
        #         cameras=self.cameras,
        #         lights=self.lights))

        blend_params = BlendParams(sigma=1e-8, gamma=1e-8)
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        verts, faces = verts.to(self.device), faces.to(self.device)
        verts_rgb = torch.ones_like(verts)
        verts_rgb[:, :] = torch.from_numpy(colors).cuda()#.unsqueeze(1)
        textures = TexturesVertex(verts_features=verts_rgb)
        verts[:, :, :2] *= -1
        meshes = Meshes(verts, faces, textures)

        if merge_meshes:
            meshes = join_meshes_as_scene(meshes)
        if cam_params is not None:
            if self.perps:
                R, T, fov = cam_params
                new_cam = FoVPerspectiveCameras(R=R, T=T, fov=fov, device=self.device)
                # new_cam = PerspectiveCameras(focal_length=focal_length, principal_point=((H/2, W/2),), R=R, T=T, device=self.device)
            else:
                R, T, xyz_ranges = cam_params
                new_cam = FoVOrthographicCameras(R=R, T=T, **xyz_ranges, device=self.device)

            # images = renderer(meshes, cameras=new_cam)
            masks = silhouette_renderer(meshes, cameras=new_cam)
        else:
            # images = renderer(meshes)
            masks = silhouette_renderer(meshes)
        images = None
        # images[:, :, :-1] *= 255
        # masks[:, :, 3] *= 255

        return images, masks[:, :, :, 3]


# def get_renderer(test=False, **kwargs):
#     renderer = Renderer(**kwargs)
#     if test:
#         import cv2
#         dist = 1 / np.tan(np.radians(args().FOV / 2.))
#         print('dist:', dist)
#         model = pickle.load(open(os.path.join(args().smpl_model_path, 'smpl', 'SMPL_NEUTRAL.pkl'), 'rb'),
#                             encoding='latin1')
#         np_v_template = torch.from_numpy(np.array(model['v_template'])).cuda().float()[None]
#         face = torch.from_numpy(model['f'].astype(np.int32)).cuda()[None]
#         np_v_template = np_v_template.repeat(2, 1, 1)
#         np_v_template[1] += 0.3
#         np_v_template[:, :, 2] += dist
#         face = face.repeat(2, 1, 1)
#         result = renderer(np_v_template, face).cpu().numpy()
#         for ri in range(len(result)):
#             cv2.imwrite('test{}.png'.format(ri), (result[ri, :, :, :3] * 255).astype(np.uint8))
#     return renderer
#
#
# if __name__ == '__main__':
#     get_renderer(test=True, perps=True)