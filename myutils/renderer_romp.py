import os
import pyrender
import trimesh
import numpy as np
import torch
import pytorch3d.transforms

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags


def add_light(scene, light):
    # Use 3 directional lights
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)


def add_light_gpu(scene, light):
    # Use 3 directional lights
    light_pose = torch.eye(4)
    light_pose[:3, 3] = torch.Tensor([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = torch.Tensor([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = torch.Tensor([1, 1, 2])
    scene.add(light, pose=light_pose)


class Py3DR_SQMesh(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        self.colors = [
            [255, 51, 51],  # red
            [255, 153, 51],  # orange
            [255, 255, 51],  # yellow
            [153, 255, 51],  # light green
            [51, 255, 51],  # green
            [51, 255, 153],  # green-blue
            [51, 255, 255],  # light blue
            [51, 153, 255],  # blue
            [51, 51, 255],  # deep blue
            [153, 51, 255],  # purple
            [255, 51, 255],  # pink-purple
            [255, 51, 153],  # pink # below: lighter version of the above
            [255, 153, 153],
            [255, 204, 153],
            [255, 255, 153],
            [204, 255, 153],
            [153, 255, 153],
            [153, 255, 204],
            [153, 255, 255],
            [153, 204, 255],
            [153, 153, 255],
            [204, 153, 255],
            [255, 153, 255],
            [255, 153, 204]
        ]

    def sort_mesh_nodes_by_name(self, scene):
        trans_nodes = []
        solid_nodes = []
        # print ('len(scene.mesh_nodes) =', len(scene.mesh_nodes))
        for i in range(len(scene.mesh_nodes)):
            node_name = 'mesh_' + str(i)
            node = scene.get_nodes(name=node_name)
            node = next(iter(node))
        # for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        return solid_nodes + trans_nodes

    def focal_length_to_fovy(self, focal_length, sensor_height):
        return 2 * np.arctan(0.5 * sensor_height / focal_length)

    def __call__(self, vertices, triangles, image, mesh_colors=None, f=None, persp=True, camera_pose=None):
        img_height, img_width = image.shape[:2]
        self.renderer.viewport_height = img_height
        self.renderer.viewport_width = img_width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0],
                                ambient_light=(0.5, 0.5, 0.5))

        if camera_pose is None:
            camera_pose = np.eye(4)
        if persp:
            if f is None:
                f = self.focal_length * max(img_height, img_width) / 2
            yfov = self.focal_length_to_fovy(f, 2 * (img_width / 2.))
            camera = PerspectiveCamera(yfov=yfov)
            # camera = pyrender.camera.IntrinsicsCamera(fx=f, fy=f, cx=img_width / 2., cy=img_height / 2.)
        else:
            # xmag = ymag = np.abs(vertices[:,:,:2]).max() * 1.05
            verts_human = vertices[0]
            xmag = ymag = np.abs(verts_human[:, :, :2]).max() * 1.05
            camera = pyrender.camera.OrthographicCamera(xmag, ymag, znear=0.05, zfar=100.0, name=None)
        scene.add(camera, pose=camera_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # print ('len(vertices) =', len(vertices))
        # for every person in the scene
        for n in range(len(vertices)):
            the_verts = np.reshape(vertices[n], (-1, 3))
            the_faces = np.reshape(triangles[n], (-1, 3))
            mesh = trimesh.Trimesh(the_verts, the_faces)
            mesh.apply_transform(self.rot)
            mesh_color = self.colors[n % len(self.colors)]
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh_'+str(n))
            # mesh = pyrender.Mesh.from_trimesh(mesh)
            # mesh_node = pyrender.Node(mesh=mesh)
            # scene.add_node(mesh_node)
            # seg_node_map[str(n)] = self.colors[n]

        add_light(scene, light)
        sorted_mesh_nodes = self.sort_mesh_nodes_by_name(scene)
        # print ('Start print node......')
        # for node in sorted_mesh_nodes:
        #     print (node.name)
        # os._exit(0)

        # nm = {node: 10*(i+1) for i, node in enumerate(scene.mesh_nodes)}  # Node->Seg Id map
        # nm = {node: self.colors[i] for i, node in enumerate(scene.mesh_nodes)}  # Node->Seg Id map
        nm = {node: self.colors[i] for i, node in enumerate(sorted_mesh_nodes)}  # Node->Seg Id map
        # render_flag = pyrender.RenderFlags.SEG
        render_flag = pyrender.RenderFlags.RGBA
        # seg = self.renderer.render(scene, render_flag, nm)[0]
        img, depth = self.renderer.render(scene, render_flag)

        # seg = seg.astype(np.float32)
        # depth = depth.astype(np.float32)

        # return depth * 255
        # np.save('test_seg_np_2.npy', seg)
        # print ('seg.shape is', seg.shape)

        output_image = img[:, :, :3]
        # print (output_image)

        return output_image

    def delete(self):
        self.renderer.delete()


class Py3DR(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        self.colors = [
            [255, 51, 51],  # red
            [255, 153, 51],  # orange
            [255, 255, 51],  # yellow
            [153, 255, 51],  # light green
            [51, 255, 51],  # green
            [51, 255, 153],  # green-blue
            [51, 255, 255],  # light blue
            [51, 153, 255],  # blue
            [51, 51, 255],  # deep blue
            [153, 51, 255],  # purple
            [255, 51, 255],  # pink-purple
            [255, 51, 153],  # pink # below: lighter version of the above
            [255, 153, 153],
            [255, 204, 153],
            [255, 255, 153],
            [204, 255, 153],
            [153, 255, 153],
            [153, 255, 204],
            [153, 255, 255],
            [153, 204, 255],
            [153, 153, 255],
            [204, 153, 255],
            [255, 153, 255],
            [255, 153, 204]
        ]

    def __call__(self, vertices, triangles, image, mesh_colors=None, f=None, persp=True, camera_pose=None):
        img_height, img_width = image.shape[:2]
        self.renderer.viewport_height = img_height
        self.renderer.viewport_width = img_width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(0.3, 0.3, 0.3))

        if camera_pose is None:
            camera_pose = np.eye(4)
        if persp:
            if f is None:
                f = self.focal_length * max(img_height, img_width) / 2
            camera = pyrender.camera.IntrinsicsCamera(fx=f, fy=f, cx=img_width / 2., cy=img_height / 2.)
        else:
            # xmag = ymag = np.abs(vertices[:,:,:2]).max() * 1.05
            verts_human = vertices[0]
            xmag = ymag = np.abs(verts_human[:, :, :2]).max() * 1.05
            camera = pyrender.camera.OrthographicCamera(xmag, ymag, znear=0.05, zfar=100.0, name=None)
        scene.add(camera, pose=camera_pose)

        # if len(triangles.shape) == 2:
        #     triangles = [triangles for _ in range(len(vertices))]

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # for every person in the scene
        for n in range(len(vertices)):
            print (vertices[n].shape)
            os._exit(0)
            the_verts = np.reshape(vertices[n], (-1, 3))
            the_faces = np.reshape(triangles[n], (-1, 3))
            mesh = trimesh.Trimesh(the_verts, the_faces)
            mesh.apply_transform(self.rot)
            if mesh_colors is None:
                mesh_color = self.colors[n % len(self.colors)]
            else:
                mesh_color = mesh_colors[n % len(mesh_colors)]
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')

            add_light(scene, light)
        # Alpha channel was not working previously need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        color = color.astype(np.float32)
        valid_mask = (rend_depth > 0)[:, :, None]
        output_image = (color[:, :, :3] * valid_mask +
                        (1 - valid_mask) * image).astype(np.uint8)

        # return color, rend_depth
        return output_image, rend_depth

    def delete(self):
        self.renderer.delete()


class Py3DR_mask(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        self.colors = [
            [255, 51, 51],  # red
            [255, 153, 51],  # orange
            [255, 255, 51],  # yellow
            [153, 255, 51],  # light green
            [51, 255, 51],  # green
            [51, 255, 153],  # green-blue
            [51, 255, 255],  # light blue
            [51, 153, 255],  # blue
            [51, 51, 255],  # deep blue
            [153, 51, 255],  # purple
            [255, 51, 255],  # pink-purple
            [255, 51, 153],  # pink # below: lighter version of the above
            [255, 153, 153],
            [255, 204, 153],
            [255, 255, 153],
            [204, 255, 153],
            [153, 255, 153],
            [153, 255, 204],
            [153, 255, 255],
            [153, 204, 255],
            [153, 153, 255],
            [204, 153, 255],
            [255, 153, 255],
            [255, 153, 204]
        ]
        # self.colors = [
        #     [255, 255, 255] for i in range(22)
        # ]

    def sort_mesh_nodes_by_name(self, scene):
        trans_nodes = []
        solid_nodes = []
        # print ('len(scene.mesh_nodes) =', len(scene.mesh_nodes))
        for i in range(len(scene.mesh_nodes)):
            node_name = 'mesh_' + str(i)
            node = scene.get_nodes(name=node_name)
            node = next(iter(node))
        # for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        return solid_nodes + trans_nodes


    def __call__(self, vertices, triangles, image, mesh_colors=None, f=None, persp=True, camera_pose=None):
        img_height, img_width = image.shape[:2]
        self.renderer.viewport_height = img_height
        self.renderer.viewport_width = img_width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(1., 1., 1.))

        if camera_pose is None:
            camera_pose = np.eye(4)
        if persp:
            if f is None:
                f = self.focal_length * max(img_height, img_width) / 2
            camera = pyrender.camera.IntrinsicsCamera(fx=f, fy=f, cx=img_width / 2., cy=img_height / 2.)
        else:
            # xmag = ymag = np.abs(vertices[:,:,:2]).max() * 1.05
            verts_human = vertices[0]
            xmag = ymag = np.abs(verts_human[:, :, :2]).max() * 1.05
            camera = pyrender.camera.OrthographicCamera(xmag, ymag, znear=0.05, zfar=100.0, name=None)
        scene.add(camera, pose=camera_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # print ('len(vertices) =', len(vertices))
        # for every person in the scene
        for n in range(len(vertices)):
            the_verts = np.reshape(vertices[n], (-1, 3))
            the_faces = np.reshape(triangles[n], (-1, 3))
            mesh = trimesh.Trimesh(the_verts, the_faces)
            mesh.apply_transform(self.rot)
            if mesh_colors is None:
                mesh_color = self.colors[n % len(self.colors)]
            else:
                mesh_color = mesh_colors[n % len(mesh_colors)]
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh_'+str(n))
            # mesh = pyrender.Mesh.from_trimesh(mesh)
            # mesh_node = pyrender.Node(mesh=mesh)
            # scene.add_node(mesh_node)
            # seg_node_map[str(n)] = self.colors[n]

        add_light(scene, light)
        sorted_mesh_nodes = self.sort_mesh_nodes_by_name(scene)
        # print ('Start print node......')
        # for node in sorted_mesh_nodes:
        #     print (node.name)
        # os._exit(0)

        # nm = {node: 10*(i+1) for i, node in enumerate(scene.mesh_nodes)}  # Node->Seg Id map
        # nm = {node: self.colors[i] for i, node in enumerate(scene.mesh_nodes)}  # Node->Seg Id map
        nm = {node: self.colors[i] for i, node in enumerate(sorted_mesh_nodes)}  # Node->Seg Id map
        render_flag = pyrender.RenderFlags.SEG
        # render_flag = pyrender.RenderFlags.RGBA
        seg = self.renderer.render(scene, render_flag, nm)[0]
        # img, depth = self.renderer.render(scene, render_flag)

        seg = seg.astype(np.float32)
        # depth = depth.astype(np.float32)

        # return depth * 255
        # np.save('test_seg_np_2.npy', seg)
        # print ('seg.shape is', seg.shape)

        output_image = seg
        # print (output_image)

        return output_image

    def delete(self):
        self.renderer.delete()


class Py3DR_mask_d3dhoi(object):
    def __init__(self, height=512, width=512):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        self.colors = [
            [255, 51, 51],  # red
            [255, 153, 51],  # orange
            [255, 255, 51],  # yellow
            [153, 255, 51],  # light green
            [51, 255, 51],  # green
            [51, 255, 153],  # green-blue
            [51, 255, 255],  # light blue
            [51, 153, 255],  # blue
            [51, 51, 255],  # deep blue
            [153, 51, 255],  # purple
            [255, 51, 255],  # pink-purple
            [255, 51, 153],  # pink # below: lighter version of the above
            [255, 153, 153],
            [255, 204, 153],
            [255, 255, 153],
            [204, 255, 153],
            [153, 255, 153],
            [153, 255, 204],
            [153, 255, 255],
            [153, 204, 255],
            [153, 153, 255],
            [204, 153, 255],
            [255, 153, 255],
            [255, 153, 204]
        ]
        # self.colors = [
        #     [255, 255, 255] for i in range(22)
        # ]

    def sort_mesh_nodes_by_name(self, scene):
        trans_nodes = []
        solid_nodes = []
        # print ('len(scene.mesh_nodes) =', len(scene.mesh_nodes))
        for i in range(len(scene.mesh_nodes)):
            node_name = 'mesh_' + str(i)
            node = scene.get_nodes(name=node_name)
            node = next(iter(node))
        # for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        return solid_nodes + trans_nodes


    def __call__(self, vertices, triangles, image, fx, fy, mesh_colors=None):
        img_height, img_width = image.shape[:2]
        self.renderer.viewport_height = img_height
        self.renderer.viewport_width = img_width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(1., 1., 1.))


        camera_pose = np.eye(4)
        camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=img_width / 2., cy=img_height / 2.)
        scene.add(camera, pose=camera_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        for n in range(len(vertices)):
            the_verts = np.reshape(vertices[n], (-1, 3))
            the_faces = np.reshape(triangles[n], (-1, 3))
            mesh = trimesh.Trimesh(the_verts, the_faces)
            mesh.apply_transform(self.rot)
            if mesh_colors is None:
                mesh_color = self.colors[n % len(self.colors)]
            else:
                mesh_color = mesh_colors[n % len(mesh_colors)]
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh_'+str(n))
            # mesh = pyrender.Mesh.from_trimesh(mesh)
            # mesh_node = pyrender.Node(mesh=mesh)
            # scene.add_node(mesh_node)
            # seg_node_map[str(n)] = self.colors[n]

        add_light(scene, light)
        sorted_mesh_nodes = self.sort_mesh_nodes_by_name(scene)
        # print ('Start print node......')
        # for node in sorted_mesh_nodes:
        #     print (node.name)
        # os._exit(0)

        # nm = {node: 10*(i+1) for i, node in enumerate(scene.mesh_nodes)}  # Node->Seg Id map
        # nm = {node: self.colors[i] for i, node in enumerate(scene.mesh_nodes)}  # Node->Seg Id map
        nm = {node: self.colors[i] for i, node in enumerate(sorted_mesh_nodes)}  # Node->Seg Id map
        render_flag = pyrender.RenderFlags.SEG
        # render_flag = pyrender.RenderFlags.RGBA
        seg = self.renderer.render(scene, render_flag, nm)[0]
        # img, depth = self.renderer.render(scene, render_flag)

        seg = seg.astype(np.float32)
        # depth = depth.astype(np.float32)

        # return depth * 255
        # np.save('test_seg_np_2.npy', seg)
        # print ('seg.shape is', seg.shape)

        output_image = seg
        # print (output_image)

        return output_image

    def delete(self):
        self.renderer.delete()


class Py3DR_mask_syn(object):
    def __init__(self, white=False):
        if not white:
            self.colors = [
            [255, 51, 51],  # red
            [255, 153, 51],  # orange
            [255, 255, 51],  # yellow
            [153, 255, 51],  # light green
            [51, 255, 51],  # green
            [51, 255, 153],  # green-blue
            [51, 255, 255],  # light blue
            [51, 153, 255],  # blue
            [51, 51, 255],  # deep blue
            [153, 51, 255],  # purple
            [255, 51, 255],  # pink-purple
            [255, 51, 153],  # pink # below: lighter version of the above
            [255, 153, 153],
            [255, 204, 153],
            [255, 255, 153],
            [204, 255, 153],
            [153, 255, 153],
            [153, 255, 204],
            [153, 255, 255],
            [153, 204, 255],
            [153, 153, 255],
            [204, 153, 255],
            [255, 153, 255],
            [255, 153, 204]
        ]
        else:
            self.colors = [
                [255, 255, 255] for i in range(22)
            ]

    def sort_mesh_nodes_by_name(self, scene):
        trans_nodes = []
        solid_nodes = []
        # print ('len(scene.mesh_nodes) =', len(scene.mesh_nodes))
        for i in range(len(scene.mesh_nodes)):
            node_name = 'mesh_' + str(i)
            node = scene.get_nodes(name=node_name)
            node = next(iter(node))
        # for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        return solid_nodes + trans_nodes

    def focal_length_to_fovy(self, focal_length, sensor_height):
        return 2 * np.arctan(0.5 * sensor_height / focal_length)

    def ccw_to_cw_rotation_matrix(self, rotation_matrix_ccw):
        # Extract the rotation angle from the given rotation matrix
        theta = np.arccos(rotation_matrix_ccw[0, 0])

        # Convert the counterclockwise rotation angle to clockwise rotation angle
        theta_cw = theta * (-1)

        # Compute the clockwise rotation matrix using the new angle
        cos_theta_cw = np.cos(theta_cw)
        # sin_theta_cw = -np.sin(theta_cw)  # Flip the sign of sine component
        sin_theta_cw = np.sin(theta_cw)

        rotation_matrix_cw = rotation_matrix_ccw.copy()
        rotation_matrix_cw[0][0] = cos_theta_cw
        rotation_matrix_cw[0][2] = sin_theta_cw
        rotation_matrix_cw[2][0] = sin_theta_cw * (-1)
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
        # try_new_rot = old_rot
        try_new_rot = np.matmul(mat, try_new_rot)
        try_new_rot = np.matmul(try_new_rot, np.linalg.inv(mat))

        rotmat_x = trimesh.transformations.rotation_matrix(
            np.radians(90), [1, 0, 0])
        new_rot = np.matmul(rotmat_x[:3, :3], try_new_rot)

        new_rot_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.Tensor(new_rot), 'XYZ')
        print ('new_rot_euler =', torch.rad2deg(new_rot_euler))

        # pi = torch.acos(torch.zeros(1)).item() * 2

        new_rot = self.ccw_to_cw_rotation_matrix(new_rot)

        new_rot_euler = pytorch3d.transforms.matrix_to_euler_angles(new_rot, 'XYZ')
        print ('new_rot_euler =', torch.rad2deg(new_rot_euler))
        # os._exit(0)

        new_x = old_trans[0] * (-1)
        new_y = old_trans[2]
        new_z = old_trans[1] * (-1)
        new_trans = np.array([new_x, new_y, new_z])

        new_campose[:3, :3] = new_rot
        new_campose[:3, 3] = new_trans

        return new_campose

    def __call__(self, vertices, triangles, camera_intrisic, camera_extrisic):

        fx, fy = camera_intrisic[0][0], camera_intrisic[1][1]
        cx, cy = camera_intrisic[0][2], camera_intrisic[1][2]

        yfov = self.focal_length_to_fovy(fx, 2 * cx)

        cam = PerspectiveCamera(yfov=yfov)
        cam_pose = self.preprocess_campose_v2(camera_extrisic)

        renderer = OffscreenRenderer(viewport_width=cx * 2, viewport_height=cy * 2)

        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(1., 1., 1.))
        cam_node = scene.add(cam, pose=cam_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # print ('len(vertices) =', len(vertices))
        # for every person in the scene
        for n in range(len(vertices)):
            the_verts = np.reshape(vertices[n], (-1, 3))
            the_faces = np.reshape(triangles[n], (-1, 3))
            mesh = trimesh.Trimesh(the_verts, the_faces)

            mesh_color = self.colors[n % len(self.colors)]

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh_'+str(n))

        add_light(scene, light)
        sorted_mesh_nodes = self.sort_mesh_nodes_by_name(scene)

        nm = {node: self.colors[i] for i, node in enumerate(sorted_mesh_nodes)}  # Node->Seg Id map
        render_flag = pyrender.RenderFlags.SEG
        seg = renderer.render(scene, render_flag, nm)[0]

        seg = seg.astype(np.float32)

        output_image = seg

        renderer.delete()

        return output_image


class Py3DR_mask_test_align(object):
    def __init__(self, white=False):
        if not white:
            self.colors = [
            [255, 51, 51],  # red
            [255, 153, 51],  # orange
            [255, 255, 51],  # yellow
            [153, 255, 51],  # light green
            [51, 255, 51],  # green
            [51, 255, 153],  # green-blue
            [51, 255, 255],  # light blue
            [51, 153, 255],  # blue
            [51, 51, 255],  # deep blue
            [153, 51, 255],  # purple
            [255, 51, 255],  # pink-purple
            [255, 51, 153],  # pink # below: lighter version of the above
            [255, 153, 153],
            [255, 204, 153],
            [255, 255, 153],
            [204, 255, 153],
            [153, 255, 153],
            [153, 255, 204],
            [153, 255, 255],
            [153, 204, 255],
            [153, 153, 255],
            [204, 153, 255],
            [255, 153, 255],
            [255, 153, 204]
        ]
        else:
            self.colors = [
                [255, 255, 255] for i in range(22)
            ]
        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

    def sort_mesh_nodes_by_name(self, scene):
        trans_nodes = []
        solid_nodes = []
        # print ('len(scene.mesh_nodes) =', len(scene.mesh_nodes))
        for i in range(len(scene.mesh_nodes)):
            node_name = 'mesh_' + str(i)
            node = scene.get_nodes(name=node_name)
            node = next(iter(node))
        # for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        return solid_nodes + trans_nodes

    def focal_length_to_fovy(self, focal_length, sensor_height):
        return 2 * np.arctan(0.5 * sensor_height / focal_length)

    def ccw_to_cw_rotation_matrix(self, rotation_matrix_ccw):
        # Extract the rotation angle from the given rotation matrix
        theta = np.arccos(rotation_matrix_ccw[0, 0])

        # Convert the counterclockwise rotation angle to clockwise rotation angle
        theta_cw = theta * (-1)

        # Compute the clockwise rotation matrix using the new angle
        cos_theta_cw = np.cos(theta_cw)
        # sin_theta_cw = -np.sin(theta_cw)  # Flip the sign of sine component
        sin_theta_cw = np.sin(theta_cw)

        rotation_matrix_cw = rotation_matrix_ccw.copy()
        rotation_matrix_cw[0][0] = cos_theta_cw
        rotation_matrix_cw[0][2] = sin_theta_cw
        rotation_matrix_cw[2][0] = sin_theta_cw * (-1)
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

    def __call__(self, vertices, triangles, camera_intrisic, camera_extrisic):

        fx, fy = camera_intrisic[0][0], camera_intrisic[1][1]
        cx, cy = camera_intrisic[0][2], camera_intrisic[1][2]

        # print ('fx =', fx, ', fy =', fy, ', cx =', cx, ', cy =', cy)

        yfov = self.focal_length_to_fovy(fx, 2 * cx)

        cam = PerspectiveCamera(yfov=yfov)
        # cam = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy)
        cam_pose = camera_extrisic

        renderer = OffscreenRenderer(viewport_width=cx * 2, viewport_height=cy * 2)

        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(1., 1., 1.))
        cam_node = scene.add(cam, pose=cam_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # print ('len(vertices) =', len(vertices))
        # for every person in the scene
        for n in range(len(vertices)):
            the_verts = np.reshape(vertices[n], (-1, 3))

            # the_verts = np.matmul(self.rot[:3, :3], np.transpose(the_verts, (1, 0)))
            # the_verts = np.transpose(the_verts, (1, 0))

            the_faces = np.reshape(triangles[n], (-1, 3))
            mesh = trimesh.Trimesh(the_verts, the_faces)

            mesh_color = self.colors[n % len(self.colors)]

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh_'+str(n))

        add_light(scene, light)
        sorted_mesh_nodes = self.sort_mesh_nodes_by_name(scene)

        nm = {node: self.colors[i] for i, node in enumerate(sorted_mesh_nodes)}  # Node->Seg Id map
        render_flag = pyrender.RenderFlags.SEG
        seg = renderer.render(scene, render_flag, nm)[0]

        seg = seg.astype(np.float32)

        output_image = seg

        renderer.delete()

        return output_image


class Py3DR_mask_align4romp(object):
    def __init__(self, white=False):
        if not white:
            self.colors = [
            [255, 51, 51],  # red
            [255, 153, 51],  # orange
            [255, 255, 51],  # yellow
            [153, 255, 51],  # light green
            [51, 255, 51],  # green
            [51, 255, 153],  # green-blue
            [51, 255, 255],  # light blue
            [51, 153, 255],  # blue
            [51, 51, 255],  # deep blue
            [153, 51, 255],  # purple
            [255, 51, 255],  # pink-purple
            [255, 51, 153],  # pink # below: lighter version of the above
            [255, 153, 153],
            [255, 204, 153],
            [255, 255, 153],
            [204, 255, 153],
            [153, 255, 153],
            [153, 255, 204],
            [153, 255, 255],
            [153, 204, 255],
            [153, 153, 255],
            [204, 153, 255],
            [255, 153, 255],
            [255, 153, 204]
        ]
        else:
            self.colors = [
                [255, 255, 255] for i in range(22)
            ]
        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

    def sort_mesh_nodes_by_name(self, scene):
        trans_nodes = []
        solid_nodes = []
        # print ('len(scene.mesh_nodes) =', len(scene.mesh_nodes))
        for i in range(len(scene.mesh_nodes)):
            node_name = 'mesh_' + str(i)
            node = scene.get_nodes(name=node_name)
            node = next(iter(node))
        # for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        return solid_nodes + trans_nodes

    def focal_length_to_fovy(self, focal_length, sensor_height):
        return 2 * np.arctan(0.5 * sensor_height / focal_length)

    def __call__(self, vertices, triangles, camera_intrisic, camera_extrisic):

        fx, fy = camera_intrisic[0][0], camera_intrisic[1][1]
        cx, cy = camera_intrisic[0][2], camera_intrisic[1][2]

        yfov = self.focal_length_to_fovy(fx, 2 * cx)

        # cam = PerspectiveCamera(yfov=yfov)
        cam = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        cam_pose = np.eye(4)

        renderer = OffscreenRenderer(viewport_width=cx * 2, viewport_height=cy * 2)

        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(1., 1., 1.))
        cam_node = scene.add(cam, pose=cam_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # print ('len(vertices) =', len(vertices))
        # for every person in the scene
        for n in range(len(vertices)):
            the_verts = np.reshape(vertices[n], (-1, 3))
            the_faces = np.reshape(triangles[n], (-1, 3))
            mesh = trimesh.Trimesh(the_verts, the_faces)

            mesh.apply_transform(self.rot)

            mesh_color = self.colors[n % len(self.colors)]

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh_'+str(n))

        add_light(scene, light)
        sorted_mesh_nodes = self.sort_mesh_nodes_by_name(scene)

        nm = {node: self.colors[i] for i, node in enumerate(sorted_mesh_nodes)}  # Node->Seg Id map
        render_flag = pyrender.RenderFlags.SEG
        seg = renderer.render(scene, render_flag, nm)[0]

        seg = seg.astype(np.float32)

        output_image = seg

        renderer.delete()

        return output_image


class Py3DR_mask_GPU(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        # self.rot = trimesh.transformations.rotation_matrix(
        #     np.radians(180), [1, 0, 0])

        self.colors = [
            [255, 51, 51],  # red
            [255, 153, 51],  # orange
            [255, 255, 51],  # yellow
            [153, 255, 51],  # light green
            [51, 255, 51],  # green
            [51, 255, 153],  # green-blue
            [51, 255, 255],  # light blue
            [51, 153, 255],  # blue
            [51, 51, 255],  # deep blue
            [153, 51, 255],  # purple
            [255, 51, 255],  # pink-purple
            [255, 51, 153],  # pink # below: lighter version of the above
            [255, 153, 153],
            [255, 204, 153],
            [255, 255, 153],
            [204, 255, 153],
            [153, 255, 153],
            [153, 255, 204],
            [153, 255, 255],
            [153, 204, 255],
            [153, 153, 255],
            [204, 153, 255],
            [255, 153, 255],
            [255, 153, 204]
        ]

    def sort_mesh_nodes_by_name(self, scene):
        trans_nodes = []
        solid_nodes = []
        # print ('len(scene.mesh_nodes) =', len(scene.mesh_nodes))
        for i in range(len(scene.mesh_nodes)):
            node_name = 'mesh_' + str(i)
            node = scene.get_nodes(name=node_name)
            node = next(iter(node))
        # for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        return solid_nodes + trans_nodes

    def unit_vector_gpu(self, data, axis=None, out=None):
        if out is None:
            data = torch.Tensor(data).clone()
            if len(data.size()) == 1:
                data = data / torch.sqrt(torch.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = torch.Tensor(data)
            data = out
        length = torch.atleast_1d(torch.sum(data * data, axis))
        torch.sqrt(length, length)
        data = data / length
        if out is None:
            return data

    def rotation_matrix_gpu(self, angle, direction):
        sina = torch.sin(angle)
        cosa = torch.cos(angle)

        direction = self.unit_vector_gpu(direction[:3]).cuda()
        # rotation matrix around unit vector
        M = torch.diag(torch.Tensor([cosa[0], cosa[0], cosa[0], 1.0])).cuda()
        M[:3, :3] = M[:3, :3] + torch.outer(direction, direction) * (1.0 - cosa)

        direction = direction * sina[0]
        M[:3, :3] = M[:3, :3] + torch.Tensor([[0.0, -direction[2], direction[1]],
                               [direction[2], 0.0, -direction[0]],
                               [-direction[1], direction[0], 0.0]]).cuda()

        # # if point is specified, rotation is not around origin
        # if point is not None:
        #     point = np.array(point[:3], dtype=np.float64, copy=False)
        #     M[:3, 3] = point - np.dot(M[:3, :3], point)

        return M[:3, :3]

    def rot(self, verts):
        M = self.rotation_matrix_gpu(torch.Tensor([np.radians(180)]).cuda(), [1, 0, 0])
        # print ('M.size is', M.size(), ', verts.size is', verts.size())
        # M.size is torch.Size([3, 3]) , verts.size is torch.Size([num_points, 3])
        # os._exit(0)
        verts = torch.matmul(M, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))

        return verts


    def __call__(self, vertices, img_height, img_width, mesh_colors=None, f=None, persp=True, camera_pose=None):
        self.renderer.viewport_height = img_height
        self.renderer.viewport_width = img_width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(1., 1., 1.))

        if camera_pose is None:
            camera_pose = np.eye(4)
        if f is None:
            f = self.focal_length * max(img_height, img_width) / 2
        camera = pyrender.camera.IntrinsicsCamera(fx=f, fy=f, cx=img_width / 2., cy=img_height / 2.)
        scene.add(camera, pose=camera_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # print ('len(vertices) =', len(vertices))
        # for every person in the scene
        for n in range(len(vertices)):
            the_verts = torch.reshape(vertices[n], (-1, 3))
            the_verts = self.rot(the_verts)
            mesh = pyrender.Mesh.from_points(the_verts.cpu())
            scene.add(mesh, 'mesh_'+str(n))

        add_light(scene, light)
        sorted_mesh_nodes = self.sort_mesh_nodes_by_name(scene)
        # print ('Start print node......')
        # for node in sorted_mesh_nodes:
        #     print (node.name)
        # os._exit(0)

        # nm = {node: 10*(i+1) for i, node in enumerate(scene.mesh_nodes)}  # Node->Seg Id map
        # nm = {node: self.colors[i] for i, node in enumerate(scene.mesh_nodes)}  # Node->Seg Id map
        nm = {node: self.colors[i] for i, node in enumerate(sorted_mesh_nodes)}  # Node->Seg Id map
        render_flag = pyrender.RenderFlags.SEG
        # render_flag = pyrender.RenderFlags.RGBA
        seg = self.renderer.render(scene, render_flag, nm)[0]
        print ('seg.is_cuda =', seg.is_cuda, ', seg.requires_grad =', seg.requires_grad)
        os._exit(0)
        # img, depth = self.renderer.render(scene, render_flag)

        # seg = seg.astype(np.float32)
        # depth = depth.astype(np.float32)

        # return depth * 255
        # np.save('test_seg_np_2.npy', seg)
        # print ('seg.shape is', seg.shape)

        output_image = seg
        # print (output_image)

        return output_image

    def delete(self):
        self.renderer.delete()