import cv2
import os
import torch
import scipy.io
import numpy as np
import torchvision.transforms as T
from PIL import Image
import trimesh
import pytorch3d
import pytorch3d.structures as p3dstr
import torchvision.transforms.functional as F
import pickle
import glob

# import sys
# sys.path.append('/Disk2/siqi/NewPrimReg')
# from smpl_webuser.serialization import load_model


class Resize_with_pad:
    def __init__(self, w=224, h=224):
        self.w = w
        self.h = h

    def __call__(self, image):

        _, w_1, h_1 = image.size()
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (hp, 0, hp, 0), 0, "constant")
                return F.resize(image, (self.h, self.w))

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (0, wp, 0, wp), 0, "constant")
                return F.resize(image, (self.h, self.w))

        else:
            return F.resize(image, (self.h, self.w))


class Datasets(object):
    def __init__(self, datamat_path, train, image_size, data_load_ratio):
        self.datamat_path = datamat_path
        self.train = train
        self.image_size = image_size
        self.transform = T.Resize((self.image_size, self.image_size))

        self.data_list = self.get_all_subdirectories(datamat_path) # TODO

    def get_all_subdirectories(self, path):
        data_list = []
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                data_list.append(os.path.join(root, dir))
                data_list.sort()
        
        if self.train:
            # 加载除前六个以外的所有子目录
            return data_list[6:]
        else:
            # 加载前六个子目录
            return data_list[:6]
    
    def __len__(self):
        return len(self.data_list)

    def padding_image(self, image):
        h, w = image.shape[:2]
        side_length = max(h, w)
        pad_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
        top, left = int((side_length - h) // 2), int((side_length - w) // 2)
        bottom, right = int(top + h), int(left + w)
        pad_image[top:bottom, left:right] = image
        image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
        return pad_image, image_pad_info

    def img_preprocess(self, image, input_size=224):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pad_image, image_pad_info = self.padding_image(image)
        input_image = torch.from_numpy(cv2.resize(pad_image, (input_size, input_size), interpolation=cv2.INTER_CUBIC))[
            None].float()
        return input_image, image_pad_info

    def load_images(self, path):
        path = path.strip()
        img = cv2.imread(path)

        img, img_pad_info = self.img_preprocess(img)
        img = img.squeeze(0)
        img = torch.permute(img, (2, 0, 1))

        return img, img_pad_info

    # def load_images_v2(self, path):
    #     path = path.strip()
    #     img = Image.open(path).convert('RGB')
    #     img = self.transform(img)
    #     img = torch.Tensor(np.array(img))
    #     img = torch.permute(img, (2, 0, 1))

    #     return img

    def load_masks(self, path):
        path = path.strip()
        img = Image.open(path).convert('L')
        img = np.array(img)
        img = np.reshape(img, (1, img.shape[0], img.shape[1]))
        img = torch.Tensor(img)
        rwp = Resize_with_pad()
        img = rwp(img)

        return img

    def load_whole_mesh(self, path):
        path = path.strip()

        mesh = trimesh.load(path, force='mesh')
        vertices = mesh.vertices
        faces = mesh.faces

        vertices = torch.Tensor(vertices)
        faces = torch.Tensor(faces)

        return vertices, faces

    def cal_box(self, path):
        mesh = trimesh.load(path, force='mesh')
        vertices = mesh.vertices
        faces = mesh.faces
        vertices = torch.Tensor(vertices)
        faces = torch.Tensor(faces)
        vertices = vertices.unsqueeze(0)
        faces = faces.unsqueeze(0)

        the_mesh = pytorch3d.structures.Meshes(vertices, faces)
        bbox = the_mesh.get_bounding_boxes()
        bbox = bbox.squeeze(0)

        bbox = bbox.cpu().detach().data.numpy()

        return bbox

    def cal_box_center(self, path):
        box = self.cal_box(path)

        mid_x = (box[0, 1] + box[0, 0]) / 2.
        mid_y = (box[1, 1] + box[1, 0]) / 2.
        mid_z = (box[2, 1] + box[2, 0]) / 2.
        mid = [mid_x, mid_y, mid_z]

        return mid

    def load_part_meshes(self, path):
        ply_path = os.path.join(path, 'image-0')
        part_num = len(os.listdir(ply_path))
        vs, fs = [], []
        for i in range(part_num):
            ply_p = os.path.join(ply_path, str(i) + '.ply')
            mesh = trimesh.load(ply_p, force='mesh')
            vertices = mesh.vertices
            faces = mesh.faces
            vertices = torch.Tensor(vertices)
            faces = torch.Tensor(faces)

            vs.append(vertices)
            fs.append(faces)

        return vs, fs

    def load_gt_sqs(self, path, is_human):
        if is_human:
            h_sqs_rots = np.load(os.path.join(path, 'pred_rots.npy'))
            ply_path = os.path.join(path, 'image-0')
            part_num = len(os.listdir(ply_path))
            h_sqs_trans = []
            for i in range(part_num):
                ply_p = os.path.join(ply_path, str(i)+'.ply')
                center = self.cal_box_center(ply_p)
                h_sqs_trans.append(center)
            h_sqs_trans = np.array(h_sqs_trans)

            return h_sqs_rots, h_sqs_trans
        else:
            o_sqs_rots = np.load(os.path.join(path, 'pred_rots.npy'))
            o_sqs_trans = np.load(os.path.join(path, 'pred_trans.npy'))
            o_sqs_scale = np.load(os.path.join(path, 'pred_scale.npy'))

            o_sqs_rots = torch.Tensor(o_sqs_rots)
            o_sqs_trans = torch.Tensor(o_sqs_trans)
            o_sqs_scale = torch.Tensor(o_sqs_scale)

            return o_sqs_rots, o_sqs_trans, o_sqs_scale

    def load_targets(self, path, num_parts):
        path = path.strip()
        vs, fs = [], []
        for i in range(num_parts):
            prim_p = os.path.join(path, str(i) + '.ply')
            mesh = trimesh.load(prim_p, force='mesh', process=False)
            vertices = mesh.vertices
            faces = mesh.faces
            vertices = torch.Tensor(vertices)
            faces = torch.Tensor(faces)

            vs.append(vertices)
            fs.append(faces)

        # calculate centers
        part_centers = self.cal_bbox_center(vs, fs)

        return vs, fs, part_centers

    def __getitem__(self, index):
        # TODO: load data by index and write to data_dict to be returned

        data_path = self.data_list[index]
        rgb_image_path = sorted(glob.glob(os.path.join(data_path, 'frames', '*.jpg')))
        mask_image_path = sorted(glob.glob(os.path.join(data_path, 'gt_mask', '*object_mask.npy')))

        rgb_image = self.load_images(rgb_image_path[0])
        mask = self.load_masks(mask_image_path[0])

        vs, fs, part_centers = self.load_targets('./SQ_templates/laptop/plys/SQ_ply', 2)
        data_dict = {
            'image_name':rgb_image_path,
            'rgb': rgb_image,
            'o_mask': mask,
            'vs': vs,
            'fs': fs,
            'part_centers': part_centers
        }
        

        return data_dict


