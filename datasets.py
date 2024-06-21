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
    def __init__(self, data_path, train, image_size, target_path, num_parts=2):
        self.data_path = data_path # TODO
        self.train = train
        self.image_size = image_size
        self.transform = T.Resize((self.image_size, self.image_size))

        videos = os.listdir(self.data_path)
        videos.sort()
        videos = videos[6:] if train else videos[:6]

        self.data_list = []
        for video in videos:
            cur_video_path = os.path.join(self.data_path, video)

            frames = os.listdir(os.path.join(cur_video_path, 'frames'))
            frames.sort()
            omasks = os.listdir(os.path.join(cur_video_path, 'gt_mask'))
            omasks.sort()
            assert len(frames) == len(omasks), 'The number of frames and masks should be the same'

            jointstate_path = os.path.join(cur_video_path, 'jointstate.txt')
            with open(jointstate_path, "r") as f:
                joint_states = f.readlines()
            
            _3dinfo_path = os.path.join(cur_video_path, '3d_info.txt')
            with open(_3dinfo_path, "r") as f:
                content = f.readlines()
            
            _3d_info = dict()
            for line in content:
                line = line.strip().split(':')
                if len(line) == 2:
                    _3d_info[line[0].strip()] = line[1].strip()

            n_frames = len(frames)
            for i in range(n_frames):
                f = frames[i]
                m = omasks[i]
                joint_state = (float(joint_states[i].strip()) - 90) / 180.0 * np.pi
                self.data_list.append((os.path.join(cur_video_path, 'frames', f), os.path.join(cur_video_path, 'gt_mask', m), 
                                       joint_state, _3d_info))

        self.load_targets(target_path, num_parts)
    
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

    def img_preprocess(self, image, input_size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pad_image, image_pad_info = self.padding_image(image)
        input_image = torch.from_numpy(cv2.resize(pad_image, (input_size, input_size), interpolation=cv2.INTER_CUBIC))[
            None].float()
        return input_image, image_pad_info

    def load_image(self, path):
        path = path.strip()
        img = cv2.imread(path)

        ori_shape = np.shape(img)

        img, img_pad_info = self.img_preprocess(img, self.image_size)
        img = img.squeeze(0)
        img = torch.permute(img, (2, 0, 1))

        return img, img_pad_info, torch.Tensor(ori_shape)

    def load_masks(self, path):
        path = path.strip()
        mask = np.expand_dims(np.load(path), 0)
        mask = torch.Tensor(mask)

        rwp = Resize_with_pad()
        return rwp(mask).squeeze(0)

    def load_whole_mesh(self, path):
        path = path.strip()

        mesh = trimesh.load(path, force='mesh')
        # trimesh.Geometry
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

    def load_targets(self, template_path, num_parts):
        template_path = template_path.strip()
        path = os.path.join(template_path, 'plys', 'SQ_ply')
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

        self.vs = vs
        self.fs = fs    
        
        # calculate centers
        self.part_centers = np.load(os.path.join(template_path, 'part_centers.npy'))

    def __getitem__(self, index):
        # TODO: load data by index and write to data_dict to be returned

        image_path, omask_path, joint_state, _3d_info = self.data_list[index]
        image, _, ori_shape = self.load_image(image_path)
        mask = self.load_masks(omask_path)

        obj_image = image.clone().detach()
        obj_image[:, mask == 0] = 0

        # cv2.imwrite('./test/obj_image.png', obj_image.permute(1, 2, 0).numpy())
        # cv2.imwrite('./test/rbg.png',image.permute(1, 2, 0).numpy())
        # exit(0)

        data_dict = {
            'image_name': image_path,
            'rgb': image,
            'o_mask': mask,
            'o_rgb': obj_image,
            'vs': self.vs,
            'fs': self.fs,
            'part_centers': self.part_centers,
            'joint_state': torch.tensor([joint_state]),
            '3d_info': _3d_info,
            'shape': ori_shape
        }

        return data_dict

if __name__ == '__main__':
    train_dataset = Datasets(data_path='./datasets/d3dhoi_video_data/laptop', train=True, image_size=224, target_path='./SQ_templates/laptop')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)
