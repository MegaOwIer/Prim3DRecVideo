import argparse
import json
import os
import random
import string

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns
import torch
import torch.utils.data
import trimesh
import yaml
from PIL import Image
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import datasets
from LossFunc import compute_loss
from myutils.graphAE_param import Parameters
from myutils.visualization_utils import _from_primitive_parms_to_mesh_v2
from networks.baseline_network import Network_pts
from renderer_nvdiff import Nvdiffrast
from visualize import visualize_mesh

sns.set()


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='Determine if we run the code for training or testing. Chosen from [train, test]')
parser.add_argument('--log_dir', type=str, default='./test/baseline/logs')
parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--output_directory', type=str, default='../../NewPrimReg_outputs_iccv/baseline/output_dir')
parser.add_argument('--output_directory', type=str, default='./test/baseline/output_dir')
parser.add_argument('--experiment_tag', type=str, default='laptop')
parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--vit_f_dim', type=int, default=3025) # dino
parser.add_argument('--vit_f_dim', type=int, default=384) # dinov2
# parser.add_argument('--res', type=int, default=112)
parser.add_argument('--res', type=int, default=224)
parser.add_argument('--batch_size_train', type=int, default=32, help='Batch size of the training dataloader.')
parser.add_argument('--batch_size_val', type=int, default=1, help='Batch size of the val dataloader.')
parser.add_argument('--data_path', type=str, default='./datasets/d3dhoi_video_data/laptop')
parser.add_argument('--datamat_path', type=str, default='./SQ_templates/laptop/joint_info.mat')
parser.add_argument('--continue_from_epoch', type=int, default=0)
parser.add_argument('--save_every', type=int, default=5)
parser.add_argument('--val_every', type=int, default=200)
parser.add_argument('--config_file', type=str, default='./config/tmp_config.yaml')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--annealing_lr', type=bool, default=True)
parser.add_argument('--checkpoint_model_path', type=str, default=None)

args = parser.parse_args()


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


class OptimizerWrapper(object):
    def __init__(self, optimizer, aggregate=1):
        self.optimizer = optimizer
        self.aggregate = aggregate
        self._calls = 0

    def zero_grad(self):
        if self._calls == 0:
            self.optimizer.zero_grad()

    def step(self):
        self._calls += 1
        if self._calls == self.aggregate:
            self._calls = 0
            self.optimizer.step()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer == "SGD":
        return OptimizerWrapper(
            torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                            weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "Adam":
        return OptimizerWrapper(
            torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "RAdam":
        return OptimizerWrapper(
            torch.optim.RAdam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    else:
        raise NotImplementedError()


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def save_checkpoints(epoch, model, optimizer, experiment_directory, args):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    # The optimizer is wrapped with an object implementing gradient
    # accumulation
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    if args.checkpoint_model_path is None:
        model_files = [
            f for f in os.listdir(experiment_directory)
            if f.startswith("model_")
        ]

        if len(model_files) == 0:
            return
        ids = [int(f[6:]) for f in model_files]
        max_id = max(ids)
        model_path = os.path.join(
            experiment_directory, "model_{:05d}"
        ).format(max_id)
        opt_path = os.path.join(experiment_directory, "opt_{:05d}").format(max_id)
        if not (os.path.exists(model_path) and os.path.exists(opt_path)):
            return

        print("Loading model checkpoint from {}".format(model_path))
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loading optimizer checkpoint from {}".format(opt_path))
        optimizer.load_state_dict(
            torch.load(opt_path, map_location=device)
        )
        args.continue_from_epoch = max_id+1
    else:
        print("Loading model checkpoint from {}".format(args.checkpoint_model_path))
        model.load_state_dict(torch.load(args.checkpoint_model_path, map_location=device))
        print("Loading optimizer checkpoint from {}".format(args.checkpoint_opt_path))
        optimizer.load_state_dict(
            torch.load(args.checkpoint_opt_path, map_location=device)
        )


def load_init_template_data(path):
    data_path_dict = scipy.io.loadmat(path)
    data_dict = {}

    data_dict['joint_tree'] = torch.Tensor(data_path_dict['joint_tree']).type(torch.IntTensor).cuda()
    data_dict['primitive_align'] = torch.Tensor(data_path_dict['primitive_align']).type(torch.IntTensor).cuda()
    data_dict['joint_parameter_leaf'] = torch.Tensor(data_path_dict['joint_parameter_leaf']).cuda()

    return data_dict

def mesh_to_mask_with_camera(mesh, image_size=(224, 224), camera_matrix=None, dist_coeffs=None, rvec=None, tvec=None):
    # 加载3D网格

    # 设置默认的相机参数
    if camera_matrix is None:
        camera_matrix = np.array([[image_size[0], 0, image_size[0] / 2],
                                  [0, image_size[1], image_size[1] / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))  # 无畸变

    if rvec is None:
        rvec = np.zeros((3, 1), dtype=np.float32)  # 没有旋转

    if tvec is None:
        tvec = np.zeros((3, 1), dtype=np.float32)  # 没有平移

    # 获取3D顶点
    vertices = mesh.vertices

    # 投影3D顶点到2D平面
    projected_points, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2).astype(np.int32)

    # 创建一个空的二维数组来存储掩码图像
    mask = np.zeros(image_size, dtype=np.uint8)

    # 将2D顶点转换为掩码图像
    for face in mesh.faces:
        pts = projected_points[face]
        # 使用OpenCV的fillPoly函数将多边形填充到掩码图像中
        cv2.fillPoly(mask, [pts], 255)

    # 调整掩码图像的大小到224x224
    mask_image = Image.fromarray(mask)
    mask_image = mask_image.resize(image_size, Image.ANTIALIAS)

    return mask_image


def test():
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d"%(args.gpu_id))
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Get the parameters and their ordering for the spreadsheet
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_tag))

    # Set log_dir for tensorboard
    log_dir = os.path.join(args.log_dir, experiment_tag)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    config = load_config(args.config_file)
    epochs = config["training"].get("epochs", 500)

    graphAE_param = Parameters()
    graphAE_param.read_config('./config/graphAE.config')

    # Create the network
    net = Network_pts(graphAE_param=graphAE_param, test_mode=False, model_type='dinov2_vits14', 
                      stride=4, device=device, vit_f_dim=args.vit_f_dim)
    net.cuda()


    # Build an optimizer object to compute the gradients of the parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.annealing_lr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(net, optimizer, experiment_directory, args, device)

    test_dataset = datasets.Datasets(data_path=args.data_path, train=False, image_size=args.res, target_path='./SQ_templates/laptop')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=1, drop_last=True)
    
    init_template_data_dict = load_init_template_data(args.datamat_path)
    print ('Dataloader finished!')

    renderer = Nvdiffrast(FOV=39.6)
    print ('Renderer set!')

    print ('Start Testing!')

    angle_list = ""

    for imgi, X in enumerate(test_dataloader):
        data_dict = X

        image_names = data_dict['image_name']
        rgb_image = data_dict['rgb'].cuda()
        o_image = data_dict['o_rgb'].cuda()
        # object_white_mask = data_dict['o_mask'].cuda()
        object_input_pts = data_dict['vs']
        object_input_fs = data_dict['fs']
        for i, _ in enumerate(object_input_pts):
            object_input_pts[i] = object_input_pts[i].cuda()
        init_object_old_center = data_dict['part_centers'].cuda()
        object_num_bones = 2
        object_joint_tree = init_template_data_dict['joint_tree'].cuda()
        object_primitive_align = init_template_data_dict['primitive_align'].cuda()
        object_joint_parameter_leaf = init_template_data_dict['joint_parameter_leaf'].cuda()
        _3d_info = data_dict['3d_info']

        pred_dict = net(rgb_image, o_image, object_input_pts, init_object_old_center, 
                    object_num_bones, object_joint_tree, object_primitive_align, object_joint_parameter_leaf)

        out_path = os.path.join(args.output_directory, experiment_tag)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path = os.path.join(out_path, 'visualize_results')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # TODO: visualze the predicted results
        # pred_vs = reformMeshData(pred_dict['deformed_object'])
        pred_vs = pred_dict['deformed_object']
        visualize_mesh(img_name=image_names, vertices = pred_vs, faces = object_input_fs, 
                        out_path = out_path)
        
        # Rendering 
        colors = sns.color_palette("Paired")

        images = []
        for bone_idx in range(object_num_bones):
            vi = pred_dict['deformed_object'][bone_idx] # (bs, num_points, 3)
            images.append(vi)
        
        m = None
        for bone_idx in range(object_num_bones):
            # out_path = os.path.join(filename, 'bone-' + str(bone_idx) + '.ply')
            vi = images[bone_idx].cpu().detach().data.numpy()
            vi = vi.squeeze(0)

            _m = _from_primitive_parms_to_mesh_v2(vi, (colors[bone_idx % len(colors)]) + (1.0,))
            m = trimesh.util.concatenate(_m, m)
            
        render_img = renderer(m).squeeze(0).detach().cpu().numpy()
        ori_img = rgb_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        render_img = cv2.cvtColor(render_img, cv2.COLOR_BGRA2BGR)

        mask = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        plt.imsave('test/mask.png', mask / 256)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2RGB)

        ori_img = cv2.bitwise_and(ori_img, mask_inv)
        img = cv2.add(ori_img, render_img)

        ori_shape = data_dict['shape'].detach().cpu().numpy().astype(int).reshape(3)
        #print(ori_shape)

        if ori_shape[0] < ori_shape[1]:
            tmp = int((ori_shape[1] - ori_shape[0])/2)
            img = cv2.resize(img, [ori_shape[1], ori_shape[1]])
            img = img[tmp:(ori_shape[1]-tmp), :]
        else:
            tmp = int((ori_shape[0] - ori_shape[1])/2)
            img = cv2.resize(img, [ori_shape[0], ori_shape[0]])
            img = img[:, tmp:(ori_shape[0]-tmp)]

        image_names = str(image_names).split('/')
        pred_angle = float(pred_dict['object_pred_angle_leaf'].detach().cpu().numpy())
        pred_angle = pred_angle / np.pi * 180 + 90

        tmp_angle_list = image_names[-3] + "_" + image_names[-1][:-6] + ": " + str(pred_angle) + "\n"
        #print(tmp_angle_list)
        angle_list = angle_list + tmp_angle_list


        
        # plt.imsave('test/rendered.png', render_img / 256)
        # plt.imsave('test/ori.png', ori_img / 256)
        plt.imsave("test/final/result_{0}_{1}.png".format(image_names[-3], image_names[-1][:-4]), img / 256)
        # exit(0)

        file = open("./test/angle.txt", 'w')
        file.write(str(angle_list))
        file.close()
        
        

    
    print ('To be finished...')


def train():
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d"%(args.gpu_id))
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Get the parameters and their ordering for the spreadsheet
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_tag))

    # Set log_dir for tensorboard
    log_dir = os.path.join(args.log_dir, experiment_tag)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    config = load_config(args.config_file)
    epochs = config["training"].get("epochs", 500)

    graphAE_param = Parameters()
    graphAE_param.read_config('./config/graphAE.config')

    # TODO: Create the network
    net = Network_pts(graphAE_param=graphAE_param, test_mode=False, model_type='dinov2_vits14', 
                      stride=4, device=device, vit_f_dim=args.vit_f_dim)
    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net)
    net.cuda()

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.annealing_lr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(net, optimizer, experiment_directory, args, device)

    # TODO: create the dataloader
    train_dataset = datasets.Datasets(data_path=args.data_path, train=True, image_size=args.res, target_path='./SQ_templates/laptop')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=1, drop_last=True)

    val_dataset = datasets.Datasets(data_path=args.data_path, train=False, image_size=args.res, target_path='./SQ_templates/laptop')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=1)
    
    init_template_data_dict = load_init_template_data(args.datamat_path)
    print ('Dataloader finished!')

    # TODO: create the differtiable renderer
    renderer = Nvdiffrast(FOV=39.6)
    print ('Renderer set!')

    print ('Start Training!')
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.continue_from_epoch, epochs):
        net.train()

        total_loss = 0.
        iter_num = 0.
        for _, X in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            data_dict = X

            # TODO: load all the data you need from dataloader, not limited
            image_names = data_dict['image_name']
            rgb_image = data_dict['rgb'].cuda()
            o_image = data_dict['o_rgb'].cuda()
            # object_white_mask = data_dict['o_mask'].cuda()
            object_input_pts = data_dict['vs']
            object_input_fs = data_dict['fs']
            for i, _ in enumerate(object_input_pts):
                object_input_pts[i] = object_input_pts[i].cuda()
            init_object_old_center = data_dict['part_centers'].cuda()
            object_num_bones = 2
            object_joint_tree = init_template_data_dict['joint_tree'].cuda()
            object_primitive_align = init_template_data_dict['primitive_align'].cuda()
            object_joint_parameter_leaf = init_template_data_dict['joint_parameter_leaf'].cuda()

            # TODO: pass the input data to the network and generate the predictions
            pred_dict = net(rgb_image, o_image, object_input_pts, init_object_old_center, 
                            object_num_bones, object_joint_tree, object_primitive_align, object_joint_parameter_leaf)
            
            # print(pred_dict['object_pred_angle_leaf'])
            # for name, param in net.named_parameters():
            #     if param.grad is None:
            #         print(f'No gradient for {name}')
            #     elif torch.all(param.grad == 0):
            #         print(f'Gradient is zero for {name}')
            
            # for name, param in net.named_parameters(): 
            #     print(name, param.requires_grad)

            # TODO: compute loss functions

            # camera_matrix = np.array([[983, 0, 112],
            #               [0, 983, 112],
            #               [0, 0, 1]], dtype=np.float32)
            # dist_coeffs = np.zeros((4, 1))

            colors = sns.color_palette("Paired")

            # mesh_vs = Reform(data_dict, init_template_data_dict, args.batch_size_train, object_num_bones)
            # mesh_fs = data_dict['fs']

            pred_seg_imgs = torch.empty(args.batch_size_train, 224, 224)
            images = [[] for i in range(args.batch_size_train)]
            for bone_idx in range(object_num_bones):
                vs = pred_dict['deformed_object'][bone_idx] # (bs, num_points, 3)
                for bsi in range(args.batch_size_train):
                    vi = vs[bsi] # (num_points, 3)
                    images[bsi].append(vi)

            for bsi in range(args.batch_size_train):
                image_curr = images[bsi]
                m = None
                for bone_idx in range(object_num_bones):
                    # out_path = os.path.join(filename, 'bone-' + str(bone_idx) + '.ply')
                    vi = image_curr[bone_idx].cpu().detach().data.numpy()

                    _m = _from_primitive_parms_to_mesh_v2(vi, (colors[bone_idx % len(colors)]) + (1.0,))
                    m = trimesh.util.concatenate(_m, m)
                    
                rendered_img = renderer(m).squeeze(0).detach().cpu().numpy()
                rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2GRAY)
                _, rendered_img = cv2.threshold(rendered_img, 125, 255, cv2.THRESH_BINARY)
                
                #print(np.shape(rendered_img))
                #pred_seg_imgs.append(torch.Tensor(rendered_img))
                pred_seg_imgs[bsi] = torch.Tensor(rendered_img/255)
                #print(pred_seg_imgs[bsi].size())
                #exit(0)
            #plt.imsave('test/test.png', pred_seg_imgs[5])
            #exit(0)

            loss = compute_loss(pred_dict, data_dict, init_template_data_dict, pred_seg_imgs, args.batch_size_train)

            # TODO: write the loss to tensorboard
            writer.add_scalar('train/loss', loss, epoch)

            total_loss += loss.item()

            iter_num += 1
            
            loss.backward()
            
            # for name, param in net.named_parameters(): 
            #       print(name, param.grad)

            if args.annealing_lr:
                scheduler.step()
            optimizer.step()

        total_loss = float(total_loss) / float(iter_num)
        print ('[Epoch %d/%d] Total_loss = %f.' % (epoch, epochs, total_loss))

        if epoch % args.save_every == 0:
            save_checkpoints(
                epoch,
                net,
                optimizer,
                experiment_directory,
                args
            )

        if epoch % args.val_every == 0:
            print("====> Validation Epoch ====>")
            net.eval()

            total_eval_loss = 0.
            iter_num = 0.
            for imgi, X in enumerate(val_dataloader):
                # TODO: load data and generate the predictions, loss
                iter_num += 1

                data_dict = X

                pred_dict = net(rgb_image, o_image, object_input_pts, init_object_old_center, 
                            object_num_bones, object_joint_tree, object_primitive_align, object_joint_parameter_leaf)

                if epoch % args.save_every == 0 or True:
                    out_path = os.path.join(args.output_directory, experiment_tag)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    out_path = os.path.join(out_path, 'visualiza_results_epoch_%d' % (epoch))
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    # TODO: visualze the predicted results
                    # pred_vs = reformMeshData(pred_dict['deformed_object'])
                    pred_vs = pred_dict['deformed_object']
                    visualize_mesh(img_name=image_names, vertices = pred_vs, faces = object_input_fs, 
                                   out_path = out_path)

            print("====> Validation Epoch ====>")


    print("Saved statistics in {}".format(experiment_tag))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        print ('Bad Mode!')
        os._exit(0)