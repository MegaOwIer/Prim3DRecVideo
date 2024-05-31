'''
The main code for training the model
'''
import argparse
import json
import os
import random
import string

import numpy as np
import scipy.io
import torch
import torch.utils.data
import yaml
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import datasets
from model import Prim3DModel
from renderer_nvdiff import Nvdiffrast
from networks.baseline_network import Network_pts

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
parser.add_argument('--log_dir', type=str, default='/Disk3/siqi/NewPrimReg_outputs_aaai/baseline/logs')
parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--output_directory', type=str, default='../../NewPrimReg_outputs_iccv/baseline/output_dir')
parser.add_argument('--output_directory', type=str, default='/Disk3/siqi/NewPrimReg_outputs_aaai/baseline/output_dir')
parser.add_argument('--experiment_tag', type=str, default='laptop')
parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--vit_f_dim', type=int, default=3025) # dino
parser.add_argument('--vit_f_dim', type=int, default=384) # dinov2
# parser.add_argument('--res', type=int, default=112)
parser.add_argument('--res', type=int, default=224)
parser.add_argument('--batch_size_train', type=int, default=8, help='Batch size of the training dataloader.')
parser.add_argument('--batch_size_val', type=int, default=1, help='Batch size of the val dataloader.')
parser.add_argument('--data_path', type=str, default='/Disk3/siqi/Data/NewPrimReg_collected_matdata_syn/laptop.mat')
parser.add_argument('--save_every', type=int, default=200)
parser.add_argument('--val_every', type=int, default=200)
# parser.add_argument('--eval_images'

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
    # ! RAdam is available since torch 1.10 while the environment gives torch 1.9.x
    # ! temporarily disable these lines
    # elif optimizer == "RAdam":
    #     return OptimizerWrapper(
    #         torch.optim.RAdam(parameters, lr=lr, weight_decay=weight_decay),
    #         config.get("aggregate", 1)
    #     )
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

    joint_tree = np.load(data_path_dict['joint_tree'][0])
    data_dict['joint_tree'] = torch.Tensor(joint_tree).cuda()
    primitive_align = np.load(data_path_dict['primitive_align'][0])
    data_dict['primitive_align'] = torch.Tensor(primitive_align).cuda()
    joint_parameter_leaf = np.load(data_path_dict['joint_parameter_leaf'][0])
    data_dict['joint_parameter_leaf'] = torch.Tensor(joint_parameter_leaf).cuda()

    return data_dict


def test():
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

    # TODO: Create the network
    net = Network_pts(vit_f_dim=args.vit_f_dim, res=args.res)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.annealing_lr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(net, optimizer, experiment_directory, args, device)

    # TODO: create the dataloader
    train_dataset = datasets.Datasets(datamat_path=args.data_path, train=True, image_size=args.res, data_load_ratio=args.data_load_ratio)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=1, drop_last=True)
    val_dataset = datasets.Datasets(datamat_path=args.data_path, train=False, image_size=args.res, data_load_ratio=args.data_load_ratio)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=1)
    init_template_data_dict = load_init_template_data(args.data_path)
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
            data_dict = X

            # TODO: load all the data you need from dataloader, not limited
            # image_names = data_dict['image_name']
            rgb_image = data_dict['rgb'].cuda()
            o_image = data_dict['o_rgb'].cuda()
            # object_white_mask = data_dict['o_mask'].cuda()
            object_input_pts = data_dict['vs'].cuda()
            init_object_old_center = data_dict['part_centers'].cuda()
            object_num_bones = 2
            object_joint_tree = init_template_data_dict['joint_tree']
            object_primitive_align = init_template_data_dict['primitive_align']
            object_joint_parameter_leaf = init_template_data_dict['joint_parameter_leaf']

            # TODO: pass the input data to the network and generate the predictions
            pred_dict = net(rgb_image, o_image, object_input_pts, init_object_old_center, 
                            object_num_bones, object_joint_tree, object_primitive_align, object_joint_parameter_leaf)

            # TODO: compute loss functions
            loss = ...

            # TODO: write the loss to tensorboard
            writer.add_scalar('train/loss', loss, epoch)

            total_loss += loss.item()

            iter_num += 1
            optimizer.zero_grad()
            loss.backward()
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

                pred_dict = net(...)

                if epoch % args.save_every == 0:
                    out_path = os.path.join(args.output_directory, experiment_tag)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    out_path = os.path.join(out_path, 'visualiza_results_epoch_%d' % (epoch))
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    # TODO: visualze the predicted results

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