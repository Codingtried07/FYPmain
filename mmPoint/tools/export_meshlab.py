import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
import numpy as np
import torch
import random
import importlib
import munch
import yaml
import argparse
from utils.model_utils import *

warnings.filterwarnings("ignore")


def write_ply(filepath, points):
    """Write a point cloud to PLY format for MeshLab."""
    n = points.shape[0]
    with open(filepath, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for i in range(n):
            f.write(f'{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n')


def predict_one_pc(input_radar_path, load_model, args):
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.mmPoint', 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()

    ckpt = torch.load(load_model)
    net.module.load_state_dict(ckpt['net_state_dict'])
    net.module.eval()

    radar = torch.Tensor(np.load(input_radar_path, allow_pickle=True))
    radar = radar.unsqueeze(0).cuda()

    template_name = os.path.join(os.path.dirname(__file__), '..', 'human_template', 'human_template_256.xyz')
    template_points = np.loadtxt(template_name)
    template_points = pc_normalize(template_points, 0.5)

    _, _, deformed_points_3, _, _, _ = net(radar, template_points)
    return deformed_points_3.squeeze().cpu().detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='../cfgs/mmPoint.yaml')
    parser.add_argument('-gpu', '--gpu_id', default=0)
    parser.add_argument('--input', default=None, help='Single .npy file to predict')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of frames to export')
    arg = parser.parse_args()

    args = munch.munchify(yaml.safe_load(open(arg.config)))

    model_path = os.path.join(os.path.dirname(__file__), '..', 'S-PMP-output', 'mmPoint-0315_1102', 'checkpoints', '50network.pth')
    hdf5_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'hdf5')
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'meshlab_output')
    os.makedirs(save_dir, exist_ok=True)

    if arg.input:
        # Single file mode
        pc = predict_one_pc(arg.input, model_path, args)
        out_path = os.path.join(save_dir, 'output.ply')
        write_ply(out_path, pc)
        print(f'Saved: {out_path}')
    else:
        # Batch mode — export first N frames
        npy_files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith('.npy')])
        npy_files = npy_files[:arg.num_frames]

        for i, npy_file in enumerate(npy_files):
            input_path = os.path.join(hdf5_dir, npy_file)
            pc = predict_one_pc(input_path, model_path, args)

            name = os.path.splitext(npy_file)[0]
            ply_path = os.path.join(save_dir, f'{name}.ply')
            write_ply(ply_path, pc)
            print(f'[{i+1}/{len(npy_files)}] Saved: {ply_path}')

    print(f'\nDone')
