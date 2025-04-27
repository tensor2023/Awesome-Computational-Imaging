```python
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
import configargparse
from run_nerf import batchify, run_network, batchify_rays, render, render_path, create_nerf, raw2outputs, render_rays, config_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
```


```python
def train():

    parser = config_parser()
    # args = parser.parse_args()
    

    # parser = config_parser()
    args = parser.parse_args([
        '--config', '/home/xqgao/2025/MIT/code/NeRF/nerf-pytorch-master/configs/lego.yaml'
    ])

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    


    print('RENDER ONLY')
    with torch.no_grad():
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        return


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
```

    /home/xqgao/anaconda3/envs/inr/lib/python3.12/site-packages/torch/__init__.py:1236: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at /pytorch/torch/csrc/tensor/python_tensor.cpp:434.)
      _C._set_default_tensor_type(t)


    Loaded blender (138, 400, 400, 4) torch.Size([40, 4, 4]) [400, 400, np.float64(555.5555155968841)] /home/xqgao/2025/MIT/Datasets/NeRF/nerf_synthetic/lego
    Found ckpts ['null']
    Not ndc!
    RENDER ONLY
    test poses shape torch.Size([40, 4, 4])


      0%|          | 0/40 [00:00<?, ?it/s]

    0 0.004059314727783203


    /home/xqgao/anaconda3/envs/inr/lib/python3.12/site-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
      2%|▎         | 1/40 [00:06<04:05,  6.29s/it]

    torch.Size([400, 400, 3]) torch.Size([400, 400])
    1 6.288769245147705


      5%|▌         | 2/40 [00:12<03:53,  6.13s/it]

    2 6.023773431777954


      8%|▊         | 3/40 [00:18<03:45,  6.09s/it]

    3 6.03666353225708


     10%|█         | 4/40 [00:24<03:38,  6.08s/it]

    4 6.0545337200164795


     12%|█▎        | 5/40 [00:30<03:32,  6.07s/it]

    5 6.06751275062561


     15%|█▌        | 6/40 [00:36<03:26,  6.08s/it]

    6 6.082669258117676


     18%|█▊        | 7/40 [00:42<03:20,  6.09s/it]

    7 6.109835863113403


     20%|██        | 8/40 [00:48<03:15,  6.10s/it]

    8 6.124653100967407


     22%|██▎       | 9/40 [00:54<03:09,  6.11s/it]

    9 6.136879205703735


     25%|██▌       | 10/40 [01:01<03:03,  6.12s/it]

    10 6.139219045639038


     28%|██▊       | 11/40 [01:07<02:57,  6.13s/it]

    11 6.154747009277344


     30%|███       | 12/40 [01:13<02:51,  6.14s/it]

    12 6.1579673290252686


     32%|███▎      | 13/40 [01:19<02:45,  6.15s/it]

    13 6.162424802780151


     35%|███▌      | 14/40 [01:25<02:39,  6.15s/it]

    14 6.16940450668335


     38%|███▊      | 15/40 [01:31<02:34,  6.16s/it]

    15 6.181149005889893


     40%|████      | 16/40 [01:38<02:27,  6.16s/it]

    16 6.1707072257995605


     42%|████▎     | 17/40 [01:44<02:21,  6.17s/it]

    17 6.168968677520752


     45%|████▌     | 18/40 [01:50<02:15,  6.17s/it]

    18 6.1801393032073975


     48%|████▊     | 19/40 [01:56<02:09,  6.17s/it]

    19 6.179092645645142


     50%|█████     | 20/40 [02:02<02:03,  6.17s/it]

    20 6.178071975708008


     52%|█████▎    | 21/40 [02:08<01:57,  6.18s/it]

    21 6.184280157089233


     55%|█████▌    | 22/40 [02:15<01:51,  6.18s/it]

    22 6.181585311889648


     57%|█████▊    | 23/40 [02:21<01:45,  6.18s/it]

    23 6.191661357879639


     60%|██████    | 24/40 [02:27<01:38,  6.19s/it]

    24 6.196957349777222


     62%|██████▎   | 25/40 [02:33<01:32,  6.19s/it]

    25 6.18497109413147


     65%|██████▌   | 26/40 [02:39<01:26,  6.19s/it]

    26 6.186652660369873


     68%|██████▊   | 27/40 [02:46<01:20,  6.18s/it]

    27 6.180799722671509


     70%|███████   | 28/40 [02:52<01:14,  6.19s/it]

    28 6.185983896255493


     72%|███████▎  | 29/40 [02:58<01:08,  6.18s/it]

    29 6.177866458892822


     75%|███████▌  | 30/40 [03:04<01:01,  6.18s/it]

    30 6.167341470718384


     78%|███████▊  | 31/40 [03:10<00:55,  6.18s/it]

    31 6.167719602584839


     80%|████████  | 32/40 [03:16<00:49,  6.17s/it]

    32 6.165492296218872


     82%|████████▎ | 33/40 [03:23<00:43,  6.17s/it]

    33 6.165848970413208


     85%|████████▌ | 34/40 [03:29<00:37,  6.17s/it]

    34 6.16896390914917


     88%|████████▊ | 35/40 [03:35<00:30,  6.16s/it]

    35 6.150846719741821


     90%|█████████ | 36/40 [03:41<00:24,  6.16s/it]

    36 6.150238037109375


     92%|█████████▎| 37/40 [03:47<00:18,  6.16s/it]

    37 6.1558897495269775


     95%|█████████▌| 38/40 [03:53<00:12,  6.16s/it]

    38 6.153788805007935


     98%|█████████▊| 39/40 [04:00<00:06,  6.15s/it]

    39 6.148360729217529


    100%|██████████| 40/40 [04:06<00:00,  6.15s/it]


    Done rendering /home/xqgao/2025/MIT/code/NeRF/nerf-pytorch-master/logs/blender_paper_lego/renderonly_path_000000

