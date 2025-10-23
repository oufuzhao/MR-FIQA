'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda'
cfg.device_id = '0'
cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'deca_model.tar')
cfg.output_dir = ''
cfg.rasterizer_type = 'pytorch3d'
# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()

cfg.model.tex_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'FLAME_texture.npz')
cfg.model.topology_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'head_template.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'texture_data_256.npy')
cfg.model.fixed_displacement_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'fixed_displacement_256.npy')
cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'generic_model.pkl') 
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'landmark_embedding.npy') 
cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'uv_face_mask.png') 
cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'uv_face_eye_mask.png') 
cfg.model.mean_tex_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'mean_texture.jpg') 
cfg.model.tex_type = 'FLAME'
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.use_tex = True
cfg.model.jaw_type = 'aa'
cfg.model.fr_model_path = os.path.join(cfg.deca_dir, 'Pretrained-Models/DECA', 'resnet50_ft_weight.pkl')

## details
cfg.model.n_detail = 128
cfg.model.max_z = 0.01

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.batch_size = 10
cfg.dataset.K = 4
cfg.dataset.isSingle = False
cfg.dataset.num_workers = 4
cfg.dataset.image_size = 224
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.trans_scale = 0.

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default = 'train', help='deca mode')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.mode
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
