import os
import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
import random
import json

from PIL import Image
from pytorch_lightning import seed_everything
from torchvision import transforms as pth_transforms
from src.cldm.model import create_model, load_state_dict
from src.cldm.ddim_hacked import DDIMSampler

from libs.decalib.deca import DECA
from libs.decalib.utils.config import cfg as deca_cfg
from libs.decalib.datasets import datasets as deca_dataset
from libs.face_parsing import FaceParser
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
from libs.decalib.datasets import detectors
from libs.decalib.datasets.similarity_transform.align_trans import warp_and_crop_face, reference_landmark

from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utilities.add_noise import add_noise_img
from Utilities.pose_estimation import pose_estimation_ext, get_pose_model
from annotation.utils import run_vl, run_fr, run_sd, get_deg_ref_files


def get_all_files(root, extension_list=['.png', '.jpg', '.jpeg'], sort=False):
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    if sort:
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        all_files = sorted(all_files, key=alphanum_key)
    return all_files


@torch.no_grad()
def get_id_feats(img_path_list, faceparser, mtcnn, facenet):
    print(f"---------------- Extracting ID Features -------------------")
    DICT_ID_IMG = {}
    transform = pth_transforms.Compose([
        pth_transforms.Resize((160, 160)),
        pth_transforms.ToTensor(),
    ])
    for img_path in tqdm(img_path_list):
        img = Image.open(img_path)
        img_name = os.path.splitext('/'.join(img_path.split('/')[-2:]))[0]
        DICT_ID_IMG[img_name] = {}
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            img_cropped = img_cropped.to(facenet.device)
        else:
            print('fail to detect faces')
            img_cropped = transform(img).to(facenet.device)
        clip_image, vis_parsing = faceparser.parse(img)
        img_embedding = facenet(img_cropped.unsqueeze(0)).unsqueeze(0).cpu()
        DICT_ID_IMG[img_name]['clip_image'] = clip_image
        DICT_ID_IMG[img_name]['id_feat'] = img_embedding
    return DICT_ID_IMG

@torch.no_grad()
def get_deca_feats(img_path_list, deca):
    print(f"---------------- Extracting DECA Features -------------------")
    DICT_DECA = {}
    dataset = deca_dataset.Dataset(img_path_list, iscrop=True, size=512, face_detector='retinaface', scale=1.25)
    for idx in tqdm(range(len(img_path_list))):
        img_path = img_path_list[idx]
        img_name = os.path.splitext('/'.join(img_path.split('/')[-2:]))[0]
        DICT_DECA[img_name] = {}
        img = dataset[idx]["image"].unsqueeze(0).to("cuda")
        code = deca.encode(img)
        tform = dataset[idx]["tform"].unsqueeze(0)
        tform = torch.inverse(tform).transpose(1, 2).to("cuda")
        code["tform"] = tform
        code_cpu = {k: v.cpu() for k, v in code.items()} 
        DICT_DECA[img_name]['deca'] = code_cpu
        DICT_DECA[img_name]['original_image'] = dataset[idx]["original_image"].unsqueeze(0)
    return DICT_DECA

@torch.no_grad()
def cond_generation(save_root, DICT_ID_IMG, DICT_ID_DECA, DICT_DECA_DEG_REF):
    print(f"---------------- Stage-2 Generation -------------------")
    num_samples=10
    deg_num = 100
    ref_num = 10
    image_resolution=512
    ddim_steps=20
    strength=1.0
    scale=7.0
    eta=0.0
    tau=0.0
    controlnet_strength=0.0
    shape = (4, image_resolution // 8, image_resolution // 8)
    model.pose_control_scales = ([controlnet_strength] * 13)
    model.control_scales = ([strength] * 13)
    model.drop_control_cond_t = tau

    probabilities = [0.8, 0.1, 0.08, 0.02]             
    ranges = [(0.82,1.6), (1.6,2.5), (2.5,4), (4,8)]
    os.makedirs(save_root, exist_ok=True)
    img_list, aug_list, low_list, blur_list = [], [], [], []

    model_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Pretrained-Models/Retinaface_Resnet50.pth'
    face_detector_r = detectors.FaceDetector(output_shape=(112, 112), model_path=model_path, square_bbox=True, fallback='pass')
    pose_model = get_pose_model()

    id_list = sorted(list(DICT_ID_IMG.keys()))
    deg_all_list = sorted(list(DICT_DECA_DEG_REF["DEG"].keys()))
    ref_list = sorted(list(DICT_DECA_DEG_REF["REF"].keys()))      

    prompt_all_temp = ["", "in the park", "in the dark", "in the street", "in the rain", "in the snow", "in the forest", "in the city", "in the mountains", "in the ocean", "in the desert", "in the garden", "in the field", "in the fog", "in the crowd", 
                       "in the mirror", "in the shadows", "in the spotlight", "in the mist", "in the sunrise", "in the sunset", "in the moonlight", "in the cafe", "in the library", "in the classroom", "in the office", "in the meadow", "in the alley", 
                       "in the hallway", "in the train", "in the car", "in the kitchen", "in the parkour", "in the basement", "in the attic", "in the attic", "in the backyard", "in the beach", "in the bedroom", "in the boardroom", "in the bathroom", 
                       "in the barn", "in the basement", "in the boardroom", "in the bathroom", "in the barn", "in the bedroom", "in the backyard", "in the beach", "in the club", "in the cottage", "in the cave", "in the concert", "in the cellar", 
                       "in the corner", "in the countryside", "in the club", "in the cottage", "in the cave", "in the concert", "in the cellar", "in the corner", "in the countryside", "in the diner", "in the driveway", "in the dungeon", 
                       "in the dance", "in the dojo", "in the desert", "in the elevator", "in the embassy", "in the forest", "in the fountain", "in the farm", "in the garage", "in the gym", "in the garden", "in the park", "in the darkroom", 
                       "in the hallway", "in the hotel", "in the hospital", "in the house", "in the inn", "in the jungle", "in the jail", "in the kitchen",
                       "wearing glasses", "wearing sunglasses", "with occluded mask"
                       ]
    n_prompt = f"deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime"

    for id_name in tqdm(id_list):
        deg_list = random.sample(deg_all_list, deg_num)
        save_dir_name = os.path.join(save_root, 'data', id_name.split('/')[-1])
        os.makedirs(save_dir_name, exist_ok=True)
        code_id = DICT_ID_DECA[id_name]['deca']
        original_image = DICT_ID_DECA[id_name]['original_image'].to('cuda')
        id_feature = DICT_ID_IMG[id_name]['id_feat'].cuda()
        id_feature_list = id_feature.expand(ref_num+deg_num, -1, -1) 
        clip_image = model.control_cond_stage_model.encode([DICT_ID_IMG[id_name]['clip_image']])
        clip_image_list = clip_image.expand(ref_num+deg_num, -1, -1)

        pos_ref_prompt_list = [f"a high-resolution clear frontal face image with natural expression under good lighting condition without any occlusion" for _ in range(deg_num)]
        pos_deg_prompt_list = [f"a photo of a human face {prompt_all_temp[random.randint(0, len(prompt_all_temp)-1)]}" for _ in range(deg_num)]
        pos_prompt_list = pos_ref_prompt_list + pos_deg_prompt_list

        c_crossattn_list = model.get_learned_conditioning(pos_prompt_list)
        c_crossattn_neg_list = model.get_learned_conditioning([n_prompt] * (ref_num+deg_num))
        control_list = torch.zeros(ref_num+deg_num, 9, 512, 512).cuda()
    
        for idx, deca_token in enumerate(ref_list):
            code = {}
            for k in code_id: code[k] = code_id[k].clone()
            code["pose"]  = torch.tensor(DICT_DECA_DEG_REF["REF"][deca_token]['deca']["pose"])
            code["light"] = torch.tensor(DICT_DECA_DEG_REF["REF"][deca_token]['deca']["light"])
            code["exp"]   = torch.tensor(DICT_DECA_DEG_REF["REF"][deca_token]['deca']["exp"])
            code["tform"] = torch.tensor(DICT_DECA_DEG_REF["REF"][deca_token]['deca']["tform"])
            code["cam"]   = torch.tensor(DICT_DECA_DEG_REF["REF"][deca_token]['deca']["cam"])
            code = {k: v.cuda() for k, v in code.items()}                   # CPU to CUDA
            opdict, _ = deca.decode(code, render_orig=True, original_image=original_image, tform=code["tform"])
            rendered = opdict["rendered_images"].detach()
            normal = opdict["normal_images"].detach()
            albedo = opdict["albedo_images"].detach()
            control = torch.cat([normal, albedo, rendered], dim=1)
            control_list[idx, :, :, :] = control

        for idx, deca_token in enumerate(deg_list):
            code = {}
            for k in code_id: code[k] = code_id[k].clone()
            code["pose"]  = torch.tensor(DICT_DECA_DEG_REF["DEG"][deca_token]['deca']["pose"])
            code["light"] = torch.tensor(DICT_DECA_DEG_REF["DEG"][deca_token]['deca']["light"])
            code["exp"]   = torch.tensor(DICT_DECA_DEG_REF["DEG"][deca_token]['deca']["exp"])
            code["tform"] = torch.tensor(DICT_DECA_DEG_REF["DEG"][deca_token]['deca']["tform"])
            code["cam"]   = torch.tensor(DICT_DECA_DEG_REF["DEG"][deca_token]['deca']["cam"])
            code = {k: v.cuda() for k, v in code.items()}                  
            opdict, _ = deca.decode(code, render_orig=True, original_image=original_image, tform=code["tform"])
            rendered = opdict["rendered_images"].detach()
            normal = opdict["normal_images"].detach()
            albedo = opdict["albedo_images"].detach()
            control = torch.cat([normal, albedo, rendered], dim=1)
            control_list[idx+ref_num, :, :, :] = control

        group_num = (ref_num+deg_num) // num_samples
        last_batch_size = (ref_num+deg_num) % num_samples
        tag_list = ["REF"]*ref_num + ["DEG"]*deg_num
    
        for iter_num in range(group_num+1):
            if iter_num != group_num:    
                batch_size = num_samples
                start_idx = iter_num * num_samples
            else:
                batch_size = last_batch_size
                start_idx = group_num * num_samples
                
            end_idx = start_idx + batch_size
            cond = {"c_concat": [control_list[start_idx:end_idx]], "c_crossattn": [c_crossattn_list[start_idx:end_idx]], "c_crossattn_control": [clip_image_list[start_idx:end_idx]], "c_crossattn_id": [id_feature_list[start_idx:end_idx]]}
            un_cond = {"c_concat": [control_list[start_idx:end_idx]], "c_crossattn": [c_crossattn_neg_list[start_idx:end_idx]], "c_crossattn_control": [clip_image_list[start_idx:end_idx]], "c_crossattn_id": [id_feature_list[start_idx:end_idx]]}

            model.low_vram_shift(is_diffusing=True)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, eta=eta, log_every_t=1, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
            model.low_vram_shift(is_diffusing=False)
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(batch_size)]
            for ii in range(len(results)):
                save_img_name = f"{save_dir_name}/Ref-{start_idx+ii}.jpg" if tag_list[start_idx+ii] == "REF" else f"{save_dir_name}/{start_idx+ii-ref_num}.jpg"
                resized_image = Image.fromarray(results[ii]).resize((224, 224))
                success, cropped, bbox, crop_landmark, score = face_detector_r.detect(resized_image)
                if success:
                    if tag_list[start_idx+ii] == "REF":
                        low_scale = 0.0
                        blur_level = 0.0
                        aligned_img, cv_tfm = warp_and_crop_face(cropped, crop_landmark, reference_landmark(), crop_size=(112, 112))
                        Image.fromarray(aligned_img).save(save_img_name)
                    else:
                        selected_range = random.choices(ranges, probabilities)[0]
                        low_scale = random.uniform(selected_range[0], selected_range[1])
                        selected_range = random.choices(ranges, probabilities)[0]
                        blur_level = random.uniform(selected_range[0], selected_range[1])
                        low_scale = 1.0 if low_scale<=1.0 else low_scale
                        blur_level = 1.0 if blur_level<=1.0 else blur_level
                        cropped_noise = add_noise_img(cropped, low_scale, blur_level, (112, 112))
                        aligned_n_img, cv_tfm = warp_and_crop_face(cropped_noise, crop_landmark, reference_landmark(), crop_size=(112, 112))
                        Image.fromarray(aligned_n_img).save(save_img_name)

                        aligned_img, cv_tfm = warp_and_crop_face(cropped, crop_landmark, reference_landmark(), crop_size=(112, 112))

                        img_list.append('/'.join(save_img_name.split('/')[-2:]))
                        aug_list.append(abs(pose_estimation_ext(Image.fromarray(aligned_img), pose_model)))
                        low_list.append(low_scale)
                        blur_list.append(blur_level)
                else:
                    print('Fail to detect faces')
    return img_list, aug_list, low_list, blur_list


def quality_labeling(save_root, img_list, aug_list, low_list, blur_list):
    print(f"---------------- Quality Annotation -------------------")
    deg_data_list, ref_data_list = get_deg_ref_files(f"{save_root}/data", sort=True)
    DICT_FR = run_fr(deg_data_list, ref_data_list)
    DICT_SD = run_sd(img_list, aug_list, low_list, blur_list)
    DICT_VL = run_vl(deg_data_list)
    lambda_p = 0.5
    outfile = open(f"{save_root}/Quality-Scores.txt", 'w')
    mr_score_list = []
    for i in img_list:
        fr_score = DICT_FR[i]
        sd_score = DICT_SD[i]
        vl_score = DICT_VL[i]
        mr_score_list.append((fr_score + lambda_p*sd_score) / (1 - vl_score))
    mr_score_list = np.asarray(mr_score_list)
    for idx, img_name in enumerate(DICT_FR.keys()):
        mr_score = (mr_score_list[idx] - np.min(mr_score_list)) / (np.max(mr_score_list) - np.min(mr_score_list))
        outfile.write(f"/{img_name}\t{mr_score}\n")
    print(f"==> Done | {outfile}")
    outfile.close()
    
def get_args():
    parser = argparse.ArgumentParser('Generation and Annotation')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    args.id_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Stage-1/samples'
    args.save_path = os.path.dirname(os.path.abspath(__file__)) + '/samples'
    
    args.deca_data =  os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Pretrained-Models/DECA/DICT_DECA_DEG_REF.json'
    args.model = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Stage-2/models/cldm.yaml'
    args.control_model = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Pretrained-Models/Stage2-Models/Control-Module.pth'
    args.sd_model = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Pretrained-Models/Stage2-Models/SDiff-Stage2.safetensors'
    args.vae_ckpt = None

    id_img_list = sorted(get_all_files(args.id_data_path))
    mtcnn = MTCNN(image_size=160, device='cuda')
    facenet = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()
    faceparser = FaceParser(save_pth=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Pretrained-Models/Stage2-Models/FaceParser.pth')
    DICT_ID_IMG = get_id_feats(id_img_list, faceparser, mtcnn, facenet)

    deca = DECA(config=deca_cfg)
    DICT_ID_DECA = get_deca_feats(id_img_list, deca)
    with open(args.deca_data, "r") as f: DICT_DECA_DEG_REF = json.load(f)

    model = create_model(args.model).cpu()
    missing_keys, unexpected_keys = model.load_state_dict(load_state_dict(f'{args.control_model}', location='cuda'), strict=False)
    print(f"==> Done loading model")

    if args.sd_model is not None:
        state_dict = load_state_dict(args.sd_model, location='cuda')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"==> Done loading StableDiffusion model")
    if args.vae_ckpt is not None:
        missing_keys, unexpected_keys = model.first_stage_model.load_state_dict(load_state_dict(f'{args.vae_ckpt}', location='cuda'), strict=False)
        print(f"==> Done loading VAE model")

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    img_list, aug_list, low_list, blur_list = cond_generation(args.save_path, DICT_ID_IMG, DICT_ID_DECA, DICT_DECA_DEG_REF)
    quality_labeling(args.save_path, img_list, aug_list, low_list, blur_list)
    