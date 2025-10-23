from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch
from .fr_irse_net import build_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compcos(feats1, feats2):
    cos = np.dot(feats1,feats2)/(np.linalg.norm(feats1)*np.linalg.norm(feats2))
    return cos

def backboneSet(device):
    model_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Pretrained-Models/AdaFace_IR18_Web4m.pth'
    net = build_model('ir_18')
    net_dict = net.state_dict()     
    eval_dict = torch.load(model_path, map_location=device)
    eval_dict = {k.replace('module.', ''): v for k, v in eval_dict.items()}
    same_dict =  {k: v for k, v in eval_dict.items() if k in net_dict}
    net.load_state_dict(same_dict)
    net.to(device)
    net.eval()
    return net

def get_id_feats(data_path, face_region_dist=None, remove=False):
    fr_net = backboneSet('cuda')
    imgs_list = []
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(exts):
                imgs_list.append(os.path.join(root, f))
    print(f"---------------- Extracting ID Features -------------------")

    imgs_list = imgs_list
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, img_paths):
            self.img_paths = img_paths
            self.target_size = 112
            self.face_region_dist = face_region_dist if face_region_dist is not None else {}
        def __len__(self):
            return len(self.img_paths)
        def __getitem__(self, idx):
            path = self.img_paths[idx]
            try:
                img = Image.open(path).convert('RGB')
                if path in list(self.face_region_dist.keys()):
                    info = self.face_region_dist[path]
                    x, y, w, h = info['x'], info['y'], info['w'], info['h']
                    squz_dist = int(abs(h-w)/2)
                    img_crop = img.crop((x-squz_dist, y, x + w +squz_dist, y + h))
                    img_crop = img_crop.resize((self.target_size, self.target_size), Image.BILINEAR)
                    np_img = np.array(img_crop)
                else:
                    img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
                    np_img = np.array(img)
                brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
                tensor = torch.tensor(brg_img.transpose(2,0,1)).float()
                return tensor, path
            except Exception as e:
                return None, path

    batch_size = 120
    dataset = Dataset(imgs_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    dict_feats = {}
    with torch.no_grad():
        for tensors, paths in tqdm(dataloader, desc="Batch Extract"):
            tensors = tensors.cuda()
            valid_idx = [i for i, t in enumerate(tensors) if t is not None]
            if len(valid_idx) == 0: continue
            valid_tensors = torch.stack([tensors[i] for i in valid_idx]).cuda()
            valid_paths = [paths[i] for i in valid_idx]
            embeddings, _ = fr_net(valid_tensors)
            embeddings = embeddings.cpu().numpy()
            for p, emb in zip(valid_paths, embeddings):
                dict_feats[p] = emb

    unique_paths = filter_unique_samples(dict_feats, threshold=0.3, remove=remove)
    return unique_paths


def filter_unique_samples(dict_feats, threshold=0.3, remove=False):
    img_paths = list(dict_feats.keys())
    feats = np.array([dict_feats[p][:] for p in img_paths])
    print(f"---------------- Eliminating Samples -------------------")
    N = feats.shape[0]
    norm_feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    keep_mask = np.ones(N, dtype=bool)
    for i in tqdm(range(N), desc='Filtering'):
        if not keep_mask[i]: continue
        idx_rest = np.where(keep_mask)[0]
        idx_rest = idx_rest[idx_rest > i]
        if len(idx_rest) == 0: continue
        sims = np.dot(norm_feats[i], norm_feats[idx_rest].T)
        remove_idx = idx_rest[sims > threshold]
        if remove: 
            for idx in remove_idx: os.remove(img_paths[idx])
        keep_mask[remove_idx] = False
    return [img_paths[i] for i in range(N) if keep_mask[i]]
