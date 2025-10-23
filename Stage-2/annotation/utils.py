import os
from PIL import Image
import torch
from torchvision.transforms.functional import InterpolationMode
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.blip_itm import blip_itm
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from scipy.stats import rankdata
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms as T


class Dataset(data.Dataset):
    def __init__(self, data_list, image_size=384):
        self.imgs = data_list
        self.transforms = T.Compose([
            T.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    def __getitem__(self, index):
        sample = self.imgs[index]
        imgPath = sample.strip()
        img = Image.open(imgPath).convert("RGB")
        data = self.transforms(img)
        return imgPath, data
    def __len__(self):
        return len(self.imgs)

def get_deg_ref_files(root, extension_list=['.png', '.jpg', '.jpeg'], sort=False):
    deg_files = list()
    ref_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        for i in filenames:
            if "Ref" in i: ref_files += [os.path.join(dirpath, i)]
            else: deg_files += [os.path.join(dirpath, i)]
    if extension_list is None:
        return deg_files, ref_files
    deg_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, deg_files))
    ref_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, ref_files))
    if sort:
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        deg_files = sorted(deg_files, key=alphanum_key)
        ref_files = sorted(ref_files, key=alphanum_key)
    return deg_files, ref_files

@torch.no_grad()
def get_id_feats(img_path_list, facenet, rd_type):
    dict_Emb = {}
    if rd_type=='REF':
        for img_path in img_path_list: dict_Emb[img_path.split('/')[-2]] = list()
        print(f"==> Extracting ref. embeddings")
    else: 
        print(f"==> Extracting deg. embeddings -------------------")
    transform = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
    ])
    for img_path in tqdm(img_path_list):
        img = Image.open(img_path)
        img_name = '/'.join(img_path.split('/')[-2:])
        img = transform(img).to(facenet.device)
        img_embedding = facenet(img.unsqueeze(0)).cpu()
        if rd_type=='REF': 
            dict_Emb[img_path.split('/')[-2]].append(img_embedding)
        else: dict_Emb[img_name] = img_embedding
    if rd_type=='REF': 
        for k in dict_Emb.keys(): 
            dict_Emb[k] = torch.stack(dict_Emb[k], dim=1).squeeze().mean(dim=0)
    return dict_Emb


@torch.no_grad()
def run_vl(deg_data_list):
    print(f"---------------- Computing vl Scores -------------------")
    data_dataset = Dataset(deg_data_list)
    dataloader = DataLoader(data_dataset, batch_size=100, shuffle=False, num_workers=8)
    blip_weights = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/Pretrained-Models/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=blip_weights, image_size=384, vit='base').to('cuda')
    model.eval()
    ref_capt = f"a high-resolution clear frontal face image with natural expression under good lighting condition without any occlusion" #facing forward
    DICT_VL = {}
    for datapath, data in tqdm(dataloader):
        data = data.to('cuda')
        itm_output = model(data, ref_capt, match_head='itm')
        itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1].squeeze().data.cpu().numpy()
        itc_score = model(data, ref_capt, match_head='itc').squeeze().data.cpu().numpy()
        for i in range(len(datapath)): 
            img_name = '/'.join(datapath[i].split('/')[-2:])
            DICT_VL[img_name] = itc_score[i]
    return DICT_VL


def run_fr(deg_data_list, ref_data_list):
    print(f"---------------- Computing fr Scores -------------------")
    facenet = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()
    dict_Emb_Ref = get_id_feats(ref_data_list, facenet, rd_type='REF')
    dict_Emb_Deg = get_id_feats(deg_data_list, facenet, rd_type='DEG')
    DICT_FR = {}
    for i in deg_data_list:
        img_name = '/'.join(i.split('/')[-2:])
        ref_id = i.split('/')[-2]
        emb_ref = dict_Emb_Ref[ref_id]
        emb_deg = dict_Emb_Deg[img_name]
        cos_sim = torch.nn.functional.cosine_similarity(emb_ref, emb_deg).item()
        DICT_FR[img_name] = cos_sim
    return DICT_FR


def run_sd(img_list, aug_list, low_list, blur_list):
    print(f"---------------- Computing sd Scores -------------------")
    DICT_SD = {}   
    aug_list_rank_score  = (len(img_list) - rankdata(aug_list, method='average')) / len(img_list)
    low_list_rank_score  = (len(img_list) - rankdata(low_list, method='average')) / len(img_list)
    blur_list_rank_score = (len(img_list) - rankdata(blur_list, method='average')) / len(img_list)
    for i in range(len(img_list)):
        DICT_SD[img_list[i]] = min(aug_list_rank_score[i], low_list_rank_score[i], blur_list_rank_score[i])
    return DICT_SD
