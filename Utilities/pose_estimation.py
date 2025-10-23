import math
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch.utils import data
from tqdm import tqdm


class SixDRepNet360(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet360, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = compute_rotation_matrix_from_ortho6d(x)

        return out

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1) #batch*3
    return out

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = singular.float()
        
    x = torch.atan2(R[:,2,1], R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0],R[:,0,0])
    
    xs = torch.atan2(-R[:,1,2], R[:,1,1])
    ys = torch.atan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0
        
    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cpu'))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cuda:%d' % gpu))
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
    
    return out_euler


class Dataset(data.Dataset):
    def __init__(self, data_path):
        self.imgs = []
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.lower().endswith(exts):
                    self.imgs.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        img = Image.open(imgPath)
        img = img.convert("RGB")
        img = self.transform(img)
        return imgPath, img

    def __len__(self):
        return len(self.imgs)

def get_pose_model():
    cudnn.enabled = True
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
    saved_state_dict = torch.load(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Pretrained-Models/Pose-LP.pth')
    if 'model_state_dict' in saved_state_dict: model.load_state_dict(saved_state_dict['model_state_dict'])
    else: model.load_state_dict(saved_state_dict)
    model.cuda()
    model.eval()
    return model

def pose_estimation(data_path, remove=False):
    model = get_pose_model()
    pose_dataset = Dataset(data_path)
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=80,
        num_workers=8,
        shuffle=False)
    total_imgs = len(pose_dataset)
    results_np = np.empty((total_imgs, 4), dtype=object)
    idx = 0
    print(f"---------------- Estimating Pose -------------------")    
    with torch.no_grad():
        for data_path_batch, images in tqdm(test_loader):
            images = torch.Tensor(images).cuda()
            R_pred = model(images)
            euler = compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu().numpy()
            y_pred_deg = euler[:, 1].cpu().numpy()
            r_pred_deg = euler[:, 2].cpu().numpy()
            batch_size = len(data_path_batch)
            results_np[idx:idx+batch_size, 0] = data_path_batch
            results_np[idx:idx+batch_size, 1] = p_pred_deg.astype(float)
            results_np[idx:idx+batch_size, 2] = y_pred_deg.astype(float)
            results_np[idx:idx+batch_size, 3] = r_pred_deg.astype(float)
            idx += batch_size
    results_np = eliminate_extreme_poses(results_np, yaw_thr=20, remove=remove)
    '''
    ----------- Correction and Clarification -----------
    It is worth noting that in the description of Section 3.2.1 of the paper, 
    it states "satisfy the criteria of an absolute yaw angle < 15", 
    but the correct version should be as shown in Fig. 2 where yaw ≤ 20. 
    I am very very very sorry for this typo mistake :(
    '''
    return results_np

def eliminate_extreme_poses(results_np, yaw_thr=20, remove=False):
    keep_mask = (np.abs(results_np[:, 2].astype(float)) < yaw_thr)
    for i in range(len(keep_mask)):
        if not keep_mask[i] and remove: os.remove(results_np[i, 0])
    results = results_np[keep_mask][:, 0]
    return results

@torch.no_grad()
def pose_estimation_ext(pil_img, model):
    img = pil_img.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()
    R_pred = model(img_tensor)
    euler = compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
    p_pred_deg = float(euler[0, 0].cpu().numpy())
    y_pred_deg = float(euler[0, 1].cpu().numpy())
    r_pred_deg = float(euler[0, 2].cpu().numpy())
    return y_pred_deg
