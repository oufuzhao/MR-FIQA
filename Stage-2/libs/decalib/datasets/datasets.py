import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from skimage.transform import estimate_transform, warp, resize, rescale
from PIL import Image
from . import detectors
# from .similarity_transform.align_trans import warp_and_crop_face, reference_landmark

class Dataset(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', sample_step=10, size=256):
        self.imagepath_list = testpath
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.size = size
        if face_detector == 'fan': self.face_detector = detectors.FAN()
        elif face_detector == 'retinaface':
            model_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + '/Pretrained-Models/Retinaface_Resnet50.pth'
            self.face_detector_r = detectors.FaceDetector(
                output_shape=(self.crop_size, self.crop_size),
                model_path=model_path,
                square_bbox=True,
                fallback='pass')
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        rgb_image = Image.open(imagepath).convert('RGB')
        rgb_image = rgb_image.resize((self.size, self.size))
        h, w = rgb_image.size
        bbox = []
        if self.iscrop:
            bbox_type = 'bbox'
            try: success, cropped, bbox, crop_landmark, score = self.face_detector_r.detect(rgb_image)
            except: print(f"ERROR for {imagename}")
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0; right = h-1; top=0; bottom=w-1
            else:
                left = bbox[0]; right=bbox[2]
                top = bbox[1]; bottom=bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        image = np.array(rgb_image)/255
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }