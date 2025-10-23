from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def analyze_emotion(img_path):
    try:
        analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'], detector_backend="retinaface", enforce_detection=False, silent=True)
        confid_neutral = analysis[0]['emotion']['neutral']
    # return img_path if confid_neutral > 10 else None
        face_region = analysis[0]['region']
        r_res = True if confid_neutral > 10 else None
        return (img_path, r_res, face_region)
    except Exception as e:
        print(f"DeepFace error processing {img_path}: {e}")
        return (img_path, None, None)


def emotion_classification(data_path, remove=False):
    imgs_list, results = [], []
    face_region_dist = {}
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(exts):
                imgs_list.append(os.path.join(root, f))
    print(f"---------------- Analyzing emotion -------------------")
    # imgs_list.reverse()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for (img, res, face_region) in tqdm(executor.map(analyze_emotion, imgs_list), total=len(imgs_list)):
            if res:
                face_region_dist[img] = face_region
                results.append(res)
            elif remove and res is None:
                os.remove(img)
                # except: pass
    return results, face_region_dist
