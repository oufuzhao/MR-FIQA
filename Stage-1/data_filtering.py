import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utilities.pose_estimation import pose_estimation
from Utilities.eyeglass_detector import detect_glasses_in_folder
from Utilities.emo_clsification import emotion_classification
from Utilities.comp_similarity import get_id_feats

def eliminate_samples(data_path):
    remove = True
    print(f"# Number of remaining samples after filtering with Yaw angle = {len(pose_estimation(data_path, remove=remove))}")
    print(f"# Number of remaining samples after filtering with glasses detection = {len(detect_glasses_in_folder(data_path, remove=remove))}")
    emo_filtering_list, face_region_dist = emotion_classification(data_path, remove=remove)
    print(f"# Number of remaining samples after filtering with emotion = {len(emo_filtering_list)}")
    final_list = get_id_feats(data_path, face_region_dist=face_region_dist, remove=remove)
    print(f"# Number of remaining samples after filtering with similarity: {len(final_list)}")
    return final_list

def rename_image(final_list):
    for idx, old_path in enumerate(final_list):
        new_path = os.path.join(os.path.dirname(old_path), f"{idx}.png")
        try: os.rename(old_path, new_path)
        except Exception as e: print(f"Failed to rename {old_path} to {new_path}: {e}")

if __name__ == "__main__":
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Stage-1/samples'
    final_list = eliminate_samples(data_path)
    rename_image(final_list)
