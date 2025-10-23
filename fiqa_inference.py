from PIL import Image
from torchvision import transforms as T
from Utilities.cr_model import iresnet50
import torch

def read_img(imgPath): 
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(imgPath).convert("RGB")
    data = transform(img).unsqueeze(0)
    return data


def network(eval_model, device):
    fix_qs = True if 'SynFIQA++' in eval_model else False
    fiqa_net = iresnet50(dropout=0.4, num_features=512, use_se=False, qs=1, fix_qs=fix_qs).to(device)
    net_dict = fiqa_net.state_dict()     
    data_dict = {key.replace('module.', ''): value for key, value in torch.load(eval_model, map_location=device).items()}
    net_dict.update(data_dict)
    fiqa_net.load_state_dict(net_dict)
    fiqa_net.eval()
    return fiqa_net

@torch.no_grad()
def pred_score(img_path, model, device):
    tensor_data = read_img(img_path).to(device)
    _, pred_score = model(tensor_data)
    pred_score = pred_score.cpu().numpy().squeeze(0)[0]
    return pred_score

if __name__ == "__main__":    
    
    img_path = "Samples/1.jpg"
    model_path = "Pretrained-Models/Syn-FIQA-Models/xxx.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network(model_path, device)
    
    pred_score = pred_score(img_path, model, device)
    print(f"==> {img_path} | Quality = {pred_score}")