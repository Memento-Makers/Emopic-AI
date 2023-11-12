import torch
import numpy as np
import requests
from PIL import Image
from image_utils import preprocess_image
from model_loader import loaded_model,class_list


def predict(image_path:str) -> str:
    # 상수들
    image_size = 640
    top_k = 3
    th = 0.75
    # 이미지 전처리    
    im_resize = preprocess_image(image_path , image_size)
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half()  # float16 inference
    output = torch.squeeze(torch.sigmoid(loaded_model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    # 점수 높은 것 순으로 정렬
    idx_sort = np.argsort(-np_output)
    # class 분류 결과 
    detected_classes = np.array(class_list)[idx_sort]
    # top-k개만 뽑기
    top_k_classes = detected_classes[: top_k]
    # 점수가 th 보다 높은 것들만 남기기
    scores = np_output[idx_sort][: top_k]
    idx_th = scores > th
    top_k_classes = top_k_classes[idx_th]
    
    detected_classes = top_k_classes.tolist()
    
    return ','.join(detected_classes)