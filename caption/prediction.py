import torch

from model_loader import loaded_model
from token_reader import beam_search_kwargs,sos_idx,eos_idx,coco_tokens

from utils.language_utils import tokens2description
from utils.image_utils import preprocess_image

def predict(image_path:str) -> str:
    # 이미지 전처리    
    img_size = 384
    image = preprocess_image(image_path, img_size)
    
    print("Generating captions ...\n")
   
    with torch.no_grad():
        pred, _ = loaded_model(enc_x=image,
                               enc_x_num_pads=[0],
                               mode='beam_search', **beam_search_kwargs)
    pred = tokens2description(
        pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)

    return pred