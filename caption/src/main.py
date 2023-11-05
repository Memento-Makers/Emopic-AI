import os
import time
import argparse
import pickle
import sys
import uvicorn
import requests
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import sys
import matplotlib.pyplot as plt
import numpy as np
import nest_asyncio
from PIL import Image

from argparse import Namespace
sys.path.append("/app/src")
# config
import myenv as my

# captioning
from utils.language_utils import tokens2description
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image

# fast api
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from typing import List, Optional
from pydantic import BaseModel

caption_model_path = my.get_model_path()

with open(my.get_token_path(), 'rb') as f:
    coco_tokens = pickle.load(f)
    sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
    eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

"""
post /captioning 에서 이미지 경로를 받기위해 사용하는 class

"""

class caption_in(BaseModel):
    url: str = ""

def loadCaptionModel():
    global coco_tokens
    gpuDevice = "cuda" if torch.cuda.is_available() else "cpu"

    print('creating model {}...'.format("ExpansionNet_v2"))

    captionModel = torch.load(caption_model_path, map_location=gpuDevice)

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)

    model_args = Namespace(model_dim=512,
                           N_enc=3,
                           N_dec=3,
                           dropout=0.0,
                           drop_args=drop_args)
    img_size = 384

    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=74, drop_args=model_args.drop_args,
                                rank='cpu')

    model.load_state_dict(captionModel['model_state_dict'])

    print("Model loaded ...")

    return model


app = FastAPI()

"""
서버가 정상적으로 동작하는지 확인하는 용도
"""
@app.get("/")
def root():
    return {"hello root"}

captionModel = loadCaptionModel()

"""
1. request에서 받은 이미지 경로로 이미지를 다운받는다. 
2. 모델에 적합한 사이즈와 형태로 이미지를 변경시킨다.
3. 모델에 이미지를 넣고 결과를 추출한다.
4. 뽑아낸 결과를 반환한다.
"""
@app.post("/captioning")
def captioning(item: caption_in):
    global captionModel, coco_tokens, sos_idx, eos_idx

    img_size = 384
    image_path = item.url
    input_image = preprocess_image(image_path, img_size)

    print("Generating captions ...\n")
    result = {}

    image = input_image
    beam_search_kwargs = {'beam_size': 5,
                          'beam_max_seq_len': 74,
                          'sample_or_max': 'max',
                          'how_many_outputs': 1,
                          'sos_idx': sos_idx,
                          'eos_idx': eos_idx}
    with torch.no_grad():
        pred, _ = captionModel(enc_x=image,
                               enc_x_num_pads=[0],
                               mode='beam_search', **beam_search_kwargs)
    pred = tokens2description(
        pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)

    result["caption"] = pred

    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, port=my.get_port(), host=my.get_host())
