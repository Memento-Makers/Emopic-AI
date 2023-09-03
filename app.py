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
from PIL.ExifTags import TAGS
sys.path.append("")
sys.path.append("")

from argparse import Namespace
# classification
from src_files.models.tresnet.tresnet import InplacABN_to_ABN
from src_files.models import create_model
from src_files.helper_functions.bn_fusion import fuse_bn_recursively
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

captionModelPath = ''
classificationModelPath = ''

with open('', 'rb') as f:
    coco_tokens = pickle.load(f)
    sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
    eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

class caption_in(BaseModel):
    url: str = ""


class classification_in(BaseModel):
    pic_path: str = ""


def loadClassificationModel():
    model_args = Namespace(
        num_classes=9605,
        model_path=classificationModelPath,
        model_name='tresnet_m',
        image_size=448,
        dataset_type='MS-COCO',
        th=0.97,
        top_k=3,
        use_ml_decoder=1,
        num_of_groups=200,
        decoder_embedding=768,
        zsl=0
    )
    print('creating model {}...'.format(model_args.model_name))

    classificationModel = torch.load(
        classificationModelPath, map_location='cpu')
    model = create_model(model_args, load_head=True).cuda()
    model.load_state_dict(classificationModel['model'], strict=True)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('done')
    classList = np.array(
        list(classificationModel['idx_to_class'].values()))

    return model, classList


def loadCaptionModel():
    global coco_tokens
    gpuDevice = "cuda" if torch.cuda.is_available() else "cpu"
    print('creating model {}...'.format(model_args.model_name))
    captionModel = torch.load(captionModelPath, map_location=gpuDevice)

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


def get_snapped_at(im):
    img_info = im.getexif()

    if img_info:
        for tag_id in img_info:
            tag = TAGS.get(tag_id, tag_id)
            data = img_info.get(tag_id)
            if tag == 'DateTime' or tag == 'DateTimeOriginal':
                dateTime = data
                break
    else:
        dateTime = ""

    return dateTime


classificationModel, classList = loadClassificationModel()
captionModel = loadCaptionModel()


app = FastAPI()


@app.get("/")
def root():
    return {"hello root"}


@app.post("/classification")
def classification(item: classification_in):
    global classificationModel, classList
    args = argparse.Namespace(
        url=item.pic_path
    )
    result = {}
  
    image_size = 448
    top_k = 3
    th = 0.97

    im = Image.open(requests.get(args.url, stream=True).raw)
    im_resize = im.resize((image_size, image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(
        2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(
        tensor_img, 0).cuda().half()  # float16 inference
    output = torch.squeeze(torch.sigmoid(classificationModel(tensor_batch)))
    np_output = output.cpu().detach().numpy()

    # Top-k predictions
    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classList)[
        idx_sort][: top_k]
    scores = np_output[idx_sort][: top_k]
    idx_th = scores > th
    detected_classes = detected_classes[idx_th]
    print('done\n')

    print(("detected classes: {}".format(detected_classes)))

    print(type(detected_classes))
    print('done\n')
    detected_classes = detected_classes.tolist()
    print(type(detected_classes))

    result["categories"] = detected_classes
    result["dateTime"] = get_snapped_at(im)

    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


# 임시로 반환형식 HTML Response
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
    uvicorn.run(app, port=8000)
