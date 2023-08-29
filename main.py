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
sys.path.append("/content/drive/MyDrive/ExpansionNet_v2")
sys.path.append("/content/drive/MyDrive/ML_Decoder")
from argparse import Namespace
# ML_Decoder
from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
from src_files.models.tresnet.tresnet import InplacABN_to_ABN
from PIL import Image

# ExpansionNet_v2 
from utils.image_utils import preprocess_image
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.language_utils import tokens2description
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from typing import List, Optional
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

class caption_in(BaseModel):
    url: str = ""

class classification_in(BaseModel):
     num_classes : int = 9605
     model_path : str = '/content/drive/MyDrive/ML_Decoder/models/tresnet_m_open_images_200_groups_86_8.pth'
     pic_path : str = ""
     model_name : str = 'tresnet_m'
     image_size : int = 448
     dataset_type : str = 'MS-COCO'
     th : float = 0.97
     top_k : float = 3
     use_ml_decoder : int = 1
     num_of_groups : int = 200
     decoder_embedding : int = 768
     zsl : int = 0

app = FastAPI()

@app.get("/")
def root():
    return {"hello root"}

@app.post("/classification")
def classification(item:classification_in):
    
    print("hi")
    args = argparse.Namespace(
      num_classes = item.num_classes,
      model_path = item.model_path,
      pic_path = item.pic_path,
      model_name = item.model_name,
      image_size = item.image_size,
      dataset_type = item.dataset_type,
      th = item.th,
      top_k = item.top_k,
      use_ml_decoder = item.use_ml_decoder,
      num_of_groups = item.num_of_groups,
      decoder_embedding = item.decoder_embedding,
      zsl = item.zsl,
    )
    
    print('Inference code on a single image')

    # parsing args
    #args = parser.parse_args()

    print(args)
    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args, load_head=True).cuda()
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('done')


    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    result = {}
    
    # doing inference
    print('loading image and doing inference...')
    #im = Image.open(args.pic_path)
    im = Image.open(requests.get(args.pic_path, stream=True).raw)
    im_resize = im.resize((args.image_size, args.image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()


    ## Top-k predictions
    # detected_classes = classes_list[np_output > args.th]
    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
    scores = np_output[idx_sort][: args.top_k]
    idx_th = scores > args.th
    detected_classes = detected_classes[idx_th]
    print('done\n')

    # displaying image
    print('showing image on screen...')
    fig = plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    # plt.rcParams["axes.titlesize"] = 10
    plt.title("detected classes: {}".format(detected_classes))

    plt.show()
    print(("detected classes: {}".format(detected_classes)))

    print(type(detected_classes))
    print('done\n')
    detected_classes = detected_classes.tolist()
    print(type(detected_classes))

    result["categories"] = detected_classes

    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


#임시로 반환형식 HTML Response
@app.post("/captioning")
def captioning(item:caption_in):
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

    with open('./ExpansionNet_v2/demo_material/demo_coco_tokens.pickle', 'rb') as f:
            coco_tokens = pickle.load(f)
            sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
            eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load('./ExpansionNet_v2/checkpoint/rf_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded ...")
    #image_paths = ["./ExpansionNet_v2/demo_material/cat_girl.jpg"]
    #image_paths = [
    #     "https://storage.googleapis.com/ssh-9753/cat_girl.jpg",
    #     "https://storage.googleapis.com/ssh-9753/KakaoTalk_20230802_125042660.jpg",
    #    ]
    image_path = item.url
    input_image = preprocess_image(image_path, img_size)
    

    print("Generating captions ...\n")
    result = {}
    
    path = image_path
    image = input_image
    beam_search_kwargs = {'beam_size': 5,
                          'beam_max_seq_len': 74,
                          'sample_or_max': 'max',
                          'how_many_outputs': 1,
                          'sos_idx': sos_idx,
                          'eos_idx': eos_idx}
    with torch.no_grad():
        pred, _ = model(enc_x=image,
                        enc_x_num_pads=[0],
                        mode='beam_search', **beam_search_kwargs)
    pred = tokens2description(pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)

    result["caption"] = pred
    
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)



if __name__ == "__main__":
        import nest_asyncio
        from pyngrok import ngrok
        import uvicorn

        ngrok_tunnel = ngrok.connect(8000)
        print('Public URL:', ngrok_tunnel.public_url)
        nest_asyncio.apply()
        uvicorn.run(app, port=8000)
        #uvicorn.run(
        #    app,
        #    host="127.0.0.1",
        #    port=8000,
        #)
