import argparse
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

from argparse import Namespace
sys.path.append("/app/src")

# config
import myenv as my

# classification
from src_files.models.tresnet.tresnet import InplacABN_to_ABN
from src_files.models import create_model
from src_files.helper_functions.bn_fusion import fuse_bn_recursively

# fast api
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder


from typing import List, Optional
from pydantic import BaseModel

coco_classificationModelPath = my.get_model_path()

"""
post /classification 에서 이미지 경로를 받기위해 사용하는 class

"""
class classification_in(BaseModel):
    pic_path: str = ""

"""


"""
def loadClassificationModel():
    model_args = Namespace(
        num_classes=80,
        model_path=coco_classificationModelPath,
        model_name='tresnet_XL',
        image_size=640,
        dataset_type='MS-COCO',
        th=0.75,
        top_k=3,
        use_ml_decoder=1,
        num_of_groups=80,
        decoder_embedding=768,
        zsl=0
    )
    print('creating model {}...'.format(model_args.model_name))

    classificationModel = torch.load(
        coco_classificationModelPath, map_location=lambda storage, loc: storage)
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


app = FastAPI()
"""
서버가 정상적으로 동작하는지 확인하는 용도
"""
@app.get("/")
def root():
    return {"hello root"}

classificationModel, classList = loadClassificationModel()

"""
1. request에서 받은 이미지 경로로 이미지를 다운받는다. 
2. 모델에 적합한 사이즈와 형태로 이미지를 변경시킨다.
3. 정확도가 75% 이상인 것들 중 제일 높은 정확도를 가진 3개의 class를 뽑아낸다. 
4. 뽑아낸 결과를 반환한다.
"""
@app.post("/classification")
def classification(item: classification_in):
    global classificationModel, classList
    args = argparse.Namespace(
        url=item.pic_path
    )
    result = {}

    image_size = 640
    top_k = 3
    th = 0.75

    im = Image.open(requests.get(args.url, stream=True).raw)
    im_resize = im.resize((image_size, image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(
        2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(
        tensor_img, 0).cuda().half()  # float16 inference
    output = torch.squeeze(torch.sigmoid(classificationModel(tensor_batch)))
    np_output = output.cpu().detach().numpy()

    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classList)[idx_sort]
    top_k_classes = detected_classes[: top_k]

    scores = np_output[idx_sort][: top_k]
    idx_th = scores > th
    top_k_classes = top_k_classes[idx_th]
    print('done\n')

    print(("top-k classes: {}".format(top_k_classes)))

    print(type(top_k_classes))
    print('done\n')
    detected_classes = top_k_classes.tolist()
    print(type(detected_classes))
    if len(detected_classes) == 0:
        detected_classes = ["default"]
    result["categories"] = detected_classes

    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, port=my.get_port(), host=my.get_host())
