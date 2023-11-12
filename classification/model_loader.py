import torch
import numpy as np
from argparse import Namespace

from src_files.models import create_model
from src_files.models.tresnet.tresnet import InplacABN_to_ABN
from src_files.helper_functions.bn_fusion import fuse_bn_recursively

from config.env_reader import ModelConfig
model_args = Namespace(
    num_classes=80,
    model_path=ModelConfig.class_model_path,
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

class_model = torch.load(
    ModelConfig.class_model_path, map_location=lambda storage, loc: storage)
model = create_model(model_args, load_head=True).cuda()
model.load_state_dict(class_model['model'], strict=True)

########### eliminate BN for faster inference ###########
model = model.cpu()
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda().half().eval()
#######################################################

print('done')
loaded_model = model
class_list = np.array(
    list(class_model['idx_to_class'].values()))

