import torch
from argparse import Namespace

from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from config.env_reader import ModelConfig
from token_reader import coco_tokens

caption_model_path = ModelConfig.caption_model_path


gpuDevice = "cuda" if torch.cuda.is_available() else "cpu"

print('creating model {}...'.format("ExpansionNet_v2"))

caption_model = torch.load(caption_model_path, map_location=gpuDevice)

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

model.load_state_dict(caption_model['model_state_dict'])

print("Model loaded ...")

loaded_model = model