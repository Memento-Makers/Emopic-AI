import pickle
from config.env_reader import ModelConfig

with open(ModelConfig.caption_token_path, 'rb') as f:
    coco_tokens = pickle.load(f)
    sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
    eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

beam_search_kwargs={'beam_size': 5,
                    'beam_max_seq_len': 74,
                    'sample_or_max': 'max',
                    'how_many_outputs': 1,
                    'sos_idx': sos_idx,
                    'eos_idx': eos_idx}