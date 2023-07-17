import argparse
import enum
import json
import struct
from pathlib import Path
import concurrent
# from types import NoneType
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
import torch
import numpy as np

import os
import sys
minigpt4_path = os.path.join(os.path.dirname(__file__), "MiniGPT-4")
sys.path.insert(0, minigpt4_path)
from minigpt4.models.blip2 import Blip2Base
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor

def align_to_next_page(pos):
    PAGE_SIZE=4096
    if (PAGE_SIZE - 1) & pos:
        return (pos + PAGE_SIZE) & ~(PAGE_SIZE - 1)
    else:
        return pos
    
class FILE_VERSION(enum.IntEnum):
    UNK = 0
    V0 = 1

STRING_ENCODING = 'UTF-8'
ENDIANNESS = ''

class DATA_TYPE(enum.IntEnum):
    F16 = 0
    F32 = 1
    I32 = 2
    L64 = 3
    Q4_0 = 4
    Q4_1 = 5
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

class FTYPE(enum.IntEnum):
    F16 = 0
    F32 = 1

def write_string(f, s):
    encoded_s = s.encode(STRING_ENCODING)
    f.write(struct.pack(f"{ENDIANNESS}i", len(encoded_s)))
    f.write(encoded_s)

def write_int(f, v):
    f.write(struct.pack(f"{ENDIANNESS}i", v))

def load_vocab(path):
    with open(path, 'r') as f:
        j = json.load(f)

        return j

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_model(f, model_name, model, ftype):
    # write header
    write_string(f, model_name)

    num_layers = len(model.keys())
    write_int(f, num_layers)

    data_type_to_enum = {
        np.dtype('float16'): DATA_TYPE.F16,
        np.dtype('float32'): DATA_TYPE.F32,
        np.dtype('int32'): DATA_TYPE.I32,
        np.dtype('int64'): DATA_TYPE.L64,
    }

    name_to_ndarrays = {}

    # write the layers
    for layer_name, layer in model.items():
        ndarray = model[layer_name]
        ndarray = ndarray.squeeze().numpy()

        data_type_name = ndarray.dtype
        shape = [*ndarray.shape]
        shape.reverse()
        ndims = len(shape)

        # print(data_type_name, data_type_name == np.float32, data_type_to_enum[np.float32])

        data_type = data_type_to_enum[data_type_name]

        performed_conversion = False
        if ftype == FTYPE.F16:
            if model_name != 'query_tokens' and model_name != 'ln_vision' \
                and ('norm' not in model_name or 'Norm' not in model_name):
                if layer_name.endswith('weight') and ndims >= 2:
                    ndarray = ndarray.astype('float16')
                    data_type = DATA_TYPE.F16
                    performed_conversion = True
        
        # TODO: ggml doesn't support f32 conv2d..., force f16
        elif layer_name == 'patch_embed.proj.weight':
            ndarray = ndarray.astype('float16')
            data_type = DATA_TYPE.F16
            performed_conversion = True

        if not performed_conversion and ndarray.dtype != np.dtype('float32'):
            ndarray = ndarray.astype('float32')
            data_type = DATA_TYPE.F32

        # name
        write_string(f, layer_name)
        
        # shape
        write_int(f, len(shape))
        f.write(struct.pack(f"{ENDIANNESS}{len(shape)}i", *shape))

        # datatype
        write_int(f, data_type)

        name_to_ndarrays[layer_name] = ndarray

    # write it actually
    print(f'=== {model_name} ===')
    padi = len(str(len(model)))
    for i, (layer_name, layer) in enumerate(model.items()):
        cur_pos = f.tell()
        f.seek(align_to_next_page(cur_pos))
        ndarray = name_to_ndarrays[layer_name]
        print(f"[{i+1:{padi}d}/{len(model)}] Writing tensor {model_name + '.' + layer_name:48s} | size {ndarray.nbytes:16} | type {str(ndarray.dtype):8s} | shape {ndarray.shape}")
        ndarray.tofile(f)
    print('======================')

def write_file(outfile, minigpt4, ftype_string):
    ftype = FTYPE.F32
    if ftype_string == 'f16':
        ftype = FTYPE.F16
    elif ftype_string == 'f32':
        ftype = FTYPE.F32
    else:
        print(f'Invalid ftype: {ftype_string}')

    with open(outfile, 'wb+') as f:
        # file header
        f.write(b'ggml')
        write_int(f, FILE_VERSION.V0)
        write_int(f, ftype)

        # write config
        config = {}
        config['ftype'] = ftype_string
        config['Qformer'] = minigpt4.Qformer.config.__dict__
        config_json = json.dumps(config)
        write_string(f, config_json)
        print(json.dumps(config, indent=2))

        visual_encoder = minigpt4.visual_encoder.state_dict()
        ln_vision = minigpt4.ln_vision.state_dict()
        query_tokens = { 'weight': minigpt4.query_tokens.detach() }
        qformer = minigpt4.Qformer.state_dict()
        llama_proj = minigpt4.llama_proj.state_dict()

        # write models
        write_model(f, 'visual_encoder', visual_encoder, ftype)
        write_model(f, 'ln_vision', ln_vision, ftype)
        write_model(f, 'query_tokens', query_tokens, ftype)
        write_model(f, 'Qformer', qformer, ftype)
        write_model(f, 'llama_proj', llama_proj, ftype)

class MiniGPT4(Blip2Base):
    """
    MiniGPT4 model from https://github.com/Vision-CAIR/MiniGPT-4
    """
    def __init__(self,
        pretrained_minigpt4_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0
    ):
        super().__init__()
        self.img_size = img_size
        self.low_resource = low_resource
        self.preprocessor = Blip2ImageEvalProcessor(img_size)

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        print('Loading VIT Done')
        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        print('Loading Q-Former Done')
        llama_hidden_size = 5120
        self.is_7b = False
        if '7b.pth' in str(pretrained_minigpt4_path):
            self.is_7b = True
            llama_hidden_size = 4096
        self.llama_proj = torch.nn.Linear(
            self.Qformer.config.hidden_size, llama_hidden_size
        )
        self.load_projection(pretrained_minigpt4_path)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def load_projection(self, path):
        state = torch.load(path, map_location=torch.device('cpu'))["model"]
        self.llama_proj.load_state_dict({
            "weight": state["llama_proj.weight"],
            "bias": state["llama_proj.bias"]})

def main():
    parser = argparse.ArgumentParser(description='Combine models into format')
    parser.add_argument("--pretrained_minigpt4", type=Path, help='Path to pretrained_minigpt4 model', nargs='?', default='pretrained_minigpt4.pth')
    parser.add_argument("--ftype", type=str, help='output type (f16 or f32)', nargs='?', default='f16')
    args = parser.parse_args()

    pretrained_minigpt4_path = args.pretrained_minigpt4

    ftype_string = args.ftype

    minigpt4 = MiniGPT4(pretrained_minigpt4_path)
    model_size = '13B'
    if minigpt4.is_7b:
        model_size = '7B'

    cwd = Path.cwd()
    outfile = cwd / f'minigpt4-{model_size}-{ftype_string}.bin'

    write_file(outfile, minigpt4, ftype_string)
    print(f'Wrote to {outfile}')

if __name__ == '__main__':
    main()


