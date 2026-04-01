import argparse
import contextlib
import os.path
import sys
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterator
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
import logging
from torch import Tensor
try:
    import gguf
except ImportError:
    print("please run command : pip install gguf")

logger = logging.getLogger("hf-to-gguf")

class mode(ABC):
    def __init__(self, mode_path: Path, ftype: int, out_path: Path, is_big_endian: bool):
        super().__init__()
        self.dir_model = mode_path
        self.ftype = ftype
        self.fname_out = out_path
        self.is_big_endian = is_big_endian
        self.endianess = (
            gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        )
        self.model_checkpoint = "model.pt"
        self.gguf_writer = gguf.GGUFWriter(
            out_path,
            "FunAsr",
            endianess=self.endianess,
            use_temp_file=False,
        )

    def set_vocab(self):
        raise NotImplementedError

    def find_hparam(self, keys: str, default=None, config=None) -> Any:
        map_key = keys.split(".")
        config = config or self.hparams

        if not isinstance(config, dict):
            return default  # 如果 config 不是字典，直接返回 default

        if len(map_key) == 1:
            return config.get(map_key[0], default)
        else:
            # 递归查找
            next_config = config.get(map_key[0])
            if next_config is None:
                return default
            option = ".".join(map_key[1:])
            return self.find_hparam(option, default=default, config=next_config)

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        print(f"gguf: loading model part '{self.model_checkpoint}'")
        file_path = self.dir_model / self.model_checkpoint
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        
        model_part = torch.load(
            str(file_path),
            map_location="cpu",
            weights_only=True, # 安全加载
        )

        for name, data in model_part.items():
            yield name, data

    def set_gguf_parameters(self):
        raise NotImplementedError

    def write_tensors(self):
        raise NotImplementedError

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()
        print(f"Model written to {self.fname_out}")

    @staticmethod
    def load_hparams(dir_model):
        try:
            import yaml
        except ImportError:
            print("please run: pip install pyyaml")
        
        config_path = dir_model / "config.yaml"
        if not config_path.exists():
             # Fallback logic needed?
             return {}
             
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def load_qwen_haparams(dir_model):
        try:
            config = AutoConfig.from_pretrained(dir_model, trust_remote_code=False).to_dict()
        except Exception as e:
            import json
            logger.warning(f"Failed to load via AutoConfig: {e}")
            json_path = dir_model / "config.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                config = {}
        return config


class FunAsr(mode):
    def __init__(self, mode_path, ftype, out_path, is_big_endian):
        super().__init__(mode_path, ftype, out_path, is_big_endian)

        self.hparams = self.load_hparams(mode_path)
        self.model_name = "FunAsr"
        self.tokpre = "qwen2"
   
        self.lm_dir = self.dir_model / "Qwen3-0.6B" 
        if not self.lm_dir.exists():
             logger.warning(f"LM Dir {self.lm_dir} not found, falling back to root")
             self.lm_dir = self.dir_model

    
        self.llm_config = self.load_qwen_haparams(self.lm_dir)

    def set_gguf_parameters(self):
        # front
        self.gguf_writer.add_int32("frontend.sample_rate", self.find_hparam("frontend_conf.fs", default=16000))
        self.gguf_writer.add_string("frontend.window", self.find_hparam("frontend_conf.window", default="hamming"))
        self.gguf_writer.add_int32("frontend.num_mels", self.find_hparam("frontend_conf.n_mels", default=80))
        self.gguf_writer.add_int32("frontend.frame_length", self.find_hparam("frontend_conf.frame_length", default=25))
        self.gguf_writer.add_int32("frontend.frame_shift", self.find_hparam("frontend_conf.frame_shift", default=10))
        self.gguf_writer.add_int32("frontend.lfr_m", self.find_hparam("frontend_conf.lfr_m", default=1))
        self.gguf_writer.add_int32("frontend.lfr_n", self.find_hparam("frontend_conf.lfr_n", default=1))

        # encoder
        self.gguf_writer.add_int32("encoder.output_size", self.find_hparam("audio_encoder_conf.output_size"))
        self.gguf_writer.add_int32("encoder.attention_heads", self.find_hparam("audio_encoder_conf.attention_heads"))
        self.gguf_writer.add_int32("encoder.linear_units", self.find_hparam("audio_encoder_conf.linear_units"))
        self.gguf_writer.add_int32("encoder.num_blocks", self.find_hparam("audio_encoder_conf.num_blocks"))
        self.gguf_writer.add_int32("encoder.tp_blocks", self.find_hparam("audio_encoder_conf.tp_blocks"))
        self.gguf_writer.add_int32("encoder.kernel_size", self.find_hparam("audio_encoder_conf.kernel_size"))
        self.gguf_writer.add_int32("encoder.sanm_shift", self.find_hparam("audio_encoder_conf.sanm_shfit"))
        #llm
        self.gguf_writer.add_block_count(self.find_hparam("num_hidden_layers", config=self.llm_config))

        self.gguf_writer.add_embedding_length(self.find_hparam("hidden_size", config=self.llm_config))
        
   
        self.gguf_writer.add_context_length(self.find_hparam("max_position_embeddings", config=self.llm_config))
        
        self.gguf_writer.add_feed_forward_length(self.find_hparam("intermediate_size", config=self.llm_config, default=self.find_hparam("head_dim", config=self.llm_config)*4)) 
        
        self.gguf_writer.add_head_count(self.find_hparam("num_attention_heads", config=self.llm_config))
        self.gguf_writer.add_head_count_kv(self.find_hparam("num_key_value_heads", config=self.llm_config))
        self.gguf_writer.add_rope_freq_base(self.find_hparam("rope_theta", config=self.llm_config))
        
        head_dim = self.find_hparam("head_dim", config=self.llm_config)
        if head_dim is None:
    
             hidden = self.find_hparam("hidden_size", config=self.llm_config)
             heads = self.find_hparam("num_attention_heads", config=self.llm_config)
             if hidden and heads:
                head_dim = hidden // heads
        
        if head_dim:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)
            
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        print("| Layer name | n_dims | torch type | gguf type | parameters size|")
        print("|:|:|:|:|:|")

        for name, data_torch in self.get_tensors():
            if name.endswith(("Loss", "loss")) :
                continue
            if name.startswith("ctc"):
                continue
            if 'linear_q_k_v' in name:

                q_k_v = data_torch.split(data_torch.size(0) // 3)
                self.write_one_tensor(q_k_v[0], name.replace('linear_q_k_v', 'linear_q'))
                self.write_one_tensor(q_k_v[1], name.replace('linear_q_k_v', 'linear_k'))
                self.write_one_tensor(q_k_v[2], name.replace('linear_q_k_v', 'linear_v'))

            elif 'fsmn_block.weight' in name:
                self.write_one_tensor(data_torch.to(torch.float16), name)
            else:
                self.write_one_tensor(data_torch, name)

     

    def does_token_look_special(self, token: str | bytes) -> bool:
        if isinstance(token, (bytes, bytearray)):
            token_text = token.decode(encoding="utf-8")
        elif isinstance(token, memoryview):
            token_text = token.tobytes().decode(encoding="utf-8")
        else:
            token_text = token

        seems_special = token_text in (
            "<pad>", "<mask>", "<2mass>", "[@BOS@]",
        )
        seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
        seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))
        seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))
        return seems_special

    def set_vocab(self):
        tokens: list[str] = []
        toktypes: list[int] = []

        tokenizer = AutoTokenizer.from_pretrained(self.lm_dir, trust_remote_code=True)
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        

        vocab_size = self.find_hparam("vocab_size", config=self.llm_config, default=len(tokenizer.vocab))
        
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    if i in added_tokens_decoder and not added_tokens_decoder[i].normalized:
                        previous_token = token
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                    
                    is_special = False
                    if i in added_tokens_decoder:
                        is_special = added_tokens_decoder[i].special
                    
                    if is_special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        self.gguf_writer.add_tokenizer_model("gpt2") 
        self.gguf_writer.add_tokenizer_pre(self.tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        
        # Special Vocab handling usually requires special_tokens_map.json or similar
        try:
            special_vocab = gguf.SpecialVocab(self.lm_dir, load_merges=True)
            special_vocab.add_to_gguf(self.gguf_writer)
        except Exception as e:
            logger.warning(f"Could not load SpecialVocab: {e}")

    def write_one_tensor(self, data_torch, name):
        old_dtype = data_torch.dtype
        if data_torch.dtype == torch.bfloat16:
    
            data = data_torch.view(torch.uint16).numpy()  # shape 不变，dtype=uint16
            target_qtype = gguf.GGMLQuantizationType.BF16
            n_dims = data_torch.ndim
            self.gguf_writer.add_tensor(name, data, raw_dtype=target_qtype)
            # print(f"✅ {name} | {old_dtype} → BF16 (保留原始比特)")
            return
        
       
        if data_torch.dtype not in (torch.float16, torch.float32):
            data_torch = data_torch.to(torch.float32)

        data = data_torch.numpy()
        n_dims = len(data.shape)
        data_dtype = data.dtype

    
        if self.ftype == 0 and data_dtype == np.float16 and 'fsmn_block.weight' not in name:
            data = data.astype(np.float32)

        if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            data = data.astype(np.float32)
        if (
                self.ftype == 1
                and data_dtype == np.float32
                and name.endswith(".weight")
                and n_dims == 2
        ):
            data = data.astype(np.float16)
        print(f"Writing tensor {name} | shape {data.shape} | {old_dtype} -> {data.dtype}")
        self.gguf_writer.add_tensor(name, data)

if __name__ == "__main__":

    model_path = Path("/home/vncuser/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512")
    out_path = Path("./FunAsr.bin")
    

    asr = FunAsr(model_path, gguf.GGMLQuantizationType.F16, out_path, True)
    with torch.inference_mode():

        print("Setting params...")
        asr.set_gguf_parameters()
    
        print("Setting vocab...")
        asr.set_vocab()
    
        print("Writing model...")
        asr.write()
    
        print("Done.")