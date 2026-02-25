"""
A bare-bones GPT-2 style transformer.
"""

import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub

from utils import state_dict_converter


# TODO: Add in attention mask to the entire assignment
# TODO: Maybe add KV caching


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Using attention dim from attention is all you need
        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)

        #self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        B, T, C = x.shape
        H=self.config.n_heads
        q = self.W_q(x) #(B,T,C)
        k = self.W_k(x) #(B,T,C)
        v = self.W_v(x) #(B,T,C)
        q = rearrange(q,'b t (h d) -> b h t d',h=H)
        k = rearrange(k,'b t (h d) -> b h t d',h=H)
        v = rearrange(v,'b t (h d) -> b h t d',h=H)
        attn_scores = (q@k.transpose(-2,-1))*(self.d_attention**-0.5)
        mask = self.causal_mask[:,:,:T,:T]
        attn_scores = attn_scores.masked_fill(mask==0,float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights@v    #(B,H,T,d)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.W_o(out)
        



class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
        

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        x = x + self.attention(self.pre_layer_norm(x))
        x = x + self.mlp(self.post_layer_norm(x))
        return x

        


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        device = x.device
        B, T = x.shape
        # Token嵌入
        tok_emb = self.embeddings(x)    #(B,T,d_model)
        # 位置嵌入
        pos = torch.arange(0,T,dtype = torch.long, device = device)
        pos_emb = self.position_embeddings(pos)     #(T,d_model)
        x = tok_emb + pos_emb
        #经过所有decoder
        for block in self.backbone:
            x = block(x)
        #最终层归一化与线性层映射回词表大小
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits        


    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:

        for _ in range(num_new_tokens):
            x_cond = x if x.size(1) <=self.config.context_length else x[:,-self.config.context_length:]
            #前向传播得到logits
            logits = self.forward(x_cond)
            #取最后一个logits
            last_logits = logits[:, -1, :]
            next_token = torch.argmax(last_logits, dim = -1, keepdim = True)
            x = torch.cat((x,next_token), dim = 1)
        return x


    def get_loss_on_batch(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"], 
    ) -> Float[Tensor, ""]:
        logits = self.forward(input_ids)    #(B,T,Vocab)
        #错位对齐，t位置的logits应该是原输入中t+1位置的token
        #logits舍弃最后一位，labels舍弃第一位
        shift_logits = logits[:,:-1,:].contiguous()
        shift_labels = input_ids[:,1:].contiguous()
        #展平向量以符合cross_entropy的输入要求
        loss = F.cross_entropy(
            shift_logits.view(-1,shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss
    @classmethod
    def from_pretrained(cls):
        """
        We simply always load up the GPT-2 model
        """

        # Config for GPT-2
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # Load weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":

    # Uncomment this if you are not logged in
    # huggingface_hub.login()
    
    model = Transformer.from_pretrained()
