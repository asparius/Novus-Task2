import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def scaled_dot_product(q,k,v,mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask==0, -1e16)

    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention,v)
    return values, attention


class MultiHeadAttention(nn.Module):


    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension musty be a factor of number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads


        self.qkv_proj  = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,x,mask=None,return_attention=False):
        batch_size, seq_length, _ = x.size()

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size,seq_length,self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0,2,1,3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        values, attention = scaled_dot_product(q,k,v,mask=mask)
        values = values.permute(0,2,1,3)
        values = values.reshape(batch_size,seq_length,self.embed_dim)

        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        

class EncoderBlock(nn.Module):

    def __init__(self, d_dim, num_heads, d_ff, dropout):
        
        super().__init__()

        self.self_attn = MultiHeadAttention(d_dim,d_dim,num_heads)

        self.MLP = nn.Sequential(
            nn.Linear(d_dim,d_ff),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_ff,d_dim)
        )
        self.ln_1 = nn.LayerNorm(d_dim)
        self.ln_2 = nn.LayerNorm(d_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = self.ln_1(x + self.dropout(attn_out))
        x = self.ln_2(x + self.dropout(self.MLP(x)))
        
        return x