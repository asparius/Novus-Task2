import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_block import EncoderBlock
import math
import pytorch_lightning as pl
class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        d_dim = block_args.pop("input_dim")
        num_heads = block_args.pop("num_heads")
        d_ff = block_args.pop("dim_feedforward")
        dropout = block_args.pop("dropout")
        self.layers = nn.ModuleList([EncoderBlock(d_dim=d_dim,num_heads=num_heads,d_ff=d_ff,dropout=dropout) for _ in range(num_layers)])

    
    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)

        return x
    
    def get_attn_maps(self,x, mask=None):
        attn_maps = []

        for l in self.layers:
            _,attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attn_maps.append(attn_map)

            x = l(x)

        return attn_maps
        

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    

class Sorter(pl.LightningModule):

    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr,max_iters,dropout=0.0,input_dropout=0.0):

        super().__init__()
        self.save_hyperparameters()
        self.build_model()
    
    
    def build_model(self):

        self.input_network = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)

        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim = self.hparams.model_dim,
            dim_feedforward = 2 * self.hparams.model_dim,
            num_heads = self.hparams.num_heads,
            dropout = self.hparams.dropout
        )
        self.output_network = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

    def forward(self,x,mask=None,add_positional_encoding=True):
        x = self.input_network(x)

        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_network(x)
        return x
    
    @torch.no_grad()
    def get_attention_maps(self,x, mask=None, add_positional_encoding=True):
        x = self.input_network(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attn_maps(x, mask=mask)

        return attention_maps
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,self.hparams.max_iters)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
    
    def calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch
        inp_data = F.one_hot(inp_data, num_classes= self.hparams.num_classes).float()

        preds = self.forward(inp_data, add_positional_encoding=True)
        loss = F.cross_entropy(preds.view(-1,preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim = -1) == labels).float().mean()

        self.log(f"{mode}_loss",loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc

    
    def training_step(self,batch,batch_idx):
        loss,_ = self.calculate_loss(batch,mode="train")
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch,batch_idx):
        _ = self.calculate_loss(batch, mode="val")
    def test_step(self, batch, batch_idx):
        _ = self.calculate_loss(batch, mode="test")
        
        


        