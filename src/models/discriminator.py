from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_confusion_matrix

from .blocks import Conv3x3, Downsample, ResBlocks
from data import Batch
from utils import init_lstm, LossAndLogs


@dataclass
class DiscriminatorConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[int]
    backup_every: int
    num_actions: Optional[int] = None


class Discriminator(nn.Module):
    def __init__(self, cfg: DiscriminatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = RewEndEncoder(2 * cfg.img_channels, cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)
        self.act_emb = nn.Embedding(cfg.num_actions, cfg.cond_channels)
        self.rew_emb = nn.Embedding(3, cfg.cond_channels)
        self.end_emb = nn.Embedding(2, cfg.cond_channels)
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (len(cfg.depths) - 1)) ** 2
        self.lstm = nn.LSTM(input_dim_lstm, cfg.lstm_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
            nn.SiLU(),
            nn.Linear(cfg.lstm_dim, 2, bias=False),
        )
        init_lstm(self.lstm)
    
    def setup_training(self, env_loop):
        self.env_loop = env_loop

    def predict_real_fake(
        self,
        obs: Tensor,
        act: Tensor,
        rew: Tensor,
        end: Tensor,
        next_obs: Tensor,
        reset_hidden: Tensor,
        hx_cx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        b, t, c, h, w = obs.shape
        obs, act, next_obs = obs.reshape(b * t, c, h, w), act.reshape(b * t), next_obs.reshape(b * t, c, h, w)
        rew, end = rew.reshape(b * t), end.reshape(b * t)
        cond = self.act_emb(act) + self.rew_emb(rew) + self.end_emb(end)
        x = self.encoder(torch.cat((obs, next_obs), dim=1), cond)
        x = x.reshape(b, t, -1)  # (b t) e h w -> b t (e h w)
        
        X = []
        for i in range(x.shape[1]):
            out, hx_cx = self.lstm(x[:, i:i+1], hx_cx)
            X.append(out)
            # reset if new episod starts
            hx_cx = hx_cx[0] * (1 - reset_hidden[:, i:i+1][None]), hx_cx[1] * (1 - reset_hidden[:, i:i+1][None]) 
        x = torch.concat(X, dim=1)
        logits = self.head(x)
        return logits, hx_cx

    def forward(self, batch: Batch) -> LossAndLogs:
        obs = batch.obs[:, :-1]
        act = batch.act[:, :-1]
        next_obs = batch.obs[:, 1:]
        rew = batch.rew[:, :-1]
        end = batch.end[:, :-1]
        mask = batch.mask_padding[:, :-1]

        # When dead, replace frame (gray padding) by true final obs
        dead = end.bool().any(dim=1)
        if dead.any():
            final_obs = torch.stack([i["final_observation"] for i, d in zip(batch.info, dead) if d]).to(obs.device)
            next_obs[dead, end[dead].argmax(dim=1)] = final_obs

        logits_real, _ = self.predict_real_fake(obs, act, rew.long(), end.long(), next_obs, reset_hidden=1 - mask.float())
        logits_real = logits_real[mask]

        loss_real = F.cross_entropy(logits_real, torch.ones(logits_real.shape[:-1], device=logits_real.device))
        
        obs, act, rew, end, trunc, _, _, _, _ = self.env_loop.send(self.cfg.backup_every)
        logits_fake, _ = self.predict_real_fake(obs[:, :-1], act[:, :-1], rew[:, :-1].long(), end[:, :-1].long(), obs[:, 1:], reset_hidden=(end + trunc).clip(max=1)[:, :-1])
        logits_fake = logits_fake.view(-1, 2)
        loss_fake = F.cross_entropy(logits_fake, torch.zeros(logits_fake.shape[:-1], device=logits_fake.device))

        loss = (loss_real + loss_fake) / 2

        accuracy_real = (logits_real.argmax() == 1).float().mean()
        accuracy_fake = (logits_fake.argmax() == 1).float().mean()
        
        metrics = {
            "loss_real": loss_real.detach(),
            "loss_fake": loss_fake.detach(),
            "loss_total": loss.detach(),
            
            "accuracy_real": accuracy_real,
            "accuracy_fake": accuracy_fake,
            "accuracy": (accuracy_real + accuracy_fake) / 2,
        }
        return loss, metrics


class RewEndEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self.conv_in = Conv3x3(in_channels, channels[0])
        blocks = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        blocks.append(
            ResBlocks(
                list_in_channels=[channels[-1]] * 2,
                list_out_channels=[channels[-1]] * 2,
                cond_channels=cond_channels,
                attn=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]] + [nn.Identity()])

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block, down in zip(self.blocks, self.downsamples):
            x = down(x)
            x, _ = block(x, cond)
        return x
