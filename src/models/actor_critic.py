from collections import namedtuple
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .blocks import Conv3x3, SmallResBlock
from coroutines.env_loop import make_env_loop
from envs import TorchEnv, WorldModelEnv
from utils import init_lstm, LossAndLogs
from data import Batch


ActorCriticOutput = namedtuple("ActorCriticOutput", "logits_act val hx_cx logits_disc")


@dataclass
class ActorCriticLossConfig:
    backup_every: int
    gamma: float
    lambda_: float
    weight_value_loss: float
    weight_entropy_loss: float
    weight_discriminator_loss: float


@dataclass
class ActorCriticConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    channels: List[int]
    down: List[int]
    num_actions: Optional[int] = None


class ActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.encoder = ActorCriticEncoder(cfg)
        self.lstm_dim = cfg.lstm_dim
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (sum(cfg.down))) ** 2
        self.lstm = nn.LSTMCell(input_dim_lstm, cfg.lstm_dim)
        self.critic_linear = nn.Linear(cfg.lstm_dim, 1)
        self.discriminator_linear = nn.Linear(cfg.lstm_dim, 1)
        self.actor_linear = nn.Linear(cfg.lstm_dim, cfg.num_actions)

        self.actor_linear.weight.data.fill_(0)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        self.discriminator_linear.weight.data.fill_(0)
        self.discriminator_linear.bias.data.fill_(0)
        init_lstm(self.lstm)

        self.discriminator_act = nn.Sigmoid()
        
        self.env_loop = None
        self.loss_cfg = None
        self.data_iter_discriminant = None
        self._rl_env = None

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device

    def setup_training(self, rl_env: Union[TorchEnv, WorldModelEnv], loss_cfg: ActorCriticLossConfig, dl_actor_critic_discriminant: DataLoader = None) -> None:
        assert self.env_loop is None and self.loss_cfg is None and self.data_iter_discriminant is None
        self._rl_env = rl_env
        self.env_loop = make_env_loop(rl_env, self)
        self.loss_cfg = loss_cfg
        
        if dl_actor_critic_discriminant is not None:
            self.data_iter_discriminant = iter(dl_actor_critic_discriminant)

    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]) -> ActorCriticOutput:
        assert obs.ndim == 4
        x = self.encoder(obs)
        x = x.flatten(start_dim=1)
        hx, cx = self.lstm(x, hx_cx)
        return ActorCriticOutput(self.actor_linear(hx), self.critic_linear(hx).squeeze(dim=1), (hx, cx), self.discriminator_act(self.discriminator_linear(hx)))

    def calulate_discriminator_loss(self, probs_disc_real, probs_disc_fake, mask_real, mask_fake):
        assert probs_disc_real.shape == probs_disc_fake.shape
        loss_D_real = F.binary_cross_entropy(
            probs_disc_real, torch.ones_like(probs_disc_real), reduction="none"
        )
        loss_D_fake = F.binary_cross_entropy(
            probs_disc_fake, torch.zeros_like(probs_disc_fake), reduction="none"
        )
        mask_real = mask_real.float()
        mask_fake = mask_fake.float()
        loss_D_real = (loss_D_real[..., 0] * mask_real).sum() / mask_real.sum()
        loss_D_fake = (loss_D_fake[..., 0] * mask_fake).sum() / mask_fake.sum()
        loss_D = (loss_D_real + loss_D_fake) / 2
        return loss_D
    
    def predict_discriminator_seq(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]=None):
        assert obs.ndim == 5
        if hx_cx is None:
            hx = torch.zeros(self._rl_env.num_envs, self.lstm_dim, device=self.device)
            cx = torch.zeros(self._rl_env.num_envs, self.lstm_dim, device=self.device)
            hx_cx = (hx, cx)
        
        b, t, c, h, w = obs.shape
        obs = obs.reshape(b * t, c, h, w)
        x = self.encoder(obs)
        x = x.reshape(b, t, -1)  # (b t) e h w -> b t (e h w)
        
        hx_hist = []
        
        for t_i in range(t):
            #_test_hx_cx = self.predict_act_value(obs.reshape(b, t, c, h, w)[:, t_i], hx_cx)[2]
            hx_cx = self.lstm(x[:, t_i], hx_cx)
            #assert ((hx_cx[0] - _test_hx_cx[0]).abs().max() < 0.001).item()
            hx_hist.append(hx_cx[0])
            
        hx = torch.stack((hx_hist), dim=1)
        return ActorCriticOutput(None, None, None, self.discriminator_act(self.discriminator_linear(hx)))
        
            
    
    def forward(self) -> LossAndLogs:
        c = self.loss_cfg
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, _, probs_disc_fake = self.env_loop.send(c.backup_every)
        assert ((probs_disc_fake <= 1).all() & (probs_disc_fake >= 0).all()).item()
        d = Categorical(logits=logits_act)
        entropy = d.entropy().mean()

        lambda_returns = compute_lambda_returns(rew, end, trunc, val_bootstrap, c.gamma, c.lambda_)

        loss_actions = (-d.log_prob(act) * (lambda_returns - val).detach()).mean()
        loss_values = c.weight_value_loss * F.mse_loss(val, lambda_returns)
        loss_entropy = -c.weight_entropy_loss * entropy
        
        batch_real: Batch = next(self.data_iter_discriminant).to(self.device)
        _, _, _, probs_disc_real = self.predict_discriminator_seq(batch_real.obs)
        
        loss_disc = c.weight_discriminator_loss * self.calulate_discriminator_loss(
            probs_disc_real, 
            probs_disc_fake, 
            mask_real=batch_real.mask_padding,
            mask_fake=torch.ones_like(end),
        )
        
        loss = loss_actions + loss_entropy + loss_values + loss_disc

        metrics = {
            "policy_entropy": entropy.detach() / math.log(2),
            "loss_actions": loss_actions.detach(),
            "loss_entropy": loss_entropy.detach(),
            "loss_values": loss_values.detach(),
            "loss_discriminator": loss_disc.detach(),
            "accuracy_discriminator_fake": (probs_disc_fake < 0.5).float().mean().item(),
            "accuracy_discriminator_real": (probs_disc_real >= 0.5).float().mean().item(),
            "loss_total": loss.detach(),
        }

        return loss, metrics


class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        assert len(cfg.channels) == len(cfg.down)
        encoder_layers = [Conv3x3(cfg.img_channels, cfg.channels[0])]
        for i in range(len(cfg.channels)):
            encoder_layers.append(SmallResBlock(cfg.channels[max(0, i - 1)], cfg.channels[i]))
            if cfg.down[i]:
                encoder_layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


@torch.no_grad()
def compute_lambda_returns(
    rew: Tensor,
    end: Tensor,
    trunc: Tensor,
    val_bootstrap: Tensor,
    gamma: float,
    lambda_: float,
) -> Tensor:
    assert rew.ndim == 2 and rew.size() == end.size() == trunc.size() == val_bootstrap.size()

    rew = rew.sign()  # clip reward

    end_or_trunc = (end + trunc).clip(max=1)
    not_end = 1 - end
    not_trunc = 1 - trunc

    lambda_returns = rew + not_end * gamma * (not_trunc * (1 - lambda_) + trunc) * val_bootstrap

    if lambda_ == 0:
        return lambda_returns

    last = val_bootstrap[:, -1]
    for t in reversed(range(rew.size(1))):
        lambda_returns[:, t] += end_or_trunc[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns
