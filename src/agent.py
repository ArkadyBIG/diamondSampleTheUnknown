from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from envs import TorchEnv, WorldModelEnv
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from models.discriminator import Discriminator, DiscriminatorConfig
from utils import extract_state_dict
from coroutines.env_loop import make_env_loop


@dataclass
class AgentConfig:
    denoiser: DenoiserConfig
    rew_end_model: RewEndModelConfig
    actor_critic: ActorCriticConfig
    discriminator: DiscriminatorConfig
    num_actions: int

    def __post_init__(self) -> None:
        self.denoiser.inner_model.num_actions = self.num_actions
        self.rew_end_model.num_actions = self.num_actions
        self.actor_critic.num_actions = self.num_actions
        self.discriminator.num_actions = self.num_actions


class Agent(nn.Module):
    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.denoiser = Denoiser(cfg.denoiser)
        self.rew_end_model = RewEndModel(cfg.rew_end_model)
        self.actor_critic = ActorCritic(cfg.actor_critic)
        self.discriminator = Discriminator(cfg.discriminator)

    @property
    def device(self):
        return self.denoiser.device

    def setup_training(
        self,
        sigma_distribution_cfg: SigmaDistributionConfig,
        actor_critic_loss_cfg: ActorCriticLossConfig,
        rl_env: Union[TorchEnv, WorldModelEnv],
    ) -> None:
        self.denoiser.setup_training(sigma_distribution_cfg)
        
        env_loop = make_env_loop(rl_env, self.actor_critic)
        self.actor_critic.setup_training(env_loop, actor_critic_loss_cfg)
        self.discriminator.setup_training(env_loop)

    def load(
        self,
        path_to_ckpt: Path,
        load_denoiser: bool = True,
        load_rew_end_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        sd = torch.load(Path(path_to_ckpt), map_location=self.device)
        sd = {k: extract_state_dict(sd, k) for k in ("denoiser", "rew_end_model", "actor_critic")}
        if load_denoiser:
            self.denoiser.load_state_dict(sd["denoiser"])
        if load_rew_end_model:
            self.rew_end_model.load_state_dict(sd["rew_end_model"])
        if load_actor_critic:
            self.actor_critic.load_state_dict(sd["actor_critic"])
