
#%%
import sys
sys.path.append('../')
from agent import Agent
from envs import make_atari_env
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

if __name__ == '__main__':
    with hydra.initialize(version_base="1.3", config_path="../../config"):
        cfg = hydra.compose(config_name='trainer')

    device = torch.device(cfg.common.device)
    test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)
    num_actions = int(test_env.num_actions)
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
# %%
