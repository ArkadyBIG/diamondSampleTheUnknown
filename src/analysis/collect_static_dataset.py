import sys
sys.path.append('../')

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)
from agent import Agent
from envs import make_atari_env
from hydra.utils import instantiate

import hydra
from data import collate_segments_to_batch, Dataset, Batch, Episode
from batch_sampler_full_episode import BatchSamplerFullEpisode
import torch
from pathlib import Path
from coroutines.collector import make_collector, NumToCollect
from torch.utils.data import DataLoader
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from tqdm import tqdm
import torch.nn.functional as F
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('actor_path')
    parser.add_argument('dataset_save_dir')
    parser.add_argument('--steps-to-collect', type=int, default=100_000)
    parser.add_argument('--epsilon', type=float, default=0)
    
    args = parser.parse_args()
    print(__file__)
    print(vars(args))
    
    with hydra.initialize(version_base="1.3", config_path="../../config"):
        cfg = hydra.compose(config_name='trainer')

    device = torch.device(cfg.common.device)
    test_env = make_atari_env(num_envs=4, device=device, **cfg.env.test)
    num_actions = int(test_env.num_actions)
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    
    actor_path = Path(args.actor_path)
    n_ep = args.steps_to_collect
    dataset = Dataset(Path(args.dataset_save_dir))
    dataset.load_from_default_path()
    
    if len(dataset) == 0:
        agent.load(path_to_ckpt=actor_path)
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=args.epsilon)
        collector.send(NumToCollect(steps=n_ep))
        dataset.save_to_default_path()
        print("\nSummary of collect:")
        print(f"Num steps: {n_ep} ")
        print(f"Reward counts: {dict(dataset.counter_rew)}")
        print(f"Average duration of an eposode: {n_ep / dataset.num_episodes}")