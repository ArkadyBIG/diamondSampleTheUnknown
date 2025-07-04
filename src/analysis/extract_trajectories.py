#%%
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



# %%
@torch.no_grad
def predict_autoregressive(sampler, batch: Batch):
    steps_cond = sampler.denoiser.inner_model.cfg.num_steps_conditioning
    obs_buffer = batch.obs[:, :steps_cond]
    act_buffer = batch.act[:, :steps_cond]
    
    pred_obs = [torch.zeros_like(batch.obs[:, 0])] * steps_cond
    for i in (range(batch.obs.size(1) - steps_cond)):
        next_obs, _ = sampler.sample(obs_buffer, act_buffer)
        
        pred_obs.append(next_obs)
        
        obs_buffer = obs_buffer.roll(-1, dims=1)
        act_buffer = act_buffer.roll(-1, dims=1)
        obs_buffer[:, -1] = next_obs
        act_buffer[:, -1] = batch.act[:, steps_cond + i]
    return torch.stack(pred_obs, dim=1)
        
def extract_autoregressive_trajectories(
        dl: DataLoader, sampler, save_to_dataset: Dataset
    ):
    for batch in tqdm(dl):
        batch: Batch = batch.to(device)
        pred_obs = predict_autoregressive(sampler, batch)
        episode = Episode(
            batch.obs[0],
            batch.act[0],
            batch.rew[0],
            batch.end[0],
            batch.trunc[0],
            {
                'pred_obs': pred_obs[0],
                'num_steps_conditioning': sampler.denoiser.inner_model.cfg.num_steps_conditioning
            }
        )
        save_to_dataset.add_episode(
            episode
        )
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('actor_path')
    parser.add_argument('denoiser_path')
    parser.add_argument('save_dir')
    parser.add_argument('--episodes-to-collect', type=int, default=5)
    parser.add_argument('--autoregresive-length', type=int, default=100)
    
    args = parser.parse_args()
    print(__file__)
    print(vars(args))
    
    with hydra.initialize(version_base="1.3", config_path="../../config"):
        cfg = hydra.compose(config_name='trainer')

    device = torch.device(cfg.common.device)
    test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)
    num_actions = int(test_env.num_actions)
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    
    actor_path = Path(args.actor_path)
    n_ep = args.episodes_to_collect
    dataset = Dataset(Path(f"dataset/{str(actor_path).split('/')[-4]}/{actor_path.stem}_{n_ep}"))
    dataset.load_from_default_path()
    
    if len(dataset) == 0:
        agent.load(path_to_ckpt=actor_path)
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
        collector.send(NumToCollect(episodes=n_ep))
        dataset.save_to_default_path()
    steps_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    bs = BatchSamplerFullEpisode(dataset, 0, 1, 1, steps_cond + args.autoregresive_length, steps_cond + 1, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
    print(f'Total number of episodes is {dataset.num_episodes}. Approx num of segemnts: {len(bs)}')
    
    denoiser_path = Path(args.denoiser_path)
    agent.load(path_to_ckpt=denoiser_path)
    sampler = DiffusionSampler(agent.denoiser, wm_env_cfg.diffusion_sampler)
    
    save_to_dataset = Dataset(args.save_dir)
    
    extract_autoregressive_trajectories(
        dl, sampler, save_to_dataset
    )
    save_to_dataset.save_to_default_path()
