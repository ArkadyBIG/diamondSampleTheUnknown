#%%
import sys
sys.path.append('../')
from agent import Agent
from envs import make_atari_env
from data import Batch
import torch
from coroutines.env_loop import make_env_loop
import matplotlib.pyplot as plt
from functools import reduce
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
OmegaConf.register_new_resolver("eval", eval)
from tqdm import tqdm
#%%
if __name__ == '__main__':
    with hydra.initialize(version_base="1.3", config_path="../../config"):
        cfg = hydra.compose(config_name='trainer')

    device = torch.device(cfg.common.device)
    test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)
    num_actions = int(test_env.num_actions)
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
# %%
# exit()
p = '/Users/arkady/Desktop/dev/diamond/outputs/PaperLike/checkpoints/agent_versions/agent_epoch_00400.pt'
agent.load(path_to_ckpt=p, )

sampler = DiffusionSampler(agent.denoiser, wm_env_cfg.diffusion_sampler)

# %%
env_loop = make_env_loop(test_env, agent.actor_critic)
# %%
obs, act, rew, end, trunc, logits_act, val, val_bootstrap, _ = env_loop.send(1000)
end_or_trunc = (end + trunc) > 0
end_or_trunc = end_or_trunc[0]

#%%
end_indxs = (torch.arange(len(end_or_trunc), device=end_or_trunc.device)[end_or_trunc] + 1)
prev_end = end_indxs.roll(1)
prev_end[0] = 0
sizes = end_indxs - prev_end
#%%



# %%
plt.imshow((obs[0, 10].transpose(0, 2).transpose(0, 1) + 1).cpu() / 2, vmin=0, vmax=1)
# %%
n = agent.denoiser.cfg.inner_model.num_steps_conditioning
obs = obs[:, :cut_i]
act = act[:, :cut_i]
obs_buffer = obs[:, :n]
act_buffer = act[:, :n]
pred_obs = [torch.zeros_like(obs[:, 0])] * n
for i in tqdm(range(len(obs[0]) - n)):
    next_obs, _ = sampler.sample(obs_buffer, act_buffer)
    
    pred_obs.append(next_obs)
    
    obs_buffer = obs_buffer.roll(-1, dims=1)
    act_buffer = act_buffer.roll(-1, dims=1)
    obs_buffer[:, -1] = next_obs
    act_buffer[:, -1] = act[:, n + i]
    


#%%
pred_obs = pred_obs[::25]
obs = obs[:, ::25]
#%%


#%%
top_row = torch.cat(list(obs[0]), 2)
bot_row = torch.cat(list(pred_obs), 3)[0]
#%%
img = torch.cat([top_row, bot_row], 1)
img = (img.transpose(0, 2).transpose(0, 1) + 1).cpu() / 2
plt.figure(figsize=[6.4 * 3, 4.8 * 3])
plt.imshow(img, vmin=0, vmax=1)
plt.savefig('test.jpg')

# %%
