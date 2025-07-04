#%%
import sys
sys.path.append('../')
from data import Dataset, Batch, Episode
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import os.path as osp
import torch.nn.functional as F
from itertools import zip_longest
import numpy as np

# %%
trajectories = {
    osp.split(p)[-1]: p for p in 
    glob('/Users/arkady/Desktop/dev/diamond/analysis_data/extracted_trajectories/aN*')
}
trajectories

# %%
for name, dataset_path in tqdm(trajectories.items()):
    dataset = Dataset(dataset_path)
    dataset.load_from_default_path()
    losses = []
    for episode_id in (range(dataset.num_episodes)):
        episode = dataset.load_episode(episode_id)
        pred_obs = episode.info['pred_obs']
        n = episode.info['num_steps_conditioning']
        loss_traj = F.mse_loss(
            pred_obs[n:].cpu(), episode.obs[n:],
            reduction='none'
        ).flatten(2).mean(-1)
        losses.append(loss_traj)
    
    losses_mean = [
        np.mean([v for v in vs if v is not None])
        for vs in zip_longest(*losses)
    ]
        
    plt.plot(losses_mean, label=name)

plt.legend(loc='best')
plt.grid()
    
# %%
