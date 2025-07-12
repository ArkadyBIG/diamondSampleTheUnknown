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
import torch as t
# %%
trajectories = {
    osp.split(p)[-1]: p for p in 
    glob('/scratch/gpfs/at3549/tmp/extracted_trajectories/*')
}
trajectories

# %%
data = {}
for name, dataset_path in (trajectories.items()):
    dataset = Dataset(dataset_path)
    dataset.load_from_default_path()
    losses = []
    for episode_id in tqdm(range(dataset.num_episodes)):
        episode = dataset.load_episode(episode_id)
        pred_obs = episode.info['pred_obs']
        n = episode.info['num_steps_conditioning']
        loss_traj = F.mse_loss(
            pred_obs[n:], episode.obs[n:].cuda(),
            reduction='none'
        ).flatten(2).mean(-1).cpu()
        losses.append(loss_traj)
    
    losses_mean = np.array([
        np.mean([v for v in vs if v is not None])
        for vs in zip_longest(*losses)
    ])
    data[name] = t.from_numpy(losses_mean)
#%%
t.save(
    data, 'data.pt'
)
#%%
data = t.load('data.pt')
#%%

for name, losses_mean in data.items():
    plt.plot(losses_mean, label=name)

plt.legend(loc='best')
plt.grid()
    
# %%
