from typing import Generator, List, Optional

import numpy as np
import torch

from data.dataset import Dataset
from data.segment import SegmentId


class BatchSamplerFullEpisode(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: Dataset,
        rank: int,
        world_size: int,
        batch_size: int,
        max_seq_length: int,
        min_seq_length: int,
        sample_weights: Optional[List[float]] = None,
        can_sample_beyond_end: bool = False,
    ) -> None:
        super().__init__(dataset)
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        assert self.batch_size == 1
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.can_sample_beyond_end = can_sample_beyond_end

    def __len__(self):
        return int(np.ceil(self.dataset.lengths / self.max_seq_length).sum())

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        for i in range(self.dataset.num_episodes):
            segments = self.sample(i)
            yield from [[s] for s in segments]

    def sample(self, episode_id) -> List[SegmentId]:
        rng = np.random.default_rng(seed=episode_id)
        start = rng.integers(0, self.max_seq_length)
        segments = []
        if start != 0:
            segments.append(SegmentId(episode_id, 0, start))
        
        for pos in range(start, self.dataset.lengths[episode_id], self.max_seq_length):
            segments.append(SegmentId(episode_id, pos, min(pos + self.max_seq_length, self.dataset.lengths[episode_id])))
        segments = [
            s for s in segments
            if s.stop - s.start >= self.min_seq_length
        ]
        
        return segments