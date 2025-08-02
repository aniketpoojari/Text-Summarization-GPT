import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class PretrainingDataset(Dataset):
    def __init__(self, data_path, T):
        with open(data_path, "rb") as file:
            self.data = torch.tensor(pickle.load(file), dtype=torch.long)
        self.T = T
        self.num_tokens = len(self.data)
        self.num_samples = self.num_tokens - self.T  # -1 to ensure we have a target for each input

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.T]
        y = self.data[idx+1:idx+self.T+1]
        return x, y

class DeepSpeedDataLoaderPretraining:
    def __init__(self, batch_size, seq_length, data_path, device, rank, world):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device
        self.rank = rank
        
        # Initialize dataset
        self.dataset = PretrainingDataset(data_path, seq_length)
        
        # Create sampler for distributed training
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset,
            num_replicas=world,
            rank=self.rank
        )
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler
        )
        
        self.iterator = iter(self.dataloader)
    
    def next_batch(self):
        try:
            x, y = next(self.iterator)
        except StopIteration:
            # Reset iterator at the end of epoch
            self.reset()
            x, y = next(self.iterator)
        
        return x.to(self.device), y.to(self.device)
    
    def reset(self):
        # Set epoch for the sampler
        self.sampler.set_epoch(self.sampler.epoch + 1)
        
        # Create new iterator
        self.iterator = iter(self.dataloader)
        
    def __len__(self):
        return len(self.dataloader)