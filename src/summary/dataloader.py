import torch
import numpy as np
import pandas as pd

class DataLoaderSummary:
    def __init__(self, B, data, device):
        self.B = B
        self.data = data
        self.device = device
        self.num_sentences = self.data.shape[0]
        self.num_batches = self.data.shape[0] // self.B
        self.current_position = 0
        self.current_batch = 0

    def next_batch(self):
        B = self.B

        # Slice the data to create the current batch
        rows = self.data[self.current_position : self.current_position + B]
        input = rows[:, 0, :]
        target = rows[:, 1, :] 
        mask = rows[:, 2, :]
        
        # advance the position in the tensor
        self.current_position += B
        self.current_batch += 1

        # if loading the next batch would be out of bounds, advance to next shard
        if (
            self.current_position > self.num_sentences
            or self.current_batch >= self.num_batches
        ):
            self.reset()

        return input.to(self.device), target.to(self.device), mask.to(self.device)

    def reset(self):
        self.current_position = 0
        self.current_batch = 0