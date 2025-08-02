class DataLoaderSummary:
    def __init__(self, B, full, summ, mask, device):
        self.B = B
        self.full = full
        self.summ = summ
        self.mask = mask
        self.device = device
        self.num_sentences = len(self.full)
        self.num_batches = len(self.full) // self.B
        self.current_position = 0
        self.current_batch = 0

    def next_batch(self):
        B = self.B

        # Slice the data to create the current batch
        input = self.full[self.current_position : self.current_position + B]
        target = self.summ[self.current_position : self.current_position + B]
        mask = self.mask[self.current_position : self.current_position + B]

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
