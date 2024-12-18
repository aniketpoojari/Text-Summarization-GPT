class DataLoaderPretraining:
    def __init__(self, B, T, data, device):
        self.B = B
        self.T = T
        self.data = data
        self.current_position = 0
        self.device = device
        self.num_tokens = len(data)
        self.num_sentences = self.num_tokens // self.T
        self.num_batches = self.num_sentences // self.B
        self.current_position = 0
        self.current_batch = 0

    def next_batch(self):
        B, T = self.B, self.T

        # Slice the data to create the current batch
        buf = self.data[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        # advance the position in the dataset
        self.current_position += B * T + 1
        self.current_batch += 1

        # If we've exhausted the data, reset position
        if (
            self.current_position >= self.num_tokens
            or self.current_batch >= self.num_batches
        ):
            self.reset()

        return x.to(self.device), y.to(self.device)

    def reset(self):
        self.current_position = 0
        self.current_batch = 0
