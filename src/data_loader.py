import torch


# data loading for pretraining step
def get_batch(
    step,
    split,
    block_size,
    batch_size,
    device,
    pre_train=None,
    pre_val=None,
    train_full=None,
    train_summ=None,
    val_full=None,
    val_summ=None,
):
    if step == "pretraining":
        data = pre_train if split == "train" else pre_val

        ix = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        x, y = x.to(device), y.to(device)

    elif step == "summary":

        if split == "train":
            ix = torch.randint(0, len(train_full), (batch_size,))

            x = train_full[ix]
            y = train_summ[ix]

            x, y = x.to(device), y.to(device)

        else:

            ix = torch.randint(0, len(val_full), (batch_size,))

            x = val_full[ix]
            y = val_summ[ix]

            x, y = x.to(device), y.to(device)

    return x, y
