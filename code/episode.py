import random
import torch

def sample_episode(x, y, n_way=5, k_shot=5, q_query=10):

    classes = random.sample(list(set(y.tolist())), n_way)

    sx, sy, qx, qy = [], [], [], []

    for i, c in enumerate(classes):

        idx = (y == c).nonzero().squeeze()
        idx = idx[torch.randperm(len(idx))]

        chosen = idx[:k_shot + q_query]

        sx.append(x[chosen[:k_shot]])
        sy += [i] * k_shot

        qx.append(x[chosen[k_shot:]])
        qy += [i] * q_query

    return torch.cat(sx), torch.tensor(sy), torch.cat(qx), torch.tensor(qy)