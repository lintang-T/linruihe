import torch
import torch.nn.functional as F

def euclidean(a, b):
    return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)


def run_episode(model, episode):

    sx, sy, qx, qy = episode

    emb_s = model(sx)
    emb_q = model(qx)

    proto = []
    for c in torch.unique(sy):
        proto.append(emb_s[sy == c].mean(0))

    proto = torch.stack(proto)

    dist = euclidean(emb_q, proto)
    logp = F.log_softmax(-dist, dim=1)

    loss = F.nll_loss(logp, qy)

    acc = (logp.argmax(1) == qy).float().mean()

    return loss, acc, emb_q, qy