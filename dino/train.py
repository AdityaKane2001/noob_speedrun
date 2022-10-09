from multiprocessing.sharedctypes import Value
from model import DINOModel
import torch
import torch.nn as nn

def train_fn():
    gt = DINOModel().eval()
    gs = DINOModel()
    tps, tpt, C = 1, 1, 1
    C = torch.zeros((1, 768))
    m = 1
    bs = 1024 # 1024 in paper
    optim = torch.optim.AdamW(gs.parameters(), lr=5e-4 * bs / 256)

    for i, param_group in enumerate(optim.param_groups):
        print(i)
        # print(param_group)
    raise ValueError()
    seq_sched = torch.optim.lr_scheduler.SequentialLR(optim, [
        torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10),
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100 - 10)
    ], milestones=[10])

    # optim_scheduler = torch.optim.lr_scheduler.LinearLR(optim, )
    # lambda1 = lambda epoch: epoch
    # optim_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda1)
    epochs = 100
    for epoch in epochs:
        for datasample in dataloader:
            optim.zero_grad()

            x1, x2 = augment(x1), augment(x2)

            s1, s2 = gs(x1), gs(x2)
            with torch.no_grad():
                t1, t2 = gt(x1), gt(x2)

            loss = H_loss(t1, s2, tps, tpt, C) / 2 + H_loss(t2, s1, tps, tpt, C) / 2
            loss.backward()

            # update(gs)
            optim.step()

            # update(gt)
            gt.update_ema(student=gs, l=0.996)

            # update(C)
            C = m * C + (1 - m) * torch.mean(torch.cat([t1, t2], dim=0), dim=0)

    
def H_loss(t, s, tps, tpt, C):
    s = nn.Softmax(dim=1)(s / tps)
    t = nn.Softmax(dim=1)((t - C) / tpt)
    _u = t * torch.log(s)
    _v = torch.sum(_u, dim=1)
    _w = - torch.mean(_v)
    # print(_u.shape, _v.shape, _w.shape, sep='\n')
    # print(_w)
    return _w
    


if __name__ == "__main__":
    train_fn()
    bs = 16
    out_class = 10
    t = torch.randn(bs, out_class)
    s = torch.randn(bs, out_class)
    H_loss(t, s, tps=1, tpt=1, C=1)