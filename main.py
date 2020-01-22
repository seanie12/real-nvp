import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from torch.distributions import MultivariateNormal
from torchvision import datasets, transforms

from datasets import DatasetMoons
from models import CouplingLayer, NormalizingFlow, NormalizingFlowModel


def main(args):
    device = torch.cuda.current_device()
    if args.cpu:
        device = "cpu"
    prior = MultivariateNormal(torch.zeros(2).to(device),
                               torch.eye(2).to(device))
    d = DatasetMoons()

    flows = [CouplingLayer(args.input_dim, args.hidden_dim, parity=i % 2)
             for i in range(args.num_blocks)]
    model = NormalizingFlowModel(prior, flows)
    optimizer = optim.Adam(model.parameters(), args.lr)
    model = model.to(device)
    model.train()
    
    for epoch in range(0, args.epochs+1):
        x = d.sample(128)
        x = x.to(device)
        zs, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.sum(logprob)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(loss.item())
    model.eval()
    x = d.sample(128)
    x = x.to(device)
    with torch.no_grad():
        zs, prior_logprob, log_det = model(x)
    z = zs[-1]

    x = x.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    p = model.prior.sample([128, 2]).squeeze(1).cpu()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(p[:, 0], p[:, 1], c='g', s=5)
    plt.scatter(z[:, 0], z[:, 1], c='r', s=5)
    plt.scatter(x[:, 0], x[:, 1], c='b', s=5)
    plt.legend(['prior', 'x->z', 'data'])
    plt.axis('scaled')
    plt.title('x -> z')

    with torch.no_grad():
        zs = model.sample(128*8)
    z = zs[-1]
    z = z.detach().cpu().numpy()
    plt.subplot(122)
    plt.scatter(x[:, 0], x[:, 1], c='b', s=5, alpha=0.5)
    plt.scatter(z[:, 0], z[:, 1], c='r', s=5, alpha=0.5)
    plt.legend(['data', 'z->x'])
    plt.axis('scaled')
    plt.title('z -> x')
    plt.savefig("./figures/fig.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--hidden_dim", type=int, default=24)
    args = parser.parse_args()

    main(args)
