import torch
import torch.nn.functional as F
import torch.nn as nn


class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, parity):
        super(CouplingLayer, self).__init__()
        self.dim = dim
        self.parity = parity
        self.scale = nn.Sequential(nn.Linear(dim//2, hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_dim, dim//2))
        self.translate = nn.Sequential(nn.Linear(dim//2, hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_dim, dim//2))

    def forward(self, x):
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        z0 = x0
        s = self.scale(x0)
        t = self.translate(x0)
        z1 = torch.exp(s) * x1 + t
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)

        return z, log_det

    def backward(self, z):
        z0, z1 = z[:, ::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        x0 = z0
        s = self.scale(z0)
        t = self.translate(z0)
        x1 = (z1 - t) * torch.exp(-s)
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)

        return x, log_det


class NormalizingFlow(nn.Module):
    def __init__(self, flows):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        batch_size = x.size(0)
        log_det = torch.zeros(batch_size).to(x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        batch_size = z.size(0)
        log_det = torch.zeros(batch_size).to(z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    def __init__(self, prior, flows):
        super(NormalizingFlowModel, self).__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_log_prob = self.prior.log_prob(zs[-1])
        
        return zs, prior_log_prob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        device = next(self.flow.parameters())
        z = self.prior.sample((num_samples,)).to(device)
        xs, _ = self.flow.backward(z)
        return xs
