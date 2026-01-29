import torch
import torch.nn as nn

class TweedieLoss(nn.Module):
   
    def __init__(self, p=1.5, epsilon=1e-8):
        super().__init__()
        assert 1 < p < 2
        self.p = p
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = pred + self.epsilon
        # L = -target * (pred^(1-p) / (1-p)) + (pred^(2-p) / (2-p))
        term1 = -target * torch.pow(pred, 1 - self.p) / (1 - self.p)
        term2 = torch.pow(pred, 2 - self.p) / (2 - self.p)
        return torch.mean(term1 + term2)

class BiomassLoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.weights = torch.tensor([1/3, 1/3, 1/3]).to(device=device)
        self.loss = TweedieLoss(p=1.2)
       
    def forward(self, outputs, labels):
        green, dead, clover, gdm, total = outputs
        l_green  = self.loss(green.squeeze(), labels[:,0])
        l_dead   = self.loss(dead.squeeze(), labels[:,1])
        l_clover = self.loss(clover.squeeze(), labels[:,2])
        losses = torch.stack([l_green, l_dead, l_clover])
        return (losses * self.weights).sum(), losses.squeeze()
    