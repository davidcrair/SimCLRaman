"""Supervised Contrastive Loss and InfoNCE Loss
"""


import torch
import torch.nn.functional as F
from torch import nn

class InfoNCELoss(torch.nn.Module):
    '''

    NOTE: THIS IS COPIED/MODIFIED FROM THE 2nd HOMEWORK ASSIGNMENT

    Implement the InfoNCE loss.
    This is Equation 1 in the SimCLR paper https://arxiv.org/abs/2002.05709

    For the two batched vectors z1 and z2,
    z1: B x D
    z2: B x D
    They are both a batch of N latent vectors, such that
    each matching index forms a positive pair,
    while any unmatching index forms a negative pair.
        (z1[i], z2[j]) is positive, if i == j
        (z1[i], z2[j]) is negative, if i != j
        (z1[i], z1[j]) is negative, if i != j
        (z2[i], z2[j]) is negative, if i != j
    1. We will construct a matrix z by concatenating z1 and z2.
    2. We will take its self matrix muliplication after L2 normalization
       to form a 2B x 2B cosine similarity matrix.
    3. We perform the exp and 1/temperature right after, since these are commutative.
    4. Then we identify the locations corresponding to the positive pairs and negative pairs,
       and compute `score_pos` and `score_neg`.
    5. Finally, we compute the InfoNCE loss with these values.


    NOTE: Additional explanation on the formulation and masking.

    Suppose we have
    z1 = [z1aa z1ab z1ac      <---- z1a
          z1ba z1bb z1bc]     <---- z1b
    z2 = [z2aa z2ab z2ac      <---- z2a
          z2ba z2bb z2bc]     <---- z2b

    z = torch.cat((z1, z2), dim=0)
    z = [z1aa z1ab z1ac
         z1ba z1bb z1bc
         z2aa z2ab z2ac
         z2ba z2bb z2bc]

    Let's then look at torch.matmul(z, z.T). It will be 2B x 2B in shape.
    If we use A*B to denote torch.matmul(A, B), the resulting matrix will be:

    [z1a*z1a z1a*z1b z1a*z2a z1a*z2b
     z1b*z1a z1b*z1b z1b*z2a z1b*z2b
     z2a*z1a z2a*z1b z2a*z2a z2a*z2b
     z2b*z1a z2b*z1b z2b*z2a z2b*z2b]

    Positive pairs are: (z1a, z2a) and (z1b, z2b)
    Negative pairs are everything besides positive pairs or self-self pairs.

    Positive mask will be
    [F F T F
     F F F T
     T F F F
     F T F F]
    Negative mask will be
    [F T F T
     T F T F
     F T F T
     T F T F]

    Think carefully why this is the case, and how this principle extends to other values of B.
    '''

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, labels=None):
        B, _ = z1.shape
        # normalize and concatenate
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z  = torch.cat([z1, z2], dim=0)

        # similarity matrix
        sim = (z @ z.T) / self.temperature

        # mask out self-self pairs
        mask = torch.eye(2*B, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        preds  = sim.argmax(dim=1)
        labels = (torch.arange(2*B, device=sim.device) + B) % (2*B)
        contrastive_acc = (preds == labels).float().mean().item()

        # compute exponent of similarity
        exp_sim = torch.exp(sim)

        # construct positive and negative masks
        pos_mask = torch.zeros_like(mask)
        for i in range(B):
            pos_mask[i, i+B] = pos_mask[i+B, i] = True
        neg_mask = ~(pos_mask | mask)

        # compute loss for positive and negative pairs
        pos = exp_sim[pos_mask].view(2*B,1)
        neg = exp_sim[neg_mask].view(2*B,-1).sum(1, keepdim=True)
        loss = -torch.log(pos/(pos+neg)).mean()

        return loss #, contrastive_acc


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss from https://arxiv.org/abs/2004.11362 (Supervised Contrastive Learning, Khosla et al. 2020)
    
    This loss is used for supervised contrastive learning, where we have
    multiple views of the same data point and we want to maximize the similarity
    between these views and data points with the same label, while minimizing the similarity
    between data points with different labels. The loss is computed as the negative log
    likelihood of the positive pairs divided by the sum of the negative pairs.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, labels):
        """
        z1, z2: two augmented views, each [B, D]
        labels: ground-truth labels for the B samples [B]
        """
        B, D = z1.shape

        # normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # construct a 2B x 2B matrix
        z = torch.cat([z1, z2], dim=0)
        lab = torch.cat([labels, labels], dim=0)

        # compute similarity matrix
        sim = (z @ z.T) / self.temperature

        # mask out self-similarities
        mask = torch.eye(2*B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -1e9)

        # get positive pairs
        labels_eq = lab.unsqueeze(0) == lab.unsqueeze(1)
        pos_mask = labels_eq & ~mask # remove self-similarities

        # compute log softmax over each row
        logprob = F.log_softmax(sim, dim=1)

        # loss = − average over i of (average log-prob of its positives)
        mean_log_prob_pos = (pos_mask * logprob).sum(1) / pos_mask.sum(1)

        loss = - mean_log_prob_pos.mean()
        return loss