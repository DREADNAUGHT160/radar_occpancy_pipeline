import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for highly imbalanced occupancy grids."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce  = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt   = torch.exp(-bce)
        at   = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = at * (1 - pt) ** self.gamma * bce
        return loss.mean()


class FocalDiceLoss(nn.Module):
    """Focal loss + Dice loss combined."""

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal      = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal = self.focal(inputs, targets)
        probs = torch.sigmoid(inputs).view(-1)
        t     = targets.view(-1)
        dice  = 1.0 - (2 * (probs * t).sum() + 1e-6) / (probs.sum() + t.sum() + 1e-6)
        return focal + self.dice_weight * dice


class BCEDiceLoss(nn.Module):
    """Weighted BCE + Dice loss combined."""

    def __init__(self, pos_weight=10.0, dice_weight=0.5):
        super().__init__()
        self.pos_weight  = pos_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        pw   = torch.tensor([self.pos_weight], device=inputs.device, dtype=inputs.dtype)
        bce  = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pw)
        p    = torch.sigmoid(inputs).view(-1)
        t    = targets.view(-1)
        dice = 1.0 - (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
        return bce + self.dice_weight * dice


class TverskyLoss(nn.Module):
    """
    Tversky loss — asymmetric Dice.
    alpha < 0.5 penalises false positives less; beta > 0.5 forces higher recall.
    """

    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, inputs, targets):
        p  = torch.sigmoid(inputs).view(-1)
        t  = targets.view(-1)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        return 1.0 - (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)


def get_loss_function(config, device):
    """Return the configured loss function from config['training']['loss']."""
    loss_type = config['training'].get('loss', 'focal').lower()
    alpha     = config['training'].get('focal_alpha', 0.25)
    gamma     = config['training'].get('focal_gamma', 2.0)

    if loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    if loss_type == 'focal_dice':
        return FocalDiceLoss(alpha=alpha, gamma=gamma,
                             dice_weight=config['training'].get('dice_weight', 0.5))
    if loss_type == 'bce_dice':
        return BCEDiceLoss(pos_weight=config['training'].get('pos_weight', 10.0),
                           dice_weight=config['training'].get('dice_weight', 0.5))
    if loss_type == 'tversky':
        return TverskyLoss(alpha=config['training'].get('tversky_alpha', 0.3),
                           beta=config['training'].get('tversky_beta', 0.7))
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    if loss_type == 'weighted_bce':
        pw = torch.tensor([config['training'].get('pos_weight', 10.0)]).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unknown loss: {loss_type}")
