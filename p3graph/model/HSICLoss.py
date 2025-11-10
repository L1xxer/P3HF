import torch
import torch.nn as nn


class HSICLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(HSICLoss, self).__init__()
        self.sigma = sigma

    def _rbf_kernel(self, X):
        # Calculate pairwise distances
        X_norm = torch.sum(X * X, dim=1, keepdim=True)
        X_dist = X_norm + X_norm.t() - 2 * torch.mm(X, X.t())
        # Apply RBF kernel
        return torch.exp(-X_dist / (2 * self.sigma**2))

    def forward(self, X, Y):
        # Inputs: X, Y are batches of features [batch_size, feature_dim]
        batch_size = X.size(0)

        # Center the kernel matrices
        K_X = self._rbf_kernel(X)
        K_Y = self._rbf_kernel(Y)

        # Center kernel matrices
        H = torch.eye(batch_size, device=X.device) - 1.0/batch_size * torch.ones((batch_size, batch_size), device=X.device)
        K_X_centered = torch.mm(torch.mm(H, K_X), H)
        K_Y_centered = torch.mm(torch.mm(H, K_Y), H)

        # Compute HSIC
        hsic_value = torch.trace(torch.mm(K_X_centered, K_Y_centered)) / (batch_size - 1)**2

        return hsic_value  # Return as a loss to minimize