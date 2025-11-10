import torch
import torch.nn as nn


class EventDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(EventDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 3)  # 3 events to classify
        )
        class_weights = torch.tensor([1.0, 1.0, 1.0], device='cuda')  # Adjust if needed
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion = nn.CrossEntropyLoss()
        # Initialize metrics counters
        self.total_samples = 0
        self.correct_samples = 0

    def forward(self, x):
        return self.model(x)

    def get_loss(self, x, labels):
        logits = self.forward(x)
        loss = self.criterion(logits, labels)

        # Update accuracy metrics during loss calculation
        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            self.total_samples += labels.size(0)
            self.correct_samples += (predicted == labels).sum().item()

        return loss

    def reset_metrics(self):
        self.total_samples = 0
        self.correct_samples = 0

    def get_accuracy(self):
        if self.total_samples == 0:
            return 0.0
        return 100.0 * self.correct_samples / self.total_samples