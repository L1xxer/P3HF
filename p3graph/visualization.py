import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    def __init__(self):
        self.losses = []
        self.disc_losses = []
        self.disc_accs = []  # New: track discriminator accuracy
        self.f1_scores = []
        self.epochs = []
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))  # Changed to 3 subplots
        plt.tight_layout()

    def start(self):
        # Just initialize the plots
        self._update_plots()
        self.save_figure()

    def add_loss(self, epoch, loss):
        self.losses.append(loss)
        self.epochs.append(epoch)
        self._update_plots()
        self.save_figure()

    def add_disc_loss(self, epoch, disc_loss):
        self.disc_losses.append(disc_loss)
        self._update_plots()
        self.save_figure()

    def add_disc_acc(self, epoch, disc_acc):
        self.disc_accs.append(disc_acc)
        self._update_plots()
        self.save_figure()

    def add_f1(self, epoch, f1):
        self.f1_scores.append(f1)
        self._update_plots()
        self.save_figure()

    def _update_plots(self):
        # Clear the axes
        for ax in self.axs:
            ax.clear()

        # Plot losses
        if self.losses:
            self.axs[0].plot(self.epochs, self.losses, 'b-', label='Training Loss')
            if self.disc_losses and len(self.disc_losses) == len(self.epochs):
                self.axs[0].plot(self.epochs, self.disc_losses, 'r-', label='Discriminator Loss')
            self.axs[0].set_title('Training Losses')
            self.axs[0].set_xlabel('Epoch')
            self.axs[0].set_ylabel('Loss')
            self.axs[0].legend()
            self.axs[0].grid(True)

        # Plot discriminator accuracy
        if self.disc_accs:
            if len(self.disc_accs) != len(self.epochs):
                epochs_acc = list(range(1, len(self.disc_accs) + 1))
            else:
                epochs_acc = self.epochs
            self.axs[1].plot(epochs_acc, self.disc_accs, 'c-', label='Discriminator Accuracy')
            self.axs[1].set_title('Discriminator Accuracy')
            self.axs[1].set_xlabel('Epoch')
            self.axs[1].set_ylabel('Accuracy (%)')
            self.axs[1].set_ylim([0, 100.0])  # Accuracy from 0-100%
            self.axs[1].legend()
            self.axs[1].grid(True)

        # Plot F1 scores
        if self.f1_scores:
            epochs_f1 = list(range(1, len(self.f1_scores) + 1))
            self.axs[2].plot(epochs_f1, self.f1_scores, 'g-', label='F1 Score')
            self.axs[2].set_title('Validation F1 Score')
            self.axs[2].set_xlabel('Epoch')
            self.axs[2].set_ylabel('F1 Score')
            self.axs[2].set_ylim([0, 1.0])  # F1 is between 0 and 1
            self.axs[2].legend()
            self.axs[2].grid(True)

        self.fig.tight_layout()

    def save_figure(self):
        plt.savefig('training_progress.png')

    def stop(self):
        plt.close(self.fig)