import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt


class ReportGenerator:
    """Saves per-epoch metrics to CSV and plots loss / accuracy curves."""

    def __init__(self, output_dir, run_id):
        self.output_dir = output_dir
        self.run_id     = run_id
        self.metrics    = []

    def log_metric(self, epoch, train_loss, val_loss, val_acc):
        self.metrics.append({
            'run_id':     self.run_id,
            'timestamp':  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'val_acc':    val_acc,
        })

    def save_report(self):
        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.output_dir, 'metrics.csv'), index=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
        ax1.plot(df['epoch'], df['val_loss'],   label='Val Loss')
        ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss Curve')
        ax1.legend(); ax1.grid(True)

        ax2.plot(df['epoch'], df['val_acc'], color='orange', label='Val Accuracy')
        ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy Curve')
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_accuracy_curve.png'))
        plt.close()

        with open(os.path.join(self.output_dir, 'report.md'), 'w') as f:
            f.write("# Training Report\n\n")
            f.write(f"**Best Val Loss:** {df['val_loss'].min():.4f}\n")
            f.write(f"**Best Val Accuracy:** {df['val_acc'].max():.4f}\n")
            f.write(f"**Final Val Loss:** {df['val_loss'].iloc[-1]:.4f}\n\n")
            f.write("![Curves](loss_accuracy_curve.png)\n\n## Metrics\n")
            f.write(df.to_markdown(index=False))
