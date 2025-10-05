import os
import matplotlib.pyplot as plt
import json


class LossTracker:
    def __init__(self, exp_dir, save_freq=100, window_size=100):
        self.exp_dir = exp_dir
        self.save_freq = save_freq
        self.window_size = window_size

        self.steps = []
        self.losses = []
        self.ema_losses = []
        self.moving_avg_losses = []
        self.moving_avg_ema_losses = []

        # Create directories if they don't exist
        os.makedirs(exp_dir, exist_ok=True)

        # For moving average calculation
        self.window = []
        self.ema_window = []

    def update(self, step, loss, ema_loss=None, batch_count=1):
        """Update loss tracker with new loss value and optional ema_loss"""
        self.steps.append(step)

        if batch_count > 1:
            avg_loss = loss / batch_count
            self.losses.append(avg_loss)
            if ema_loss is not None:
                avg_ema_loss = ema_loss / batch_count
                self.ema_losses.append(avg_ema_loss)
            else:
                self.ema_losses.append(None)
        else:
            self.losses.append(loss)
            self.ema_losses.append(ema_loss)

        # Update moving averages
        self.window.append(loss)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        self.moving_avg_losses.append(self.get_moving_average(self.window))

        if ema_loss is not None:
            self.ema_window.append(ema_loss)
            if len(self.ema_window) > self.window_size:
                self.ema_window.pop(0)
            self.moving_avg_ema_losses.append(self.get_moving_average(self.ema_window))
        else:
            self.moving_avg_ema_losses.append(None)

        # Save periodically
        if len(self.steps) % self.save_freq == 0:
            self.save_loss_plot()
            self.save_loss_data()

    def get_moving_average(self, window):
        if not window:
            return 0.0
        return sum(window) / len(window)

    def save_loss_plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, "b-", alpha=0.3, label="Loss")
        plt.plot(
            self.steps,
            self.moving_avg_losses,
            "b-",
            label=f"Loss MA (window={self.window_size})",
        )
        if any(l is not None for l in self.ema_losses):
            plt.plot(self.steps, self.ema_losses, "g-", alpha=0.3, label="EMA Loss")
            plt.plot(
                self.steps,
                self.moving_avg_ema_losses,
                "g-",
                label=f"EMA Loss MA (window={self.window_size})",
            )
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss and EMA Loss")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.exp_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()

    def save_loss_data(self):
        loss_data = {
            "steps": self.steps,
            "losses": self.losses,
            "ema_losses": self.ema_losses,
            "moving_avg_losses": self.moving_avg_losses,
            "moving_avg_ema_losses": self.moving_avg_ema_losses,
        }
        for key in loss_data:
            if (
                isinstance(loss_data[key], list)
                and loss_data[key]
                and hasattr(loss_data[key][0], "item")
            ):
                loss_data[key] = [
                    x.item() if hasattr(x, "item") else x for x in loss_data[key]
                ]
        json_path = os.path.join(self.exp_dir, "loss_data.json")
        with open(json_path, "w") as f:
            json.dump(loss_data, f)
