import os
import numpy as np
import matplotlib.pyplot as plt
import json

class LossTracker:
    def __init__(self, exp_dir, save_freq=100, window_size=100):
        self.exp_dir = exp_dir
        self.save_freq = save_freq
        self.window_size = window_size
        
        self.steps = []
        self.losses = []
        self.moving_avg_losses = []
        
        # Create directories if they don't exist
        os.makedirs(exp_dir, exist_ok=True)
        
        # For moving average calculation
        self.window = []
    
    def update(self, step, loss, batch_count=1):
        """Update loss tracker with new loss value"""
        # If batch_count > 1, this is an epoch-level update
        self.steps.append(step)
        
        if batch_count > 1:
            # This is an epoch-level average
            avg_loss = loss / batch_count
            self.losses.append(avg_loss)
        else:
            # This is a step-level loss
            self.losses.append(loss)
        
        # Update moving average
        self.window.append(loss)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        
        self.moving_avg_losses.append(self.get_moving_average())
        
        # Save periodically
        if len(self.steps) % self.save_freq == 0:
            self.save_loss_plot()
            self.save_loss_data()
    
    def get_moving_average(self):
        """Get moving average of losses"""
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)
    
    def save_loss_plot(self):
        """Save loss plot to file"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, 'b-', alpha=0.3, label='Loss')
        plt.plot(self.steps, self.moving_avg_losses, 'r-', label=f'Moving Average (window={self.window_size})')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.exp_dir, 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
    
    def save_loss_data(self):
        """Save loss data to JSON file"""
        loss_data = {
            'steps': self.steps,
            'losses': self.losses,
            'moving_avg_losses': self.moving_avg_losses
        }
        
        # Convert numpy values to Python native types for JSON serialization
        for key in loss_data:
            if isinstance(loss_data[key], list) and loss_data[key] and hasattr(loss_data[key][0], 'item'):
                loss_data[key] = [x.item() if hasattr(x, 'item') else x for x in loss_data[key]]
        
        json_path = os.path.join(self.exp_dir, 'loss_data.json')
        with open(json_path, 'w') as f:
            json.dump(loss_data, f) 