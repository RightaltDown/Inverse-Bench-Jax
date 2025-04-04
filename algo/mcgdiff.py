import torch
import tqdm
from .base import Algo
import numpy as np
import wandb
from utils.scheduler import Scheduler
    
class MCG_diff(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 scheduler_config,
                 num_particles):
        super(MCG_diff, self).__init__(net, forward_op)
        self.scheduler = Scheduler(**scheduler_config)
        self.num_particles = num_particles

    def score(self, model, x, sigma):
        """
            Computes the score function for the given model.

            Parameters:
                model (DiffusionModel): Diffusion model.
                x (torch.Tensor): Input tensor.
                sigma (float): Sigma value.

            Returns:
                torch.Tensor: The computed score.
        """
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (d - x) / sigma**2
    
    def K(self, t):
        if t == self.scheduler.num_steps:
            return 1
        return self.scheduler.factor_steps[t] / (self.scheduler.factor_steps[t]+ self.scheduler.sigma_steps[t]**2)
    
    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        
        sigma_y = self.forward_op.sigma_noise
        observation_t = self.forward_op.Ut(observation).unsqueeze(1).repeat(1, self.num_particles, 1, 1, 1) / self.forward_op.S
        z = torch.randn(observation.shape[0], self.num_particles, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device)
        x_t = self.scheduler.sigma_max * z * (1 - self.forward_op.M) + self.forward_op.M * self.scheduler.sigma_max * self.K(0) * z 

        pbar = tqdm.trange(self.scheduler.num_steps)


        for step in pbar:
            sigma, sigma_next, factor, scaling_factor, scaling_step = self.scheduler.sigma_steps[step], self.scheduler.sigma_steps[step + 1], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step], self.scheduler.scaling_steps[step]
            x = self.forward_op.V(x_t)
            denoised_t = self.forward_op.Vt(self.net(x.flatten(0,1)/scaling_step, torch.as_tensor(sigma).to(x.device))).view(observation.shape[0], self.num_particles, self.net.img_channels, self.net.img_resolution, self.net.img_resolution)
            score = (denoised_t - x_t / scaling_step) / sigma ** 2 / scaling_step
            x_next_t = x_t * scaling_factor + factor * score
            log_prob = -torch.linalg.norm(((observation_t - x_next_t) * self.forward_op.M).flatten(2), dim=-1)**2 / (2 * (sigma_next**2 + factor)) 
            log_prob += torch.linalg.norm(((observation_t - x_t)*self.forward_op.M).flatten(2), dim=-1)**2 / (2 * sigma**2)
            
            log_prob -= log_prob.min(dim=1, keepdim=True)[0]
            log_prob = torch.clamp(log_prob, max=60)
            indices = torch.multinomial(torch.exp(log_prob), self.num_particles, replacement=True)
            

            K = self.K(step+1)
            # print(x_next_t[indices].shape, x_next_t.shape, indices.shape)
            x_next_t = torch.gather(x_next_t, 1, indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.net.img_channels, self.net.img_resolution, self.net.img_resolution))
            x_masked = K * observation_t + (1 - K) * x_next_t + np.sqrt(K) * sigma_next * torch.randn_like(x_t)
            x_unmasked = x_next_t + np.sqrt(factor) * torch.randn_like(x_t)
            x_t = self.forward_op.M * x_masked + (1 - self.forward_op.M) * x_unmasked


        return self.forward_op.V(x_t)[:,0]
