import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
    
class PiGDM(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 scheduler_config,):
        super(PiGDM, self).__init__(net, forward_op)
        self.scheduler = Scheduler(**scheduler_config)
        self.sde = False

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x = torch.randn(observation.shape[0], self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max   
        
        pbar = tqdm.trange(self.scheduler.num_steps)
        
        for i in pbar:

            sigma, scaling_factor, factor = self.scheduler.sigma_steps[i], self.scheduler.scaling_factor[i], self.scheduler.factor_steps[i]
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                denoised = self.net(x / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x.device))
                inverse_vec = self.forward_op.pseudo_inverse(observation) - self.forward_op.pseudo_inverse(self.forward_op.forward(denoised))
                loss = (inverse_vec.detach() * denoised).sum()
            grad = torch.autograd.grad(loss, x)[0]    
            
            # coeff = (sigma**2 + 1) / sigma ** 2
            score = (denoised - x / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]

            # score = score + coeff * grad
            if self.sde:
                epsilon = torch.randn_like(x)
                x = x * scaling_factor + factor * score + np.sqrt(factor) * epsilon + grad * self.scheduler.scaling_steps[i]
            else:
                x = x * scaling_factor + factor * score * 0.5 + grad * self.scheduler.scaling_steps[i]

            difference = observation - self.forward_op.forward(denoised)
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Avg. Error: {difference.abs().mean().item()}')
        return x