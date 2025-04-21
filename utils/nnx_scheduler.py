import jax
import jax.numpy as jnp

'''
    Scheduler for diffusion sampling following EDM framework.
    schedule (σ(t)): linear, sqrt, vp
    timestep (discretization of t): log, poly-n, vp
    scaling: none, vp

    Example:
    VP: Scheduler(num_steps=1000, schedule='vp', timestep='vp', scaling='vp')
    VE: Scheduler(num_steps=1000, schedule='sqrt', timestep='log', scaling='none')
    EDM: Scheduler(num_steps=200, schedule='linear', timestep='poly-7', scaling='none')
    
    Example Usage: See DiffusionSampler for unconditional diffusion sampling.
'''

class Scheduler:
    """
    Scheduler for diffusion sigma(t) and discretization step size Delta t
    JAX-compatible implementation
    """

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear',
                 timestep='poly-7', scaling='none'):
        """
        Initializes the scheduler with the given parameters.

        Parameters:
            num_steps (int): Number of steps in the schedule.
            sigma_max (float): Maximum value of sigma.
            sigma_min (float): Minimum value of sigma.
            sigma_final (float): Final value of sigma, defaults to sigma_min.
            schedule (str): Type of schedule for sigma ('linear', 'sqrt', or 'vp').
            timestep (str): Type of timestep function ('log', 'poly-n', or 'vp').
            scaling (str): Type of scaling function ('none' or 'vp').
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final if sigma_final is not None else sigma_min
        self.schedule = schedule
        self.timestep = timestep

        # Create all the required functions
        steps = jnp.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)
        scaling_fn, scaling_derivative_fn = self.get_scaling_fn(scaling)
        if self.schedule == 'vp':
            self.sigma_max = sigma_fn(1.0) * scaling_fn(1.0)        
        
        # JAX-compatible approach to compute the arrays
        time_steps = jnp.array([time_step_fn(s) for s in steps])
        time_steps = jnp.append(time_steps, sigma_inv_fn(self.sigma_final))
        
        sigma_steps = jnp.array([sigma_fn(t) for t in time_steps])
        scaling_steps = jnp.array([scaling_fn(t) for t in time_steps])
        
        # Calculate scaling factors
        def calc_scaling_factor(i):
            return 1 - scaling_derivative_fn(time_steps[i]) / scaling_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1])
        # scaling_factor = 1 - \dot s(t)/s(t) * \Delta t
        scaling_factor = jnp.array([calc_scaling_factor(i) for i in range(num_steps)])
        
        # Calculate factor steps
        def calc_factor_steps(i):
            return 2 * scaling_fn(time_steps[i])**2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1])
        
        factor_steps = jnp.array([calc_factor_steps(i) for i in range(num_steps)])
        
        # Store computed arrays
        self.sigma_steps = sigma_steps
        self.time_steps = time_steps
        self.factor_steps = factor_steps
        self.scaling_factor = scaling_factor
        self.scaling_steps = scaling_steps
        
        # Ensure factor_steps are non-negative (using jnp.maximum instead of a list comprehension)
        self.factor_steps = jnp.maximum(self.factor_steps, 0)

    def get_sigma_fn(self, schedule):
        """
        Returns the sigma function, its derivative, and its inverse based on the given schedule.
        Uses JAX's pure function approach.
        """
        if schedule == 'sqrt':
            sigma_fn = lambda t: jnp.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / (2 * jnp.sqrt(t))
            sigma_inv_fn = lambda sigma: sigma ** 2

        elif schedule == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: jnp.ones_like(t)
            sigma_inv_fn = lambda t: t
        
        elif schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            
            def sigma_fn(t):
                return jnp.sqrt(jnp.exp(beta_d * t**2/2 + beta_min * t) - 1)
            
            def sigma_derivative_fn(t):
                sigma = sigma_fn(t)
                return (beta_d * t + beta_min) * jnp.exp(beta_d * t**2/2 + beta_min * t) / (2 * sigma)
            
            def sigma_inv_fn(sigma):
                return jnp.sqrt(beta_min**2 + 2*beta_d*jnp.log(sigma**2 + 1))/beta_d - beta_min/beta_d

        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented")
            
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_scaling_fn(self, schedule):
        """
        Returns the scaling function and its derivative based on the given schedule.
        Uses JAX's pure function approach.
        """
        if schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            
            def scaling_fn(t):
                return 1 / jnp.sqrt(jnp.exp(beta_d * t**2/2 + beta_min * t))
            
            def scaling_derivative_fn(t):
                return -(beta_d * t + beta_min) / (2 * jnp.sqrt(jnp.exp(beta_d * t**2/2 + beta_min * t)))
        else:
            scaling_fn = lambda t: jnp.ones_like(t) if isinstance(t, jnp.ndarray) else 1.0
            scaling_derivative_fn = lambda t: jnp.zeros_like(t) if isinstance(t, jnp.ndarray) else 0.0
            
        return scaling_fn, scaling_derivative_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        """
        Returns the time step function based on the given timestep type.
        Uses JAX's pure function approach.
        """
        if timestep == 'log':
            return lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            return lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        
        elif timestep == 'vp':
            return lambda r: 1 - r * (1 - 1e-3)
        
        else:
            raise NotImplementedError(f"Timestep {timestep} not implemented")

    @classmethod
    def get_partial_scheduler(cls, scheduler, new_sigma_max):
        """
        Generates a new scheduler with the given sigma_max value.
        
        In JAX, we create a new instance rather than deeply copying.
        """
        # Find how many steps to include
        mask = scheduler.sigma_steps < new_sigma_max
        num_steps = jnp.sum(mask) + 1
        
        # Create a new scheduler
        new_scheduler = cls(
            num_steps=num_steps-1,
            sigma_max=new_sigma_max,
            sigma_min=scheduler.sigma_min,
            sigma_final=scheduler.sigma_final,
            schedule=scheduler.schedule,
            timestep=scheduler.timestep
        )
        
        # Set arrays directly (in JAX, prefer functional assignment rather than mutation)
        new_scheduler.sigma_steps = scheduler.sigma_steps[-num_steps:]
        new_scheduler.time_steps = scheduler.time_steps[-num_steps:]
        new_scheduler.factor_steps = scheduler.factor_steps[-num_steps:]
        new_scheduler.scaling_factor = scheduler.scaling_factor[-num_steps:]
        new_scheduler.scaling_steps = scheduler.scaling_steps[-num_steps:]
        
        return new_scheduler