import torch


class StochasticWeightAveraging:
    """
    Maintains a **running arithmetic mean** of a set of parameters
    (a.k.a. Stochastic Weight Averaging).

    Public interface is identical to `ExponentialMovingAverage`:
      • move_shadow_params_to_device
      • update
      • copy_to
      • store / restore
      • state_dict / load_state_dict
    """

    def __init__(self,
                 parameters,
                 start_step: int,
                 avg_frequency: int = 1,
                 device=None):
        """
        Args
        ----
        parameters     : iterable of `torch.nn.Parameter`
        start_step     : optimizer step at which averaging begins
        avg_frequency  : average every *avg_frequency* optimizer steps
        device         : optional target device for the shadow copy
        """
        if start_step < 0 or avg_frequency < 1:
            raise ValueError("start_step must be ≥0 and avg_frequency ≥1")

        self.start_step = int(start_step)
        self.avg_frequency = int(avg_frequency)

        self.num_updates = 0            # number of models in the mean
        self._step_counter = 0          # counts every call to update()

        self.shadow_params = [
            p.clone().detach().to(device or p.device)
            for p in parameters if p.requires_grad
        ]
        self.collected_params = []

    # ------------------------------------------------------------
    # Optional helper if you move the model to a new device
    # ------------------------------------------------------------
    def move_shadow_params_to_device(self, device):
        self.shadow_params = [p.to(device) for p in self.shadow_params]

    # ------------------------------------------------------------
    # CORE: call once per *optimizer* step
    # ------------------------------------------------------------
    def update(self, parameters):
        self._step_counter += 1

        if self._step_counter < self.start_step:
            return  # still in burn-in phase

        if (self._step_counter - self.start_step) % self.avg_frequency != 0:
            return  # not a snapshot step

        self.num_updates += 1
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for avg_p, p in zip(self.shadow_params, parameters):
                # running mean:  avg ← avg + (p - avg) / num_updates
                avg_p.add_(p - avg_p, alpha=1.0 / self.num_updates)

    # ------------------------------------------------------------
    # Swap averaged weights into a live model
    # ------------------------------------------------------------
    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for avg_p, p in zip(self.shadow_params, parameters):
            p.data.copy_(avg_p.data)

    # ------------------------------------------------------------
    # Save / restore raw weights (identical to EMA helper)
    # ------------------------------------------------------------
    def store(self, parameters):
        self.collected_params = [p.clone() for p in parameters]

    def restore(self, parameters):
        if not self.collected_params:
            raise RuntimeError("No parameter values stored. "
                               "Call store() before restore().")
        for c_p, p in zip(self.collected_params, parameters):
            p.data.copy_(c_p.data)

    # ------------------------------------------------------------
    # Check-/restore the SWA state itself
    # ------------------------------------------------------------
    def state_dict(self):
        return dict(start_step=self.start_step,
                    avg_frequency=self.avg_frequency,
                    num_updates=self.num_updates,
                    step_counter=self._step_counter,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state):
        self.start_step = state['start_step']
        self.avg_frequency = state['avg_frequency']
        self.num_updates = state['num_updates']
        self._step_counter = state['step_counter']
        self.shadow_params = state['shadow_params']
