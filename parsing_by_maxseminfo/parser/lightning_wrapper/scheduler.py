from lightning.pytorch import Callback


class WarmupScheduler(Callback):
    def __init__(self, warmup_steps, coeff_name, initial_coeff=0.0, target_coeff=1.0, start_step = 0):
        super().__init__()
        self.coeff_name = coeff_name
        self.warmup_steps = warmup_steps
        self.start_step = start_step
        self.initial_coeff = initial_coeff
        self.target_coeff = target_coeff
        self.coefficient = initial_coeff

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        assert self.coeff_name in dir(pl_module), f"Attribute {self.coeff_name} not found in the model"

        total_warmup_steps = self.warmup_steps + self.start_step
        if trainer.global_step < total_warmup_steps:
            # Linearly increase the coefficient
            self.coefficient = (
                self.initial_coeff +
                (self.target_coeff - self.initial_coeff) * (max(0, (trainer.global_step-self.start_step)) / self.warmup_steps)
            )
            
            # Update the attribute in your model
            pl_module.__dict__[self.coeff_name] = self.coefficient
        else:
            # Update the attribute in your model
            pl_module.__dict__[self.coeff_name] = self.target_coeff