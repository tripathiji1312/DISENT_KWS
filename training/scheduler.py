import math

def grl_lambda_schedule(epoch: int, max_epochs: int) -> float:
    """Sigmoid ramp-up schedule for GRL lambda weight coefficient.
    
    Gradually increases lambda from 0.0 to 1.0 during training to avoid
    instability in the early stages when features are not well formed.
    """
    if max_epochs <= 0:
        return 0.0
    
    # Sigmoid scaling logic: p goes from 0.0 to 1.0
    p = float(epoch) / float(max_epochs)
    # Scaled to range [0.0, 1.0]
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
