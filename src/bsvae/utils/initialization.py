import torch.nn as nn

def weights_init(m, activation: str = "relu"):
    """
    Initialize weights for a given module.
    - Supports nn.Linear, can be extended for Conv layers later.
    - Chooses init based on activation for correct variance scaling.

    Parameters
    ----------
    m : nn.Module
        Layer to initialize.
    activation : str
        Activation name ('relu', 'leaky_relu', 'tanh', 'sigmoid', or 'linear').
    """
    if isinstance(m, nn.Linear):
        act = activation.lower() if activation is not None else "linear"

        if act == "relu":
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif act == "leaky_relu":
            nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="leaky_relu")
        elif act in ("tanh", "sigmoid"):
            gain = nn.init.calculate_gain(act)
            nn.init.xavier_uniform_(m.weight, gain=gain)
        else:  # linear or unknown
            nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:
            nn.init.zeros_(m.bias)
