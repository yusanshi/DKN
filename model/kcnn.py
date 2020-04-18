import torch


class KCNN(torch.nn.Module):
    """
    Knowledge-aware CNN (KCNN) based on Kim CNN.
    Input a news sentence (e.g. its title), produce its embedding vector.
    """

    def __init__(self):
        super(KCNN, self).__init__()

    def forward(self, x):
        pass
