import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import settings

class IrisNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3, p_dropout=0.1):
        """Initialized model

        Args:
            input_dim (int, optional): Input dimension. Defaults to 4.
            hidden_dim (int, optional): Hidden dimension. Defaults to 16.
            output_dim (int, optional): Output dimension. Defaults to 3.
            p_dropout (float, optional): dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass, takes input data, 
        process it through the network's layers,
        and returns the logits.

        Args:
            x (torch.Tensor): Input tensor to process. 

        Returns:
            torch.Tensor: The trained logits. 
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)
    
_model: IrisNet | None = None # module-scope cache

def get_model(device: str = "cpu") -> IrisNet:
    """Returns a singleton IrisNet initialized with saved weights.

    Args:
        device (str, optional): Preferred device to use. Defaults to "cpu".

    Returns:
        IrisNet: The model with saved weights.
    """
    global _model
    if _model is None:
        # instantiate and load weights
        model = IrisNet()
        state = torch.load(settings.model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        _model = model
    return model