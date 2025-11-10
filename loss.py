import torch
import torch.nn as nn

class MEMTOLoss(nn.Module):
    def __init__(self, lambda_entropy: float):
        super().__init__()
        self.lambda_entropy = lambda_entropy
    
    def forward(self, reconstructed_output: torch.Tensor, original_input: torch.Tensor, attention_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        epsilon = 1e-12
        l_rec = nn.MSELoss()(reconstructed_output, original_input)
        entropy = -attention_weights * torch.log(attention_weights + epsilon)
        l_entr = entropy.sum() / attention_weights.size(0)
        total_loss = l_rec + self.lambda_entropy * l_entr
        return total_loss, l_rec, l_entr
