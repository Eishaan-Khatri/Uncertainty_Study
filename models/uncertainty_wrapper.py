import torch
import torch.nn as nn

class MCDropoutWrapper(nn.Module):
    def __init__(self, model, n_passes=20):
        super().__init__()
        self.model = model
        self.n_passes = n_passes
        
    def forward(self, x):
        # Crucial: MC Dropout requires dropout layers to be active.
        # However, we must be careful not to affect BatchNorm stats (if we had them).
        # Since our backbone ONLY has Dropout and Conv, .train() is 'safe enough' 
        # BUT we must acknowledge this design decision.
        self.model.train() 
        
        outputs = []
        for _ in range(self.n_passes):
            with torch.no_grad():
                outputs.append(self.model(x))
                
        outputs = torch.stack(outputs) # (T, B, C, L)
        
        # Mean prediction
        mean_pred = outputs.mean(dim=0)
        
        # Predictive Variance (Uncertainty)
        # Squeeze channel dim: (B, 1, L) -> (B, L)
        variance = outputs.var(dim=0).squeeze(1)
        
        return mean_pred, variance
