import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # --- ENCODER ---
        # Input: (Batch, 1, 256)
        self.encoder = nn.Sequential(
            # Layer 1: 256 -> 128
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Layer 2: 128 -> 64
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # --- DECODER ---
        # Input: (Batch, 64, 64)
        self.decoder = nn.Sequential(
            # Layer 1: 64 -> 128 (Kernel=2, Stride=2 acts as perfect 2x upsampler)
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(True),
            
            # Layer 2: 128 -> 256
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid() # Force [0, 1] range
        )

    def forward(self, x):
        # Explicit Shape Check (Fix 1)
        original_len = x.shape[-1]
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Safety Assertion: If this fails, the architecture is wrong for the input size
        assert decoded.shape[-1] == original_len, \
            f"Shape Mismatch! Input: {original_len}, Output: {decoded.shape[-1]}"
            
        return decoded
