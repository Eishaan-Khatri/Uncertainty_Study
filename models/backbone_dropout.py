import torch
import torch.nn as nn

class ConvAutoencoderMCDropout(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ConvAutoencoderMCDropout, self).__init__()
        
        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate), # Explicit Dropout
            nn.MaxPool1d(2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate), # Explicit Dropout
            nn.MaxPool1d(2, stride=2),
        )
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate), # Explicit Dropout
            
            nn.ConvTranspose1d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Shape Safety Check (Restored)
        original_len = x.shape[-1]
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # 2. Shape Assertion
        assert decoded.shape[-1] == original_len, \
            f"Shape Mismatch! Input: {original_len}, Output: {decoded.shape[-1]}"
            
        return decoded
