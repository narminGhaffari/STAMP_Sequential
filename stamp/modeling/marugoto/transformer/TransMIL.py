import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=512 // 8, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.heads = heads
        self.norm = norm_layer(dim)
        self.mhsa = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.repeat(self.heads, 1, 1)

        x = self.norm(x)
        attn_output, _ = self.mhsa(x, x, x, need_weights=False, attn_mask=mask)
        return attn_output

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, norm_layer=norm_layer, dropout=dropout),
                FeedForward(dim, mlp_dim, norm_layer=norm_layer, dropout=dropout)
            ]))
        self.norm = norm_layer(dim)

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x_attn = attn(x, mask=mask)
            x = x_attn + x
            x = ff(x) + x
        return self.norm(x)

class TransMILWithSequencePrediction(nn.Module):
    def __init__(self, *, 
        num_classes: int, sequence_length: int, input_dim: int = 1024, bag_size: int = 512,
        dim: int = 512, depth: int = 2, heads: int = 8, dim_head: int = 64, mlp_dim: int = 2048,
        lstm_hidden_dim: int = 256, lstm_layers: int = 1,
        dropout: float = 0., emb_dropout: float = 0.
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.bag_size = bag_size

        # Transform each feature vector in the bag
        self.fc = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.GELU())
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer for feature extraction across the bag
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, nn.LayerNorm, dropout)
        
        # LSTM layer for autoregressive sequence prediction based on bag representation
        self.lstm = nn.LSTM(input_size=dim + num_classes, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        
        # Final layer to produce predictions at each step
        self.mlp_head = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x, lens=None):
        # x has shape (batch_size, bag_size, input_dim), where bag_size = 512 and input_dim = 1024
        b, n, d = x.shape

        # Map each feature vector in the bag to a lower-dimensional space
        x = self.fc(x)  # Shape (batch_size, bag_size, dim)
        
        # Pass through the Transformer to aggregate information across the bag
        x = self.transformer(x)  # Shape remains (batch_size, bag_size, dim)
        
        # Mean pooling over the bag dimension to get a single representation per bag
        x = x.mean(dim=1)  # Shape (batch_size, dim)

        # Initialize LSTM hidden state
        lstm_hidden = None

        # Autoregressive feedback: iteratively pass predictions as input to the next step
        predictions = []
        prev_output = torch.zeros(b, 1, self.num_classes, device=x.device)  # Start with zeros for the initial prediction

        for _ in range(self.sequence_length):
            # Concatenate previous output with the bag representation
            lstm_input = torch.cat((x.unsqueeze(1), prev_output), dim=-1)  # Shape (batch_size, 1, dim + num_classes)
            
            # Pass through LSTM
            lstm_out, lstm_hidden = self.lstm(lstm_input, lstm_hidden)  # Shape (batch_size, 1, lstm_hidden_dim)
            
            # Predict for this time step
            output = self.mlp_head(lstm_out.squeeze(1))  # Shape (batch_size, num_classes)
            predictions.append(output)

            # Update previous output for the next time step
            prev_output = output.unsqueeze(1)  # Shape (batch_size, 1, num_classes)

        # Stack predictions for the entire sequence
        return torch.stack(predictions, dim=1)  # Shape (batch_size, sequence_length, num_classes)
