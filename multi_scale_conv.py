import torch
import torch.nn as nn

class MultiScaleConvModel(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super(MultiScaleConvModel, self).__init__()
        
        # First scale: Original resolution (H x W)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU()
        )
        
        # Second scale: (H/2 x W/2)
        self.downsample1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim*2),
            nn.ReLU(),
            nn.Conv3d(hidden_dim*2, hidden_dim*2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim*2),
            nn.ReLU()
        )
        
        # Third scale: (H/4 x W/4)
        self.downsample2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv3d(hidden_dim*2, hidden_dim*4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim*4),
            nn.ReLU(),
            nn.Conv3d(hidden_dim*4, hidden_dim*4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim*4),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, time, height, width)
        
        # First scale
        feat1 = self.conv1(x)  # Original resolution
        
        # Second scale
        x2 = self.downsample1(feat1)
        feat2 = self.conv2(x2)  # Half resolution
        
        # Third scale
        x3 = self.downsample2(feat2)
        feat3 = self.conv3(x3)  # Quarter resolution
        
        return feat1, feat2, feat3

# Example usage
def example_usage():
    # Example input dimensions
    batch_size = 4
    channels = 3  # RGB video
    time_frames = 16
    height = 224
    width = 224
    
    # Create model
    model = MultiScaleConvModel(in_channels=channels)
    
    # Create example input
    x = torch.randn(batch_size, channels, time_frames, height, width)
    
    # Forward pass
    feat1, feat2, feat3 = model(x)
    
    # Print output shapes
    print(f"Original scale feature shape: {feat1.shape}")
    print(f"Half scale feature shape: {feat2.shape}")
    print(f"Quarter scale feature shape: {feat3.shape}")

example_usage()
