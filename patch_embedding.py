import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossScalePatchEmbedding(nn.Module):
    def __init__(self, 
                 scale1_channels, 
                 scale2_channels, 
                 scale3_channels, 
                 embed_dim=256, 
                 patch_size=4):
        """
        Cross-scale patch embedding module
        
        Args:
            scale1_channels (int): Number of channels in the first (original) scale
            scale2_channels (int): Number of channels in the second (half) scale
            scale3_channels (int): Number of channels in the third (quarter) scale
            embed_dim (int): Dimension to embed all scales into
            patch_size (int): Size of patches to create
        """
        super(CrossScalePatchEmbedding, self).__init__()
        
        # Projection layers for each scale to match embedding dimension
        self.scale1_proj = nn.Conv3d(scale1_channels, embed_dim, kernel_size=1)
        self.scale2_proj = nn.Conv3d(scale2_channels, embed_dim, kernel_size=1)
        self.scale3_proj = nn.Conv3d(scale3_channels, embed_dim, kernel_size=1)
        
        # Patch embedding layers
        self.patch_embed1 = self._create_patch_embedding(embed_dim, patch_size)
        self.patch_embed2 = self._create_patch_embedding(embed_dim, patch_size * 2)
        self.patch_embed3 = self._create_patch_embedding(embed_dim, patch_size * 4)
        
        # Optional learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(3))
        
    def _create_patch_embedding(self, channels, patch_size):
        """
        Create a patch embedding layer
        
        Args:
            channels (int): Number of output channels
            patch_size (int): Size of patches
        
        Returns:
            nn.Sequential: Patch embedding layers
        """
        return nn.Sequential(
            nn.Conv3d(channels, channels, 
                      kernel_size=(1, patch_size, patch_size), 
                      stride=(1, patch_size, patch_size)),
            nn.LayerNorm([channels, None, None, None])
        )
    
    def forward(self, scale1_feat, scale2_feat, scale3_feat):
        """
        Forward pass for cross-scale patch embedding
        
        Args:
            scale1_feat (torch.Tensor): Features from original scale
            scale2_feat (torch.Tensor): Features from half scale
            scale3_feat (torch.Tensor): Features from quarter scale
        
        Returns:
            torch.Tensor: Concatenated and weighted patch embeddings
        """
        # Project each scale to embedding dimension
        proj1 = self.scale1_proj(scale1_feat)
        proj2 = self.scale2_proj(scale2_feat)
        proj3 = self.scale3_proj(scale3_feat)
        
        # Create patch embeddings for each scale
        patch1 = self.patch_embed1(proj1)
        patch2 = self.patch_embed2(proj2)
        patch3 = self.patch_embed3(proj3)
        
        # Reshape patches to a consistent format
        batch_size = proj1.size(0)
        patch1 = patch1.flatten(2)  # Flatten spatial and temporal dimensions
        patch2 = patch2.flatten(2)
        patch3 = patch3.flatten(2)
        
        # Apply learnable scale weights
        weighted_patches = [
            patch1 * self.scale_weights[0],
            patch2 * self.scale_weights[1],
            patch3 * self.scale_weights[2]
        ]
        
        # Concatenate patches
        combined_patches = torch.cat(weighted_patches, dim=-1)
        
        return combined_patches

# Example usage
def example_usage():
    # Simulate features from multi-scale convolution model
    batch_size = 4
    scale1_channels = 64
    scale2_channels = 128
    scale3_channels = 256
    
    # Create example input features
    scale1_feat = torch.randn(batch_size, scale1_channels, 16, 224, 224)
    scale2_feat = torch.randn(batch_size, scale2_channels, 16, 112, 112)
    scale3_feat = torch.randn(batch_size, scale3_channels, 16, 56, 56)
    
    # Create cross-scale patch embedding module
    cross_scale_embedding = CrossScalePatchEmbedding(
        scale1_channels=scale1_channels,
        scale2_channels=scale2_channels,
        scale3_channels=scale3_channels
    )
    
    # Forward pass
    combined_patches = cross_scale_embedding(scale1_feat, scale2_feat, scale3_feat)
    
    print(f"Combined patches shape: {combined_patches.shape}")

# Run example if script is executed directly
if __name__ == "__main__":
    example_usage()
