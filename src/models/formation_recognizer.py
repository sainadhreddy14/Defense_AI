"""
Formation recognition model using CNN to identify pattern features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ..utils.constants import UNIT_TYPES, GRID_HEIGHT, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE


class FormationDataset(Dataset):
    """Dataset for training the formation recognizer."""
    
    def __init__(self, formations, transform=None):
        """
        Initialize dataset with a list of formations.
        
        Args:
            formations: List of 3D numpy arrays representing formations
            transform: Optional transform to apply to samples
        """
        self.formations = formations
        self.transform = transform
    
    def __len__(self):
        """Return the number of formations in the dataset."""
        return len(self.formations)
    
    def __getitem__(self, idx):
        """Get a formation by index."""
        formation = self.formations[idx]
        
        # Convert to PyTorch tensor
        formation_tensor = torch.tensor(formation, dtype=torch.float32)
        
        if self.transform:
            formation_tensor = self.transform(formation_tensor)
        
        return formation_tensor


class FormationRecognizer(nn.Module):
    """
    CNN model for recognizing formation patterns.
    The model works as an autoencoder to extract meaningful embeddings
    from formation data without requiring labeled data.
    """
    
    def __init__(self, embedding_size=64):
        """
        Initialize the formation recognizer model.
        
        Args:
            embedding_size: Size of the embedding vector
        """
        super(FormationRecognizer, self).__init__()
        
        # Input shape: (batch_size, channels, height, width)
        # where channels = len(UNIT_TYPES)
        
        # Encoder layers
        self.conv1 = nn.Conv2d(len(UNIT_TYPES), 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate feature size after convolutions and pooling
        # For a 25x10 input:
        # After conv1 + pool1: 12x5
        # After conv2 + pool2: 6x2 (rounded down)
        feature_height = GRID_HEIGHT // 4
        feature_width = 10 // 4  # Base width for both sides
        feature_size = feature_height * feature_width * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, 256)
        self.fc2 = nn.Linear(256, embedding_size)
        
        # Decoder layers (for autoencoder training)
        self.fc3 = nn.Linear(embedding_size, 256)
        self.fc4 = nn.Linear(256, feature_size)
        
        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, len(UNIT_TYPES), kernel_size=2, stride=2)
        
        self.embedding_size = embedding_size
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Reconstructed formation and embedding
        """
        # Input tensor has shape [batch_size, height, width, channels]
        # Rearrange to [batch_size, channels, height, width] for convolution
        x = x.permute(0, 3, 1, 2)
        
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        
        # Generate embedding
        x = F.relu(self.fc1(x))
        embedding = self.fc2(x)
        
        # Decoder (for autoencoder training)
        x = F.relu(self.fc3(embedding))
        x = F.relu(self.fc4(x))
        
        # Reshape back to [batch_size, channels, height/4, width/4]
        feature_height = GRID_HEIGHT // 4
        feature_width = 10 // 4
        x = x.reshape(batch_size, 64, feature_height, feature_width)
        
        # Upsample
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # Sigmoid for 0-1 values
        
        # Ensure exact dimensions match the input
        # Get current size
        _, ch, h, w = x.shape
        
        # Resize if needed to match input dimensions
        if h != GRID_HEIGHT or w != 10:
            # Use interpolate to resize to exact dimensions
            x = F.interpolate(x, size=(GRID_HEIGHT, 10), mode='nearest')
        
        # Rearrange back to [batch_size, height, width, channels]
        x = x.permute(0, 2, 3, 1)
        
        return x, embedding
    
    def get_embedding(self, formation):
        """
        Extract the embedding from a formation without reconstructing it.
        
        Args:
            formation: Formation tensor or numpy array
            
        Returns:
            Embedding vector
        """
        # Handle numpy input
        if isinstance(formation, np.ndarray):
            formation = torch.tensor(formation, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(formation.shape) == 3:
            formation = formation.unsqueeze(0)
        
        # Switch to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Get embedding (forward pass through encoder only)
            x = formation.permute(0, 3, 1, 2)
            
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)
            
            x = F.relu(self.fc1(x))
            embedding = self.fc2(x)
        
        # Return as numpy array if single sample
        if embedding.size(0) == 1:
            return embedding.squeeze(0).cpu().numpy()
        
        return embedding.cpu().numpy()


def train_formation_recognizer(data_collector, epochs=TRAINING_EPOCHS):
    """
    Train the formation recognition model.
    
    Args:
        data_collector: BattleDataCollector instance
        epochs: Number of training epochs
        
    Returns:
        Trained FormationRecognizer model
    """
    try:
        # Get training data
        battles = data_collector.get_training_data(limit=5000)
        
        # Extract formations
        enemy_formations = [battle["enemy_formation"] for battle in battles]
        home_formations = [battle["home_formation"] for battle in battles]
        
        # Combine both sides
        formations = enemy_formations + home_formations
        
        print(f"Training on {len(formations)} formations")
        
        # Verify dimensions of all formations
        for i, formation in enumerate(formations):
            if formation.shape != (GRID_HEIGHT, 10, len(UNIT_TYPES)):
                print(f"Warning: Formation {i} has shape {formation.shape}, expected ({GRID_HEIGHT}, 10, {len(UNIT_TYPES)})")
                # Fix the shape if needed
                fixed_formation = np.zeros((GRID_HEIGHT, 10, len(UNIT_TYPES)))
                h, w, c = min(formation.shape[0], GRID_HEIGHT), min(formation.shape[1], 10), min(formation.shape[2], len(UNIT_TYPES))
                fixed_formation[:h, :w, :c] = formation[:h, :w, :c]
                formations[i] = fixed_formation
        
        # Create dataset and dataloader
        dataset = FormationDataset(formations)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize model
        model = FormationRecognizer()
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                # Ensure batch has correct shape
                if batch.shape[1:] != (GRID_HEIGHT, 10, len(UNIT_TYPES)):
                    print(f"Warning: Batch has shape {batch.shape}, expected (batch_size, {GRID_HEIGHT}, 10, {len(UNIT_TYPES)})")
                    continue
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass (autoencoder)
                try:
                    reconstructed, _ = model(batch)
                    
                    # Check if reconstructed and batch have the same shape
                    if reconstructed.shape != batch.shape:
                        print(f"Shape mismatch: reconstructed {reconstructed.shape} vs batch {batch.shape}")
                        # Resize reconstructed to match batch
                        reconstructed = F.interpolate(
                            reconstructed.permute(0, 3, 1, 2),  # Change to channels-first format
                            size=(GRID_HEIGHT, 10),
                            mode='nearest'
                        ).permute(0, 2, 3, 1)  # Change back to channels-last format
                    
                    # Calculate loss
                    loss = criterion(reconstructed, batch)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Error during training: {e}")
                    continue
            
            # Print progress
            avg_loss = total_loss / max(1, len(dataloader))
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Early stopping if loss is very low
            if avg_loss < 0.001:
                print("Loss is very low, stopping early")
                break
        
        # Set model to evaluation mode
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error in train_formation_recognizer: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Simple test of model architecture
    model = FormationRecognizer()
    
    # Create a random formation
    formation = np.random.rand(25, 10, len(UNIT_TYPES))
    formation_tensor = torch.tensor(formation, dtype=torch.float32).unsqueeze(0)
    
    # Forward pass
    reconstructed, embedding = model(formation_tensor)
    
    print(f"Input shape: {formation_tensor.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Embedding shape: {embedding.shape}")
    
    # Test get_embedding
    embedding = model.get_embedding(formation)
    print(f"Embedding shape from get_embedding: {embedding.shape}") 