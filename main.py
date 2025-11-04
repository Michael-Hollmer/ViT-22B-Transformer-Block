import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange

class ViT_22B_Block(torch.nn.Module):

    def __init__(self, input_features: int, embedding_dim: int, attention_heads: int):
        super().__init__()

        # Query, Key and Value projection. Biases are omitted, see paper page 3 in the paper
        self.query_projection = torch.nn.Linear(in_features=input_features, out_features=embedding_dim, bias=False) 
        self.key_projection = torch.nn.Linear(in_features=input_features, out_features=embedding_dim, bias=False)
        self.value_projection = torch.nn.Linear(in_features=input_features, out_features=embedding_dim, bias=False)

        # Biases in the layer norms are also omitted
        self.input_layer_norm = torch.nn.LayerNorm(normalized_shape=input_features, bias = False)
        self.query_layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim, bias = False)
        self.key_layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim, bias= False)

        self.attention = torch.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True)

        # Bias in the Linear layer that is parallel to the attention, is kept
        self.mlp = torch.nn.Linear(in_features=input_features, out_features=embedding_dim, bias=True) 
        self.gelu = torch.nn.GELU()

    def forward(self, x):

        input = x

        # Normalize input
        input_normalized = self.input_layer_norm(x)

        # Get query, key, value from normalized input
        query, key, value = self.query_projection(input_normalized), self.key_projection(input_normalized), self.value_projection(input_normalized)
        
        # Apply layer norm to query and key
        query, key = self.query_layer_norm(query), self.key_layer_norm(key)

        # Calculate attention values
        attention_output, _ = self.attention(query, key, value)

        # Run normalized input through MLP - this is the parallel operation that the ViT-22B architecture improves
        mlp_out = self.gelu(self.mlp(input_normalized))

        # Add attention output + mlp output. The input is also added because of the residual connection
        return attention_output + mlp_out + input


class ViT(torch.nn.Module):

    # These are small values, so we can train it on standard hardware and datasets
    def __init__(self, block_count = 8, patch_size=16, patch_count_per_side = 14, embedding_dim = 128, number_output_classes = 10):
        super().__init__()

        self.block_count = block_count
        self.patch_size = patch_size
        self.patch_count = patch_count_per_side
        self.embedding_dim = embedding_dim
        self.number_output_classes = number_output_classes

        self.patch_projection = torch.nn.Linear(in_features=patch_size**2 * 3, out_features=embedding_dim)

        # Create ViT-22 transformer blocks
        self.transformer_blocks = torch.nn.ModuleList([
            ViT_22B_Block(input_features=embedding_dim, embedding_dim=embedding_dim, attention_heads=4)
            for _ in range(block_count)
        ])
        
        # Learnable class token for classification
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Create learnable positional encoding
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, patch_count_per_side**2 + 1, embedding_dim))

        # Classification head
        self.classification_head = torch.nn.Linear(embedding_dim, number_output_classes)

    def forward(self, img):

        B, C, H, W = img.shape
        assert H == W == self.patch_count * self.patch_size

        patches = rearrange(img, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=self.patch_size, pw=self.patch_size)

        # Run through patch projection
        patches = self.patch_projection(patches)

        # Add learnable class token
        cls_token = self.cls_token.expand(B, -1, -1) 
        tokens = torch.cat([cls_token, patches], dim=1)

        # Add positional embedding
        pos_embedding = self.pos_embedding.expand(B, -1, -1) 
        tokens = tokens + pos_embedding

        for block in self.transformer_blocks:
            tokens = block(tokens)

        return self.classification_head(tokens[:, 0, :]) 

def train_vit_on_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Grayscale(num_output_channels=3), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    model = ViT(block_count=8, patch_size=16, patch_count_per_side=14, embedding_dim= 128, number_output_classes=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(5):  # 5 epochs for demo
        model.train()
        total_loss = 0
        step = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            total_loss += loss.item()
            print("Loss:", total_loss / step, end='\r')


        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}, Loss: {avg_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_vit_on_mnist()