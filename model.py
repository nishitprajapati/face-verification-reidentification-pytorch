import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FaceEmbeddingModel(nn.Module):
    """ResNet18 that maps face images to normalized 512-D embeddings."""

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # ResNet18's final fully connected layer produces 512 features.
        self.backbone.fc = nn.Identity()
        self.embedding_dim = 512

    def forward(self, images):
        features = self.backbone(images)
        embeddings = F.normalize(features, p=2, dim=1)
        return embeddings


if __name__ == "__main__":
    model = FaceEmbeddingModel(pretrained=False)
    sample = torch.randn(2, 3, 224, 224)
    output = model(sample)

    print("Input shape:", sample.shape)
    print("Embedding shape:", output.shape)
    print("Embedding norms:", output.norm(p=2, dim=1).detach().tolist())
