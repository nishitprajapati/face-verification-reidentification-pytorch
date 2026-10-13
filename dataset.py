import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class FaceIdentityDataset(Dataset):
    """Loads face images and their identity labels from a split JSON file."""

    def __init__(self, split_json):
        self.split_json = Path(split_json)

        with self.split_json.open("r", encoding="utf-8-sig") as file:
            self.records = json.load(file)

        self.identity_names = sorted(
            {record["identity"] for record in self.records},
            key=lambda value: int(value),
        )
        self.identity_to_label = {
            identity: index
            for index, identity in enumerate(self.identity_names)
        }

        self.transform = IMAGE_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = Path(record["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = self.identity_to_label[record["identity"]]
        return image, label


if __name__ == "__main__":
    dataset = FaceIdentityDataset("data/splits/train.json")
    image, label = dataset[0]

    print("Training images:", len(dataset))
    print("Training identities:", len(dataset.identity_names))
    print("Image tensor shape:", image.shape)
    print("Identity label:", label)
    print("Image tensor dtype:", image.dtype)
