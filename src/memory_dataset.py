import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

class MemoryDataset(Dataset):
    def __init__(self, images: list[Image.Image], processor: TrOCRProcessor):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].convert("RGB")
        image_tensor: torch.Tensor = self.processor(image, return_tensors="pt").pixel_values[0]

        # create fake label
        label_tensor: torch.Tensor = self.processor.tokenizer(
            "",
            return_tensors="pt",
        ).input_ids[0]

        return {"idx": idx, "input": image_tensor, "label": label_tensor}
