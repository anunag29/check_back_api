from PIL import Image
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from .configs import constants
from .memory_dataset import MemoryDataset
from .util import load_model, load_processor


class TrocrPredictor:
    def __init__(self, use_local_model: bool = True, use_custom_decoder: bool = False):
        self.processor = load_processor()
        self.model = load_model(use_local_model, use_custom_decoder)

    def predict_for_image_paths(self, image_paths: list[str]) -> list[tuple[str, float]]:
        images = [Image.open(path) for path in image_paths]
        return self.predict_images(images)

    def predict_images(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        dataset = MemoryDataset(images, self.processor)
        dataloader = DataLoader(dataset, constants.batch_size)
        predictions, confidence_scores = self.predict(self.processor, self.model, dataloader)
        return zip([p[1] for p in sorted(predictions)], [p[1] for p in sorted(confidence_scores)])
    
    def predict(self, processor: TrOCRProcessor, model: VisionEncoderDecoderModel, dataloader: DataLoader) -> tuple[list[tuple[int, str]], list[float]]:
        output: list[tuple[int, str]] = []
        confidence_scores: list[tuple[int, float]] = []

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(dataloader):
                inputs: torch.Tensor = batch["input"].to(constants.device)

                generated_ids = model.generate(inputs, return_dict_in_generate=True, output_scores = True)
                generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)

                ids = [t.item() for t in batch["idx"]]
                output.extend(zip(ids, generated_text))

                # Compute confidence scores
                batch_confidence_scores = self.get_confidence_scores(generated_ids)
                confidence_scores.extend(zip(ids, batch_confidence_scores))

                return output, confidence_scores

    def get_confidence_scores(self, generated_ids) -> list[float]:
        # Get raw logits, with shape (examples,tokens,token_vals)
        logits = generated_ids.scores
        logits = torch.stack(list(logits),dim=1)

        # Transform logits to softmax and keep only the highest (chosen) p for each token
        logit_probs = F.softmax(logits, dim=2)
        char_probs = logit_probs.max(dim=2)[0]

        # Only tokens of val>2 should influence the confidence. Thus, set probabilities to 1 for tokens 0-2
        mask = generated_ids.sequences[:,:-1] > 2
        char_probs[mask] = 1

        # Confidence of each example is cumulative product of token probs
        batch_confidence_scores = char_probs.cumprod(dim=1)[:, -1]
        return [v.item() for v in batch_confidence_scores]
