from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .configs import paths
from .configs import constants
from src.yaml.config import Config
from src.logger.logger import Logger
from .custom_decoder import CustomTrOCRForCausalLM

log = Logger("account_ocr_service", "util", Config.get("logs.path"), Config.get("logs.level"))

def load_processor() -> TrOCRProcessor:
    return TrOCRProcessor.from_pretrained(paths.trocr_repo)


def load_model(from_disk: bool, use_custom_decoder: bool = False) -> VisionEncoderDecoderModel:
    if from_disk:
        # assert paths.trocr_repo.exists(), f"No model existing at {paths.trocr_repo}"
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(paths.model_path)
        log.debug(f"Loaded local model from {paths.model_path}")

        if use_custom_decoder:
            decoder_config = model.decoder.config
            custom_decoder = CustomTrOCRForCausalLM(decoder_config)
            model.decoder = custom_decoder

            from    .torch import load_model
            load_model(model, paths.model_path)
            log.debug(f"Loaded custom decoder from {paths.model_path}")
    else:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(paths.trocr_repo)
        log.debug(f"Loaded pretrained model from huggingface ({paths.trocr_repo})")

    log.debug(f"Using device {constants.device}.")
    model.to(constants.device)
    return model


def init_model_for_training(model: VisionEncoderDecoderModel, processor: TrOCRProcessor):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
