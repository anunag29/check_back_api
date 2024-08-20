import os
import time

from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, Request
from PIL import Image

from src.yaml.config import Config
from src.logger.logger import Logger
from src.exception.cheque_exception import ChequeException
from .trocr_predictor import TrocrPredictor
from .easyocr_detect import EasyOCRDetect

app = FastAPI()
log = Logger("cheque_back_ocr_service", "api", Config.get("logs.path"), Config.get("logs.level"))

# Define the directory where the images will be saved
UPLOAD_DIRECTORY = "input_image"

model = TrocrPredictor()

easyocr_model = EasyOCRDetect()

# @app.middleware("http")
# async def add_process_time_header(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     process_time = time.time() - start_time
#     response.headers["X-Process-Time"] = str(process_time)
#     log.info(f"API '{request.url.path}' took: {process_time} seconds")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    try:
        # Create a directory for input image
        upload_dir = f"{UPLOAD_DIRECTORY}"
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        image_path = os.path.join(upload_dir, image_file.filename)
        with open(image_path, "wb") as f:
            f.write(await image_file.read())

        image = Image.open(image_path).convert("RGB")
        # Use EasyOCR to extract bounding boxes
        bounding_boxes = easyocr_model.extract_bounding_boxes(image)
        
        ocr_results = []
        for bbox_info in bounding_boxes:
            # Crop image using the bounding box
            cropped_image = easyocr_model.crop_image(image, bbox_info['bbox'])
            # Use TrOCR to perform OCR on the cropped image
            trocr_text, trocr_conf = list(model.predict_images([cropped_image]))[0]
            log.info(f"Detected {trocr_text} with confidence: {trocr_conf}")
            log.info(f"bbox_info {bbox_info}")
        
            # Combine results from EasyOCR and TrOCR
            ocr_results.append({
                # 'easyocr_text': bbox_info['text'],s
                'text': trocr_text,
                'bbox': list(bbox_info['bbox']),
                'text_conf': trocr_conf
            })
            
        #post-processing
        text_seq = []
        for seq in ocr_results:
              text_seq.append(seq['text'])

        text_seq = sorted(text_seq, key=len)
        i=0
        while (len(text_seq[i]) < 10):
          i+=1

        phone = text_seq[i]
        acct = text_seq[i+1]

        phone = phone.replace(" ", "")
        acct = acct.replace(" ", "")

        ocr_results.append({
            'Phone': phone,
            'Account': acct
        })

        os.remove(image_path)
        # log.info(f"ocr_results {ocr_results}")
        return ocr_results
        
    except Exception as ex:
        log.error(f'Error: {ex}', ex)
        return {'ocr_results': [], 'error': str(ex)}

@app.exception_handler(ChequeException)
async def handle_custom_error(request: Request, ex: ChequeException):
    log.error(f'Error: {ex}', ex)
    return JSONResponse(content={"error": str(ex)}, status_code = ex.status_code)

@app.exception_handler(Exception)
async def handle_uncaught_exception(request: Request, ex: Exception):
    log.error(f'Error: {ex}', ex)
    return JSONResponse(content={"error": str(ex)}, status_code=500)