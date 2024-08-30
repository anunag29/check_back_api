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
from .util import post_process


app = FastAPI()
log = Logger("cheque_back_ocr_service", "api", Config.get("logs.path"), Config.get("logs.level"))

# Define the directory where the images will be saved
UPLOAD_DIRECTORY = "input_image"
# OUTPUT_DIR = "/app/output"

model = TrocrPredictor(use_custom_decoder=True)

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
    ocr_results = []
    try:
        # Create a directory for input image
        upload_dir = f"{UPLOAD_DIRECTORY}"
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        image_path = os.path.join(upload_dir, image_file.filename)
        with open(image_path, "wb") as f:
            f.write(await image_file.read())

        image = Image.open(image_path).convert("RGB")
        latencies = {}
        # Use EasyOCR to extract bounding boxes
        start_time = time.time()
        bounding_boxes = easyocr_model.extract_bounding_boxes(image)
        process_time = time.time() - start_time
        latencies["easyocr_process_latency"] = process_time
        
        i=1
        TrOCR_latency_tot = 0
        for j,bbox in enumerate(bounding_boxes):
            # Crop image using the bounding box
            cropped_image = easyocr_model.crop_image(image, bbox)
            # save_path = os.path.join(OUTPUT_DIR,image_file.filename[:-4])
            # cropped_image.save(os.path.join(save_path, f"crop{j}.jpeg"))
            width, _ = cropped_image.size
            if (200 <= width and width <= 800): #only passing relavent crops i.e 200 <= width <= 800 
                start_time = time.time()
                # Use TrOCR to perform OCR on the cropped image
                trocr_text, trocr_conf = list(model.predict_images([cropped_image]))[0]
                process_time = time.time() - start_time
                latencies[f"TrOCR_process_crop{i}_latency"] = process_time
                i =+ 1
                TrOCR_latency_tot =+ process_time
                log.info(f"Detected {trocr_text} with confidence: {trocr_conf}")
                log.info(f"bbox_info {bbox}")
                # Combine results from EasyOCR and TrOCR
                ocr_results.append({
                    # 'easyocr_text': bbox_info['text'],s
                    'text': trocr_text,
                    'bbox': str(bbox),
                    'text_conf': trocr_conf
                })
            
        #post-processing
        ocr_results = post_process(ocr_results)

        latencies["TrOCR_latency_total"] = TrOCR_latency_tot
        ocr_results.append(latencies)

        os.remove(image_path)
        # log.info(f"ocr_results {ocr_results}")
        return ocr_results
        
    except Exception as ex:
        log.error(f'Error: {ex}', ex)
        return {'ocr_results': ocr_results, 'error': str(ex)}

@app.exception_handler(ChequeException)
async def handle_custom_error(request: Request, ex: ChequeException):
    log.error(f'Error: {ex}', ex)
    return JSONResponse(content={"error": str(ex)}, status_code = ex.status_code)

@app.exception_handler(Exception)
async def handle_uncaught_exception(request: Request, ex: Exception):
    log.error(f'Error: {ex}', ex)
    return JSONResponse(content={"error": str(ex)}, status_code=500)