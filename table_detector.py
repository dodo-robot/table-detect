
import pyarrow.fs
import ray     
from src.TableDetectionModel import TableDetectionModel
from src.util import bytes2img
import os, json
# Set your AWS credentials (replace with your own credentials)
from typing import List, Dict
import numpy as np
from pydantic import BaseModel
import torch
from minio import Minio 
from mmdet.apis.inference import inference_detector, init_detector
import os
from src.util import send_msg_to_queue
# Set your AWS credentials (replace with your own credentials)
import numpy as np
import cv2
import pypdfium2 as pdfium
from io import BytesIO

# os.environ["RAY_memory_monitor_refresh_ms"] = "0"
ray.init()

def bytes2img(file_bytes):
    nparr = np.frombuffer(pdf_bytes_to_jpeg(file_bytes["bytes"]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # You may need to further process the image data here if necessary
    # For example, converting it to a different format or encoding
    return {
        "path": file_bytes["path"],
        "transformed": image
    }

def pdf_bytes_to_jpeg(pdf_bytes):
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf[0]  # load a page
    bitmap = page.render(
        scale=1,
        rotation=0,  
    )
    pil_image = bitmap.to_pil()
    jpeg_bytes = BytesIO()
    pil_image.save(jpeg_bytes, format="JPEG")
    jpeg_bytes.seek(0)
    return jpeg_bytes.getbuffer().tobytes()


def notify_conductor(orchestratorRef, tenant, username) -> dict:
    payload = {
            "orchestratorRef": orchestratorRef,
            "taskName": 'wait_table'
        }
        
    send_msg_to_queue(os.getenv('PROCESS_REQUEST_QUEUE', "process-request-queue"), payload, 'NOTIFY_TASK', tenant, False, username)
    


class TableDetectionModel:
    def __init__(self):
            minio_endpoint = os.getenv("AWS_ENDPOINT_URL", "localhost:9000")
            minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "Altilia.2021") 
            minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "Altilia.2021") 
            model_bucket = os.getenv("MODEL_BUCKET", "triton") 
            model_name = os.getenv("MODEL_NAME", "mmdet-obj-det-financial-statements-table-detector") 
            model_weight = os.getenv("MODEL_WEIGHT", "epoch_12.pth") 
            model_config = os.getenv("MODEL_CONFIG", "deformable_detr_r50_16x2_50e_coco.py") 
            self.client = Minio(
                        minio_endpoint,
                        secure = False,
                        access_key = minio_access_key,
                        secret_key = minio_secret_key)
            # Download data of an object.
            self.client.fget_object(model_bucket, f"{model_name}/{model_weight}", "weights.pth")
            self.client.fget_object(model_bucket, f"{model_name}/{model_config}", "config.py")

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.table_detector = init_detector(
                "config.py",
                "weights.pth",
                device=self.device
            ) 

            self.classes = self.table_detector.CLASSES
    
    def __call__(self, input_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Define the per-batch inference code in `__call__`.
        batch = [image for image in input_batch["transformed"]]
        paths = [path for path in input_batch["path"]]
        return self.detect(self.table_detector, batch, paths, 0.3)
    

    def process_batch_results(self, results, paths, confidence_thr):
        pp_result = []

        for i, result in enumerate(results):
            b_k_result = []

            for label_id, preds_in_label in enumerate(result):
                for pred_in_label in preds_in_label:
                    if len(pred_in_label) != 5:
                        raise ValueError("Each prediction should have 5 elements")
                    x, y, z, w, confidence = pred_in_label

                    if confidence > confidence_thr:
                        b_k_result.append(DetectTableResult(path=paths[i], bbox=[x,y,z,w],label=self.classes[label_id],confidence=confidence).__dict__)

            pp_result.append(b_k_result)

        return pp_result

    def detect(self, detector, image, paths,  confidence):
        raw_results = inference_detector(detector, image)
        pp_results = self.process_batch_results(raw_results, paths, confidence)
        return { "predictions": pp_results } 


class DetectTableResult(BaseModel):
    path: str
    bbox: List[float]
    label: str
    confidence: float

    class Config:
        arbitrary_types_allowed = True


