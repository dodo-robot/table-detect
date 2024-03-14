import torch
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.responses import Response
from fastapi import FastAPI, HTTPException, Request
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from typing import List
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import torch
import os
from minio import Minio
from minio.sse import SseCustomerKey 
import pypdfium2 as pdfium
from uuid import uuid4
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

serve.start(detached=True, http_options={"location":"EveryNode"})

app = FastAPI()

class DetectionRequest(BaseModel):
    tenant: str
    filename: str

class DetectTableResult(BaseModel):
    path: str
    bbox: List[float]
    label: str
    confidence: float

    class Config:
        arbitrary_types_allowed = True

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, minio_endpoint, minio_access_key, minio_secret_key, object_detection_handle) -> None: 
        self.client = Minio(
            minio_endpoint,
            secure = False,
            access_key = minio_access_key,
            secret_key = minio_secret_key)
        
        self.handle: DeploymentHandle = object_detection_handle.options(
            use_new_handle_api=True,
        )
    
    def pdf_bytes_to_jpeg(self, pdf_bytes):
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

    def bytes2img(self, file_bytes):
        import cv2
        nparr = np.frombuffer(file_bytes, np.uint8)
        return [cv2.imdecode(nparr, cv2.IMREAD_COLOR)]
    
    @app.post(
        "/detect"
    )
    async def detect(self, bytes):
        try:
            # Read the content of the object into bytes
            # Process the PDF bytes
            detection = await self.handle.detect.remote(self.bytes2img(self.pdf_bytes_to_jpeg(bytes)))
            return JSONResponse(content=jsonable_encoder(detection))

        except Exception as e:
            print(e)
            raise HTTPException(status_code=400, detail="Failed to process the content as an image.")

        
@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 6},
)
class TableDetection:
  def __init__(self, bucket, name, weight, conf, minio_endpoint, minio_access_key, minio_secret_key):
    import torch
    import os
    from minio import Minio
    from minio.sse import SseCustomerKey
    from mmdet.apis.inference import init_detector
    
    self.client = Minio(
                minio_endpoint,
                secure = False,
                access_key = minio_access_key,
                secret_key = minio_secret_key)
    # Download data of an object.
    weight = f"{name}/{weight}"
    conf = f"{name}/{conf}"

    print(weight, conf)

    self.client.fget_object(bucket, weight, "weights.pth")
    self.client.fget_object(bucket, conf, "config.py")

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.table_detector = init_detector(
        "config.py",
        "weights.pth",
        device=self.device
    ) 

    self.classes = self.table_detector.CLASSES
    

  def detect(self, image, confidence=0.3):
        from mmdet.apis.inference import inference_detector
        # Convert PIL image to bytes
        raw_results = inference_detector(self.table_detector, image)
        pp_results = self.process_batch_results(raw_results, confidence)
        return pp_results[0]
  
  def get_classes(self):
      return self.classes
   
  def process_batch_results(self, results, confidence_thr):
      pp_result = []

      for result in results:
          b_k_result = []

          for label_id, preds_in_label in enumerate(result):
              for pred_in_label in preds_in_label:
                  if len(pred_in_label) != 5:
                      raise ValueError("Each prediction should have 5 elements")
                  x, y, z, w, confidence = pred_in_label

                  if confidence > confidence_thr:
                      b_k_result.append(DetectTableResult(path="",bbox=[x,y,z,w],label=self.classes[label_id],confidence=confidence).__dict__)

          pp_result.append(b_k_result)

      return pp_result


from pydantic import BaseModel

from ray.serve import Application


class ComposedArgs(BaseModel):
    bucket: str
    name: str
    weight: str 
    conf: str 
    minio_endpoint: str 
    minio_access_key: str 
    minio_secret_key: str

def typed_app_builder(args: ComposedArgs) -> Application:
    return APIIngress.bind(
        args.minio_endpoint, 
        args.minio_access_key, 
        args.minio_secret_key,
        TableDetection.bind( 
            args.bucket, 
            args.name, 
            args.weight, 
            args.conf, 
            args.minio_endpoint, 
            args.minio_access_key, 
            args.minio_secret_key
            )
    )

 





  

