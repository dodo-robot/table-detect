import ray 
from typing import List
from pydantic import BaseModel

class DetectTableResult(BaseModel):
    path: str
    bbox: List[float]
    label: str
    confidence: float

    class Config:
        arbitrary_types_allowed = True

@ray.remote(num_gpus=1)
class RayTableDetector:
  def __init__(self, bucket, model, config):
    import torch
    from minio import Minio
    from minio.sse import SseCustomerKey
    from mmdet.apis.inference import init_detector
    client = Minio(
                "minio.triton.svc.cluster.local:9000",
                secure = False,
                access_key = 'Altilia.2021',
                secret_key = 'Altilia.2021')

    # Download data of an object.
    client.fget_object(bucket, model, "epoch_12.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.table_detector = init_detector(
        config,
        "epoch_12.pth",
        device=device
    ) 

    self.classes = self.table_detector.CLASSES
    


  def detect(self, image, path, confidence):
      from mmdet.apis.inference import inference_detector
      raw_results = inference_detector(self.table_detector, image)
      pp_results = self.process_batch_results(raw_results, path, confidence)
      return pp_results[0]
  
  def get_classes(self):
      return self.classes
   
  
  def process_batch_results(self, results, path, confidence_thr):
      pp_result = []

      for result in results:
          b_k_result = []

          for label_id, preds_in_label in enumerate(result):
              for pred_in_label in preds_in_label:
                  if len(pred_in_label) != 5:
                      raise ValueError("Each prediction should have 5 elements")
                  x, y, z, w, confidence = pred_in_label

                  if confidence > confidence_thr:
                      b_k_result.append(DetectTableResult(path=path, bbox=[x,y,z,w],label=self.classes[label_id],confidence=confidence).__dict__)

          pp_result.append(b_k_result)

      return pp_result




