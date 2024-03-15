import ray
# from src.utils import delivery_report, DetectTableResult
import json

def delivery_report(err, msg):
  if err is not None:
    print('Message delivery failed: {}'.format(err))
  else:
    print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

from typing import List

from pydantic import BaseModel
from datetime import datetime


class DetectTableResult(BaseModel):
    bbox: List[float]
    label: str
    confidence: float

    class Config:
        arbitrary_types_allowed = True
    

@ray.remote(num_cpus=0.5)
class RayProducer:
  def __init__(self, kafka, sink):
    from confluent_kafka import Producer
    self.producer = Producer(kafka)
    self.sink = sink

  def produce(self, data):
    self.producer.produce(self.sink, json.dumps(data).encode('utf-8'), callback=delivery_report)
    self.producer.poll(0)

  def destroy(self):
    self.producer.flush(30)

  
