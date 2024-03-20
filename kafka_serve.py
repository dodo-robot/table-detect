import ray
import torch
import cv2
import numpy as np 
import os
from producer import RayProducer
import s3fs
from pyarrow.fs import PyFileSystem, FSSpecHandler
from ray import serve
import pypdfium2 as pdfium
from io import BytesIO

from fastapi.responses import Response, JSONResponse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ray.init()

from configparser import ConfigParser
config_parser = ConfigParser()
config_parser.read("conf.ini")
config = dict(config_parser['default'])

kafka = {
  'consumer': config,
  'producer': config
}

producer = RayProducer.options(name='producer').remote(kafka['producer'], 'tables')

def bytes2img(file_bytes):
    nparr = np.frombuffer(pdf_bytes_to_jpeg(file_bytes['bytes']), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # You may need to further process the image data here if necessary
    # For example, converting it to a different format or encoding
    return {
        "path": file_bytes["path"],
        "transformed": [image]
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

def load_images_from_s3(bucket_url):
    minio_endpoint = 'http://minio.triton.svc.cluster.local:9000'
    minio_access_key = 'Altilia.2021'
    minio_secret_key = 'Altilia.2021'
    
    fs = s3fs.S3FileSystem(
        key=minio_access_key,
        secret=minio_secret_key,
        client_kwargs={
            'endpoint_url': minio_endpoint
        }
    )

    pa_fs = PyFileSystem(FSSpecHandler(fs))

    ds = ray.data.read_binary_files(bucket_url,
        include_paths=True,
        partition_filter=ray.data.datasource.FileExtensionFilter("pdf"),
        filesystem=pa_fs)
    
    return ds



@ray.remote(num_cpus=1)
def perform_table_detection(data, created_at):
    result = []
    handle = serve.get_app_handle("table_detect") 
    
    for item in data.iter_rows():
        path = item['path']
        img = item['bytes']

        detected = handle.detect.remote(img).result()
        if detected.status_code == 200:
            tables = json.loads(detected.body.decode('utf-8'))
            res = {
                "path": path,
                "tables": tables,
                "created_at": created_at
            }
            result.append(res) 
    
    ray.get(producer.produce.remote({"arrayTables": result}))
    print("sent tables")
    return result



import json
from datetime import datetime, timezone 
from confluent_kafka import Consumer, KafkaError
# Create a Kafka consumer
consumer = Consumer(kafka["consumer"])
consumer.subscribe(['splitted_files'])

# Continuously consume and process Kafka messages

result_refs = [] 
MAX_NUM_RUNNING_TASKS = 6
NUM_TASKS = 0

while True:
    msg = consumer.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            print('Reached end of partition')
        else:
            print('Error while consuming message: {}'.format(msg.error()))
    else:
        print('Received message: {}'.format(msg.value()))
        # Extract the bucket_url from the Kafka message
        msg_value = msg.value().decode('utf-8')
        # Parse the JSON string to extract the bucket_url
        data = json.loads(msg_value)
        bucket_url = data.get('documentUri')
        created_at = data.get('createdAt')
        created_at_datetime = datetime.fromtimestamp(int(created_at)/1000, tz=timezone.utc)  # Convert the Unix timestamp to a datetime object
        formatted_created_at = created_at_datetime.strftime('%Y-%m-%dT%H:%M:%S.%fZ')  # Format the datetime object as desired
        ds = load_images_from_s3(data.get('tenant') + "/" + bucket_url)
        result_refs.append(perform_table_detection.remote(ds, formatted_created_at))
        
        if len(result_refs) >= MAX_NUM_RUNNING_TASKS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1) 
 

