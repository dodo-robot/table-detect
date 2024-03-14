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

def perform_table_detection(data):
    for item in data.iter_rows():
        path = item['path']
        img = item['transformed']
        # # Open the PDF document (assuming it's one page) 
        handle = serve.get_app_handle("table_detect")
        detected = ray.get(handle.detect.remote(img, path))
        # ray.get(table_detector.detect.remote(image, confidence=0.3))
        res = {
            "path": path,
            "tables": detected,
        }

        ray.get(producer.produce.remote(res))

    return res



import json
# Function to perform table detection on a Kafka message
def perform_table_detection_kafka(msg):
    # Extract the bucket_url from the Kafka message
    msg_value = msg.value().decode('utf-8')
    # Parse the JSON string to extract the bucket_url
    data = json.loads(msg_value)
    bucket_url = data.get('documentUri')
    ds = load_images_from_s3(data.get('tenant') + "/" + bucket_url)
    ds = ds.map(bytes2img)
    ray.remote(perform_table_detection).remote(ds)

# Kafka consumer configuration

from confluent_kafka import Consumer, KafkaError
# Create a Kafka consumer
consumer = Consumer(kafka["consumer"])
consumer.subscribe(['splitted_files'])

# Continuously consume and process Kafka messages
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
        perform_table_detection_kafka(msg)  # Process the message
