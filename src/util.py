# Set your AWS credentials (replace with your own credentials)
import numpy as np
import cv2
import pypdfium2 as pdfium
from io import BytesIO


from message_broker_api.environment.rabbitmq import BlockingConnectionEnvFactory
from message_broker_api.producer.rabbitmq import RabbitMQProducer
from message_broker_api.queue_param.rabbitmq import RabbitMQQueueParam
from message_broker_api.message import Message


import os
import json


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




def send_msg_to_queue(queue_name: str, payload: dict, operation: str, tenant: str, queue_is_durable: bool = True,
                      user: str = ''):
    connection = BlockingConnectionEnvFactory.create_connection(os.getenv('MESSAGE_BROKER_TYPE', 'rabbitmq'))
    queue_param = RabbitMQQueueParam(queue_name, durable=queue_is_durable)
    producer = RabbitMQProducer(connection=connection, queue_param=queue_param)
    msg = {
        "dataJsonString": payload,
        "operation": operation,
        "tenant": tenant
    }
    msg_payload = {
        "dataJsonString": json.dumps(msg['dataJsonString']), 'requestContext': {'user': user, 'tenant': msg['tenant']},
        "operation": msg['operation'],
        "tenant": msg['tenant']
    }
    producer.send(message=Message(msg_payload))
    producer.close()