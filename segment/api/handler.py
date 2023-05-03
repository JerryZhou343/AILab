# coding=utf-8
from loguru import logger
from tornado.web import RequestHandler
from app.service import ServiceInstance
from infra.utils.image import decode_to_pil,encode_to_base64

class PromptSegment(RequestHandler):
    def post(self):
        file_path = "/home/jerry/go/src/github.com/JerryZhou343/AILab/demo/base/01.jpg"
        image_pil =  decode_to_pil(file_path)
        mask_images,masks_gallery,  matted_images = ServiceInstance.segment_by_prompt(image_pil=image_pil,prompt_text="human",file_path=file_path)

        self.set_status(200)




class InactiveSegment(RequestHandler):
    def post(self):
        pass