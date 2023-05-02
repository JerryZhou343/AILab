# coding=utf-8
from loguru import logger
from tornado.web import RequestHandler
from app.service import ServiceInstance
from PIL import Image

class PromptSegment(RequestHandler):
    def post(self):
        file_path = "/home/jerry/go/src/github.com/JerryZhou343/AILab/demo/base/01.jpg"
        image_pil =  Image.open(file_path).convert("RGBA")
        ServiceInstance.segment_by_prompt(image_pil=image_pil,prompt_text="human")
        self.write("hello")
        self.set_status(200)




class InactiveSegment(RequestHandler):
    def post(self):
        pass