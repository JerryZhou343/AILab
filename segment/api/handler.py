# coding=utf-8
from loguru import logger
from tornado.web import RequestHandler
from app.service import ServiceInstance

class PromptSegment(RequestHandler):
    def post(self):
        self.write("hello")
        self.set_status(200)
        ServiceInstance.initializer()
