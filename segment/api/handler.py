# coding=utf-8
from loguru import logger
from tornado.web import RequestHandler

class SegmentPrompt(RequestHandler):
    def post(self):
        self.write("hello")
        self.set_status(200)
