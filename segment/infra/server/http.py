#coding=utf-8
import asyncio

import tornado.ioloop
from tornado.web import Application
from tornado.routing import RuleRouter
from tornado.httpserver import HTTPServer
from infra.config.config import Http
from infra.router.stub import _handlers
class HttpServer():
    def __init__(self,opt:Http):
        self.opt = opt
        self.server = Application(_handlers)
    def run(self):
        self.server.listen(port=self.opt.port, address=self.opt.host)
        tornado.ioloop.IOLoop.current().start()