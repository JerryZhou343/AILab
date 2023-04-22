# coding=utf-8

from tornado.web import Application, RequestHandler
from tornado.routing import Rule, PathMatches, Router
from tornado.httpserver import httputil
from api.handler import Helloworld

_handlers = [
    Rule(PathMatches("/hello_world"),Helloworld)
]



