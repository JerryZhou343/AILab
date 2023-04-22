# coding=utf-8

from tornado.web import Application, RequestHandler
from tornado.routing import Rule, PathMatches, Router
from tornado.httpserver import httputil
from  api.handler import *

_handlers = [
    Rule(PathMatches("/prompt_segment"),PromptSegment)
]



