#coding=utf-8

import uuid

trace_key = "x-trace_id"

def generate_trace_id():
    return uuid.uuid4().hex

class Context():
    def __init__():
        self.meta = {}
    def Set(self,k,v):
        self.meta[k] = v
    def get(self,k):
        if k in self.meta:
            return self.meta[k]
        else:
            return  None

    def set_trace_id(self,id:str):
        self.meta[trace_key] = id

    def get_trace_id(self)->str:
        if trace_key in self.meta:
            return self.meta[trace_key]
        else:
            return ""
        

def NewContext():
    ret = Context()
    ret.set(generate_trace_id())
    return ret

def NewContextFromHeader():
    pass