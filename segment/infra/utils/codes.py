#coding=utf-8


def unmarshal(obj,data):
    for k, v in obj.__dict__.items():
        if k in data:
            if isinstance(v, object) and hasattr(v,'unmarshal'):
                v.unmarshal(data[k])
            else:
                obj.__dict__[k] = data[k]

def marshal(obj:object)->str:
    pass