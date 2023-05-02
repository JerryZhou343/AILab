# coding=utf-8
import yaml


def unmarshal(obj,data):
    for k, v in obj.__dict__.items():
        if k in data:
            if isinstance(v, object) and hasattr(v,'unmarshal'):
                v.unmarshal(data[k])
            else:
                obj.__dict__[k] = data[k]

class Log:
    def __init__(self):
        self.level = "INFO"
        self.output = "./output"
        self.unmarshal=unmarshal.__get__(self)
    def __str__(self):
        return f"level:{self.level},output:{self.output}"

class Http():
    def __init__(self):
        self.port = 8080
        self.host = "127.0.0.1"
        self.unmarshal = unmarshal.__get__(self)

class Config:
    def __init__(self):
        self.unmarshal = unmarshal.__get__(self)
        self.log = Log()
        self.http = Http()
        self.models = Models()

class Models:
    def __init__(self):
        self.unmarshal = unmarshal.__get__(self)
        self.sam_onnx_path = ""
        self.sam_check_point_path = ""
        self.dino_check_point_path = ""
        self.device = "cpu"
        self.dino_conf_file=""
        self.mask_threshold = 0.6


def load_config_file(path:str)->Config:
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        conf = Config()
        conf.unmarshal(data)
        return conf