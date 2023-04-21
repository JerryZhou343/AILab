# coding=utf-8
import yaml


class Log:
    def __init__(self):
        self.level = "INFO"
        self.output = "./output"

    def unmarshal(self, data):
        for k, v in self.__dict__.items():
            if k in data:
                self.__dict__[k] = data[k]

class Config:
    def __init__(self):
        self.log = Log()

    def unmarshal(self, data):
        for k, v in self.__dict__.items():
            if k in data:
                if isinstance(v, object):
                    v.unmarshal(data[k])
                else:
                    self.__dict__[k] = data[k]


def load_config_file(path:str)->Config:
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        conf = Config()
        conf.unmarshal(data)
        return conf