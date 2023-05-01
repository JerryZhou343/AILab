#coding=utf-8
import os
import sys
import torch
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
from infra.config.config import Models



class ApplicationService():
    def initializer(self, config:Models):
        self.build_dion_model_from_pth(config)

    def build_sam_model_from_pth(self,path:str):
        pass
    def build_dion_model_from_pth(self,config:Models):
        ''''''
        args = SLConfig.fromfile(sys.path[0]+config.dino_conf_file)
        args.device = config.device
        model = build_model(args)
        checkpoint = torch.load(config.dino_check_point_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        self.dino_model = model 

