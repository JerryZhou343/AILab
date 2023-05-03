# coding=utf-8
from loguru import logger
from tornado.web import RequestHandler
from app.service import ServiceInstance
from infra.utils.image import decode_to_pil,encode_to_base64
from infra.utils.codes import unmarshal
import json
import os

class PromptSegmentArgs(object):
    def __init__(self):
        self.unmarshal = unmarshal.__get__(self)
        self.prompt_text = "human"
        self.raw_image = ""
        self.format = ""
    def invalidate(self):
        if len(self.raw_image) == 0:
            raise Exception("invalid arguments")
        if len(self.prompt_text) == 0:
            self.prompt_text = "human"
        if len(self.format) == 0:
            self.format = "png"




class PromptSegment(RequestHandler):
    def post(self):
        try:
            #file_path = "/home/jerry/go/src/github.com/JerryZhou343/AILab/demo/base/01.jpg"
            self.prepare()
            image_pil =  decode_to_pil(self.args.raw_image)
            mask_images,masks_gallery,  matted_images = ServiceInstance.segment_by_prompt(image_pil=image_pil,prompt_text = self.args.prompt_text)
            ret_mask_images, ret_masks_gallery, ret_matted_images = [],[],[]
        
            for idx, mask in enumerate(mask_images):
                ret_mask_images.append(encode_to_base64(mask,self.args.format))
                #mask.save("./output/",f"{idx}.{self.args.format}")

            for  m in masks_gallery:
                ret_masks_gallery.append(encode_to_base64(m,self.args.format))

            for  m in matted_images:
                ret_matted_images.append(encode_to_base64(m,self.args.format))

            rsp = {
                "mask_images":ret_mask_images,
                "mask_gallery":ret_masks_gallery,
                "matted_images":ret_matted_images
            }
            
            self.write(json.dumps(rsp))
        #TODO(JerryZhou): 精细化错误处理
        except Exception as e:
            logger.error(e)
            self.set_status(500)
        else:
            self.set_status(200)

    def prepare(self):
        content_type = self.request.headers['Content-Type']
        content_type = content_type.lower()
        content_type = content_type.strip()
        if content_type == 'application/json' or content_type == 'application/x-json':
            #logger.info(self.request.body)
            data =  json.loads(self.request.body)
            self.args = PromptSegmentArgs()
            self.args.unmarshal(data)
            self.args.invalidate()
            return 
        raise Exception("not accept content type")