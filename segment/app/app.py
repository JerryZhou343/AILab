#coding=utf-8
import os
import sys
import torch
from loguru import logger
import numpy as np
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
from infra.config.config import Models
from infra.utils.files import check_file
from segment_anything import build_sam,SamPredictor

from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from .utils import show_masks, show_boxes
from PIL import Image
import copy

class ApplicationService():
    def initializer(self, config:Models):
        self.config = config
        #self.build_dion_model_from_pth(config)
        self.build_sam_model(self.config)
        self.build_dino_model(self.config)


    def build_sam_model(self,config:Models):
        #self.sam_onnx_model = SamOnnxModel(config.sam_onnx_path,return_single_mask=True)
        #onnx_file_path = os.path.join(sys.path[0],config.sam_onnx_path)
        #if not check_file(onnx_file_path):
        #    logger.fatal(f"onnx model not exits. path:{onnx_file_path}")

        #self.ort_session = onnxruntime.InferenceSession(onnx_file_path)

        if not check_file(config.sam_check_point_path):
            logger.fatal(f"sam check point not exits.{config.sam_check_point_path}")

        sam_checkpoint = config.sam_check_point_path 
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=config.device)
        sam.eval()
        self.sam_model = sam

    def segment_by_prompt(self,image_pil,prompt_text):
        #1. dino 生成 prompt box
        boxes_filt = self.get_prompt_mask(prompt_text,image_pil,self.config.mask_threshold)

        #2. prompt box input to sam,get boxes
        masks,score = self.segment_image(image_pil,boxes_filt)
        #3. output 
        return self.create_mask_out(image_pil,masks,boxes_filt)
 

    def build_dino_model(self,config:Models):
        ''''''
        dino_conf_file = os.path.join(sys.path[0],config.dino_conf_file)
        logger.info(f"load config file path:{dino_conf_file}")
        args = SLConfig.fromfile(dino_conf_file)
        args.device = config.device
        model = build_model(args)

        if not check_file(config.dino_check_point_path):
            logger.fatal(f"check point file not exits{config.dino_check_point_path}")

        checkpoint = torch.load(config.dino_check_point_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        self.dino_model = model 

    def get_prompt_mask(self,prompt_text,image_pil,box_threshold):
        # 转换 dino image
        dino_image = self.load_dino_image(image_pil.convert("RGB"))

        # 提示词获得 mask prompt
        boxes_filt = self.get_grounding_output(dino_image, prompt_text, box_threshold)

        # 转换边框
        H, W = image_pil.size[1], image_pil.size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]


        return boxes_filt

    def segment_image(self,image_pil,boxes_filt):
        ''''''
        image_np = np.array(image_pil.convert("RGBA"))

        image_np_rgb = image_np[...,:3]
        # 图片预处理
        predictor = SamPredictor(self.sam_model)
        logger.info(f"device:{self.config.device}, boxes: {boxes_filt}")
        predictor.set_image(image_np_rgb)
        #
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        # 处理
        masks, score, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(self.config.device),
        multimask_output=False)
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        return masks,score

    def create_mask_out(self,image_pil, masks, boxes_filt):
        image_np = np.array(image_pil.convert("RGBA"))
        mask_images, masks_gallery, matted_images = [], [], []
        boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
        for mask in masks:
            masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
            blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
            mask_images.append(Image.fromarray(blended_image))
            image_np_copy = copy.deepcopy(image_np)
            image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
            matted_images.append(Image.fromarray(image_np_copy))
        return mask_images , masks_gallery , matted_images


    def load_dino_image(self,image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image
    
    def get_grounding_output(self,image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        image = image.to(self.config.device)

        with torch.no_grad():
            outputs = self.dino_model(image[None], captions=[caption])
        
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        return boxes_filt.cpu()