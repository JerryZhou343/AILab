#!/usr/bin/env python
from skimage import io,data
import argparse
import os
import cv2
import os
import copy
import numpy as np
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

model_dir = "/home/jerry/workbench/download"
dino_batch_dest_dir="/home/jerry/go/src/github.com/JerryZhou343/AILab/demo/base/" 
input_image_path = "/home/jerry/go/src/github.com/JerryZhou343/AILab/demo/base/20230415145253.jpg"
device = "cpu"
dino_batch_save_mask = 1

def get_grounding_output(model, image, caption, box_threshold):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    return boxes_filt.cpu()

def dino_predict_internal(input_image, dino_model, text_prompt, box_threshold):
    '''阈值过高可能导致无法检测到对象'''
    dino_image = load_dino_image(input_image.convert("RGB"))

    boxes_filt = get_grounding_output(
        dino_model, dino_image, text_prompt, box_threshold
    )

    H, W = input_image.size[1], input_image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    #gc.collect()
    #torch_gc()
    return boxes_filt

def load_sam_model(sam_checkpoint):
    model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
    sam_checkpoint = os.path.join(model_dir, sam_checkpoint)
    #torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    #torch.load = load
    return sam


def load_dino_image(image_pil):
    import groundingdino.datasets.transforms as T
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def load_dino_model(dino_checkpoint):
    args = SLConfig.fromfile("grd.cfg.py")
    args.device = device 
    dino = build_model(args)
    checkpoint =  torch.load(os.path.join(model_dir,dino_checkpoint), map_location="cpu")
    dino.load_state_dict(clean_state_dict(
        checkpoint['model']), strict=False)
    dino.to(device=device)
    dino.eval()
    return dino

if __name__ == "__main__":
    parser = argparse.ArgumentParser("example", add_help=True)
    #parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    #parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    sam = load_sam_model("sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)


    dino_model = load_dino_model("groundingdino_swinb_cogcoor.pth")


    args = parser.parse_args()

    input_image =  Image.open(input_image_path).convert("RGBA")
    image_np = np.array(input_image)    
    image_np_rgb = image_np[...,:3]

    boxes_filt = dino_predict_internal(input_image,dino_model,"human",0.3)

    #print(type(boxes_filt))

    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
    masks, score, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    print(type(masks)) 
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    for idx, mask in enumerate(masks):
        #blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        merged_mask = np.any(mask, axis=0)
        #if batch_dilation_amt:
        #    _, merged_mask = dilate_mask(merged_mask, batch_dilation_amt)
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~merged_mask] = np.array([0, 0, 0, 0])
        output_image = Image.fromarray(image_np_copy)
        output_image.save(os.path.join(dino_batch_dest_dir, f"11_{idx}_output.png"))
        if dino_batch_save_mask:
            output_mask = Image.fromarray(merged_mask)
            output_mask.save(os.path.join(dino_batch_dest_dir, f"11_{idx}_mask.png"))