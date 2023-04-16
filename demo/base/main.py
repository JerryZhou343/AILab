#/usr/bin/env python3
#coding=utf-8
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
from scipy.ndimage import binary_dilation
#from modules.devices import device, torch_gc, cpu
#from modules.safe import unsafe_torch_load, load

model_dir = "/home/jerry/workbench/download"
dino_batch_dest_dir="/home/jerry/go/src/github.com/JerryZhou343/AILab/demo/base/" 
input_image_path = "/home/jerry/go/src/github.com/JerryZhou343/AILab/demo/base/20230415145253.png"
device = "cpu"
dino_batch_save_mask = True
dino_batch_save_image_with_mask=True
batch_dilation_amt= 10
dino_batch_output_per_image = 1

def dilate_mask(mask, dilation_amt):
    # Create a dilation kernel
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)

    # Dilate the image
    dilated_binary_img = binary_dilation(mask, dilation_kernel)

    # Convert the dilated binary numpy array back to a PIL image
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)

    return dilated_mask, dilated_binary_img

def show_boxes(image_np, boxes, color=(255, 0, 0, 255), thickness=2, show_index=False):
    if boxes is None:
        return image_np

    image = copy.deepcopy(image_np)
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (w, h), color, thickness)
        if show_index:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(idx)
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(image, text, (x, y+textsize[1]), font, 1, color, thickness)

    return image

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)

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


def load_sam_model(sam_checkpoint):
    model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
    sam_checkpoint = os.path.join(model_dir, sam_checkpoint)
    #torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    #torch.load = load
    return sam


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

    boxes_filt = dino_predict_internal(input_image,dino_model,"eyes,neck,face",0.3)

    print(type(boxes_filt))

    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=(dino_batch_output_per_image == 1),
    )
    
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    #boxes_filt = boxes_filt.cpu().numpy().astype(int)

    filename, ext = os.path.splitext(os.path.basename(input_image_path))

    #for idx, mask in enumerate(masks):
        #blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
    merged_mask = np.any(masks, axis=0)
    #if batch_dilation_amt:
    #    _, merged_mask = dilate_mask(merged_mask, batch_dilation_amt)
    image_np_copy = copy.deepcopy(image_np)
    image_np_copy[~merged_mask] = np.array([0, 0, 0, 0])
    output_image = Image.fromarray(image_np_copy)
    output_image.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_output{ext}"))
        #if dino_batch_save_mask:
        #    output_mask = Image.fromarray(merged_mask)
        #    output_mask.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_mask{ext}"))
        #if dino_batch_save_image_with_mask:
        #    output_blend = Image.fromarray(blended_image)
        #    output_blend.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_blend{ext}"))
    
    #if shared.cmd_opts.lowvram:
    #    sam.to("cpu")
    #gc.collect()
    #torch_gc()
    
    #return "Done"
                #cropped_image.save(f"path/to/your/output_{i}.jpg") 