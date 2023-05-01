#!/bin/bash/env python3

import torch.onnx
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig


#Function to Convert to ONNX
def Convert_ONNX(input):

    # set the model to inference mode
    #model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(input,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "ImageClassifier.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


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
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    model = load_dino_model("/home/jerry/workbench/download/groundingdino_swinb_cogcoor.pth")

    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX()