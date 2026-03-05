### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union

### External Imports ###
import torch as tc
from torch.nn import Module
from torchvision import transforms
import timm


### Internal Imports ###
import slide_encoder

########################



def parse_model_raw():
    encoder = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
    raw_model_path = "/home/mw/Projects/BigPicture/CBIR/models/provgigapath_slide_raw.pkl"
    with open(raw_model_path, "wb") as f:
        tc.save(encoder, f)
    # tc.save(encoder, raw_model_path)
    print(f"Raw model saved to {raw_model_path}")
    
    
def parse_model_torchscript():
    encoder = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536).to("cuda:0")
    example_embedding = tc.randn((1, 8, 1536), dtype=tc.float32).to("cuda:0")
    example_coords = tc.randint(low=0, high=1024, size=(1, 8, 2)).to(tc.float32).to("cuda:0")
    output = encoder(example_embedding, example_coords)
    print(f"Output length: {len(output)}")
    print(f"Output shape: {output[0].shape}")
    # encoder = tc.jit.trace(encoder, example_inputs=[example_embedding, example_coords])
    # raw_model_path = "/home/mw/Projects/BigPicture/CBIR/models/provgigapath_slide_jit.pth"
    # tc.jit.save(encoder, raw_model_path)
    # print(f"Raw model saved to {raw_model_path}")

    
    
    
if __name__ == "__main__":
    parse_model_raw()
    # parse_model_torchscript()
    pass