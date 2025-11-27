from sam3.model_builder import build_sam3_image_model
import torch

model = build_sam3_image_model(
    bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
    device="cpu",
)

print(">>> Listing all modules containing 'proj' or 'attn'")
for name, module in model.named_modules():
    lname = name.lower()
    if ("proj" in lname) or ("attn" in lname) or ("qkv" in lname):
        print(name, "->", module.__class__.__name__)
