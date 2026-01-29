import os
import pathlib
import torch
from model import BiomassModel

def main():
    device = torch.device("cuda:0")
    inp = torch.randn(2, 3, 512, 512).to(device)
    ckpt_path = pathlib.Path("./checkpoints/ver0")
    
    all_files = [i for i in os.listdir(ckpt_path) if "fold_" in i]
    save_path = ckpt_path / "merged_weights"
    save_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(len(all_files)):
        pr = torch.load(ckpt_path / all_files[i] / "best_ema.bin")
        model = BiomassModel()
        model = model.to(device)
        model.eval()
        model.load_state_dict(pr)
        out = model(inp, inp)
        model.backbone = model.backbone.merge_and_unload()
        out1 = model(inp, inp)
        print(out[0], out1[0])
        fold_idx = all_files[i].split("fold_")[-1]
        torch.save(model.state_dict(), save_path/f"model_{fold_idx}.bin")
        
if __name__ == "__main__":
    main()