"""Package a trained model for inference: CHECKPOINT_DIR/<tag>/last.pt -> MODEL_DIR/<tag>.pt = {"model": state dict,
"config": {arch, dim, att_temp}}, the format inference/infer.py loads (the 13 zoomed views and every other setting are
fixed in the inference code, so they are checked here against the run's args.json). The packaged weights are loaded
into inference/model.py with strict keys before they are written.
usage: python training/package.py TAG --att_temp T
"""
import argparse, importlib.util, json, os, sys
import torch

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from train import path, ROI_GENERAL, ROI_MENISCI, ROI_PF  # noqa: E402
from data import parse_rois  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--att_temp", type=float, required=True, help="inference attention temperature (1.75 EfficientNetV2-S, "
                "1.25 ConvNeXt)")
e = ap.parse_args()
a = json.load(open(path("CHECKPOINT_DIR", e.tag, "args.json")))
assert a["roi"] == f"{ROI_GENERAL};{ROI_MENISCI};{ROI_PF}" and a["res"] == 384 and a["span"] == "0.04,0.96", \
    "inference/ fixes the 13 zoomed views, 384 px and the window span"
sd = torch.load(path("CHECKPOINT_DIR", e.tag, "last.pt"), map_location="cpu", weights_only=True)
cfg = {"arch": a["arch"], "dim": int(a["D"]), "att_temp": e.att_temp}
spec = importlib.util.spec_from_file_location("inference_model", os.path.join(os.path.dirname(H), "inference", "model.py"))
im = importlib.util.module_from_spec(spec); spec.loader.exec_module(im)
m = im.KQNet(cfg["arch"], [r["plane"] for r in parse_rois(a["roi"])], D=cfg["dim"])
m.load_state_dict(sd, strict=True)
os.makedirs(path("MODEL_DIR"), exist_ok=True)
torch.save({"model": sd, "config": cfg}, path("MODEL_DIR", f"{e.tag}.pt"))
print(f"{e.tag}: {cfg} -> {path('MODEL_DIR', e.tag + '.pt')}")
