"""FasterRCNN 行人检测 Demo：COCO 预训练，只保留 person 类。"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图片目录")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--max", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=args.threshold,
    ).eval().to(device)

    records = []
    images = sorted(Path(args.images).glob("*.jpg"))[: args.max]
    for i, path in enumerate(images, 1):
        img = Image.open(path).convert("RGB")
        tensor = F.pil_to_tensor(img).float().div(255).to(device)
        t0 = time.time()
        with torch.no_grad():
            pred = model([tensor])[0]
        ms = round((time.time() - t0) * 1000)
        keep = (pred["labels"] == 1) & (pred["scores"] >= args.threshold)
        boxes, scores = pred["boxes"][keep].cpu(), pred["scores"][keep].cpu()
        n = int(keep.sum())
        records.append({"image": path.name, "persons": n,
                        "avg_score": round(float(scores.mean()), 3) if n else 0,
                        "infer_ms": ms})
        annotated = draw_bounding_boxes(
            F.pil_to_tensor(img), boxes,
            labels=[f"person {s:.2f}" for s in scores],
            colors=["#13e2da"] * n, width=3) if n else F.pil_to_tensor(img)
        F.to_pil_image(annotated).save(out / f"detect-{i}.png")
        print(records[-1])

    summary = {
        "device": str(device),
        "threshold": args.threshold,
        "images": len(records),
        "avg_infer_ms": round(sum(r["infer_ms"] for r in records) / len(records)) if records else 0,
        "detail": records,
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
