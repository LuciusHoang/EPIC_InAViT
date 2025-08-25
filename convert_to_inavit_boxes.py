#!/usr/bin/env python3
# convert_to_inavit_boxes.py
import argparse, json, os

def clamp01(x): return 0.0 if x < 0 else (1.0 if x > 1.0 else x)

def norm_box(b, W=224, H=224):
    x1, y1, x2, y2 = map(float, b[:4])
    x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
    y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [clamp01(x1/W), clamp01(y1/H), clamp01(x2/W), clamp01(y2/H)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to bboxes_combined.json")
    ap.add_argument("--output", required=True, help="path to output bboxes.json for InAViT")
    ap.add_argument("--O", type=int, default=4)
    ap.add_argument("--U", type=int, default=1)
    ap.add_argument("--W", type=int, default=224)
    ap.add_argument("--H", type=int, default=224)
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    out = {}
    for frame_id, entry in data.items():
        objs = entry.get("obj", [])
        hands = entry.get("hand", [])

        # normalize and trim/pad
        objs = [norm_box(b, args.W, args.H) for b in objs][:args.O]
        while len(objs) < args.O:
            objs.append([0,0,0,0])

        if args.U > 0:
            if len(hands) > 0:
                # pick the largest-hand box if multiple
                hands_n = [norm_box(b, args.W, args.H) for b in hands]
                # area sort
                hands_n = sorted(hands_n, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]), reverse=True)
                hands = [hands_n[0]]
            else:
                hands = [[0,0,0,0]]
        else:
            hands = []

        out[frame_id] = {"obj": objs, "hand": hands}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote InAViT bboxes to {args.output}")

if __name__ == "__main__":
    main()