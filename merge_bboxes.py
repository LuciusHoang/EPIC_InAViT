#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def merge_bboxes(obj_json: Dict[str, Any], hand_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge object detections (obj_json) with hand detections (hand_json).
    For each frame:
      - Keep both "obj" and "hand"
      - If one is missing, insert an empty list
    """
    all_keys = set(obj_json.keys()) | set(hand_json.keys())
    merged = {}

    for k in sorted(all_keys):
        obj_entry = obj_json.get(k, {"obj": []})
        hand_entry = hand_json.get(k, {"hand": []})

        # normalize: always lists
        objs = obj_entry.get("obj", [])
        hands = hand_entry.get("hand", [])

        merged[k] = {
            "obj": objs if isinstance(objs, list) else [],
            "hand": hands if isinstance(hands, list) else []
        }

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge object and hand detection JSONs")
    parser.add_argument("--obj", required=True, help="Path to bboxes.json (object only)")
    parser.add_argument("--hand", required=True, help="Path to bboxes_hand.json (hand only)")
    parser.add_argument("--out", required=True, help="Output merged JSON path")
    args = parser.parse_args()

    obj_json = load_json(args.obj)
    hand_json = load_json(args.hand)

    merged = merge_bboxes(obj_json, hand_json)

    save_json(args.out, merged)
    print(f"[DONE] wrote merged JSON → {args.out}")
    print(f"[STATS] frames: {len(merged)}")


if __name__ == "__main__":
    main()