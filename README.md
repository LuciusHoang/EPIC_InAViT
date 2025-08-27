# EPIC_InAViT
Inferencing the methodology and performance metrics of InAViT model to the subset of dataset EPIC-KITCHENS100.

## Disclaimer
This repository contains code only and does not include the EPIC-KITCHENS-100 dataset. Users must obtain the dataset separately from the official source and comply with its license terms as Non-Commercial Government License for public sector information.
More detail at: https://data.bris.ac.uk/data/dataset/3h91syskeag572hl6tvuovwv4d.

## Inference Result:
- "Top5_Recall_Verb": 0.777027027027027
- "Top5_Recall_Noun": 0.7027027027027027
- "Top5_Recall_Action_Derived": 0.36486486486486486
- "num_eval_action": 148
- "overlap": 148

## How to run pipeline
1. Extract test segments from .pkl:  
python extract_pkl.py

2. Create test labels from test segments:  
python create_test_label.py

4. Detect hands from data:  
python detect_hands_to_bboxes.py \  
  --images-dir data/object_detection_images/P01_11 \  
  --out data/object_detection_images/P01_11/bboxes_hand.json \  
  --no-motion \  
  --sample-log  

5. Detect objects from data:  
python detect_objects_to_bboxes.py \  
  --images-dir data/object_detection_images/P01_11 \  
  --out data/object_detection_images/P01_11/bboxes_obj.json \  
  --thr 0.6 \  
  --max-obj 4 \  
  --sample-log  

6. Merge hands and objects bboxes:  
python merge_bboxes.py \  
  --obj  data/object_detection_images/P01_11/bboxes_obj.json \  
  --hand data/object_detection_images/P01_11/bboxes_hand.json \  
  --out  data/object_detection_images/P01_11/bboxes_combined.json  

8. Convert combined bboxes to inavit bboxes:  
python convert_to_inavit_boxes.py \  
  --input data/object_detection_images/P01_11/bboxes_combined.json \  
  --output data/object_detection_images/P01_11/bboxes.json  

9. Install InAViT package:  
git clone https://github.com/LAHAproject/InAViT.git  

10. Amend setup in InAViT package:  
- In /InAViT/slowfast/models/video_model_builder.py, delete "<<<<<<< HEAD", "=======", ">>>>>>> e0ef9a0442f6ba31ffe45ac06f6b3bf13782c7de", 
and comment out below:  
_"from .HOIVIT import ORViT as ORViT  
from .HOIVIT import HoIViT as HoIViT  
from .HOIVIT import STDViT as STDViT  
from .HOIVIT import UNIONHOIVIT as UNIONHOIVIT  
from .HOIVIT import ObjectsCrops"_  
- In /InAViT/slowfast/datasets/ek_MF/frame_loader.py, comment out below:  
_"import decord  
decord.bridge.set_bridge('torch')  
from decord import VideoReader"_  
- In /InAViT/slowfast/datasets/egt a_MF/frame_loader.py, comment out below:  
_"import decord  
decord.bridge.set_bridge('torch')  
from decord import VideoReader"_  

9. Run main.py to infer InAViT model:  
python main.py \  
  --csv EPIC_100_test_segments.csv \  
  --frames-base data/frames_rgb \  
  --obj-base data/object_detection_images \  
  --bboxes data/object_detection_images/P01_11/bboxes.json \  
  --cfg EK_INAVIT_MF_ant.yaml \  
  --weights checkpoints/checkpoint_epoch_00081.pyth \  
  --out results/inavit_predictions.jsonl \  
  --skip 0  

10. Run evaluation:  
python evaluate_predictions.py  
