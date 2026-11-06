# Models Directory

This directory stores the object detection model checkpoints for the Campus Cleanliness Scoring System.

## Checkpoints

1. **Pretrained Baseline (`yolov8n.pt`)**:
   - Official YOLOv8 nano model checkpoint pretrained on the 80-class MS COCO dataset.
   - Used as the starting backbone/weights for transfer learning.
   - Also serves as the baseline model in `src/model/compare_models.py`.

2. **Fine-Tuned Model (`best.pt`)**:
   - YOLOv8n fine-tuned on the single-class TACO (`litter`, class ID 0) dataset.
   - Selected automatically by Ultralytics based on highest validation mAP@0.5:0.95.
   - This checkpoint is used for production inference in the Flask application.

3. **Last Checkpoint (`last.pt`)**:
   - Checkpoint from the final epoch of training, useful for resuming interrupted training runs.

## Important Note on Checkpoint Integrity

Pretrained weights and fine-tuned weights are strictly distinct files. Pretrained COCO weights detect general objects (bottles, cups, persons, chairs) across 80 classes, whereas the fine-tuned weights detect `litter` specifically with a single class output (ID 0). The system never silently substitutes pretrained weights for missing fine-tuned weights in final evaluations.

