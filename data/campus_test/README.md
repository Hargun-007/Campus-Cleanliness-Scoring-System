# Campus Test Photographs

This directory holds real photographs captured around the university campus (corridors, gardens, parking lots, canteen, classrooms, and sports grounds).

## Testing Campus Photographs

These images are kept strictly separate from the TACO training and validation dataset to evaluate real-world generalization under novel campus illumination, camera sensors, and surfaces.

### How to Add Real Photographs
1. Take photos around your campus using your smartphone camera.
2. Save them in `.jpg` or `.png` format.
3. Copy them into `data/campus_test/`.
4. Run the batch evaluation script:
   ```bash
   python src/model/batch_test_campus.py --images-dir data/campus_test
   ```
5. View annotated output images in `reports/campus_test_runs/` and structured score summary in `reports/campus_test_results.json`.

### Observed Real-World Failure Modes
- **Small Litter Items**: Objects smaller than 16×16 pixels (bottle caps, cigarette butts) may not be detected by YOLOv8n.
- **Occlusion**: Litter partially hidden behind stairs, bushes, or furniture.
- **Lighting and Shadows**: Strong midday sunlight or night fluorescent flicker affecting edge detection.
- **Textured Terrains**: Dry leaves, fallen blossoms, or speckled marble tiles occasionally mistaken for wrappers.

