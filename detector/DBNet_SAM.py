import cv2
import numpy as np
from mmocr.apis import init_detector, model_inference
from segment_anything import sam_model_registry, SamPredictor

# 1. Load image
img_path = "path/to/image.jpg"
image = cv2.imread(img_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Initialize DBNet (replace with your config/checkpoint)
dbnet_config = "configs/dbnet/dbnet_r50_icdar2015.py"
dbnet_ckpt   = "checkpoints/dbnet_r50_icdar2015.pth"
dbnet_model  = init_detector(dbnet_config, dbnet_ckpt, device="cuda:0")

# 3. Detect text polygons
result = model_inference(dbnet_model, img_path)
# result["boundary_result"]["boundary_list"] is a list of polygons: [[x1,y1],…]
polygons = result["boundary_result"]["boundary_list"]

# 4. Initialize SAM
sam = sam_model_registry["default"](checkpoint="checkpoints/sam_vit_b.pth")
predictor = SamPredictor(sam)
predictor.set_image(rgb_image)

# 5. For each detected polygon, get a mask
masks = []
for poly in polygons:
    poly = np.array(poly)
    x_min, y_min = poly[:,0].min(), poly[:,1].min()
    x_max, y_max = poly[:,0].max(), poly[:,1].max()
    # Add a bit of padding if you like:
    pad = 5
    box = [max(0, x_min-pad), max(0, y_min-pad),
           min(image.shape[1], x_max+pad), min(image.shape[0], y_max+pad)]
    
    # SAM inference
    mask, score, logit = predictor.predict(
        box=box,
        multimask_output=False
    )
    masks.append(mask[0])  # mask is returned as a list of masks

# 6. Visualize results
vis = image.copy()
for m in masks:
    vis[m == 1] = (0, 255, 0)  # paint mask area green
cv2.imwrite("text_masks_overlay.png", vis)
