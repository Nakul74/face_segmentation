import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import openvino as ov
core = ov.Core()

ir_model_path = Path('selfie_multiclass_256x256.xml')
ov_model = core.read_model(ir_model_path)
compiled_model = core.compile_model(ov_model, 'CPU')
BG_COLOR = (192, 192, 192)

# Define Label and SegmentationMap classes for visualization
class Label:
    def __init__(self, index, color, name):
        self.index = index
        self.color = color
        self.name = name

class SegmentationMap:
    def __init__(self, labels):
        self.labels = labels

    def get_colormap(self):
        colormap = {}
        for label in self.labels:
            colormap[label.index] = label.color
        return colormap

labels = [
    Label(index=0, color=(192, 192, 192), name="background"),
    Label(index=1, color=(128, 0, 0), name="hair"),
    Label(index=3, color=(255, 204, 204), name="face skin"),
]
SegmentationLabels = SegmentationMap(labels)

# img = cv2.imread('/content/image001.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_and_pad(image:np.ndarray, height:int = 256, width:int = 256):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (width, np.floor(h / (w / width)).astype(int)))
    else:
        img = cv2.resize(image, (np.floor(w / (h / height)).astype(int), height))

    r_h, r_w = img.shape[:2]
    right_padding = width - r_w
    bottom_padding = height - r_h
    padded_img = cv2.copyMakeBorder(img, 0, bottom_padding, 0, right_padding, cv2.BORDER_CONSTANT)
    return padded_img, (bottom_padding, right_padding)


def postprocess_mask(out: np.ndarray, pad_info: Tuple[int, int], orig_img_size: Tuple[int, int]):
    hair_idx = 1  
    face_skin_idx = 3  

    label_mask = np.argmax(out, axis=-1)[0]

    filtered_mask = np.where(
        (label_mask == hair_idx) | (label_mask == face_skin_idx),
        label_mask,
        0 
    )

    pad_h, pad_w = pad_info
    unpad_h = filtered_mask.shape[0] - pad_h
    unpad_w = filtered_mask.shape[1] - pad_w
    label_mask_unpadded = filtered_mask[:unpad_h, :unpad_w]

    orig_h, orig_w = orig_img_size
    label_mask_resized = cv2.resize(label_mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return label_mask_resized

def get_segment_mask(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    padded_img, pad_info = resize_and_pad(np.array(img))
    normalized_img = np.expand_dims(padded_img.astype(np.float32) / 255, 0)
    out = compiled_model(normalized_img)[0]

    image_data = np.array(img)
    orig_img_shape = image_data.shape
    postprocessed_mask = postprocess_mask(out, pad_info, orig_img_shape[:2])
    bg_image = np.full(orig_img_shape, BG_COLOR, dtype=np.uint8)
    condition = np.stack((postprocessed_mask,) * 3, axis=-1) > 0
    output_image = np.where(condition, image_data, bg_image).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.jpg',output_image)
    return 'output.jpg'

if __name__ == '__main__':
    output_img_path = get_segment_mask('sample/sample.png')
    print(output_img_path)
