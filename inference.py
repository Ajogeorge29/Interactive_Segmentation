"""
inference.py
ToothFairy3 IAC Segmentation Inference Script
- Supports batch inference and single image inference
- Loads click maps from JSON
- Saves predictions, transformed images, and visualization
"""

import json
import os
from glob import glob

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Resized,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)
from scipy.ndimage import gaussian_filter


# -------------------------
# CONFIGURATION
# -------------------------
class Config:
    MODEL_PATH = "best_model.pth"
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    NUM_CLASSES = 3
    SPATIAL_SIZE = (384, 384, 192)
    PIXDIM = (0.3, 0.3, 0.3)
    CLICK_SIGMA = 3
    SW_BATCH_SIZE = 4
    OVERLAP = 0.5
    CLICK_JSON = "clicks.json"  # path to click map JSON


config = Config()


# -------------------------
# CUSTOM TRANSFORM
# -------------------------
class AddIACClickMaps(MapTransform):
    """Add interactive click maps from JSON for inference"""

    def __init__(self, keys, sigma=3, click_json=None):
        super().__init__(keys)
        self.sigma = sigma
        self.click_json = click_json
        self.left_clicks = []
        self.right_clicks = []
        self._load_clicks()

    def _load_clicks(self):
        if self.click_json and os.path.exists(self.click_json):
            with open(self.click_json, "r") as f:
                data = json.load(f)
            self.left_clicks = data.get("left", [])
            self.right_clicks = data.get("right", [])

    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        left_map = np.zeros_like(img[0], dtype=np.float32)
        right_map = np.zeros_like(img[0], dtype=np.float32)

        for x, y, z in self.left_clicks:
            if (
                0 <= x < left_map.shape[0]
                and 0 <= y < left_map.shape[1]
                and 0 <= z < left_map.shape[2]
            ):
                left_map[x, y, z] = 1
        for x, y, z in self.right_clicks:
            if (
                0 <= x < right_map.shape[0]
                and 0 <= y < right_map.shape[1]
                and 0 <= z < right_map.shape[2]
            ):
                right_map[x, y, z] = 1

        left_map = gaussian_filter(left_map, sigma=self.sigma)
        right_map = gaussian_filter(right_map, sigma=self.sigma)

        d["image"] = np.stack([img[0], left_map, right_map], axis=0)
        d["original_image"] = img[0]
        return d


# -------------------------
# MODEL
# -------------------------
def detect_model_channels(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    key = "model.0.conv.unit0.conv.weight"
    if key in checkpoint["model_state_dict"]:
        return checkpoint["model_state_dict"][key].shape[1]
    return 3


def load_model(model_path, device):
    in_channels = detect_model_channels(model_path)
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=config.NUM_CLASSES,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="batch",
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, in_channels


# -------------------------
# TRANSFORMS
# -------------------------
def get_transforms(model_channels):
    transforms = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=config.PIXDIM, mode="trilinear"),
        ScaleIntensityd(keys=["image"]),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=config.SPATIAL_SIZE, mode="trilinear"),
    ]
    if model_channels == 3:
        transforms.append(
            AddIACClickMaps(
                keys=["image"], sigma=config.CLICK_SIGMA, click_json=config.CLICK_JSON
            )
        )
    transforms += [
        CastToTyped(keys=["image"], dtype=np.float32),
        ToTensord(keys=["image"]),
    ]
    return Compose(transforms)


# -------------------------
# INFERENCE
# -------------------------
def predict(model, model_channels, image_path, device):
    data_dict = {"image": image_path}
    transforms = get_transforms(model_channels)
    data = transforms(data_dict)
    inputs = data["image"].unsqueeze(0).to(device)
    original_img = data["original_image"]

    with torch.no_grad():
        outputs = sliding_window_inference(
            inputs=inputs,
            roi_size=config.SPATIAL_SIZE,
            sw_batch_size=config.SW_BATCH_SIZE,
            predictor=model,
            overlap=config.OVERLAP,
        )
    pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    return pred.cpu().numpy()[0], original_img, data["image"].cpu().numpy()[0]


# -------------------------
# SAVE FUNCTIONS
# -------------------------
def save_nifti(arr, ref_path, save_path, dtype=np.uint8):
    ref_img = nib.load(ref_path)
    nib.save(
        nib.Nifti1Image(arr.astype(dtype), ref_img.affine, ref_img.header), save_path
    )
    print(f"Saved: {save_path}")


# -------------------------
# VISUALIZATION
# -------------------------
def visualize(original_img, prediction, save_path=None):
    colors = ["black", "red", "blue"]
    cmap = ListedColormap(colors)
    H, W, D = original_img.shape
    slices = {
        "Sagittal": (original_img[H // 2, :, :], prediction[H // 2, :, :]),
        "Coronal": (original_img[:, W // 2, :], prediction[:, W // 2, :]),
        "Axial": (original_img[:, :, D // 2], prediction[:, :, D // 2]),
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    for idx, (name, (img_slice, pred_slice)) in enumerate(slices.items()):
        axes[idx, 0].imshow(img_slice, cmap="gray")
        axes[idx, 0].set_title(f"{name} - Input")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(img_slice, cmap="gray", alpha=0.5)
        axes[idx, 1].imshow(pred_slice, cmap=cmap, alpha=0.6, vmin=0, vmax=2)
        axes[idx, 1].set_title(f"{name} - Prediction")
        axes[idx, 1].axis("off")

    fig.legend(
        [Patch(facecolor="red"), Patch(facecolor="blue")],
        ["Left IAC", "Right IAC"],
        loc="lower center",
        ncol=2,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# -------------------------
# BATCH INFERENCE
# -------------------------
def batch_inference():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, channels = load_model(config.MODEL_PATH, device)

    files = sorted(glob(os.path.join(config.INPUT_DIR, "*.nii*")))
    if not files:
        print("No images found.")
        return

    for f in files:
        print(f"Processing: {f}")
        pred, orig, trans_img = predict(model, channels, f, device)
        base = os.path.basename(f).replace(".nii.gz", "").replace(".nii", "")
        save_nifti(pred, f, os.path.join(config.OUTPUT_DIR, f"{base}_pred.nii.gz"))
        save_nifti(
            trans_img,
            f,
            os.path.join(config.OUTPUT_DIR, f"{base}_transformed.nii.gz"),
            dtype=np.float32,
        )
        visualize(
            orig, pred, save_path=os.path.join(config.OUTPUT_DIR, f"{base}_viz.png")
        )


# -------------------------
# SINGLE IMAGE INFERENCE
# -------------------------
def single_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, channels = load_model(config.MODEL_PATH, device)
    pred, orig, trans_img = predict(model, channels, image_path, device)
    base = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")
    save_nifti(pred, image_path, os.path.join(config.OUTPUT_DIR, f"{base}_pred.nii.gz"))
    save_nifti(
        trans_img,
        image_path,
        os.path.join(config.OUTPUT_DIR, f"{base}_transformed.nii.gz"),
        dtype=np.float32,
    )
    visualize(orig, pred, save_path=os.path.join(config.OUTPUT_DIR, f"{base}_viz.png"))


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Uncomment one of these:
    # batch_inference()
    # single_inference("/content/input/ToothFairy3F_001_0000.nii.gz")
    batch_inference()
