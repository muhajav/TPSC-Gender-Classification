import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def load_heatmap(path):
    hm = np.load(path)
    # If heatmap has channels, collapse to single channel
    if hm.ndim == 3:
        # If shape is (H,W,3) or (H,W,4) take mean across channels
        hm = hm.mean(axis=2)
    return hm.astype(float)

def normalize(hm):
    hm = np.array(hm, dtype=float)
    # replace NaNs
    hm = np.nan_to_num(hm, nan=0.0)
    mn = hm.min()
    hm = hm - mn
    mx = hm.max()
    if mx == 0:
        return hm
    return hm / mx

def resize_heatmap_to(hm, target_shape):
    """Resize heatmap (2D, values 0..1 expected) to match target_shape (H,W,...)."""
    H, W = target_shape[0], target_shape[1]
    hm_img = Image.fromarray((np.clip(hm, 0.0, 1.0) * 255).astype('uint8'))
    hm_img = hm_img.resize((W, H), resample=Image.BILINEAR)
    hm_resized = np.array(hm_img).astype(float) / 255.0
    return hm_resized

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_side_by_side(original, saliency, lime, outpath, dpi=300):
    if original is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax_sal, ax_lime = axes
        im1 = ax_sal.imshow(saliency, cmap='jet')
        ax_sal.set_title("Saliency")
        ax_sal.axis('off')
        im2 = ax_lime.imshow(lime, cmap='hot')
        ax_lime.set_title("LIME")
        ax_lime.axis('off')
        plt.colorbar(im1, ax=ax_sal, fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax_lime, fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax0, ax1, ax2 = axes
        ax0.imshow(original)
        ax0.set_title("Original")
        ax0.axis('off')
        im1 = ax1.imshow(saliency, cmap='jet')
        ax1.set_title("Saliency")
        ax1.axis('off')
        im2 = ax2.imshow(lime, cmap='hot')
        ax2.set_title("LIME")
        ax2.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def save_overlay(original, heatmap, cmap, title, outpath, alpha=0.5, dpi=300):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(original)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize heatmaps (saliency & LIME)")
    parser.add_argument("--original", type=str, default=None, help="Path to original image (png/jpg). Optional.")
    parser.add_argument("--saliency", type=str, required=True, help="Path to saliency .npy file")
    parser.add_argument("--lime", type=str, required=True, help="Path to LIME .npy file")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha (0..1)")
    parser.add_argument("--dpi", type=int, default=300, help="Saved figure DPI")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    original = None
    if args.original:
        if not os.path.exists(args.original):
            print(f"[WARN] Original image not found at {args.original}. Continuing without original overlay.")
        else:
            original = load_image(args.original)
            print(f"[INFO] Loaded original image: {args.original} shape={original.shape}")

    saliency = load_heatmap(args.saliency)
    lime = load_heatmap(args.lime)
    print(f"[INFO] Loaded heatmaps: saliency {saliency.shape}, LIME {lime.shape}")

    # Normalize and (if original exists) resize to image shape
    saliency_n = normalize(saliency)
    lime_n = normalize(lime)

    if original is not None:
        saliency_n = resize_heatmap_to(saliency_n, original.shape)
        lime_n = resize_heatmap_to(lime_n, original.shape)

    # Save side-by-side comparison
    side_path = os.path.join(args.outdir, "comparison_side_by_side.png")
    save_side_by_side(original, saliency_n, lime_n, side_path, dpi=args.dpi)
    print(f"[SAVED] {side_path}")

    # If original exists, save overlays
    if original is not None:
        ov1 = os.path.join(args.outdir, "overlay_saliency.png")
        save_overlay(original, saliency_n, cmap='jet', title='Overlay: Saliency', outpath=ov1, alpha=args.alpha, dpi=args.dpi)
        print(f"[SAVED] {ov1}")
        ov2 = os.path.join(args.outdir, "overlay_lime.png")
        save_overlay(original, lime_n, cmap='hot', title='Overlay: LIME', outpath=ov2, alpha=args.alpha, dpi=args.dpi)
        print(f"[SAVED] {ov2}")
    else:
        print("[INFO] Original image not provided — overlay images were not created.")

if __name__ == "__main__":
    main()
