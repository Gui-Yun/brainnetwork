from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil

BASE = Path(__file__).resolve().parents[1]
SRC_DIR = BASE / "result" / "paper_panels_20260331_v4"
OUT_DIR = SRC_DIR

PANEL_MAP = {
    "Fig1": [
        ["Fig1_PanelA_experimental_schematic.png", "Fig1_PanelB_representative_fov_rr_overlay.png", "Fig1_PanelC_population_mean_trace_group.png"],
        ["Fig1_PanelD_decoder_confusion_single_M73.png", "Fig1_PanelE_decoder_accuracy_group_true_vs_shuffle.png"],
    ],
    "Fig2": [
        ["Fig2_PanelA_representative_corr_matrix_sorted_single_M73.png"],
        ["Fig2_PanelB_pairwise_corr_ridge_single_M73.png", "Fig2_PanelC_strongest_decile_group.png", "Fig2_PanelD_weakest_decile_group.png"],
        ["Fig2_PanelE_decile_profile_group.png", "Fig2_PanelF_fc_decoder_full_weak_shuffle_group.png"],
    ],
    "Fig3": [
        ["Fig3_PanelA_preference_sorted_response_matrix_single_M73.png", "Fig3_PanelB_participants_ratio_group.png", "Fig3_PanelC_gini_group.png"],
        ["Fig3_PanelD_orthogonal_vs_parallel_expansion_group.png", "Fig3_PanelE_rsm_distribution_group.png", "Fig3_PanelF_mean_rsm_group.png"],
        ["Fig3_PanelG_lmm_participants_vs_rsm_group.png"],
    ],
}

PANEL_LETTERS = {
    "Fig1": ["A", "B", "C", "D", "E"],
    "Fig2": ["A", "B", "C", "D", "E", "F"],
    "Fig3": ["A", "B", "C", "D", "E", "F", "G"],
}

try:
    FONT = ImageFont.truetype("arial.ttf", 36)
except Exception:
    FONT = ImageFont.load_default()

LABEL_FILL = (255, 255, 255, 235)
LABEL_TEXT = (30, 30, 30, 255)
H_GAP = 20
V_GAP = 30


def load_image(name: str) -> Image.Image:
    path = SRC_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGBA")


def add_label(img: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), label, font=FONT)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 10
    box = (pad, pad, pad + w + 18, pad + h + 14)
    draw.rectangle(box, fill=LABEL_FILL)
    draw.text((pad + 9, pad + 7), label, font=FONT, fill=LABEL_TEXT)
    return img


def compose_figure(fig_name: str, rows):
    letters = PANEL_LETTERS[fig_name]
    letter_idx = 0
    row_imgs = []
    row_widths = []
    # First pass: load, set target height, label; record natural row width
    for row in rows:
        target_h = 900 if len(row) == 1 else 760
        imgs = []
        for fname in row:
            im = load_image(fname)
            scale = target_h / im.size[1]
            im = im.resize((int(im.size[0] * scale), target_h), Image.LANCZOS)
            label = letters[letter_idx] if letter_idx < len(letters) else ""
            letter_idx += 1
            im = add_label(im, label)
            imgs.append(im)
        nat_w = sum(i.size[0] for i in imgs) + H_GAP * (len(imgs) - 1)
        row_imgs.append((imgs, target_h))
        row_widths.append(nat_w)

    max_w = max(row_widths)
    rendered_rows = []
    # Second pass: scale each row so its total width matches max_w
    for (imgs, target_h), nat_w in zip(row_imgs, row_widths):
        if nat_w > 0:
            row_scale = max_w / nat_w
        else:
            row_scale = 1.0
        scaled_imgs = []
        for im in imgs:
            new_size = (max(1, int(im.size[0] * row_scale)), max(1, int(im.size[1] * row_scale)))
            scaled_imgs.append(im.resize(new_size, Image.LANCZOS))
        row_h = scaled_imgs[0].size[1] if scaled_imgs else target_h
        row_canvas = Image.new("RGBA", (int(round(max_w)), row_h), (255, 255, 255, 255))
        x = 0
        for im in scaled_imgs:
            row_canvas.paste(im, (x, 0), im)
            x += im.size[0] + H_GAP
        rendered_rows.append(row_canvas)

    total_h = sum(r.size[1] for r in rendered_rows) + V_GAP * (len(rendered_rows) - 1)
    canvas = Image.new("RGBA", (max_w, total_h), (255, 255, 255, 255))
    y = 0
    for r in rendered_rows:
        x = (max_w - r.size[0]) // 2
        canvas.paste(r, (x, y), r)
        y += r.size[1] + V_GAP

    out_path = OUT_DIR / f"{fig_name}_combined.png"
    canvas.convert("RGB").save(out_path, dpi=(300, 300))
    print(f"[+] wrote {out_path}")


def main():
    for fig, rows in PANEL_MAP.items():
        compose_figure(fig, rows)


if __name__ == "__main__":
    main()
