"""
Web UI: python app.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from unmask.config import Settings, effective_restoration_id, resolve_unet_weights
from unmask.pipeline import RestorationPipeline
from unmask.restoration.registry import list_names

CSS = """
footer {display: none !important;}
.gradio-container {max-width: 1560px !important; padding: 2rem 1.75rem !important;}
.unmask-title {font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.03em; margin: 0 0 0.4em 0 !important; color: #0f172a;}
.unmask-sub {font-size: 1.06rem !important; line-height: 1.6; max-width: 54rem; color: #334155;}
.panel-wrap {min-height: 720px;}
.panel-wrap img, .panel-wrap canvas {border-radius: 14px !important; box-shadow: 0 12px 40px rgba(15,23,42,0.10) !important;}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    spacing_size="md",
    radius_size="lg",
)


def _bgr_from_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _rgb_from_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _backend_choices() -> list[str]:
    return ["auto"] + list_names()


def build_demo(
    mask_mode: str,
    backend: str,
    unet_path: Path | None,
    feather: int,
    radius: int,
    opencv_algorithm: str,
    opencv_multiscale: bool,
    bilateral_refinement: bool = True,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 55.0,
    bilateral_sigma_space: float = 55.0,
) -> gr.Blocks:
    cfg = Settings(
        mask_mode=mask_mode,  # type: ignore[arg-type]
        restoration_backend=backend,  # type: ignore[arg-type]
        unet_weights=unet_path,
        feather_px=feather,
        opencv_inpaint_radius=radius,
        opencv_algorithm=opencv_algorithm,  # type: ignore[arg-type]
        opencv_multiscale=opencv_multiscale,
        bilateral_refinement=bilateral_refinement,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )
    pipe = RestorationPipeline(cfg)

    def on_run(rgb: np.ndarray | None):
        if rgb is None:
            return None, "Upload or capture an image with a visible face."
        out, ok = pipe.process_bgr(_bgr_from_rgb(rgb))
        if not ok:
            return rgb, "No face detected — ensure the face is visible and lit."
        return _rgb_from_bgr(out), "Restored lower-face region (see README for limits)."

    with gr.Blocks(title="Unmask") as demo:
        gr.Markdown(
            '<p class="unmask-title">Unmask</p>'
            '<p class="unmask-sub">Lower-face <strong>restoration</strong> is approximate: classical / small U-Net fill holes; '
            "they do not recover your exact mouth without a stronger generative prior. "
            "<code>auto</code> uses U-Net weights in <code>assets/unmask_unet.pth</code> if present, else OpenCV. "
            "<code>gan</code> is a plugin slot — set <code>UNMASK_GAN_CLASS</code> (see docs/PLUGINS.md).</p>"
        )
        with gr.Row(equal_height=True):
            inp = gr.Image(
                label="Input",
                type="numpy",
                sources=["upload", "webcam"],
                height=720,
                width=640,
                elem_classes=["panel-wrap"],
            )
            out = gr.Image(
                label="Result",
                type="numpy",
                height=720,
                width=640,
                elem_classes=["panel-wrap"],
            )
        msg = gr.Textbox(label="Status", interactive=False)
        btn = gr.Button("Unmask", variant="primary", size="lg")
        btn.click(on_run, inputs=[inp], outputs=[out, msg])

        gr.Markdown(
            "Train U-Net: `python -m training.train_unet --data-dir ./data/faces`. "
            "DCGAN in `training/dcgan.py` is research-only (not wired to this UI)."
        )

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Unmask web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    p.add_argument("--mask-mode", choices=("mesh", "bbox"), default="mesh")
    p.add_argument("--backend", choices=_backend_choices(), default="auto")
    p.add_argument("--unet-weights", type=Path, default=None)
    p.add_argument("--feather", type=int, default=22)
    p.add_argument("--radius", type=int, default=7)
    p.add_argument("--opencv-algo", choices=("ns", "telea"), default="ns")
    p.add_argument("--no-multiscale", action="store_true")
    p.add_argument("--no-bilateral", action="store_true", help="Disable edge-preserving bilateral pass on mask region")
    p.add_argument("--bilateral-d", type=int, default=7)
    p.add_argument("--bilateral-sigma-color", type=float, default=55.0)
    p.add_argument("--bilateral-sigma-space", type=float, default=55.0)
    args = p.parse_args()

    _probe = Settings(
        restoration_backend=args.backend,  # type: ignore[arg-type]
        unet_weights=args.unet_weights,
    )
    _w = resolve_unet_weights(_probe)
    _eff = effective_restoration_id(_probe)
    if args.backend == "auto":
        ckpt = str(_w.resolve()) if _w else None
        print(
            f"[Unmask] --backend auto -> effective={_eff!r}; "
            f"checkpoint={ckpt if ckpt else '(none — OpenCV inpaint)'}"
        )
    else:
        print(
            f"[Unmask] --backend {args.backend!r} -> effective={_eff!r}; "
            f"checkpoint={str(_w.resolve()) if _w else 'none'}"
        )

    demo = build_demo(
        mask_mode=args.mask_mode,
        backend=args.backend,
        unet_path=args.unet_weights,
        feather=args.feather,
        radius=args.radius,
        opencv_algorithm=args.opencv_algo,
        opencv_multiscale=not args.no_multiscale,
        bilateral_refinement=not args.no_bilateral,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
    )
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
