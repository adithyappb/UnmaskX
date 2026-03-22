"""CLI: python main.py webcam | python main.py ui"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from unmask.config import Settings
from unmask.pipeline import RestorationPipeline
from unmask.restoration.registry import list_names


def _backend_choices() -> list[str]:
    return ["auto"] + list_names()


def run_webcam(
    camera: int,
    mask_mode: str,
    backend: str,
    unet_weights: Path | None,
    feather: int,
    radius: int,
    opencv_algorithm: str,
    opencv_multiscale: bool,
    bilateral_refinement: bool = True,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 55.0,
    bilateral_sigma_space: float = 55.0,
) -> None:
    cfg = Settings(
        mask_mode=mask_mode,  # type: ignore[arg-type]
        restoration_backend=backend,  # type: ignore[arg-type]
        unet_weights=unet_weights,
        feather_px=feather,
        opencv_inpaint_radius=radius,
        opencv_algorithm=opencv_algorithm,  # type: ignore[arg-type]
        opencv_multiscale=opencv_multiscale,
        bilateral_refinement=bilateral_refinement,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Could not open camera {camera}", file=sys.stderr)
        sys.exit(1)

    with RestorationPipeline(cfg) as pipe:
        print("Webcam — press Q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, face_ok = pipe.process_bgr(frame)
            label = "Unmask" if face_ok else "No face"
            cv2.putText(
                out,
                label,
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (40, 220, 40) if face_ok else (40, 40, 220),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Unmask", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def _shared_ui_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--mask-mode", choices=("mesh", "bbox"), default="mesh")
    ap.add_argument("--backend", choices=_backend_choices(), default="auto")
    ap.add_argument("--unet-weights", type=Path, default=None)
    ap.add_argument("--feather", type=int, default=22)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--opencv-algo", choices=("ns", "telea"), default="ns")
    ap.add_argument("--no-multiscale", action="store_true")
    ap.add_argument("--no-bilateral", action="store_true")
    ap.add_argument("--bilateral-d", type=int, default=7)
    ap.add_argument("--bilateral-sigma-color", type=float, default=55.0)
    ap.add_argument("--bilateral-sigma-space", type=float, default=55.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Unmask — lower-face restoration")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("webcam", help="OpenCV window + camera")
    w.add_argument("--camera", type=int, default=0)
    _shared_ui_args(w)

    ui = sub.add_parser("ui", help="Gradio web UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--share", action="store_true")
    _shared_ui_args(ui)

    args = p.parse_args()
    if args.cmd == "webcam":
        run_webcam(
            args.camera,
            args.mask_mode,
            args.backend,
            args.unet_weights,
            args.feather,
            args.radius,
            args.opencv_algo,
            not args.no_multiscale,
            bilateral_refinement=not args.no_bilateral,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )
    else:
        from app import CSS, THEME, build_demo

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
