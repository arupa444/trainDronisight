"""Inference service for the web app: loads the 12-model set ONCE (cached), runs one image through
the full pipeline, and turns the result into a URL-bearing structured report for the frontend.

Reuses inference/pipeline.py verbatim so the app and the CLI can never drift. Device follows
CUDA -> MPS -> CPU (shared.device.select_device), threaded into every detector.
"""
import json
from pathlib import Path

from shared import config
from shared.device import select_device
from inference.pipeline import (discover_weights, build_detector, build_component_detector,
                                build_condition_detector, run_pipeline, result_to_rows, write_csv)
from inference.visualize import save_layers, LAYERS
from data_prep.preprocess import load_oriented_bgr, clahe_image


def _is_defect(condition_class):
    """A condition is a 'defect / attention' unless it is the family's *_normal class."""
    return bool(condition_class) and not condition_class.endswith("_normal")


# Component types that are themselves an attention finding regardless of any condition model.
ATTENTION_COMPONENTS = {"vegetation", "rust"}


class InspectionService:
    """Holds the loaded detectors. Build once per process; call analyze() per image."""

    def __init__(self, weights_dir, runs_dir, *, backend="yolo", use_clahe=True,
                 pole_conf=0.12, comp_conf=0.25, cond_conf=0.25,
                 pole_imgsz=640, comp_imgsz=1280, cond_imgsz=1280,
                 nms_iou=0.55, device=None):
        self.weights_dir = str(weights_dir)
        self.runs_dir = Path(runs_dir)
        self.backend = backend
        self.use_clahe = use_clahe
        self.pole_conf, self.comp_conf, self.cond_conf = pole_conf, comp_conf, cond_conf
        self.pole_imgsz, self.comp_imgsz, self.cond_imgsz = pole_imgsz, comp_imgsz, cond_imgsz
        self.nms_iou = nms_iou
        self.device = device or select_device()
        self.weights = discover_weights(self.weights_dir, config.SUBSETS)
        self._loaded = False
        self.pole_det = None
        self.component_dets = []
        self.condition_dets = {}

    # ---- discovery / readiness -------------------------------------------------
    @property
    def has_pole(self):
        return "pole" in self.weights

    def weights_status(self):
        """{subset: bool present} for every subset (drives the UI 'models' panel)."""
        return {s: (s in self.weights) for s in config.SUBSETS}

    # ---- model loading (cached) ------------------------------------------------
    def load_models(self, progress=None):
        if self._loaded:
            return
        if not self.has_pole:
            raise RuntimeError(
                f"No pole weights found under {self.weights_dir} "
                f"(expected **/pole/**/weights/best.pt). Set DRONISIGHT_WEIGHTS to your runs/ folder.")
        # reset accumulators so a retry after a partial load can't double-append detectors
        self.pole_det = None
        self.component_dets = []
        self.condition_dets = {}

        def _mk(subset, conf, imgsz):
            return build_detector(self.backend, self.weights[subset], conf, imgsz,
                                  config.SUBSET_CLASSES[subset], device=self.device)

        if progress:
            progress(f"Loading pole detector on {self.device.upper()}", 16)
        self.pole_det = _mk("pole", self.pole_conf, self.pole_imgsz)
        # component detectors honor COMPONENT_WEIGHTS_OVERRIDE (e.g. vegetation served by the
        # below_1000 detector filtered to just `vegetation`); iterate ALL slots so an overridden
        # slot loads even if its own solo weights are absent.
        for i, s in enumerate(config.COMP_SUBSETS):
            det = build_component_detector(s, self.weights_dir, self.weights, self.backend,
                                           self.comp_conf, self.comp_imgsz, self.device)
            if det is not None:
                if progress:
                    progress(f"Loading component model {s}", 18 + i)
                self.component_dets.append(det)
        # condition detectors are ensembles (specialist + old unified classifier) per CONDITION_ENSEMBLE
        for i, s in enumerate(config.COND_SUBSETS):
            det = build_condition_detector(s, self.weights_dir, self.weights, self.backend,
                                           self.cond_conf, self.cond_imgsz, self.device)
            if det is not None:
                if progress:
                    progress(f"Loading condition model {s}", 24 + i)
                self.condition_dets[s] = det
        self._loaded = True

    # ---- one image -------------------------------------------------------------
    def analyze(self, image_path, run_dir, progress=None):
        """Run the full pipeline on one image, write artifacts under run_dir, return the report dict.
        progress(stage:str, percent:int) is called as work advances (drives the loading screen)."""
        image_path = Path(image_path)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{image_path.stem}_inference"

        if progress:
            progress("Preparing image (EXIF orient + CLAHE)", 6)
        oriented = load_oriented_bgr(str(image_path))
        image = clahe_image(oriented) if self.use_clahe else oriented

        self.load_models(progress)

        if progress:
            progress("Detecting poles, components & conditions", 45)
        result = run_pipeline(image, self.pole_det, self.component_dets, self.condition_dets,
                              crop_dir=str(run_dir / "crops"), image_name=image_path.name,
                              pole_pad=config.POLE_CROP_PAD, name_stem=stem,
                              nms_iou=self.nms_iou, condition_pad=config.CONDITION_INFER_PAD,
                              component_pad=config.COMPONENT_CROP_PAD)

        if progress:
            progress("Rendering annotated views", 82)
        # draw viz on the un-CLAHE'd frame (identical geometry, natural colors)
        save_layers(oriented, result, str(run_dir / "viz"), stem)

        rows = result_to_rows(result)
        write_csv(rows, str(run_dir / "result.csv"))
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))

        if progress:
            progress("Building report", 94)
        report = self.build_report(result, run_dir, stem)
        if progress:
            progress("Done", 100)
        return report

    # ---- result dict -> frontend report (URLs + summary + attention flags) ----
    def build_report(self, result, run_dir, stem):
        run_dir = Path(run_dir)

        def url(path):
            if not path:
                return None
            return "/files/" + str(Path(path).resolve().relative_to(self.runs_dir.resolve())).replace("\\", "/")

        viz = {layer: url(run_dir / "viz" / layer / f"{stem}.jpg") for layer in LAYERS}

        class_counts, condition_counts, attention_items = {}, {}, []
        poles_out = []
        n_components = 0
        for pi, pole in enumerate(result["poles"]):
            comps_out = []
            for c in pole.get("components", []):
                n_components += 1
                cls = c["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
                has_family = cls in config.COMPONENT_TO_CONDITION_MODEL
                # multi-label: keep ALL in-family condition detections (an insulator can be both
                # broken AND chip_off), each flagged normal/defect
                conds = []
                for x in c.get("conditions", []):
                    xdef = _is_defect(x["class"])
                    condition_counts[x["class"]] = condition_counts.get(x["class"], 0) + 1
                    conds.append({"class": x["class"], "confidence": round(float(x["confidence"]), 3),
                                  "defect": xdef})
                any_defect = any(x["defect"] for x in conds)
                defect = (cls in ATTENTION_COMPONENTS) or any_defect
                best = c.get("condition")
                cond_obj = ({"class": best["class"], "confidence": round(float(best["confidence"]), 3),
                             "defect": _is_defect(best["class"])} if best else None)
                comp = {"class": cls, "confidence": round(float(c["confidence"]), 3),
                        "box_full": c["box_full"], "crop_url": url(c.get("crop_path")),
                        "has_condition_family": has_family,
                        "condition": cond_obj, "conditions": conds, "attention": defect}
                if defect:
                    defect_names = [x["class"] for x in conds if x["defect"]]
                    attention_items.append({"pole": pi, "component": cls,
                                            "condition": ", ".join(defect_names) or None,
                                            "crop_url": comp["crop_url"]})
                comps_out.append(comp)
            poles_out.append({"index": pi,
                              "confidence": round(float(pole["confidence"]), 3),
                              "box": pole["box"], "crop_url": url(pole.get("crop_path")),
                              "components": comps_out})

        return {
            "image": result["image"],
            "device": self.device,
            "layers": LAYERS,
            "viz": viz,
            "summary": {
                "poles": len(result["poles"]),
                "components": n_components,
                "attention": len(attention_items),
                "class_counts": class_counts,
                "condition_counts": condition_counts,
            },
            "attention_items": attention_items,
            "poles": poles_out,
            "downloads": {"csv": url(run_dir / "result.csv"), "json": url(run_dir / "result.json")},
        }
