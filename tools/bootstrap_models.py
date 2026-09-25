#!/usr/bin/env python3
"""
Zayed model bootstrap: weights -> verification -> TensorRT engines on THIS GB10 -> GPU smoke tests.

    tools/bootstrap_models.sh            (runs this inside zayed/inference:25.11 with the GPU)

Required models, all public pretrained weights (no Dubai custom model is used):

    yolo26l        COCO-80 detector            -> yolo26l.engine       (imgsz 640, batch 1..6)
    yolo26l-pose   COCO-17 pose                -> yolo26l-pose.engine  (imgsz 960, batch 1..4)
    yolo26m-pose   COCO-17 pose                -> yolo26m-pose.engine  (imgsz 640, batch 1..3)
    buffalo_l      InsightFace det+recognition    ONNX Runtime CUDA EP (no engine)
    osnet_ain_x1_0 torchreid Re-ID                PyTorch CUDA         (no engine)

1. Weights are looked for under MODELS_DIR; missing ones are downloaded from their public
   release (Ultralytics assets, InsightFace v0.7 release, torchreid model zoo).
2. Every weight file is hashed; the first successful bootstrap pins SHA-256 + size in
   MANIFEST.json, and later runs refuse a file that no longer matches.
3. TensorRT engines are BUILT HERE (never copied from another machine) into
   ENGINE_ROOT/<trt version>_<gpu>_sm<cc>_drv<driver>/ with a JSON sidecar recording how and
   where they were built. A cached engine is reused only if its sidecar matches this
   machine's TensorRT, GPU, compute capability, driver and source-weight hash.
4. Each model runs a smoke test on the GPU. Any CPU fallback (no CUDA, ONNX Runtime not on
   CUDAExecutionProvider, Re-ID not on cuda) is a FAILURE, not a warning.
5. The key directory is published as ENGINE_ROOT/current only after every test passed.
"""
import datetime
import hashlib
import json
import os
import platform
import shutil
import sys
import time

MODELS_DIR = os.environ.get("MODELS_DIR", "/models/zayed")
ENGINE_ROOT = os.environ.get("ENGINE_ROOT", "/engines/zayed")
FORCE_REBUILD = os.environ.get("FORCE_REBUILD", "0") == "1"

ENGINES = {
    "yolo26l": {"task": "detect", "imgsz": 640, "batch": 6},
    "yolo26l-pose": {"task": "pose", "imgsz": 960, "batch": 4},
    "yolo26m-pose": {"task": "pose", "imgsz": 640, "batch": 3},
}
REPORT = {"started": datetime.datetime.now().astimezone().isoformat(), "models": {}, "engines": {},
          "smoke": {}, "ok": False}


def log(msg):
    print(f"[bootstrap] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    REPORT["error"] = msg
    log(f"FAILED: {msg}")
    print(json.dumps(REPORT, indent=2))
    sys.exit(1)


# ------------------------------------------------------------------ environment
def machine():
    import tensorrt as trt
    import torch
    if not torch.cuda.is_available():
        fail("CUDA is not available - refusing to bootstrap for CPU")
    props = torch.cuda.get_device_properties(0)
    driver = "unknown"
    try:
        import pynvml
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
    except Exception:                                            # noqa: BLE001
        pass
    info = {"gpu": props.name, "compute_capability": f"{props.major}.{props.minor}",
            "tensorrt": trt.__version__, "driver": driver, "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__, "arch": platform.machine()}
    key = (f"trt{trt.__version__}_{props.name.replace(' ', '-')}_sm{props.major}{props.minor}"
           f"_drv{driver}")
    return info, key


# ------------------------------------------------------------------ manifest
def load_manifest():
    path = os.path.join(MODELS_DIR, "MANIFEST.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"files": {}}


def save_manifest(manifest):
    path = os.path.join(MODELS_DIR, "MANIFEST.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def pin(manifest, rel, source, licence):
    path = os.path.join(MODELS_DIR, rel)
    digest, size = sha256(path), os.path.getsize(path)
    known = manifest["files"].get(rel)
    if known and (known["sha256"] != digest or known["bytes"] != size):
        fail(f"{rel} does not match its pinned hash (expected {known['sha256'][:16]}..., got "
             f"{digest[:16]}...) - refusing a changed weight file")
    if not known:
        manifest["files"][rel] = {"sha256": digest, "bytes": size, "source": source, "licence": licence,
                                  "pinned_at": datetime.datetime.now().astimezone().isoformat()}
        log(f"pinned {rel} sha256={digest[:16]}... ({size} bytes)")
    return digest


# ------------------------------------------------------------------ weights
def ensure_yolo(manifest, name):
    from ultralytics.utils.downloads import attempt_download_asset
    rel = f"{name}.pt"
    path = os.path.join(MODELS_DIR, rel)
    if not os.path.exists(path):
        log(f"downloading {rel} (Ultralytics release asset)")
        cwd = os.getcwd()
        os.chdir(MODELS_DIR)
        try:
            got = attempt_download_asset(rel)
        finally:
            os.chdir(cwd)
        if not got or not os.path.exists(path):
            fail(f"could not download {rel}")
    digest = pin(manifest, rel, "https://github.com/ultralytics/assets/releases", "AGPL-3.0 (Ultralytics)")
    REPORT["models"][name] = {"path": path, "sha256": digest}
    return path, digest


def ensure_buffalo(manifest):
    root = os.path.join(MODELS_DIR, "insightface")
    model_dir = os.path.join(root, "models", "buffalo_l")
    if not os.path.isdir(model_dir) or not any(f.endswith(".onnx") for f in os.listdir(model_dir)):
        log("downloading InsightFace buffalo_l (v0.7 release)")
        from insightface.utils.storage import ensure_available
        ensure_available("models", "buffalo_l", root=root)
        zip_path = os.path.join(root, "models", "buffalo_l.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)                    # keep only the extracted models
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".onnx"))
    if not files:
        fail("buffalo_l has no ONNX files after download")
    digests = {f: pin(manifest, f"insightface/models/buffalo_l/{f}",
                      "https://github.com/deepinsight/insightface/releases/tag/v0.7",
                      "InsightFace pretrained models: non-commercial research use") for f in files}
    REPORT["models"]["buffalo_l"] = {"path": model_dir, "files": digests}
    return root


def ensure_osnet(manifest):
    torch_home = os.path.join(MODELS_DIR, "torch")
    os.environ["TORCH_HOME"] = torch_home
    from torchreid.utils import FeatureExtractor
    log("loading osnet_ain_x1_0 (torchreid; downloads its pretrained weights on first use)")
    extractor = FeatureExtractor(model_name="osnet_ain_x1_0", device="cuda")
    ckpt_dir = os.path.join(torch_home, "checkpoints")
    files = sorted(f for f in os.listdir(ckpt_dir) if "osnet_ain_x1_0" in f) if os.path.isdir(ckpt_dir) else []
    if not files:
        fail("osnet_ain_x1_0 weights were not cached under TORCH_HOME")
    digest = pin(manifest, f"torch/checkpoints/{files[0]}",
                 "torchreid model zoo (osnet_ain_x1_0, ImageNet-pretrained, as the Dubai reid_poc used)",
                 "MIT (torchreid)")
    REPORT["models"]["osnet_ain_x1_0"] = {"path": os.path.join(ckpt_dir, files[0]), "sha256": digest}
    return extractor


# ------------------------------------------------------------------ engines
def build_engine(name, spec, pt_path, pt_digest, info, key_dir):
    engine = os.path.join(key_dir, f"{name}.engine")
    sidecar = engine + ".json"
    want = {"source_sha256": pt_digest, "imgsz": spec["imgsz"], "batch": spec["batch"], "half": True,
            "dynamic": True, **{k: info[k] for k in ("gpu", "compute_capability", "tensorrt", "driver")}}
    if not FORCE_REBUILD and os.path.exists(engine) and os.path.exists(sidecar):
        with open(sidecar) as f:
            have = json.load(f)
        if all(have.get(k) == v for k, v in want.items()):
            log(f"{name}.engine cached and compatible - reusing")
            REPORT["engines"][name] = {"path": engine, "reused": True, **want}
            return engine
        log(f"{name}.engine cached but built differently - rebuilding")
    from ultralytics import YOLO
    log(f"building {name}.engine on this GPU (FP16, dynamic batch 1..{spec['batch']}, imgsz {spec['imgsz']})")
    t0 = time.monotonic()
    out = YOLO(pt_path, task=spec["task"]).export(format="engine", half=True, dynamic=True,
                                                  batch=spec["batch"], imgsz=spec["imgsz"], device=0,
                                                  simplify=True, verbose=False)
    built = str(out)
    if not os.path.exists(built):
        fail(f"export of {name} produced no engine")
    shutil.move(built, engine)
    onnx = os.path.splitext(pt_path)[0] + ".onnx"
    if os.path.exists(onnx):
        os.remove(onnx)                          # intermediate only
    record = {**want, "built_at": datetime.datetime.now().astimezone().isoformat(),
              "build_seconds": round(time.monotonic() - t0, 1), "cuda_runtime": info["cuda_runtime"],
              "torch": info["torch"], "task": spec["task"]}
    with open(sidecar, "w") as f:
        json.dump(record, f, indent=2)
    log(f"{name}.engine built in {record['build_seconds']} s")
    REPORT["engines"][name] = {"path": engine, "reused": False, **record}
    return engine


# ------------------------------------------------------------------ smoke tests
def smoke_yolo(name, engine, spec):
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    import cv2
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    model = YOLO(engine, task=spec["task"])
    batch = [img] * min(3, spec["batch"])
    t0 = time.perf_counter()
    results = model.predict(batch, imgsz=spec["imgsz"], conf=0.25, device=0, verbose=False)
    ms = (time.perf_counter() - t0) * 1000
    persons = int(sum(int((r.boxes.cls == 0).sum()) for r in results[:1]))
    entry = {"batch": len(batch), "first_call_ms": round(ms, 1), "persons_on_bus_jpg": persons}
    if spec["task"] == "pose":
        kp = results[0].keypoints
        entry["keypoints_shape"] = list(kp.data.shape) if kp is not None else None
        ok = persons >= 3 and kp is not None and kp.data.shape[1] == 17
    else:
        ok = persons >= 3
    t0 = time.perf_counter()
    for _ in range(5):
        model.predict(batch, imgsz=spec["imgsz"], conf=0.25, device=0, verbose=False)
    entry["warm_ms_per_batch"] = round((time.perf_counter() - t0) * 1000 / 5, 1)
    entry["ok"] = bool(ok)
    REPORT["smoke"][name] = entry
    if not ok:
        fail(f"smoke test failed for {name}: {entry}")
    log(f"{name}: {persons} persons on bus.jpg, warm {entry['warm_ms_per_batch']} ms per batch of {len(batch)}")


def smoke_face(root):
    import cv2
    from insightface.app import FaceAnalysis
    from ultralytics.utils import ASSETS
    app = FaceAnalysis(name="buffalo_l", root=root, providers=["CUDAExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
    providers = {task: model.session.get_providers() for task, model in app.models.items()}
    on_gpu = all(p and p[0] == "CUDAExecutionProvider" for p in providers.values())
    faces = app.get(cv2.imread(str(ASSETS / "zidane.jpg")))
    dims = sorted({int(f.normed_embedding.shape[0]) for f in faces}) if faces else []
    entry = {"providers": providers, "faces_on_zidane_jpg": len(faces), "embedding_dims": dims,
             "ok": bool(on_gpu and len(faces) >= 1 and dims == [512])}
    REPORT["smoke"]["buffalo_l"] = entry
    if not on_gpu:
        fail(f"InsightFace is not running on CUDAExecutionProvider: {providers}")
    if not entry["ok"]:
        fail(f"InsightFace smoke test failed: {entry}")
    log(f"buffalo_l: {len(faces)} faces, 512-D embeddings, providers {sorted({p[0] for p in providers.values()})}")


def smoke_osnet(extractor):
    import numpy as np
    import torch
    device = next(extractor.model.parameters()).device
    crops = [np.random.default_rng(i).integers(0, 255, (256, 128, 3), dtype=np.uint8) for i in range(4)]
    feats = extractor(crops)
    entry = {"device": str(device), "feature_shape": list(feats.shape),
             "ok": device.type == "cuda" and tuple(feats.shape) == (4, 512)}
    REPORT["smoke"]["osnet_ain_x1_0"] = entry
    if device.type != "cuda":
        fail(f"OSNet is on {device}, refusing CPU fallback")
    if not entry["ok"]:
        fail(f"OSNet smoke test failed: {entry}")
    log(f"osnet_ain_x1_0: features {tuple(feats.shape)} on {device}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    info, key = machine()
    REPORT["machine"], REPORT["engine_key"] = info, key
    log(f"machine: {info}")
    key_dir = os.path.join(ENGINE_ROOT, key)
    os.makedirs(key_dir, exist_ok=True)
    manifest = load_manifest()

    pts = {name: ensure_yolo(manifest, name) for name in ENGINES}
    face_root = ensure_buffalo(manifest)
    extractor = ensure_osnet(manifest)
    save_manifest(manifest)

    engines = {name: build_engine(name, spec, pts[name][0], pts[name][1], info, key_dir)
               for name, spec in ENGINES.items()}
    for name, spec in ENGINES.items():
        smoke_yolo(name, engines[name], spec)
    smoke_face(face_root)
    smoke_osnet(extractor)

    current = os.path.join(ENGINE_ROOT, "current")
    tmp = current + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(key, tmp)
    os.replace(tmp, current)
    REPORT["current"] = f"{current} -> {key}"
    REPORT["ok"] = True
    REPORT["finished"] = datetime.datetime.now().astimezone().isoformat()
    with open(os.path.join(key_dir, "bootstrap_report.json"), "w") as f:
        json.dump(REPORT, f, indent=2)
    print(json.dumps(REPORT, indent=2))
    log("ALL MODELS READY")


if __name__ == "__main__":
    main()
