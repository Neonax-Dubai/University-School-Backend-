# Zayed University AI inference image.
#
# Base: cctv/analytics-probe:25.11 = nvcr.io/nvidia/pytorch:25.11-py3 (driver 580 / CUDA 13.0,
# torch 2.10, TensorRT 10.14.1, ONNX Runtime GPU 1.29) + FFmpeg 6.1/GStreamer NVDEC +
# PyNvVideoCodec + OpenCV-headless 4.13 - the image validated on this GB10 (host setup §9).
#
# Everything added here is installed with --no-deps so pip can never replace NGC's torch,
# numpy, OpenCV or TensorRT builds; the few pure-Python dependencies they need are listed
# explicitly. `pip check` is limited to the packages added here.
FROM cctv/analytics-probe:25.11

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp

# Ultralytics (YOLO26 detection + pose, TensorRT export/runtime)
RUN pip install --no-cache-dir --no-deps "ultralytics>=8.4,<8.5" "ultralytics-thop>=2.0" "onnxslim>=0.1.60" \
      "polars>=1.0"
# InsightFace (face detection + ArcFace embedding on ONNX Runtime CUDA) and what it imports
RUN pip install --no-cache-dir --no-deps "scikit-image>=0.24" "imageio>=2.34" "tifffile>=2024.1" \
      "lazy_loader>=0.4" "networkx>=3.2" "easydict>=1.13" "prettytable>=3.10" "wcwidth>=0.2" \
      "albumentations==1.4.24" "albucore==0.0.23" "scikit-learn>=1.4" "joblib>=1.3" "threadpoolctl>=3.4" \
      "qudida>=0.0.4" "typing_extensions" "eval_type_backport" "stringzilla" "simsimd" \
 && pip install --no-cache-dir --no-deps --no-build-isolation "insightface==0.7.3"
# torchreid (OSNet Re-ID) - the deep-person-reid library the Dubai reid_poc imports
RUN pip install --no-cache-dir --no-deps "yacs>=0.1.8" "gdown>=5.2" "beautifulsoup4>=4.12" "soupsieve>=2.5" \
      "filelock" "h5py>=3.10" "six" "future" \
 && pip install --no-cache-dir --no-deps --no-build-isolation \
      "torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git"
# Qdrant client (vector store) - pure Python on top of NGC's grpc/httpx/pydantic
RUN pip install --no-cache-dir --no-deps "qdrant-client>=1.12,<2" "portalocker>=2.7" "h2>=4" "hpack>=4" \
      "hyperframe>=6" "protobuf>=4"

LABEL org.opencontainers.image.title="zayed-inference" \
      org.opencontainers.image.description="Zayed University CCTV AI inference runtime (GB10, NGC 25.11)"
WORKDIR /app
