from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "models"
YOLO11N_PATH = MODELS_PATH / "yolo11n.pt"
YOLO11N_OPEN_VINO_PATH = MODELS_PATH / "yolo11n_int8_openvino"
YOLO11N_OPEN_VINO_XML_PATH = YOLO11N_OPEN_VINO_PATH / "yolo11n.xml"

# raw source video 
RAW_VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "source.mp4"

# config OpenVINO
INT8 = True

# image size
DEFAULT_IMAGE_SIZE = 640
HALF_IMAGE_SIZE = 320

# color in BGR
BLUE = (255, 0, 0)
RED = (0, 255, 0)
GREEN = (0, 0, 255)


