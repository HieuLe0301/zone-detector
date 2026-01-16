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

# how many frames to skip - 1
FRAME_FACTOR = 5

# color in BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GREY = (100,100,100)
BLACK = (0,0,0)
WHITE = (255,255,255)

# Coordinates for texturs
TESTING_LINE = [(415,385),(1398,773)]
LINES_THICKNESS = 2 #eg
POLY = [(803,539),(618,648),(1225,988),(1394,772)]

BLENDING_FACTOR = 0.5

WINDOW_TITLE = "Zone Detector"