from config import constants as c
from ultralytics import YOLO #type: ignore
import argparse


def prepare_yolo11n():
    YOLO(c.YOLO11N_PATH)

def prepare_yolo11nopenvino(int8: bool, imgsz: int):
    name = "yolo11n_int8_openvino" if int8 else "yolo11n_fp32_openvino"
    if not c.YOLO11N_OPEN_VINO_PATH.exists():
        temp_model = YOLO(c.YOLO11N_PATH)
        temp_model.export(format = 'openvino', 
                          int8 = int8, 
                          imgsz = imgsz,
                          project = c.MODELS_PATH,
                          name = name)


def main():
    int8 = False
    imgsz = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo11n", action = "store_true")
    parser.add_argument("--yolo11nopenvino", action = "store_true")
    parser.add_argument("--int8",action = "store_true")
    parser.add_argument("--fp32",action = "store_true")
    parser.add_argument("--fullsize",action = "store_true")
    parser.add_argument("--halfsize",action = "store_true")
    parser.add_argument("--detect",action = "store_true")
    parser.add_argument("--track",action = "store_true")
    parser.add_argument("--segment",action = "store_true")
    

    args = parser.parse_args()

    if not args.yolo11n and not args.yolo11nopenvino:
        parser.error("No model specified")

    if args.yolo11nopenvino:
        if not args.int8 and not args.fp32:
            parser.error("Floats format not specified. Add --int8 or fp32")
        if not args.fullsize and not args.halfsize:
            parser.error("Image size not specified. Add --fullsize or --halfsize")
    
    if args.yolo11n:
        prepare_yolo11n()
    elif args.yolo11nopenvino:
        if args.int8:
            int8 = True
        elif args.fp32:
            int8 = False
        
        if args.fullsize:
            imgsz = c.DEFAULT_IMAGE_SIZE
        elif args.halfsize:
            imgsz = c.HALF_IMAGE_SIZE

        
        prepare_yolo11nopenvino(int8 = int8, imgsz = imgsz)



if __name__ == "__main__":
    main()
