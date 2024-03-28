#!/usr/bin/env python3

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")

    trained = model.train(data="config.yaml", epochs=1)
