from ultralytics import YOLO
from PIL import Image
import numpy as np



def train(n_epochs):
    #TRAIN
    model = YOLO("yolov8n.pt")
    # Use the model
    model.train(data="config.yaml", epochs=n_epochs)  # train the model
    return model

def main():
    n = 30
    model = train(n)

if __name__ == "__main__":
    main()
