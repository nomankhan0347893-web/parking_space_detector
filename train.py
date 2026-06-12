from ultralytics import YOLO
import shutil,os

model = YOLO("yolov8n.pt")

# fine tunning on our dataset for 50 epochs
results = model.train(
    data="data.yaml",
    epochs=50,   # trainig on 50 epochs for better learning
    imgsz=640,
    batch=8,      # it mean it will check 4 images per round in epochs
    device="cpu", 
    workers=0, 
    name="parking_detector",
    patience=10, # if losses is not improved for 10 epochs it stop running and save the
    lr0=0.001,
    plots=True
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")

#copy best model to models directory

shutil.copy(f"{results.save_dir}/weights/best.pt", "models/best.pt")

print("Best model copied to models/best.pt")
