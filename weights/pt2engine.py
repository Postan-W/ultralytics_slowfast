from ultralytics import YOLO

# Load a model
model = YOLO("climb_fall_0805.pt")
# Export the model
model.export(format="engine",device=[0])