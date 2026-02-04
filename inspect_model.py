from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\VED SHARMA\Python\Python 3.13\basic\image enhancer\venv\fastapi\DocClasfnModel.pt"
IMAGE_PATH = r"C:\Users\VED SHARMA\Python\Python 3.13\basic\image enhancer\venv\fastapi\outputs\pan6.png"
# ---------------------------------------


# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model(IMAGE_PATH)

# Get classification result
probs = results[0].probs           # Probabilities object
class_id = int(probs.top1)         # Best class index
confidence = float(probs.top1conf) # Confidence

# Class names (from model)
class_name = model.names[class_id]

print("Document Type:", class_name)
print("Confidence:", round(confidence, 4))

print("Predicted index:", class_id)
print("Predicted label:", CLASS_NAMES[class_id])
print("Confidence:", confidence)
