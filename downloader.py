import kagglehub

# Download latest version
path = kagglehub.model_download("google/yamnet/tfLite/classification-tflite")

print("Path to model files:", path)