import torch
import torch.onnx as onnx
import librosa
import numpy as np


def preprocess_audio(file_path, sr=16000, n_mfcc=40):
    # Load audio file
    y, sr = librosa.load(file_path, sr=sr)
    # Extract MFCCs (Mel-frequency cepstral coefficients) as features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Normalize or preprocess features if necessary
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc

# Example: Loading a pre-trained model
model = torch.load('embedding_model.ckpt', weights_only=True, map_location=torch.device('cpu'))
#model.eval()  # Set the model to inference mode
print(model)


audio_file_path = 'sample1.wav'
features = preprocess_audio(audio_file_path)

input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

onnx_file_path = "speaker_embedding_model.onnx"

onnx_program = torch.onnx.export(model, input_tensor, onnx_file_path, export_params=True, opset_version=11, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
onnx_program.save("embedding_model.onnx")