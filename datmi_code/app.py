from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Load model yang sudah disimpan
model = pickle.load(open('model.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))


# Definisikan input data menggunakan Pydantic
class InputData(BaseModel):
    heart_rate: float
    activity_level: float
    sleep_patterns: float
    health_status: float
    stress_level: float
    lesson_engagement: float
    progress: float
    feedback: int

# Endpoint utama untuk prediksi
@app.post("/predict")
def predict(data: InputData):
    # Mengubah inputan ke array numpy
    input_features = np.array([
        [
            data.heart_rate,
            data.activity_level,
            data.sleep_patterns,
            data.health_status,
            data.stress_level,
            data.lesson_engagement,
            data.progress,
            data.feedback
        ]
    ])
    
    # Melakukan prediksi
    prediction = model.predict(input_features)
    
    # Return hasil prediksi
    return {"predicted_health_concern_level": int(prediction[0])}
