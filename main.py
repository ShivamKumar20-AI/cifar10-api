from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="CIFAR-10 Image Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = tf.keras.models.load_model('cnn_cifar10_model.keras')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.get("/")
def root():
    return {"message": "CIFAR-10 Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    image = image.resize((32, 32))
    image = image.convert('RGB')

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    return {
        "prediction": class_names[predicted_index],
        "confidence": f"{confidence:.2%}",
        "all_scores": {
            class_names[i]: f"{float(predictions[0][i]):.2%}"
            for i in range(10)
        }
    }