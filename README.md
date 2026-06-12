# CIFAR-10 Image Classifier API

A REST API that classifies images into 10 categories using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. Built with **TensorFlow** and served via **FastAPI**.

---

## Demo Results

Real-world images tested against the live API:

| Image | Prediction | Confidence | Result |
|-------|------------|------------|--------|
| red car.jpeg | automobile | 99.54% | ✅ Correct |
| airplane flying.jpeg | airplane | 96.90% | ✅ Correct |
| ship at sea.jpeg | truck | 62.35% | ❌ Misclassified |

> **Why the ship was misclassified:** CIFAR-10 resizes all images to 32×32 pixels before classification. At that resolution, a large container ship with a boxy hull closely resembles a truck. This is a known limitation of low-resolution datasets — the model lacks the spatial detail needed to distinguish between similarly shaped objects.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Model | TensorFlow / Keras CNN |
| API | FastAPI |
| Server | Uvicorn |
| Dataset | CIFAR-10 (60,000 images, 10 classes) |
| Language | Python 3.10+ |

---

## CIFAR-10 Classes

The model classifies images into one of these 10 categories:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---

## Project Structure

```
cifar10-api/
├── main.py              # FastAPI app and /predict endpoint
├── model.py             # CNN model architecture and training script
├── cifar10_model.h5     # Trained model weights
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ShivamKumar20-AI/cifar10-api.git
cd cifar10-api
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn main:app --reload
```

The API will be live at `http://127.0.0.1:8000`

---

## API Usage

### Endpoint

```
POST /predict
```

### Request

Send a `multipart/form-data` request with your image file:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
```

### Response

```json
{
  "prediction": "automobile",
  "confidence": "99.54%",
  "all_scores": {
    "airplane": "0.00%",
    "automobile": "99.54%",
    "bird": "0.00%",
    "cat": "0.00%",
    "deer": "0.00%",
    "dog": "0.00%",
    "frog": "0.00%",
    "horse": "0.00%",
    "ship": "0.40%",
    "truck": "0.05%"
  }
}
```

### Interactive Docs

FastAPI auto-generates interactive documentation. Once the server is running, visit:

- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## Model Architecture

The CNN was trained from scratch on the CIFAR-10 training set:

- **Input:** 32×32 RGB images
- **Architecture:** Convolutional layers → MaxPooling → Dropout → Dense layers → Softmax output
- **Training:** 10 epochs on 50,000 training images
- **Validation:** Tested on 10,000 held-out images

---

## Limitations

- Images are resized to **32×32 pixels** before classification — fine detail is lost
- The model was trained on the standard CIFAR-10 dataset; domain-specific imagery (e.g., satellite photos, medical images) will not perform well
- Classes outside the 10 CIFAR-10 categories will be forced into the nearest match

---

## Author

**Shivam Kumar** — AI & ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-ShivamKumar20--AI-black?logo=github)](https://github.com/ShivamKumar20-AI)
