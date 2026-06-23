# 🌾 Crop Recommendation System via Machine Learning

A smart web application that recommends the most suitable crop to grow based on soil nutrients and environmental conditions. It uses multiple machine learning algorithms so farmers and agriculturalists can compare results and choose the best-fit crop for their land.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Models Used](#ml-models-used)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)
- [Input Parameters](#input-parameters)
- [How It Works](#how-it-works)
- [Contributing](#contributing)

---

## 📖 About the Project

Farmers often struggle to decide which crop to plant based on their soil and climate conditions. This project solves that problem by taking key soil and weather parameters as input and recommending the most suitable crop using trained machine learning models.

The system supports **5 different ML algorithms** and allows users to either pick one or compare all of them side by side — helping make better, data-driven agricultural decisions.

---

## ✨ Features

- 🌱 Recommends the best crop from **22 different crops**
- 🤖 Supports **5 ML algorithms** — Decision Tree, Random Forest, Logistic Regression, SVM, and KNN
- 📊 Displays **model accuracy** for each algorithm
- 🔄 **Compare mode** — run all algorithms at once and compare predictions
- 🌐 Flask-powered REST API with CORS support
- ⚡ Fast predictions using pre-trained `.pkl` model files

---

## 🛠️ Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Backend      | Python, Flask, Flask-CORS         |
| ML Library   | scikit-learn, joblib              |
| Data         | pandas, numpy                     |
| Models       | `.pkl` (scikit-learn), `.h5` (CNN)|
| Runtime      | Python 3.11.9                     |

---

## 📂 Dataset

- **File:** `Crop_recommendation.csv`
- **Total Records:** 2,200
- **Features:** 7 input parameters
- **Target:** 22 crop types

**Crops covered:**
Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 🤖 ML Models Used

| Algorithm           | File                                      |
|---------------------|-------------------------------------------|
| Decision Tree       | `crop_recommendation_dt.pkl`              |
| Random Forest       | `crop_recommendation_rf.pkl`              |
| Logistic Regression | `crop_recommendation_lr.pkl`              |
| SVM                 | `crop_recommendation_svm.pkl`             |
| KNN                 | `crop_recommendation_knn.pkl`             |
| Best Model          | `best_crop_model.pkl`                     |
| CNN (Deep Learning) | `crop_recommendation_cnn.h5`              |

> All models are pre-trained and loaded at startup. Accuracy is computed on a 20% test split at runtime.

---

## 📁 Project Structure

```
crop-recommendation-system/
│
├── app.py                            # Main Flask application
├── Crop_recommendation.csv           # Dataset
│
├── best_crop_model.pkl               # Best performing model
├── crop_recommendation_dt.pkl        # Decision Tree model
├── crop_recommendation_rf.pkl        # Random Forest model
├── crop_recommendation_lr.pkl        # Logistic Regression model
├── crop_recommendation_svm.pkl       # SVM model
├── crop_recommendation_knn.pkl       # KNN model
├── crop_recommendation_cnn.h5        # CNN (Deep Learning) model
│
├── scaler.pkl                        # StandardScaler for input normalization
├── encoder.pkl                       # LabelEncoder for crop labels
│
├── requirements.txt                  # Python dependencies
└── runtime.txt                       # Python version specification
```
### System Workflow Diagram
![image Alt](https://raw.githubusercontent.com/arshlansaifi956-droid/crop-recommendation-system-via-machine-learning/7981ff1fd651ef81a8e9ebb14120a50ec2217509/system%20workflow%20diagram..jpg)
## Home page
### Models selection
![image Alt]()
### Confusion matrix and models comparison
![image Alt]((https://raw.githubusercontent.com/arshlansaifi956-droid/crop-recommendation-system-via-machine-learning/a79e77f2aa56b7ad888c33ad2f967dc792e39e83/Model%20accuracy%20and%20confusion%20matrix.png)
### Results
![image Alt]()


## 🚀 Getting Started

### Prerequisites

- Python 3.11.9
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/crop-recommendation-system.git
   cd crop-recommendation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in your browser**
   ```
   http://localhost:5000
   ```

---

## 🔌 API Endpoints

### `GET /`
Returns the main web interface.

---

### `POST /recommend`
Get a crop recommendation using a selected algorithm.

**Request Body:**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9,
  "algorithm": "Random Forest"
}
```

**Response:**
```json
{
  "crop": "rice",
  "algorithm": "Random Forest",
  "accuracy": 99.32
}
```

---

### `POST /compare`
Run all algorithms and compare their predictions for the same input.

**Request Body:** *(same as `/recommend`, no `algorithm` field needed)*

**Response:**
```json
{
  "Random Forest": { "crop": "rice", "accuracy": 99.32, "error": null },
  "Decision Tree": { "crop": "rice", "accuracy": 98.18, "error": null },
  "SVM":           { "crop": "rice", "accuracy": 97.95, "error": null },
  "KNN":           { "crop": "rice", "accuracy": 97.50, "error": null },
  "Logistic Regression": { "crop": "rice", "accuracy": 95.45, "error": null }
}
```

---

## 🧪 Input Parameters

| Parameter     | Description                         | Unit      |
|---------------|-------------------------------------|-----------|
| `N`           | Nitrogen content in soil            | mg/kg     |
| `P`           | Phosphorus content in soil          | mg/kg     |
| `K`           | Potassium content in soil           | mg/kg     |
| `temperature` | Average temperature                 | °C        |
| `humidity`    | Relative humidity                   | %         |
| `ph`          | pH value of soil                    | 0–14      |
| `rainfall`    | Annual rainfall                     | mm        |

---

## ⚙️ How It Works

1. **Data Preprocessing** — The dataset is loaded and split into features (N, P, K, temperature, humidity, ph, rainfall) and labels (crop name). Features are scaled using `StandardScaler` and labels are encoded using `LabelEncoder`.

2. **Model Loading** — All pre-trained `.pkl` model files are loaded at startup. Accuracy is calculated on a held-out 20% test set.

3. **Prediction** — When the user submits soil/climate data via the API, the input is scaled using the same scaler, passed through the selected model, and the predicted label is decoded back to a crop name.

4. **Compare Mode** — All models run predictions on the same input simultaneously, allowing side-by-side comparison.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

