# Yield Prediction FastAPI

A robust machine learning API service designed to predict crop yields in India. This project leverages **FastAPI** for high-performance web serving and **XGBoost** for accurate regression predictions. It also includes a recommendation system to suggest improvements for farming practices.

## 🚀 Features

- **Accurate Predictions**: Uses a trained XGBoost regressor (R² score ~91.56%) to predict crop yield (tons/hectare).
- **Smart Recommendations**: Generates actionable advice based on fertilizer/pesticide usage, rainfall, and plot size.
- **Data Preprocessing**: Handles complex feature engineering including:
  - Target Encoding for high-cardinality categorical variables (Crop, State).
  - One-Hot Encoding for seasonal data.
  - Automatic derivation of per-hectare metrics.
- **Input Validation**: Strict schema validation using Pydantic to ensure data integrity.
- **Supabase Integration**: 
  - Logs all prediction requests for auditing and analysis.
  - Supports JWT-based authentication (optional).
- **Health Monitoring**: Dedicated health check endpoint with database connectivity status.

## 🛠️ Tech Stack

- **Framework**: FastAPI, Uvicorn
- **Machine Learning**: XGBoost, Scikit-learn, Category Encoders, Joblib
- **Data Processing**: Pandas, NumPy
- **Database/Auth**: Supabase (PostgreSQL)
- **Language**: Python 3.10+

## 📂 Project Structure

```
Yield-prediction-fastapi/
├── model/                  # Directory for ML artifacts (optional location)
├── server.py               # Main FastAPI application and logic
├── test_api.py             # Python script for testing endpoints
├── requirements.txt        # Python dependencies
├── xgb_model.joblib        # Trained XGBoost model
├── target_encoder.joblib   # Trained Target Encoder
├── payload.json            # Sample payload for testing
├── .env                    # Environment variables (not committed)
└── README.md               # Project documentation
```

## ⚡ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Yield-prediction-fastapi
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Model Files**
   Ensure `xgb_model.joblib` and `target_encoder.joblib` are present in the root directory.

## ⚙️ Configuration

Create a `.env` file in the root directory to configure the application.

```ini
# Supabase Configuration (Required for logging & auth)
SUPABASE_URL="your_supabase_url"
SUPABASE_SERVICE_ROLE_KEY="your_service_role_key"
SUPABASE_ANON_KEY="your_anon_key"
SUPABASE_JWT_SECRET="your_jwt_secret" # Optional: for local JWT decoding

# App Configuration
MODEL_PATH="xgb_model.joblib"
TARGET_ENCODER_PATH="target_encoder.joblib"
SUPABASE_REQUESTS_TABLE="prediction"

# Development
DISABLE_AUTH=1  # Set to 1 to bypass authentication for local testing
```

## 🏃‍♂️ Running the Server

Start the server using Uvicorn:

```bash
uvicorn server:app --reload
```

The API will be available at `http://localhost:8000`.

## 🔌 API Endpoints

### 1. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Checks API status and database connectivity.

### 2. Predict Yield
- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Predicts crop yield and provides recommendations.
- **Body**:
  ```json
  {
    "Crop": "Rice",
    "Crop_Year": 2021,
    "Season": "Kharif     ",
    "State": "Assam",
    "Area": 5000.0,
    "Annual_Rainfall": 2000.0,
    "Fertilizer": 500000.0,
    "Pesticide": 1500.0
  }
  ```
  *Note: `Season` field often requires specific spacing (e.g., "Kharif     ") to match the training data categories.*

## 🧪 Testing

You can test the API using the provided script or `curl`.

**Using Python Script:**
```bash
python test_api.py
```

**Using cURL (PowerShell):**
```powershell
$json = @'
{
  "Crop": "Arhar/Tur",
  "Crop_Year": 2000,
  "Season": "Kharif     ",
  "State": "Assam",
  "Area": 6637.0,
  "Annual_Rainfall": 2051.4,
  "Fertilizer": 631643.29,
  "Pesticide": 2057.47
}
'@
$json | curl.exe -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" --data-binary @-
```

## 📝 License

This project is licensed under the MIT License.
