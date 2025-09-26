# Crop Yield Prediction API

A FastAPI-based service that predicts crop yield using XGBoost machine learning model with proper preprocessing pipeline including target encoding and one-hot encoding.

## Features

- **XGBoost Model**: Trained on crop yield data with 91.56% R² score on test set
- **Target Encoding**: For Crop and State categorical variables
- **One-Hot Encoding**: For Season categorical variable
- **Feature Engineering**: Automatic calculation of Fertilizer_per_Area and Pesticide_per_Area
- **Data Validation**: Ensures input data matches training schema
- **Coconut Filtering**: Automatically rejects Coconut crops (data quality issues)

## Model Files Required

Make sure these files are in your project directory:
- `xgb_model.joblib` - The trained XGBoost model
- `target_encoder.joblib` - The target encoder for Crop and State
- `all_seasons.joblib` - (Optional) Contains the seasons list

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify model files exist:**
```bash
ls -la *.joblib
```

## Running the API

### Option 1: Direct Python execution
```bash
python server.py
```

### Option 2: Using uvicorn
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Production mode
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Crop": "Arecanut",
    "Crop_Year": 1997,
    "Season": "Whole Year ",
    "State": "Assam",
    "Area": 73814.0,
    "Annual_Rainfall": 2051.4,
    "Fertilizer": 7024878.38,
    "Pesticide": 22882.34
  }'
```

## Input Schema

The API expects the following input format:

```json
{
  "Crop": "string",           // One of 54 valid crop types
  "Crop_Year": "integer",     // Year between 1990-2030
  "Season": "string",         // One of 6 valid seasons
  "State": "string",          // One of 30 valid Indian states
  "Area": "float",            // Area in hectares (> 0)
  "Annual_Rainfall": "float", // Rainfall in mm (≥ 0)
  "Fertilizer": "float",      // Fertilizer in kg (≥ 0)
  "Pesticide": "float"        // Pesticide in kg (≥ 0)
}
```

### Valid Values

**Crops (54 total):**
Arecanut, Arhar/Tur, Castor seed, Cotton(lint), Dry chillies, Gram, Jute, Linseed, Maize, Mesta, Niger seed, Onion, Other Rabi pulses, Potato, Rapeseed &Mustard, Rice, Sesamum, Small millets, Sugarcane, Sweet potato, Tapioca, Tobacco, Turmeric, Wheat, Bajra, Black pepper, Cardamom, Coriander, Garlic, Ginger, Groundnut, Horse-gram, Jowar, Ragi, Cashewnut, Banana, Soyabean, Barley, Khesari, Masoor, Moong(Green Gram), Other Kharif pulses, Safflower, Sannhamp, Sunflower, Urad, Peas & beans (Pulses), other oilseeds, Other Cereals, Cowpea(Lobia), Oilseeds total, Guar seed, Other Summer Pulses, Moth

**States (30 total):**
Assam, Karnataka, Kerala, Meghalaya, West Bengal, Puducherry, Goa, Andhra Pradesh, Tamil Nadu, Odisha, Bihar, Gujarat, Madhya Pradesh, Maharashtra, Mizoram, Punjab, Uttar Pradesh, Haryana, Himachal Pradesh, Tripura, Nagaland, Chhattisgarh, Uttarakhand, Jharkhand, Delhi, Manipur, Jammu and Kashmir, Telangana, Arunachal Pradesh, Sikkim

**Seasons (6 total):**
"Whole Year ", "Kharif     ", "Rabi       ", "Autumn     ", "Summer     ", "Winter     "

## Response Format

```json
{
  "predicted_yield": 5.2341,
  "unit": "tons_per_hectare",
  "recommendations": [
    "Consider increasing fertilizer application per hectare",
    "Conditions look good. Maintain current practices."
  ],
  "model_version": "v1"
}
```

## Testing

### Using the test script:
```bash
python test_api.py
```

### Manual testing with curl:

1. **Test health:**
```bash
curl http://localhost:8000/health
```

2. **Test prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

3. **Test Coconut rejection:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Crop": "Coconut ",
    "Crop_Year": 2020,
    "Season": "Whole Year ",
    "State": "Kerala",
    "Area": 50.0,
    "Annual_Rainfall": 2500.0,
    "Fertilizer": 2000.0,
    "Pesticide": 10.0
  }'
```

## Preprocessing Pipeline

The API automatically applies the same preprocessing as the training pipeline:

1. **Coconut Filtering**: Rejects Coconut crops due to data quality issues
2. **Feature Engineering**: 
   - `Fertilizer_per_Area = Fertilizer / (Area + 1e-6)`
   - `Pesticide_per_Area = Pesticide / (Area + 1e-6)`
3. **Column Dropping**: Removes Production, Area, Fertilizer, Pesticide
4. **Target Encoding**: Applies learned encodings for Crop and State
5. **One-Hot Encoding**: Converts Season to 6 binary features
6. **Log Transformation**: Model predicts on log-transformed yield, then transforms back

## Error Handling

- **400 Bad Request**: Invalid input data or Coconut crop
- **500 Internal Server Error**: Model loading or prediction errors
- **401 Unauthorized**: Missing or invalid authentication (if Supabase enabled)

## Environment Variables

- `MODEL_PATH`: Path to XGBoost model file (default: "xgb_model.joblib")
- `TARGET_ENCODER_PATH`: Path to target encoder file (default: "target_encoder.joblib")
- `DISABLE_AUTH`: Set to "1" to disable authentication for testing
- `SUPABASE_URL`: Supabase URL for logging (optional)
- `SUPABASE_SERVICE_ROLE_KEY`: Supabase service key (optional)

## Model Performance

- **Training R²**: 99.09%
- **Test R²**: 91.56%
- **Test MAE**: 0.9661 tons/hectare
- **Test RMSE**: 3.6518 tons/hectare
- **Test MedAE**: 0.1770 tons/hectare
