"""Single-file Yield Prediction FastAPI server.

Environment Variables (set before running):
  SUPABASE_URL=... (required for auth & logging)
  SUPABASE_ANON_KEY=... (optional if using service role)
  SUPABASE_SERVICE_ROLE_KEY=... (server key for inserts)
  SUPABASE_JWT_SECRET=... (optional: if provided we decode locally)
  MODEL_PATH=path/to/yield_model.joblib (optional; fallback heuristic used if missing)
  DISABLE_AUTH=1 (only for local dev/testing - bypasses supabase auth)

Run: uvicorn server:app --reload
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load .env early (optional, for local dev)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional heavy deps imported lazily
try:  # noqa: SIM105
    import joblib  # type: ignore
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import category_encoders as ce  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore
    pd = None  # type: ignore
    np = None  # type: ignore
    ce = None  # type: ignore

try:  # noqa: SIM105
    from supabase import create_client, Client  # type: ignore
except Exception:  # pragma: no cover
    create_client = None  # type: ignore
    Client = Any  # type: ignore

try:  # noqa: SIM105
    import jwt  # type: ignore
except Exception:  # pragma: no cover
    jwt = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "xgb_model.joblib"))
TARGET_ENCODER_PATH = Path(os.getenv("TARGET_ENCODER_PATH", "target_encoder.joblib"))
DISABLE_AUTH = os.getenv("DISABLE_AUTH") == "1"
SUPABASE_REQUESTS_TABLE = os.getenv("SUPABASE_REQUESTS_TABLE", "prediction")

# All possible seasons for one-hot encoding
ALL_SEASONS = ['Whole Year ', 'Kharif     ', 'Rabi       ', 'Autumn     ', 'Summer     ', 'Winter     ']

# Valid crop types
VALID_CROPS = [
    'Arecanut', 'Arhar/Tur', 'Castor seed', 'Cotton(lint)', 'Dry chillies', 'Gram', 'Jute', 'Linseed', 
    'Maize', 'Mesta', 'Niger seed', 'Onion', 'Other  Rabi pulses', 'Potato', 'Rapeseed &Mustard', 
    'Rice', 'Sesamum', 'Small millets', 'Sugarcane', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric', 
    'Wheat', 'Bajra', 'Black pepper', 'Cardamom', 'Coriander', 'Garlic', 'Ginger', 'Groundnut', 
    'Horse-gram', 'Jowar', 'Ragi', 'Cashewnut', 'Banana', 'Soyabean', 'Barley', 'Khesari', 'Masoor', 
    'Moong(Green Gram)', 'Other Kharif pulses', 'Safflower', 'Sannhamp', 'Sunflower', 'Urad', 
    'Peas & beans (Pulses)', 'other oilseeds', 'Other Cereals', 'Cowpea(Lobia)', 'Oilseeds total', 
    'Guar seed', 'Other Summer Pulses', 'Moth'
]

# Valid states
VALID_STATES = [
    'Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal', 'Puducherry', 'Goa', 'Andhra Pradesh', 
    'Tamil Nadu', 'Odisha', 'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram', 'Punjab', 
    'Uttar Pradesh', 'Haryana', 'Himachal Pradesh', 'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand', 
    'Jharkhand', 'Delhi', 'Manipur', 'Jammu and Kashmir', 'Telangana', 'Arunachal Pradesh', 'Sikkim'
]

# Valid seasons
VALID_SEASONS = ['Whole Year ', 'Kharif     ', 'Rabi       ', 'Autumn     ', 'Summer     ', 'Winter     ']


# ---------------------------------------------------------------------------
# Data Schemas
# ---------------------------------------------------------------------------
class PredictionInput(BaseModel):
    Crop: str = Field(..., description="The name of the crop cultivated")
    Crop_Year: int = Field(..., ge=1990, le=2030, description="The year in which the crop was grown")
    Season: str = Field(..., description="The specific cropping season")
    State: str = Field(..., description="The Indian state where the crop was cultivated")
    Area: float = Field(..., gt=0, description="The total land area (in hectares) under cultivation")
    Annual_Rainfall: float = Field(..., ge=0, description="The annual rainfall received (in mm)")
    Fertilizer: float = Field(..., ge=0, description="The total amount of fertilizer used (in kilograms)")
    Pesticide: float = Field(..., ge=0, description="The total amount of pesticide used (in kilograms)")
    
    def model_validate(cls, values):
        # Validate crop
        if values.get('Crop') not in VALID_CROPS:
            raise ValueError(f"Crop must be one of: {VALID_CROPS}")
        
        # Validate state
        if values.get('State') not in VALID_STATES:
            raise ValueError(f"State must be one of: {VALID_STATES}")
        
        # Validate season
        if values.get('Season') not in VALID_SEASONS:
            raise ValueError(f"Season must be one of: {VALID_SEASONS}")
        
        return values


class PredictionResponse(BaseModel):
    predicted_yield: float
    unit: str = "tons_per_hectare"
    recommendations: List[str]
    model_version: Optional[str] = None


class AuthUser(BaseModel):
    user_id: str
    email: Optional[str] = None


# ---------------------------------------------------------------------------
# Model Loader (lazy singleton)
# ---------------------------------------------------------------------------
_MODEL: Any | None = None
_TARGET_ENCODER: Any | None = None


def load_model() -> tuple[Any, Any]:
    global _MODEL, _TARGET_ENCODER
    if _MODEL is not None and _TARGET_ENCODER is not None:
        return _MODEL, _TARGET_ENCODER
    
    if not (MODEL_PATH.exists() and TARGET_ENCODER_PATH.exists() and joblib is not None):
        raise FileNotFoundError("Model files not found or joblib not available")
    
    try:
        _MODEL = joblib.load(MODEL_PATH)
        _TARGET_ENCODER = joblib.load(TARGET_ENCODER_PATH)
        return _MODEL, _TARGET_ENCODER
    except Exception as e:  # pragma: no cover
        print(f"Failed to load models: {e}.")
        raise FileNotFoundError("Model not connected") from e


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_data(data: PredictionInput, target_encoder: Any) -> Any:
    """
    Preprocess the input data according to the training pipeline:
    1. Remove Coconut crops (problematic)
    2. Create derived features (Fertilizer_per_Area, Pesticide_per_Area)
    3. Apply target encoding to Crop and State
    4. One-hot encode Season
    5. Return features in correct order
    """
    if pd is None or np is None:
        raise ImportError("pandas and numpy are required for preprocessing")
    
    # Check if Coconut crop (should be filtered out)
    if data.Crop == 'Coconut ':
        raise ValueError("Coconut crop is not supported due to data quality issues")
    
    # Create DataFrame from input
    df = pd.DataFrame([{
        'Crop': data.Crop,
        'Crop_Year': data.Crop_Year,
        'Season': data.Season,
        'State': data.State,
        'Area': data.Area,
        'Annual_Rainfall': data.Annual_Rainfall,
        'Fertilizer': data.Fertilizer,
        'Pesticide': data.Pesticide
    }])
    
    # Create derived features
    df['Fertilizer_per_Area'] = df['Fertilizer'] / (df['Area'] + 1e-6)
    df['Pesticide_per_Area'] = df['Pesticide'] / (df['Area'] + 1e-6)
    
    # Drop columns that won't be used for prediction
    X = df.drop(['Area', 'Fertilizer', 'Pesticide'], axis=1)
    
    # Apply target encoding
    X_target_encoded = target_encoder.transform(X)
    
    # One-hot encode Season
    X_target_encoded['Season'] = pd.Categorical(
        X_target_encoded['Season'],
        categories=ALL_SEASONS
    )
    
    # Create one-hot encoded features
    X_final = pd.get_dummies(X_target_encoded, columns=['Season'], dtype=int)
    
    # Ensure all season columns are present (in case some are missing)
    for season in ALL_SEASONS:
        col_name = f'Season_{season}'
        if col_name not in X_final.columns:
            X_final[col_name] = 0
    
    # Return as numpy array
    return X_final.values


# ---------------------------------------------------------------------------
# Recommendation logic (simple heuristic)
# ---------------------------------------------------------------------------
def generate_recommendations(data: PredictionInput) -> List[str]:
    recs: List[str] = []
    
    # Check fertilizer per area
    fertilizer_per_area = data.Fertilizer / (data.Area + 1e-6)
    if fertilizer_per_area < 50:
        recs.append("Consider increasing fertilizer application per hectare")
    elif fertilizer_per_area > 200:
        recs.append("Consider reducing fertilizer application to avoid over-fertilization")
    
    # Check pesticide per area
    pesticide_per_area = data.Pesticide / (data.Area + 1e-6)
    if pesticide_per_area < 0.1:
        recs.append("Consider increasing pesticide application for pest control")
    elif pesticide_per_area > 1.0:
        recs.append("Consider reducing pesticide application to avoid overuse")
    
    # Check rainfall
    if data.Annual_Rainfall < 500:
        recs.append("Low rainfall: consider irrigation or drought-resistant varieties")
    elif data.Annual_Rainfall > 2000:
        recs.append("High rainfall: ensure proper drainage to prevent waterlogging")
    
    # Check area size
    if data.Area < 1:
        recs.append("Small plot: consider intensive farming techniques")
    elif data.Area > 1000:
        recs.append("Large plot: consider mechanized farming for efficiency")
    
    if not recs:
        recs.append("Conditions look good. Maintain current practices.")
    
    return recs


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
_SUPABASE_CLIENT = None  # type: ignore


def get_supabase():  # -> Optional[Client]
    global _SUPABASE_CLIENT
    if not SUPABASE_URL or not (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY):
        if not SUPABASE_URL:
            logging.info("Supabase not configured: SUPABASE_URL missing")
        if not (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY):
            logging.info("Supabase not configured: No key provided (SERVICE_ROLE or ANON)")
        return None
    if _SUPABASE_CLIENT is None and create_client is not None:
        key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY  # prefer service key for inserts
        try:
            _SUPABASE_CLIENT = create_client(SUPABASE_URL, key)  # type: ignore
            logging.info("Supabase client initialized")
        except Exception as e:
            logging.error(f"Supabase client init failed: {e}")
            _SUPABASE_CLIENT = None
    return _SUPABASE_CLIENT


async def get_current_user(authorization: str = Header(None, alias="Authorization")) -> AuthUser:
    # Dev bypass if explicitly set OR no Supabase config provided
    if DISABLE_AUTH or not SUPABASE_URL:
        return AuthUser(user_id="00000000-0000-0000-0000-000000000000", email="dev@example.com")
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    token = authorization.split()[1]
    # Try local decode first
    if SUPABASE_JWT_SECRET and jwt is not None:
        try:
            payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
            return AuthUser(user_id=payload.get("sub") or payload.get("user_id", "unknown"), email=payload.get("email"))
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=401, detail="Invalid token") from e
    # Fallback: call supabase client
    sb = get_supabase()
    if sb is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        user_resp = sb.auth.get_user(token)  # type: ignore[attr-defined]
        user = user_resp.user  # type: ignore
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return AuthUser(user_id=getattr(user, "id", "unknown"), email=getattr(user, "email", None))
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=401, detail="Token validation failed") from e


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Yield Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _supabase_status() -> dict[str, bool | str]:
    return {
        "url_set": bool(SUPABASE_URL),
        "service_key_set": bool(SUPABASE_SERVICE_ROLE_KEY),
        "anon_key_set": bool(SUPABASE_ANON_KEY),
        "client_available": create_client is not None,
        "bypass_auth": DISABLE_AUTH,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "auth": not DISABLE_AUTH, "supabase": _supabase_status()}


@app.get("/debug/supabase")
async def debug_supabase():
    sb = get_supabase()
    info = _supabase_status()
    if sb is None:
        return {"ok": False, "info": info, "error": "Supabase not configured or client init failed"}
    # Try a lightweight select to verify connectivity
    try:
        res = sb.table(SUPABASE_REQUESTS_TABLE).select("id").limit(1).execute()  # type: ignore[attr-defined]
        return {"ok": True, "info": info, "select_sample": getattr(res, "data", None)}
    except Exception as e:
        return {"ok": False, "info": info, "error": str(e)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionInput, user: AuthUser = Depends(get_current_user)):
    # Try to load models but do not short-circuit; we always attempt a Supabase insert
    model = None
    target_encoder = None
    model_connected = True
    try:
        model, target_encoder = load_model()
    except FileNotFoundError:
        model_connected = False

    pred_value: float | None = None
    recs: list[str] = []
    response_obj: PredictionResponse | None = None

    if model_connected and model is not None and target_encoder is not None:
        try:
            # Preprocess the data
            features = preprocess_data(payload, target_encoder)
            
            # Make prediction (model expects log-transformed target)
            pred_log = model.predict(features)[0]
            
            # Transform back from log scale
            pred_value = float(np.expm1(pred_log))
            
        except ValueError as e:
            # Handle Coconut crop or other validation errors
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e
        
        recs = generate_recommendations(payload)

        # Build the response object once and reuse it for DB insert and HTTP response
        response_obj = PredictionResponse(
            predicted_yield=pred_value,
            recommendations=recs,
            model_version="v1",
        )

    # Always attempt to log request to Supabase (requests table)
    sb = get_supabase()
    if sb is not None:
        payload_data = payload.model_dump()
        if pred_value is not None:
            # augment payload with result so it's visible in test table
            payload_data["predicted_yield"] = pred_value
            payload_data["recommendations"] = recs
        record = {
            "user_id": user.user_id,
            "payload": payload_data,
            "note": "predicted" if pred_value is not None else "model_not_connected",
            "predicted_yield": pred_value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            sb.table(SUPABASE_REQUESTS_TABLE).insert(record).execute()  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            logging.error(f"Supabase log insert failed: {e}")


    # Response changes based on model connectivity
    if not model_connected or pred_value is None:
        return PlainTextResponse("Model not connected", status_code=200)

    # Return the exact same variables we stored
    return response_obj


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)