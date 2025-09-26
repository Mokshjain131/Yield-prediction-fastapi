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
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

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
MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/yield_model.joblib"))
DISABLE_AUTH = os.getenv("DISABLE_AUTH") == "1"

FEATURE_ORDER = [
    "rainfall_mm",
    "temperature_c",
    "soil_moisture",
    "nitrogen",
    "phosphorus",
    "potassium",
]


# ---------------------------------------------------------------------------
# Data Schemas
# ---------------------------------------------------------------------------
class PredictionInput(BaseModel):
    rainfall_mm: float = Field(..., ge=0)
    temperature_c: float
    soil_moisture: float = Field(..., ge=0, le=1)
    nitrogen: float = Field(..., ge=0)
    phosphorus: float = Field(..., ge=0)
    potassium: float = Field(..., ge=0)
    crop_type: str
    region: str


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


def load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if MODEL_PATH.exists() and joblib is not None:
        try:
            _MODEL = joblib.load(MODEL_PATH)
            return _MODEL
        except Exception as e:  # pragma: no cover
            print(f"Failed to load model: {e}.")
            raise FileNotFoundError("Model not connected") from e
    # No model file or joblib missing
    raise FileNotFoundError("Model not connected")


# ---------------------------------------------------------------------------
# Recommendation logic (simple heuristic)
# ---------------------------------------------------------------------------
def generate_recommendations(data: PredictionInput) -> List[str]:
    recs: List[str] = []
    if data.soil_moisture < 0.3:
        recs.append("Increase irrigation (soil moisture < 30%)")
    elif data.soil_moisture > 0.8:
        recs.append("Reduce irrigation to avoid waterlogging")
    if data.nitrogen < 50:
        recs.append("Apply nitrogen-rich fertilizer")
    if data.phosphorus < 30:
        recs.append("Add phosphorus to support root growth")
    if data.potassium < 40:
        recs.append("Apply potassium for stress tolerance")
    if data.temperature_c < 15:
        recs.append("Cool conditions: consider cold-tolerant varieties")
    elif data.temperature_c > 34:
        recs.append("High heat: irrigate mornings/evenings to reduce stress")
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
        res = sb.table("prediction_requests").select("id").limit(1).execute()  # type: ignore[attr-defined]
        return {"ok": True, "info": info, "select_sample": getattr(res, "data", None)}
    except Exception as e:
        return {"ok": False, "info": info, "error": str(e)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionInput, user: AuthUser = Depends(get_current_user)):
    # Try to load model but do not short-circuit; we always attempt a Supabase insert
    model = None
    model_connected = True
    try:
        model = load_model()
    except FileNotFoundError:
        model_connected = False

    # Build features list
    try:
        features = [getattr(payload, f) for f in FEATURE_ORDER]
    except AttributeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pred_value: float | None = None
    recs: list[str] = []

    if model_connected and model is not None:
        try:
            pred = model.predict([features])[0]
            pred_value = float(pred)
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e
        recs = generate_recommendations(payload)

    # Always attempt to log request to Supabase (prediction_requests table)
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
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            sb.table("prediction_requests").insert(record).execute()  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            logging.error(f"Supabase log insert failed: {e}")

    # Response changes based on model connectivity
    if not model_connected or pred_value is None:
        return PlainTextResponse("Model not connected", status_code=200)

    return PredictionResponse(
        predicted_yield=pred_value,
        recommendations=recs,
        model_version="v1",
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)