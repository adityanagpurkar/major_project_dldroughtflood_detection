"""
3_api_backend.py
FastAPI REST API to serve localized predictions and trigger Twilio SMS alerts.
Supports 4 Multi-Modal Features: [Precipitation, NDVI, NDWI, LST].
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from twilio.rest import Client
import importlib.util
import sys
import torch
import numpy as np

# Load environment variables
load_dotenv()

# Import geographic registry
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geo_registry import VIDARBHA_GEO_DATA

app = FastAPI(title="Vidarbha Flood & Drought Localized Early Warning API (v1.0)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamically import 2_model_engine.py
model_engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_model_engine.py")
spec = importlib.util.spec_from_file_location("model_engine", model_engine_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {model_engine_path}")
model_engine = importlib.util.module_from_spec(spec)
sys.modules["model_engine"] = model_engine
spec.loader.exec_module(model_engine)

VidarbhaHydroLSTM = model_engine.VidarbhaHydroLSTM
predict_risk = model_engine.predict_risk

# Global Model Initialization (input_size=4)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vidarbha_model.pth')
model = VidarbhaHydroLSTM(input_size=4)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
        print("Loaded 4-feature trained model from vidarbha_model.pth")
    except Exception as e:
        print(f"Warning: Could not load {MODEL_PATH}: {e}. Using randomly initialized weights.")
else:
    print("Warning: Pre-trained model not found. Using untrained weights.")

model.eval()

# Pydantic models for request validation
class PredictionRequest(BaseModel):
    district: str = Field(..., json_schema_extra={"example": "Nagpur"})
    taluka: Optional[str] = Field(None, json_schema_extra={"example": "Katol"})
    phone: str = Field(..., json_schema_extra={"example": "+919876543210"})
    sequence: List[List[float]] # 14x4 nested list: [Precip, NDVI, NDWI, LST]

class PredictionResponse(BaseModel):
    district: str
    taluka: Optional[str]
    flood_probability: float
    drought_severity: int
    mean_lst_celsius: float
    alert_sent: bool
    message: str
    risk_level: str
    vulnerability_details: Optional[Dict[str, Any]] = None

class DistrictBatchRequest(BaseModel):
    district: str = Field(..., json_schema_extra={"example": "Nagpur"})
    scenario: str = Field("Random Scenario", json_schema_extra={"example": "Simulate Drought Risk"})

def trigger_sms_alert(phone: str, flood_prob: float, drought_sev: int, district: str, taluka: Optional[str] = None) -> bool:
    """Trigger Twilio SMS alert based on severity."""
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_phone = os.getenv('TWILIO_PHONE_NUMBER')
    
    if not all([account_sid, auth_token, from_phone]):
        return False
        
    if not phone or phone == "+91" or len(phone.strip()) < 8:
        return False
        
    try:
        client = Client(account_sid, auth_token)
        loc_str = f"{taluka}, {district}" if taluka else district
        msg_body = f"URGENT HYDRO-CLIMATE ALERT ({loc_str}):\n"
        if flood_prob > 0.85:
            msg_body += f"- HIGH FLOOD RISK! Probability: {flood_prob:.1%}\n"
        if drought_sev >= 2:
            msg_body += f"- CRITICAL DROUGHT WARNING! Severity Level: {drought_sev}\n"
            
        msg_body += "Vidarbha Early Warning System V1.0. Please take necessary precautions."
        
        message = client.messages.create(
            body=msg_body,
            from_=from_phone,
            to=phone
        )
        return True
    except Exception as e:
        print(f"SMS dispatch failed: {e}")
        return False

@app.get("/")
def health_check():
    return {
        "status": "ok", 
        "service": "Vidarbha Localized Flood & Drought Early Warning API",
        "version": "1.0",
        "features": ["Precipitation", "NDVI", "NDWI", "LST"]
    }

@app.get("/regions")
def get_regions():
    """Return all districts and their localized talukas."""
    return VIDARBHA_GEO_DATA

@app.post("/predict", response_model=PredictionResponse)
def predict_and_alert(request: PredictionRequest):
    """
    Accepts 14-day sequence of 4 features [Precipitation, NDVI, NDWI, LST],
    returns localized risk metrics, and triggers alerts if critical.
    """
    if len(request.sequence) != 14 or not all(len(day) == 4 for day in request.sequence):
        raise HTTPException(status_code=400, detail="Sequence must be exactly 14 days, each with 4 features (Precipitation, NDVI, NDWI, LST).")
        
    try:
        results = predict_risk(model, request.sequence)
        flood_prob = float(results['flood_probability'])
        drought_sev = int(results['drought_severity'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        
    # Calculate average Land Surface Temperature over the 14 days
    lst_values = [day[3] for day in request.sequence]
    mean_lst = float(np.mean(lst_values))
    
    # Check localized geographical profile if taluka specified
    vuln_info = None
    if request.district in VIDARBHA_GEO_DATA and request.taluka in VIDARBHA_GEO_DATA[request.district]["talukas"]:
        taluka_meta = VIDARBHA_GEO_DATA[request.district]["talukas"][request.taluka]
        vuln_info = {
            "river_proximity": taluka_meta.get("river_proximity", taluka_meta.get("river", "N/A")),
            "elevation_m": taluka_meta.get("elevation_m", taluka_meta.get("elevation", 300)),
            "base_lst_c": taluka_meta.get("base_lst_c", taluka_meta.get("base_lst", 38.0)),
            "coordinates": taluka_meta.get("coords")
        }
        
    # Determine risk category
    if flood_prob > 0.85 and drought_sev >= 2:
        risk_level = "COMPOUND EXTREME (Flood & Drought)"
    elif flood_prob > 0.85:
        risk_level = "HIGH FLOOD HAZARD"
    elif drought_sev >= 2:
        risk_level = f"CRITICAL DROUGHT (Level {drought_sev})"
    elif drought_sev == 1 or flood_prob > 0.60:
        risk_level = "MODERATE WATCH"
    else:
        risk_level = "NORMAL / LOW RISK"
        
    alert_sent = False
    msg = f"Conditions normal for {request.district}."
    if flood_prob > 0.85 or drought_sev >= 2:
        alert_sent = trigger_sms_alert(request.phone, flood_prob, drought_sev, request.district, request.taluka)
        msg = "Critical risk thresholds exceeded. Early warning alert dispatched." if alert_sent else "Alert conditions met, but SMS delivery unconfigured/failed."

    return PredictionResponse(
        district=request.district,
        taluka=request.taluka,
        flood_probability=flood_prob,
        drought_severity=drought_sev,
        mean_lst_celsius=round(mean_lst, 1),
        alert_sent=alert_sent,
        message=msg,
        risk_level=risk_level,
        vulnerability_details=vuln_info
    )

@app.post("/predict_district_zones")
def predict_district_zones(req: DistrictBatchRequest):
    """
    Computes real-time localized predictions and dynamic LST for all talukas in a district.
    """
    if req.district not in VIDARBHA_GEO_DATA:
        raise HTTPException(status_code=404, detail=f"District '{req.district}' not found in Vidarbha registry.")
        
    talukas_dict = VIDARBHA_GEO_DATA[req.district]["talukas"]
    zone_results = []
    
    for t_name, t_meta in talukas_dict.items():
        base_lst = t_meta.get("base_lst_c", t_meta.get("base_lst", 38.0))
        elev = t_meta.get("elevation_m", t_meta.get("elevation", 300))
        river_name = t_meta.get("river_proximity", t_meta.get("river", "River Basin"))
        
        # Synthesize realistic 14-day localized sequence based on scenario and localized profile
        if req.scenario == "Simulate Flood Risk":
            flood_mult = 1.3 if elev < 250 else 1.0
            precip = np.random.uniform(70, 160, (14, 1)) * flood_mult
            ndvi = np.random.uniform(0.35, 0.65, (14, 1))
            ndwi = np.random.uniform(0.40, 0.85, (14, 1))
            lst = np.random.uniform(base_lst - 5.0, base_lst - 1.0, (14, 1))
        elif req.scenario == "Simulate Drought Risk":
            precip = np.random.uniform(0, 3, (14, 1))
            ndvi = np.random.uniform(0.05, 0.22, (14, 1))
            ndwi = np.random.uniform(-0.6, -0.2, (14, 1))
            lst = np.random.uniform(base_lst + 1.0, base_lst + 5.5, (14, 1))
        elif req.scenario == "Simulate Normal Condition":
            precip = np.random.uniform(10, 35, (14, 1))
            ndvi = np.random.uniform(0.45, 0.75, (14, 1))
            ndwi = np.random.uniform(-0.1, 0.2, (14, 1))
            lst = np.random.uniform(base_lst - 2.0, base_lst + 2.0, (14, 1))
        else: # Random Localized Scenario
            is_high_risk = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
            if is_high_risk == 1: # Localized Flood
                precip = np.random.uniform(60, 140, (14, 1))
                ndvi = np.random.uniform(0.3, 0.6, (14, 1))
                ndwi = np.random.uniform(0.3, 0.75, (14, 1))
                lst = np.random.uniform(base_lst - 4.0, base_lst, (14, 1))
            elif is_high_risk == 2: # Localized Drought
                precip = np.random.uniform(0, 10, (14, 1))
                ndvi = np.random.uniform(0.1, 0.3, (14, 1))
                ndwi = np.random.uniform(-0.5, 0.0, (14, 1))
                lst = np.random.uniform(base_lst + 1.0, base_lst + 4.5, (14, 1))
            else: # Normal
                precip = np.random.uniform(5, 30, (14, 1))
                ndvi = np.random.uniform(0.4, 0.7, (14, 1))
                ndwi = np.random.uniform(-0.2, 0.2, (14, 1))
                lst = np.random.uniform(base_lst - 1.5, base_lst + 1.5, (14, 1))
                
        seq = np.concatenate([precip, ndvi, ndwi, lst], axis=1).tolist()
        pred = predict_risk(model, seq)
        
        f_prob = float(pred['flood_probability'])
        d_sev = int(pred['drought_severity'])
        mean_lst_val = float(np.mean(lst))
        
        # Classification
        if f_prob > 0.85:
            status = "Flood Hotspot"
            marker_color = "red"
        elif d_sev >= 2:
            status = "Drought Hotspot"
            marker_color = "orange"
        elif d_sev == 1 or f_prob > 0.60:
            status = "Moderate Warning"
            marker_color = "cadetblue"
        else:
            status = "Normal"
            marker_color = "green"
            
        zone_results.append({
            "taluka": t_name,
            "coords": t_meta["coords"],
            "elevation_m": elev,
            "river_proximity": river_name,
            "flood_probability": f_prob,
            "drought_severity": d_sev,
            "mean_lst_c": round(mean_lst_val, 1),
            "status": status,
            "marker_color": marker_color,
            "sequence": seq
        })
        
    return {
        "district": req.district,
        "center": VIDARBHA_GEO_DATA[req.district]["center"],
        "talukas_count": len(zone_results),
        "zones": zone_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
