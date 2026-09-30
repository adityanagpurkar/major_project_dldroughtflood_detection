# VidarbhaDrought&FloodPredictionV1 (Phase 2) 🛰️🌧️🔥

**VidarbhaDrought&FloodPredictionV1** is an AI, Remote Sensing, and GIS early warning system designed for flood and drought hazard prediction across the **11 districts and all localized talukas / sub-regions** of the Vidarbha region, Maharashtra, India.

---

## 🌟 What's New in Phase 2

1. **Sub-District & Taluka Level Localization**:
   - Micro-region granular mapping across all **100+ talukas** in Vidarbha.
   - Localized terrain metrics: Elevation (m), baseline Land Surface Temperature, and river basin proximity (e.g. Wainganga, Kanhan, Wardha, Penganga, Purna, Godavari basins).

2. **Land Surface Temperature (LST) Multi-Sensor Fusion**:
   - Integration of MODIS & Landsat Thermal Infrared (LST in °C) data alongside Sentinel-2 NDVI, NDWI, and IMD precipitation.
   - 4-Feature Deep Learning Architecture: `[Precipitation, NDVI, NDWI, LST]`.

3. **Dynamic GIS Thermal Heatmap & Hotspot Visualization**:
   - **LST Thermal Gradient Heatmap**: Visualizes surface thermal stress from cooler riparian zones (<30°C) to extreme heat stress zones (>44°C).
   - **Automated Hotspot Tagging**: Identifies and flags **🌊 Flood Hotspots** (saturated river valleys) and **🔥 Drought Hotspots** (high LST + vegetation stress).

---

## 🗺️ Monitored Districts and Localized Talukas

- **Nagpur**: Nagpur Urban, Nagpur Rural, Katol, Saoner, Ramtek, Hingna, Umred, Narkhed, Kamptee, Parseoni, Mouda, Bhiwapur, Kuhi
- **Amravati**: Amravati, Achalpur, Chandur Bazar, Morshi, Warud, Daryapur, Anjangaon Surji, Dharni (Melghat), Chikhaldara, Nandgaon Khandeshwar, Chandur Railway, Dhamangaon Railway, Tiwsa, Bhatkuli
- **Akola**: Akola Urban, Akot, Telhara, Balapur, Patur, Murtizapur, Barshitakli
- **Wardha**: Wardha, Sevagram/Seloo, Arvi, Deoli, Hinganghat, Samudrapur, Karanja (Ghadge), Ashti
- **Gadchiroli**: Gadchiroli, Chamorshi, Aheri, Sironcha, Dhanora, Kurkheda, Armori, Etapalli, Bhamragad, Korchi
- **Yavatmal**: Yavatmal, Pusad, Umarkhed, Digras, Darwha, Wani, Pandharkawada (Kelapur), Ralegaon, Ghatanji, Arni
- **Bhandara**: Bhandara, Tumsar, Mohadi, Pauni, Sakoli, Lakhani, Lakhandur
- **Gondiya**: Gondiya, Tirora, Goregaon, Amgaon, Salekasa, Deori, Sadak Arjuni, Arjuni Morgaon
- **Washim**: Washim, Risod, Malegaon, Mangrulpir, Karanja (Lad), Manora
- **Buldhana**: Buldhana, Chikhli, Mehkar, Lonar, Sindkhed Raja, Deulgaon Raja, Khamgaon, Shegaon, Malkapur, Motala, Nandura, Jalgaon Jamod, Sangrampur
- **Chandrapur**: Chandrapur, Ballarpur, Bhadrawati, Warora, Chimur, Nagbhid, Brahmapuri, Sindewahi, Mul, Saoli, Pombhurna, Gondpipri, Rajura, Korpurna, Jiwati

---

## 🚀 Getting Started

### 1. Run the Backend API
```bash
python 3_api_backend.py
```
FastAPI Interactive Docs: `http://127.0.0.1:8000/docs`

### 2. Launch the Streamlit Dashboard
```bash
streamlit run 4_dashboard.py
```
Dashboard URL: `http://localhost:8501`

---

## 📊 Model Architecture

```
         14-Day Sequential Input: [Precipitation, NDVI, NDWI, LST (°C)]
                                       │
                                       ▼
                         nn.LSTM (2 Layers, 64 Units)
                                       │
                                       ▼
                       Last Hidden Layer (64 Units)
                                       │
                                       ▼
                      Shared Linear + ReLU + Dropout (32)
                                ┌──────┴──────┐
                                │             │
                                ▼             ▼
                      Flood Head (1)    Drought Head (4)
                       Sigmoid (0-1)      Logits (0-3)
```
