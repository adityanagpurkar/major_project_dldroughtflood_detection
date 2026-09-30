"""
geo_registry.py
Comprehensive Geographic and Topographical Registry for Vidarbha Region (Maharashtra, India).
Provides full coordinates, elevation, river proximity, base LST, and micro-grid parcel generation.
"""

import numpy as np

VIDARBHA_GEO_DATA = {
    "Nagpur": {
        "center": [21.1458, 79.0882],
        "bounds": [[20.50, 78.25], [21.75, 79.95]],
        "talukas": {
            "Nagpur Urban": {"coords": [21.1458, 79.0882], "river": "Nag River Basin", "elevation": 310, "base_lst": 41.0},
            "Nagpur Rural": {"coords": [21.1800, 79.0200], "river": "Kanhan Sub-basin", "elevation": 295, "base_lst": 38.5},
            "Katol": {"coords": [21.2721, 78.5878], "river": "Jam River", "elevation": 417, "base_lst": 39.2},
            "Saoner": {"coords": [21.3857, 78.9194], "river": "Kolar River", "elevation": 305, "base_lst": 38.0},
            "Ramtek": {"coords": [21.3972, 79.3304], "river": "Sur River / Totladoh Basin", "elevation": 345, "base_lst": 36.5},
            "Hingna": {"coords": [21.0667, 78.9667], "river": "Vena River Basin", "elevation": 320, "base_lst": 39.0},
            "Umred": {"coords": [20.8547, 79.3267], "river": "Amb River", "elevation": 280, "base_lst": 39.5},
            "Narkhed": {"coords": [21.4939, 78.5303], "river": "Wardha Upper Tributary", "elevation": 398, "base_lst": 40.1},
            "Kamptee": {"coords": [21.2231, 79.1969], "river": "Kanhan & Pench Confluence", "elevation": 275, "base_lst": 37.5},
            "Parseoni": {"coords": [21.3789, 79.1764], "river": "Pench River Basin", "elevation": 290, "base_lst": 36.8},
            "Mouda": {"coords": [21.1714, 79.3878], "river": "Kanhan River Floodplain", "elevation": 260, "base_lst": 38.2},
            "Bhiwapur": {"coords": [20.7600, 79.5200], "river": "Maru River", "elevation": 265, "base_lst": 39.0},
            "Kuhi": {"coords": [20.9833, 79.3500], "river": "Nag-Kanhan Basin", "elevation": 270, "base_lst": 38.4}
        }
    },
    "Amravati": {
        "center": [20.9320, 77.7523],
        "bounds": [[20.40, 76.50], [21.80, 78.60]],
        "talukas": {
            "Amravati": {"coords": [20.9320, 77.7523], "river": "Pedhi River", "elevation": 343, "base_lst": 39.5},
            "Achalpur": {"coords": [21.2575, 77.5097], "river": "Chandrabhaga River", "elevation": 370, "base_lst": 38.1},
            "Chandur Bazar": {"coords": [21.2400, 77.7500], "river": "Purna Tributary", "elevation": 360, "base_lst": 38.6},
            "Morshi": {"coords": [21.3200, 78.0100], "river": "Upper Wardha Dam Basin", "elevation": 350, "base_lst": 39.4},
            "Warud": {"coords": [21.4600, 78.2600], "river": "Chudaman River", "elevation": 380, "base_lst": 40.2},
            "Daryapur": {"coords": [20.9200, 77.3200], "river": "Purna River Floodplain", "elevation": 288, "base_lst": 40.8},
            "Anjangaon Surji": {"coords": [21.1600, 77.3100], "river": "Shahnur River", "elevation": 374, "base_lst": 39.3},
            "Dharni (Melghat)": {"coords": [21.5200, 76.9500], "river": "Tapti River Gorge", "elevation": 420, "base_lst": 35.5},
            "Chikhaldara": {"coords": [21.4000, 77.3200], "river": "Sipna River Valley", "elevation": 1188, "base_lst": 31.2},
            "Nandgaon Khandeshwar": {"coords": [20.6800, 77.8300], "river": "Bembla River", "elevation": 315, "base_lst": 39.8},
            "Chandur Railway": {"coords": [20.8100, 77.9600], "river": "Kholat River", "elevation": 330, "base_lst": 39.5},
            "Dhamangaon Railway": {"coords": [20.7800, 78.1400], "river": "Wardha River Border", "elevation": 300, "base_lst": 39.7},
            "Tiwsa": {"coords": [20.9600, 78.0500], "river": "Wardha River Basin", "elevation": 310, "base_lst": 39.2},
            "Bhatkuli": {"coords": [20.9000, 77.5800], "river": "Purna Basin", "elevation": 305, "base_lst": 40.0}
        }
    },
    "Akola": {
        "center": [20.7059, 77.0082],
        "bounds": [[20.30, 76.60], [21.30, 77.50]],
        "talukas": {
            "Akola Urban": {"coords": [20.7059, 77.0082], "river": "Morna River Floodplain", "elevation": 282, "base_lst": 41.5},
            "Akot": {"coords": [21.0967, 77.0583], "river": "Katepurna Tributary", "elevation": 345, "base_lst": 39.8},
            "Telhara": {"coords": [21.0333, 76.8333], "river": "Wan River Basin", "elevation": 310, "base_lst": 40.5},
            "Balapur": {"coords": [20.6667, 76.7667], "river": "Man River", "elevation": 270, "base_lst": 41.5},
            "Patur": {"coords": [20.4500, 76.9333], "river": "Katepurna Hills", "elevation": 360, "base_lst": 39.5},
            "Murtizapur": {"coords": [20.7333, 77.3667], "river": "Uma & Purna Confluence", "elevation": 290, "base_lst": 40.9},
            "Barshitakli": {"coords": [20.5833, 77.0667], "river": "Katepurna Reservoir", "elevation": 315, "base_lst": 40.2}
        }
    },
    "Wardha": {
        "center": [20.7453, 78.6022],
        "bounds": [[20.25, 78.00], [21.40, 79.20]],
        "talukas": {
            "Wardha": {"coords": [20.7453, 78.6022], "river": "Dham River", "elevation": 234, "base_lst": 39.2},
            "Sevagram / Seloo": {"coords": [20.8333, 78.7000], "river": "Bor River Basin", "elevation": 250, "base_lst": 38.8},
            "Arvi": {"coords": [20.9833, 78.2333], "river": "Wardha River Basin", "elevation": 328, "base_lst": 39.8},
            "Deoli": {"coords": [20.6500, 78.4833], "river": "Yashoda River", "elevation": 240, "base_lst": 40.1},
            "Hinganghat": {"coords": [20.5500, 78.8333], "river": "Wena River Floodplain", "elevation": 215, "base_lst": 40.6},
            "Samudrapur": {"coords": [20.6000, 79.0333], "river": "Pothra River", "elevation": 225, "base_lst": 39.9},
            "Karanja (Ghadge)": {"coords": [21.1833, 78.4333], "river": "Upper Wardha Catchment", "elevation": 365, "base_lst": 38.5},
            "Ashti": {"coords": [21.2000, 78.1833], "river": "Wardha River Bank", "elevation": 310, "base_lst": 39.0}
        }
    },
    "Gadchiroli": {
        "center": [20.1849, 79.9948],
        "bounds": [[18.60, 79.50], [21.00, 80.70]],
        "talukas": {
            "Gadchiroli": {"coords": [20.1849, 79.9948], "river": "Wainganga River Basin", "elevation": 217, "base_lst": 36.5},
            "Chamorshi": {"coords": [19.9333, 79.9333], "river": "Wainganga-Dina Confluence", "elevation": 195, "base_lst": 37.0},
            "Aheri": {"coords": [19.4167, 80.0000], "river": "Pranhita River Floodplain", "elevation": 160, "base_lst": 37.5},
            "Sironcha": {"coords": [18.8333, 79.9667], "river": "Godavari & Pranhita Confluence", "elevation": 118, "base_lst": 38.2},
            "Dhanora": {"coords": [20.2500, 80.2833], "river": "Kathani River Catchment", "elevation": 270, "base_lst": 35.8},
            "Kurkheda": {"coords": [20.5833, 80.1833], "river": "Sati River", "elevation": 240, "base_lst": 36.0},
            "Armori": {"coords": [20.4667, 79.9833], "river": "Wainganga Lowland", "elevation": 205, "base_lst": 36.8},
            "Etapalli": {"coords": [19.6500, 80.3167], "river": "Bandia River Basin", "elevation": 230, "base_lst": 35.2},
            "Bhamragad": {"coords": [19.2500, 80.3500], "river": "Indravati & Parlkota Confluence", "elevation": 175, "base_lst": 34.8},
            "Korchi": {"coords": [20.7333, 80.4667], "river": "Tipagarh Hills", "elevation": 310, "base_lst": 35.0}
        }
    },
    "Yavatmal": {
        "center": [20.3888, 78.1204],
        "bounds": [[19.30, 77.30], [20.70, 79.20]],
        "talukas": {
            "Yavatmal": {"coords": [20.3888, 78.1204], "river": "Adan Tributary", "elevation": 445, "base_lst": 39.5},
            "Pusad": {"coords": [19.9000, 77.5667], "river": "Painganga Sub-basin", "elevation": 315, "base_lst": 40.5},
            "Umarkhed": {"coords": [19.6000, 77.7000], "river": "Painganga Floodplain", "elevation": 300, "base_lst": 40.8},
            "Digras": {"coords": [20.1000, 77.7167], "river": "Dhavanda River", "elevation": 330, "base_lst": 40.0},
            "Darwha": {"coords": [20.3167, 77.7667], "river": "Adan River Valley", "elevation": 355, "base_lst": 39.8},
            "Wani": {"coords": [20.0667, 78.9500], "river": "Wardha & Penganga Basin", "elevation": 228, "base_lst": 41.5},
            "Pandharkawada (Kelapur)": {"coords": [20.0167, 78.5333], "river": "Khuni River", "elevation": 260, "base_lst": 40.6},
            "Ralegaon": {"coords": [20.4167, 78.5000], "river": "Ramganga River", "elevation": 275, "base_lst": 39.9},
            "Ghatanji": {"coords": [20.1333, 78.3167], "river": "Waghadi River", "elevation": 290, "base_lst": 40.3},
            "Arni": {"coords": [20.0833, 77.9333], "river": "Arna River", "elevation": 320, "base_lst": 40.2}
        }
    },
    "Bhandara": {
        "center": [21.1714, 79.6547],
        "bounds": [[20.60, 79.40], [21.60, 80.20]],
        "talukas": {
            "Bhandara": {"coords": [21.1714, 79.6547], "river": "Wainganga River Basin", "elevation": 244, "base_lst": 37.5},
            "Tumsar": {"coords": [21.3833, 79.7333], "river": "Wainganga & Bawanthadi Basin", "elevation": 255, "base_lst": 37.0},
            "Mohadi": {"coords": [21.3167, 79.6500], "river": "Sur River Confluence", "elevation": 250, "base_lst": 37.2},
            "Pauni": {"coords": [20.7833, 79.6333], "river": "Gosekhurd Dam Catchment", "elevation": 225, "base_lst": 38.0},
            "Sakoli": {"coords": [21.0833, 79.9833], "river": "Chulband River", "elevation": 260, "base_lst": 36.8},
            "Lakhani": {"coords": [21.1333, 79.8500], "river": "Chulband Catchment", "elevation": 250, "base_lst": 37.0},
            "Lakhandur": {"coords": [20.7667, 79.8833], "river": "Wainganga Lowland Floodplain", "elevation": 220, "base_lst": 37.8}
        }
    },
    "Gondiya": {
        "center": [21.4598, 80.1961],
        "bounds": [[20.60, 79.70], [21.75, 80.80]],
        "talukas": {
            "Gondiya": {"coords": [21.4598, 80.1961], "river": "Pangoli River", "elevation": 300, "base_lst": 36.5},
            "Tirora": {"coords": [21.4167, 79.9333], "river": "Wainganga River Bank", "elevation": 260, "base_lst": 37.2},
            "Goregaon": {"coords": [21.3000, 80.2167], "river": "Bagh River Basin", "elevation": 290, "base_lst": 36.2},
            "Amgaon": {"coords": [21.3667, 80.3833], "river": "Bagh & Son Confluence", "elevation": 310, "base_lst": 36.0},
            "Salekasa": {"coords": [21.3000, 80.5833], "river": "Darekasa Hill Catchment", "elevation": 360, "base_lst": 34.8},
            "Deori": {"coords": [21.0833, 80.3667], "river": "Gadhavi River", "elevation": 320, "base_lst": 35.5},
            "Sadak Arjuni": {"coords": [21.1333, 80.1500], "river": "Chulband Upper Stream", "elevation": 280, "base_lst": 36.5},
            "Arjuni Morgaon": {"coords": [20.7833, 80.0333], "river": "Itiadoh Reservoir Basin", "elevation": 255, "base_lst": 36.8}
        }
    },
    "Washim": {
        "center": [20.1098, 77.1352],
        "bounds": [[19.80, 76.50], [20.70, 77.80]],
        "talukas": {
            "Washim": {"coords": [20.1098, 77.1352], "river": "Painganga Origin Plateau", "elevation": 546, "base_lst": 39.5},
            "Risod": {"coords": [19.9667, 76.7833], "river": "Penganga Upper Tributary", "elevation": 520, "base_lst": 40.2},
            "Malegaon": {"coords": [20.3667, 77.0333], "river": "Katepurna Catchment", "elevation": 450, "base_lst": 40.0},
            "Mangrulpir": {"coords": [20.3167, 77.3500], "river": "Aran River", "elevation": 420, "base_lst": 40.5},
            "Karanja (Lad)": {"coords": [20.4833, 77.4833], "river": "Bembla River Origin", "elevation": 395, "base_lst": 40.8},
            "Manora": {"coords": [20.2167, 77.5500], "river": "Aran Basin", "elevation": 410, "base_lst": 40.1}
        }
    },
    "Buldhana": {
        "center": [20.5312, 76.1837],
        "bounds": [[19.80, 75.80], [21.30, 77.00]],
        "talukas": {
            "Buldhana": {"coords": [20.5312, 76.1837], "river": "Penganga Ridge", "elevation": 639, "base_lst": 38.0},
            "Chikhli": {"coords": [20.3500, 76.2500], "river": "Dhanasri River", "elevation": 605, "base_lst": 39.0},
            "Mehkar": {"coords": [20.1500, 76.5667], "river": "Painganga River Floodplain", "elevation": 512, "base_lst": 40.5},
            "Lonar": {"coords": [19.9833, 76.5167], "river": "Lonar Crater Basin", "elevation": 563, "base_lst": 39.8},
            "Sindkhed Raja": {"coords": [19.9667, 76.1333], "river": "Khadakpurna Catchment", "elevation": 550, "base_lst": 40.0},
            "Deulgaon Raja": {"coords": [20.0167, 75.9333], "river": "Khadakpurna River Bank", "elevation": 525, "base_lst": 40.2},
            "Khamgaon": {"coords": [20.6833, 76.5667], "river": "Gyan Ganga River", "elevation": 290, "base_lst": 41.5},
            "Shegaon": {"coords": [20.7833, 76.6833], "river": "Man River Basin", "elevation": 276, "base_lst": 41.2},
            "Malkapur": {"coords": [20.8833, 76.2000], "river": "Nalganga & Purna Confluence", "elevation": 255, "base_lst": 41.8},
            "Motala": {"coords": [20.7000, 76.2000], "river": "Nalganga Dam Catchment", "elevation": 310, "base_lst": 40.5},
            "Nandura": {"coords": [20.8333, 76.4500], "river": "Gyan Ganga Basin", "elevation": 262, "base_lst": 41.4},
            "Jalgaon Jamod": {"coords": [21.0500, 76.5333], "river": "Satpuda Foothills / Purna Basin", "elevation": 295, "base_lst": 40.8},
            "Sangrampur": {"coords": [21.0333, 76.6833], "river": "Wan & Purna Confluence", "elevation": 285, "base_lst": 41.0}
        }
    },
    "Chandrapur": {
        "center": [19.9615, 79.2961],
        "bounds": [[19.40, 78.80], [20.80, 80.10]],
        "talukas": {
            "Chandrapur": {"coords": [19.9615, 79.2961], "river": "Erai & Zarpat River Floodplain", "elevation": 189, "base_lst": 42.5},
            "Ballarpur": {"coords": [19.8500, 79.3500], "river": "Wardha River Floodplain", "elevation": 180, "base_lst": 42.0},
            "Bhadrawati": {"coords": [20.1000, 79.1167], "river": "Erai Dam Catchment", "elevation": 210, "base_lst": 41.5},
            "Warora": {"coords": [20.2333, 78.9833], "river": "Wardha River Basin", "elevation": 220, "base_lst": 41.8},
            "Chimur": {"coords": [20.5000, 79.3667], "river": "Uma River Catchment", "elevation": 240, "base_lst": 40.2},
            "Nagbhid": {"coords": [20.5833, 79.6667], "river": "Ghodazari Reservoir", "elevation": 230, "base_lst": 39.0},
            "Brahmapuri": {"coords": [20.6000, 79.8500], "river": "Wainganga River Floodplain", "elevation": 215, "base_lst": 39.5},
            "Sindewahi": {"coords": [20.3000, 79.6333], "river": "Asolamendha Reservoir Basin", "elevation": 225, "base_lst": 39.8},
            "Mul": {"coords": [20.0667, 79.6667], "river": "Painganga-Wainganga Sub-basin", "elevation": 200, "base_lst": 40.5},
            "Saoli": {"coords": [19.9500, 79.8000], "river": "Wainganga River Basin", "elevation": 190, "base_lst": 40.2},
            "Pombhurna": {"coords": [19.7833, 79.6500], "river": "Wainganga Basin", "elevation": 185, "base_lst": 40.8},
            "Gondpipri": {"coords": [19.6000, 79.6833], "river": "Wardha-Wainganga Confluence", "elevation": 165, "base_lst": 41.0},
            "Rajura": {"coords": [19.7833, 79.3667], "river": "Wardha River Bank", "elevation": 182, "base_lst": 42.1},
            "Korpurna": {"coords": [19.7000, 79.1833], "river": "Penganga River Basin", "elevation": 210, "base_lst": 41.2},
            "Jiwati": {"coords": [19.5333, 79.0333], "river": "Manikgarh Hill Plateau", "elevation": 350, "base_lst": 39.2}
        }
    }
}

def generate_taluka_boundary_polygon(center_lat, center_lon, radius_km=10.0, num_vertices=12):
    """
    Generates a realistic smooth polygon boundary around a taluka center point.
    """
    deg_r = radius_km / 111.0
    # Seed by lat/lon for consistent shape
    seed = int((abs(center_lat) * 100 + abs(center_lon) * 10)) % 1000
    rng = np.random.RandomState(seed)
    
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    poly = []
    
    # Slight perturbation to mimic natural administrative irregular boundaries
    radii_factors = 1.0 + rng.uniform(-0.15, 0.15, size=num_vertices)
    
    for i, a in enumerate(angles):
        r = deg_r * radii_factors[i]
        lat = center_lat + r * np.cos(a)
        lon = center_lon + (r / np.cos(np.radians(center_lat))) * np.sin(a)
        poly.append([round(float(lat), 5), round(float(lon), 5)])
        
    poly.append(poly[0])  # Close the polygon
    return poly

def generate_district_taluka_polygons(district_name):
    """
    Generates dictionary of all taluka boundary polygons in the selected district.
    """
    if district_name not in VIDARBHA_GEO_DATA:
        return {}
    talukas = VIDARBHA_GEO_DATA[district_name]["talukas"]
    result = {}
    for name, info in talukas.items():
        lat, lon = info["coords"]
        result[name] = generate_taluka_boundary_polygon(lat, lon, radius_km=10.5)
    return result

def generate_hyper_local_parcels(center_lat, center_lon, base_lst, flood_prob, drought_sev, grid_dim=6, cell_size_m=400):
    """
    Generates a localized grid of micro-parcels (e.g., 400m x 400m = 160,000 m² = 16 hectares)
    with realistic spatial thermal and hydro gradients.
    """
    deg_step = cell_size_m / 111000.0
    half = grid_dim // 2
    parcels = []
    
    # Deterministic seed based on coordinate hash
    seed_val = int((abs(center_lat) * 1000 + abs(center_lon) * 100)) % 10000
    rng = np.random.RandomState(seed_val)
    
    for r in range(-half, half):
        for c in range(-half, half):
            lat1 = center_lat + r * deg_step
            lat2 = lat1 + deg_step
            lon1 = center_lon + c * deg_step / np.cos(np.radians(center_lat))
            lon2 = lon1 + deg_step / np.cos(np.radians(center_lat))
            
            dist = np.sqrt(r**2 + c**2) / (max(half, 1) * 1.414)
            t_var = float(rng.normal(0, 0.6))
            parcel_lst = round(float(base_lst + (1.0 - dist) * 1.6 + t_var), 1)
            
            f_var = float(rng.normal(0, 0.04))
            parcel_flood = round(float(np.clip(flood_prob + f_var - (dist * 0.05), 0.0, 1.0)), 2)
            
            sq_m = cell_size_m * cell_size_m
            hectares = round(sq_m / 10000.0, 1)
            
            parcels.append({
                "bounds": [[round(lat1, 5), round(lon1, 5)], [round(lat2, 5), round(lon2, 5)]],
                "center": [round((lat1 + lat2)/2, 5), round((lon1 + lon2)/2, 5)],
                "lst_c": parcel_lst,
                "flood_prob": parcel_flood,
                "drought_sev": drought_sev,
                "area_m2": sq_m,
                "area_ha": hectares
            })
            
    return parcels

# Alias for backward compatibility
generate_hyper_local_micro_grid = generate_hyper_local_parcels
