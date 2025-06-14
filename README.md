# Illegal Site Detection Tool

This Streamlit web app predicts whether a given GPS location is a legal or illegal dumping site using:

* A classification model trained on spatial and visual features
* YOLOv7-based object detection for garbage and buildings
* Real satellite tile images fetched from map servers

## Features

* Select a location by entering coordinates or pinning a point on the map
* Fetch satellite tile of the selected location
* Extract spatial features (buildings, roads, police proximity, etc.) using OpenStreetMap
* Classify site legality using a trained model (`classModel.h5`)
* Run object detection using YOLOv7 weights (`garbage.pt`, `building.pt`)
* View prediction results and detected bounding boxes directly on the app

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Models

Put your trained model files in the `model_wts/` directory:

* `garbage.pt`
* `building.pt`
* `classModel.h5`

### 4. Launch the App

```bash
streamlit run app.py
```

## Requirements

See `requirements.txt` for the full list. Major ones include:

* `streamlit`
* `torch`
* `tensorflow`
* `opencv-python`
* `osmnx`
* `shapely`
* `folium`
* `geopy`

## Notes

* The app fetches satellite tiles from public map tile servers.
* It also queries OpenStreetMap for nearby features.
* Internet connection is required for full functionality.

## Output Example

* Tile image with object detection:

  * Detected buildings
  * Detected garbage
* Final legality prediction (with score)

## Credits

Created as a smart location intelligence tool combining satellite imagery, spatial data, and AI models.
