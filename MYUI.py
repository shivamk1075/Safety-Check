# import os
# os.system("pip install streamlit-folium")

import streamlit as st
from streamlit_folium import st_folium


import streamlit as st
import os
from PIL import Image
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim

st.set_page_config(page_title="TerrainScan tool", layout="centered")

from preds import predict_from_gps
from combined import detect_objects


# load_classification_model()
# load_yolo_models()

st.title("TerrainScan tool")
st.markdown("Choose a location by typing latitude/longitude or selecting a point on the map.")

# Option selection
option = st.radio("Select Input Mode:", ["Enter Coordinates", "Pin on Map"])

lat, lon = None, None

if option == "Enter Coordinates":
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")

elif option == "Pin on Map":
    st.markdown("👇Click on the map to drop a pin.")

    col_map, col_control = st.columns([2, 1])  # 2:1 ratio for layout

    # Search bar
    with col_control:
        st.subheader("Search a Location")
        search_query = st.text_input("Enter location name (e.g., Varanasi, India):")

    # Default map center
    lat_center, lon_center = 25.262863, 82.979820

    if search_query:
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.geocode(search_query)
        if location:
            lat_center, lon_center = location.latitude, location.longitude
        else:
            st.warning("Location not found. Showing default.")

    with col_map:
        m = folium.Map(location=[lat_center, lon_center], zoom_start=14, tiles="Esri.WorldImagery")
        m.add_child(folium.LatLngPopup())
        if search_query and location:
            folium.Marker([lat_center, lon_center], tooltip="Search Result", icon=folium.Icon(color="blue")).add_to(m)
        map_data = st_folium(m, height=500, width=700)

    with col_control:
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]
            st.success(f"📍 Selected Coordinates:\n`({lat:.6f}, {lon:.6f})`")

            if st.button("Predict Site Type"):
                os.makedirs("Testing", exist_ok=True)

                try:
                    score = predict_from_gps(lat, lon, row_num=1)
                    if score is None:
                        st.error("Could not process this location.")
                    elif score > 0.5:
                        # st.error(f" Prediction: Illegal Site (Score: {score:.4f})")
                        st.error(f"Prediction: Illegal Site ")
                    else:
                        # st.success(f"Prediction: Legal Site (Score: {score:.4f})")
                        st.success(f"Prediction: Legal Site ")

                    img_path = f"Testing/test_row_1_tile.jpg"
                    if os.path.exists(img_path):

                        # Run YOLO detection
                        output_img, garbage_count, building_count = detect_objects(img_path)

                        st.image(Image.open(output_img), caption="Detected Objects")

                        # Show counts
                        st.markdown(f" **Garbage Sites Detected:** {garbage_count}")
                        # st.markdown(f" **Buildings Detected:** {building_count}")

                        # st.image(Image.open(img_path), caption="Satellite Tile")
                except Exception as e:
                    st.exception(e)
        else:
            st.warning("Click on the map to choose a location.")

# Fallback for coordinate input mode
if option == "Enter Coordinates":
    if st.button("🔍 Predict Site Type"):
        if lat is not None and lon is not None:
            os.makedirs("Testing", exist_ok=True)

            try:
                score = predict_from_gps(lat, lon, row_num=1)
                if score is None:
                    st.error("Could not process this location.")
                elif score > 0.5:
                    # st.error(f"Prediction: Illegal Site (Score: {score:.4f})")
                    st.error(f"Prediction: Illegal Site")
                else:
                    # st.success(f"Prediction: Legal Site (Score: {score:.4f})")
                    st.success(f" Prediction: Legal Site ")

                img_path = f"Testing/test_row_1_tile.jpg"
                if os.path.exists(img_path):
                    # Run YOLO detection
                    output_img, garbage_count, building_count = detect_objects(img_path)

                    st.image(Image.open(output_img), caption="Detected Objects")

                    # Show counts
                    st.markdown(f"**Garbage Sites Detected:** {garbage_count}")
                    # st.markdown(f" **Buildings Detected:** {building_count}")

                    # st.image(Image.open(img_path), caption="Satellite Tile")
            except Exception as e:
                st.exception(e)
        else:
            st.warning("Please enter valid coordinates.")
