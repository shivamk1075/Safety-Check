import streamlit as st
from streamlit_folium import st_folium


import streamlit as st
import os
from PIL import Image
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim

st.set_page_config(page_title="DumpTrace", layout="centered")

from preds import predict_from_gps
from combined import detect_objects


st.title("TerrainScan tool")
st.markdown("Choose a location by selecting a point on the map.")

st.markdown("Click on the map to drop a pin.")

col_map, col_control = st.columns([2, 1])

with col_control:
    st.subheader("Search a Location")
    search_query = st.text_input("Enter location name (e.g., Varanasi, MBMC Dumping Ground,Ghazipur Landfill, Delhi):")

lat_center, lon_center = 25.262863, 82.979820
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
        st.success(f"Selected Coordinates:\n`({lat:.6f}, {lon:.6f})`")

        if st.button("Predict Site Type"):
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            testing_dir = os.path.join(parent_dir, "Testing")
            os.makedirs(testing_dir, exist_ok=True)

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

                img_path = os.path.join(testing_dir, "test_row_1_tile.jpg")
                if os.path.exists(img_path):

                    output_img, garbage_count, building_count = detect_objects(img_path)

                    st.image(Image.open(output_img), caption="Detected Objects")
            except Exception as e:
                st.exception(e)
    else:
        st.warning("Click on the map to choose a location.")
