# #!/bin/bash
# pip install streamlit-folium==0.22.1

#!/bin/bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends $(cat packages.txt)
pip install -r requirements.txt

