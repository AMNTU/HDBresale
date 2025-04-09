import pandas as pd             # version 2.2.2
import numpy as np              # version 2.0.2
import matplotlib.pyplot as plt
import seaborn as sns           
import json
import requests
from bs4 import BeautifulSoup
from scipy.spatial.distance import cdist
import geopandas as gpd
import re
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes

# Insert the API key from OneMap by replacing the %%% (keep the "Bearer ")
headers = {"Authorization": "Bearer %%%"}

# Takes in an address and retrieve the postal code and geocodes (lng, lat) from OneMap API
def get_postal_geo(address):
    list_tokens = address.split(' ')  # split string based on space ' ' as delimiter
    check_len = len(list_tokens)      # check how many words are in the address
    if check_len > 1:
        cleaned_address = '%20'.join(list_tokens) # use '%20' for space when using HTTPS request
    else:
        cleaned_address = address

    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={cleaned_address}&returnGeom=Y&getAddrDetails=Y"

    response = requests.get(url, headers=headers)
    result = json.loads(response.content.decode('utf-8'))       # parse the JSON results
    postalcode = str(result['results'][0]['POSTAL']).zfill(6)   # save the postal code as string and fill it with leading zero
    latitude = result['results'][0]['LATITUDE']
    longitude = result['results'][0]['LONGITUDE']
    return [postalcode, latitude, longitude]

# Batch process a dataframe of addresses by calling 'get_postal_geo' function for each address
def get_postal_geo_batch(df_address):
    list_address = list(df_address)
    list_postal_geo = []

    for address in list_address:
        try:
            postal_geo = get_postal_geo(address)
        except:
            postal_geo = ['', '', '']

        if postal_geo == None:
            address = ''
        list_postal_geo.append(postal_geo)

    return list_postal_geo

# Concatenate 'results' into the df_hdbInfo data frame
df_hdbInfo = pd.concat([df_hdbInfo, results], axis=1)

# Load the GeoJSON file
gdf = gpd.read_file("LTAMRTStationExitGEOJSON.geojson")

# Function to extract MRT station name
def extract_mrt_station(description):
    try:
        soup = BeautifulSoup(description, "html.parser")
        for row in soup.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td and "STATION_NA" in th.get_text():
                return td.get_text().strip()  # Extract MRT station name
    except Exception:
        return None

# Convert 'remaining_lease' to months in numerical format
def extract_remaining_lease(lease_str):

# Extract years and months using regex
  years_match = re.search(r"(\d+) years", lease_str)
  months_match = re.search(r"(\d+) months", lease_str)

# Convert matches to integers, defaulting to 0 if no match is found
  years = int(years_match.group(1)) if years_match else 0
  months = int(months_match.group(1)) if months_match else 0

# Convert everything to total months
  return years * 12 + months

# Parse 'storey_range' and calculate the average
def calculate_avg_storey(storey_range):
    numbers = re.findall(r'\d+', storey_range)          # Extract numbers using regular expressions
    if numbers and len(numbers) == 2:
      return (int(numbers[0]) + int(numbers[1])) // 2   # Integer division to get the average whole number
    return None

# Calculate the floor level category
def calculate_height_category(row):

    if row['max_floor_lvl'] == 2:  # For two storey blocks, always return 0 (low floor category)
        return 0

    # Some 'storey_range' e.g. 10 to 12, is higher than 'max_floor_level' e.g. 10. Return 2 (high floor).
    if row['avg_storey'] > row['max_floor_lvl']:
        return 2

    # Categorise based on lower third, mid third or upper third
    percentage = row['avg_storey'] / row['max_floor_lvl']
    return 0 if percentage < 1/3 else 1 if percentage < 2/3 else 2

# Define the non-mature towns category dictionary (the rest are mature towns)
category = {
    'BUKIT BATOK': 'non-mature',
    'BUKIT PANJANG': 'non-mature',
    'CHOA CHU KANG': 'non-mature',
    'HOUGANG': 'non-mature',
    'JURONG EAST': 'non-mature',
    'JURONG WEST': 'non-mature',
    'PUNGGOL': 'non-mature',
    'SEMBAWANG': 'non-mature',
    'SENGKANG': 'non-mature',
    'TENGAH': 'non-mature',
    'WOODLANDS': 'non-mature',
    'YISHUN': 'non-mature'
}

# Pass the 'address' column to the batch function to return postal and geocodes as a list
postal_geo_list = get_postal_geo_batch(df_hdbInfo.iloc[:, df_hdbInfo.columns.get_loc('address')])

# Convert the returned list into a dataframe
results = pd.DataFrame(postal_geo_list, columns=['postalcode', 'latitude', 'longitude'])

# Cast the columns to the intended data types
results['postalcode'] = results['postalcode'].astype(str)
results['latitude'] = results['latitude'].astype(float)
results['longitude'] = results['longitude'].astype(float)

# Apply the function to extract the MRT station name
gdf["mrt_station"] = gdf["Description"].apply(extract_mrt_station)

# Extract longitude and latitude from geometry
gdf["longitude"] = gdf.geometry.x
gdf["latitude"] = gdf.geometry.y

# Extract relevant columns and remove duplicates
mrt_stations = gdf[["mrt_station", "longitude", "latitude"]].drop_duplicates(subset=["mrt_station"], keep="first").reset_index(drop=True)

geocodes = df_hdbInfo[["latitude", "longitude"]].values               # The source list of HDB blocks' geocodes
reference_geocodes = mrt_stations[["latitude", "longitude"]].values   # The target list of MRT stations' geocodes as references

# Compute Euclidean distances (direct distance without consideration of route or earth's curvature)
dist_matrix = cdist(geocodes, reference_geocodes, metric='euclidean')
shortest_distances = np.min(dist_matrix, axis=1) * 111                # Get the shortest distance between two geocodes in km
shortest_distances = [round(dist, 1) for dist in shortest_distances]  # Round to the nearest 0.1km

# Get the corresponding list of MRT stations names matched by the closest_station_indices
closest_station_indices = np.argmin(dist_matrix, axis=1)              # Get the index of the closest MRT station for each point
closest_mrt_stations = mrt_stations.iloc[closest_station_indices]["mrt_station"].values

# Add both distance and nearest MRT station back to the hdbInfo dataframe
df_hdbInfo['nearest_mrt'] = closest_mrt_stations
df_hdbInfo['distance_km'] = shortest_distances

# Save 'df_hdbInfo' as CSV for future re-use
df_hdbInfo.to_csv('HDBInfo.csv', index=False)

# Cast 'floor_area_sqm' and 'resale_price' as integer
df_hdbResale = df_hdbResale.astype({'floor_area_sqm': 'int32', 'resale_price': 'int32'})

# Apply function to 'remaining_lease' column to convert into integer number of months
df_hdbResale['remaining_lease'] = df_hdbResale['remaining_lease'].apply(extract_remaining_lease).astype('int32')

# Map the categories to a new 'mature' column: 'non-mature' -> False, 'mature' -> True
df_hdbResale['mature'] = df_hdbResale['town'].map(lambda x: category.get(x) != 'non-mature')

# Merge 'block' and 'street_name' into 'address'
df_hdbResale['address'] = df_hdbResale['block'].astype(str) + ' ' + df_hdbResale['street_name'].astype(str)

# Lookup hdbInfo dataset, by calculating the average floor level against max_floor_lvl and populate a new 'level' column
df_hdbResale['avg_storey'] = df_hdbResale['storey_range'].apply(calculate_avg_storey)                   # Get the average storey level in integer
df_hdbResale = df_hdbResale.merge(df_hdbInfo[['address', 'max_floor_lvl']], on='address', how='left')   # Merge datasets based on 'address'
df_hdbResale['level'] = df_hdbResale.apply(calculate_height_category, axis=1).astype('int32')           # Add the level category into a new column 'level'

# Add a new column 'price_sqm' by dividing 'resale_price' by 'floor_area_sqm'
df_hdbResale['price_sqm'] = (df_hdbResale['resale_price'] / df_hdbResale['floor_area_sqm']).astype('int32')

# Lookup hdbInfo dataset using 'address' and add columns 'distance_km' and 'postalcode' for each record
df_hdbResale = df_hdbResale.merge(df_hdbInfo[['address', 'distance_km', 'postalcode']], on='address', how='left')

# Remove the 8 outlier transactions of flat_type '3-room' AND more than 200 sqm floor area
df_hdbResale = df_hdbResale[~((df_hdbResale['flat_type'] == '3 ROOM') & (df_hdbResale['floor_area_sqm'] > 200))]

# Cast 'month' column into datetime format
df_hdbResale['month'] = pd.to_datetime(df_hdbResale['month'], format='%Y-%m')

# Encode 'town' and 'flat_type' in numerical format using label encoding then cast them as integer
categorical_cols = ['town', 'flat_type']    # Convert categorical columns into numerical format
label_encoders = {}                         # For storing the label encoders
for col in categorical_cols:
    le = LabelEncoder()
    df_hdbResale.loc[:, col] = le.fit_transform(df_hdbResale[col])
    label_encoders[col] = le                # Save encoders for future reference
df_hdbResale = df_hdbResale.astype({'town': 'int32', 'flat_type': 'int32'})

# Save the label encoder for 'town' as a CSV
town_encoder = label_encoders['town']

# Create a DataFrame with the mappings of encoded values and original labels
town_mapping = pd.DataFrame({
    'encoded': range(len(town_encoder.classes_)),
    'original': town_encoder.classes_
})

# Save the mapping for later reuse
town_mapping.to_csv('town_encoder.csv', index=False)

# Plot correlation heatmap for the numerical values
plt.figure(figsize=(8, 6))
sns.heatmap(df_hdbResale[["town", "flat_type", "floor_area_sqm", "price_sqm", "lease_commence_date", "remaining_lease", "mature", "distance_km", "level", "resale_price"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Columns to be dropped
df_prepared = df_hdbResale.drop(columns = ['flat_type', 'block', 'street_name', 'storey_range', 'flat_model', 'lease_commence_date', 'address', 'postalcode', 'avg_storey', 'max_floor_lvl'])

# Save a copy of the prepared data frame for future reuse
df_prepared.to_csv('Prepared.csv', index=False)

# Cast 'month' as datetime type, if not, before extracting 'years' from datetime to numerical
if not ptypes.is_datetime64_any_dtype(df_prepared['month']):
  df_prepared['month'] = pd.to_datetime(df_prepared['month'], utc=True).dt.tz_localize(None)
df_prepared['years'] = df_prepared['month'].dt.year

# Create a feature that counts the number of months from baseline of the earliest month (Jan 2017), as the models cannot accept datetime data type
df_prepared['months'] = ((df_prepared['month'].dt.year - df_prepared['month'].min().year) * 12 + (df_prepared['month'].dt.month - df_prepared['month'].min().month))

# Drop the original datetime column since we already have the count of number of months
df_prepared = df_prepared.drop(columns=['month'], axis=1)

# Convert distance from km to m then convert all features to 'int32' data type for easier working with numpy array later
df_prepared['distance_km'] = df_prepared['distance_km'] * 1000
df_prepared = df_prepared.astype('int32')

df_prepared.info(), df_prepared.head()