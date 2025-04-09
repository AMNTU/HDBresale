import pandas as pd             # version 2.2.2
import numpy as np              # version 2.0.2
import matplotlib.pyplot as plt
import seaborn as sns

# Read the HDB Resale transactions dataset into a dataframe
df_hdbResale = pd.read_csv('ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')

# Display properties of the dataframe and the first five rows of data
df_hdbResale.info(), df_hdbResale.head()

# Plot correlation heatmap for the numerical values
plt.figure(figsize=(8, 6))
sns.heatmap(df_hdbResale[["floor_area_sqm", "lease_commence_date", "resale_price"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Visualizing the relationship between floor area and resale price
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_hdbResale, x="floor_area_sqm", y="resale_price", alpha=0.5)
plt.title("Resale Price vs. Floor Area (sqm)")
plt.xlabel("Floor Area (sqm)")
plt.ylabel("Resale Price (SGD)")
plt.show()

# Filter the records with floor area sqm more than 200
print(df_hdbResale[df_hdbResale["floor_area_sqm"] > 200])

# Boxplot of resale prices by town
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_hdbResale, x="town", y="resale_price", palette="pastel", hue="town", legend=False)
plt.xticks(rotation=90)
plt.title("Resale Price Distribution by Town")
plt.xlabel("Town")
plt.ylabel("Resale Price (SGD)")
plt.show()

# Read the HDB Property Info master dataset into a dataframe
df_hdbInfo = pd.read_csv('HDBPropertyInformation.csv')

# Filter the dataframe where 'residential' is 'Y' and reindex the filtered dataframe
df_hdbInfo_filtered = df_hdbInfo[df_hdbInfo['residential'] == 'Y'].reset_index(drop=True)

# Merge 'blk_no' and 'street' into 'address' for lookup later with the transactional dataset
address = pd.Series(df_hdbInfo['blk_no'].astype(str) + ' ' + df_hdbInfo['street'].astype(str), name='address')

# Define the towns abbreviation dictionary
category = {
    'BB': 'BUKIT BATOK',
    'BP': 'BUKIT PANJANG',
    'CCK': 'CHOA CHU KANG',
    'HG': 'HOUGANG',
    'JE': 'JURONG EAST',
    'JW': 'JURONG WEST',
    'PG': 'PUNGGOL',
    'SB': 'SEMBAWANG',
    'SK': 'SENGKANG',
    'TG': 'TENGAH',
    'WL': 'WOODLANDS',
    'YS': 'YISHUN',
    'KWN': 'KALLANG/WHAMPOA',
    'BD': 'BEDOK',
    'CT': 'CENTRAL AREA',
    'PRC': 'PASIR RIS',
    'BM': 'BUKIT MERAH',
    'QT': 'QUEENSTOWN',
    'GL': 'GEYLANG',
    'TP': 'TAMPINES',
    'SGN': 'SERANGOON',
    'MP': 'MARINE PARADE',
    'BT': 'BUKIT TIMAH',
    'TAP': 'TOA PAYOH',
    'AMK': 'ANG MO KIO',
    'BH': 'BISHAN',
    'CL': 'CLEMENTI'
}

# Map the categories to their full town names
towns = pd.Series(df_hdbInfo['bldg_contract_town'].map(category), name='towns')

# Keep only the first 4 columns (blk_no, street, max_floor_lvl, year_completed)
df_hdbInfo = df_hdbInfo_filtered.iloc[:, :4]

# Add 'address' and 'towns' into the df_hdbInfo dataframe
df_hdbInfo = pd.concat([df_hdbInfo, towns, address], axis=1)

# Display the info and first five rows of the dataset
df_hdbInfo.info(), df_hdbInfo.head()