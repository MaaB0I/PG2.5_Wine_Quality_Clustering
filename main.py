import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import seaborn as sns

# Lese die Daten ein
path_red = 'venv/CSV_Files/winequality-red.csv'
path_white = 'venv/CSV_Files/winequality-white.csv'
data_red = pd.read_csv(path_red, delimiter=';')
data_white = pd.read_csv(path_white, delimiter=';')

# Füge eine Spalte für den Weintyp hinzu
data_red['type'] = 'red'
data_white['type'] = 'white'

# Kombiniere die Datensätze
data = pd.concat([data_red, data_white], ignore_index=True)

# Konvertiere den Weintyp in numerische Werte
data['type'] = data['type'].map({'red': 0, 'white': 1})

# Ermittle die optimale Gruppenzahl
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,10), timings=False)
visualizer.fit(data)
visualizer.show()

# Clustere die Daten
optimal_clusters = visualizer.elbow_value_
kmeans = KMeans(n_clusters=optimal_clusters)
data['cluster'] = kmeans.fit_predict(data)

# Deskriptive Statistik
print(data.describe())

# Boxplot für die Qualität, gruppiert nach Cluster
sns.boxplot(x='cluster', y='quality', data=data)
plt.show()

# Scatter Plot für Alkohol gegen Dichte, farbkodiert nach Cluster
sns.scatterplot(x='alcohol', y='density', hue='cluster', data=data, palette='viridis')
plt.show()
