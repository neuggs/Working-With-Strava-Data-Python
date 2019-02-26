import pandas as pd
import matplotlib.pyplot as plt

all_strava_data = pd.read_csv('All_Strava_Data.csv')
print(all_strava_data.head(10))

strava_data = all_strava_data[(all_strava_data.type == 'Ride') | (all_strava_data.type == 'VirtualRide')]


def my_histogram(data, color, edgecolor, bins, title, label):
    plt.hist(data, color=color, edgecolor=edgecolor, bins=bins)
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.show()

my_histogram(strava_data['distance_mi'], 'blue', 'black', 20, 'Distance Histogram', 'Distance in Miles')
my_histogram(strava_data['moving_time'], 'blue', 'black', 20, 'Moving Time Histogram', 'Time in Minutes')
my_histogram(strava_data['total_elevation_gain'], 'blue', 'black', 20, 'Elevation Gain Histogram', 'Elevation Gain in Feet')
#my_histogram(strava_data['gear_id'], 'blue', 'black', 10, 'Gear Id Histogram', 'Gear')
my_histogram(strava_data['average_watts'], 'blue', 'black', 10, 'Watts Histogram', 'Average Power in Watts')
my_histogram(strava_data['average_heartrate'], 'blue', 'black', 10, 'Heart Rate Histogram', 'Average HR in Beats per Minute')
my_histogram(strava_data['average_temp'], 'blue', 'black', 10, 'Temperature Histogram', 'Average Temperature in Celsius')




