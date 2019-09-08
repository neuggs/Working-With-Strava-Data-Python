import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import thinkplot
import thinkstats2
import numpy as np
import statsmodels.formula.api as smf

########### PART ONE ############
# Pull the data
all_strava_data = pd.read_csv('All_Strava_Data.csv')
#print(all_strava_data.head(10))

# I only want ride data - this is what I'm analyzing.
strava_data = all_strava_data[
    ((all_strava_data.type == 'Ride') | (all_strava_data.type == 'VirtualRide')) &
    ((all_strava_data.gear_id == 'b1477130') |
     (all_strava_data.gear_id == 'b2653090') |
     (all_strava_data.gear_id == 'b575984') |
     (all_strava_data.gear_id == 'b1395475') |
     (all_strava_data.gear_id == 'b250312') |
     (all_strava_data.gear_id == 'b249850') |
     (all_strava_data.gear_id == 'b250313') |
     (all_strava_data.gear_id == 'b266100') |
     (all_strava_data.gear_id == 'b350107') |
     (all_strava_data.gear_id == 'b635473') |
     (all_strava_data.gear_id == 'b1108192') |
     (all_strava_data.gear_id == 'b2468160')
    )
]

########### PART TWO ############
distance_mi = strava_data['distance_mi']
moving_time = strava_data['moving_time']
total_elevation_gain = strava_data['total_elevation_gain']
gear = strava_data['gear_id']
average_speed_mph = strava_data['avg_speed_mph']
average_watts = strava_data['average_watts']
average_heartrate = strava_data['average_heartrate']
average_temp = strava_data['average_temp']

########### PART THREE ############
# Reusable function to plot a histogram
def my_histogram(data, color, edgecolor, bins, title, label):
    plt.hist(data, color=color, edgecolor=edgecolor, bins=bins)
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.show()

# Plot the variables, except for gear_id.
my_histogram(distance_mi, 'blue', 'black', 20, 'Distance Histogram', 'Distance in Miles')
my_histogram(moving_time, 'blue', 'black', 20, 'Moving Time Histogram', 'Time in Seconds')
my_histogram(elevation_gain, 'blue', 'black', 20, 'Elevation Gain Histogram', 'Elevation Gain in Feet')
my_histogram(avg_speed_mph, 'blue', 'black', 20, 'Average Speed Histogram', 'Average Speed in MPH')
my_histogram(average_watts, 'blue', 'black', 20, 'Watts Histogram', 'Average Power in Watts')
my_histogram(average_heartrate, 'blue', 'black', 20, 'Heart Rate Histogram', 'Average HR in Beats per Minute')
my_histogram(average_temp, 'blue', 'black', 20, 'Temperature Histogram', 'Average Temperature in Celsius')

# Gear id is categorical and has to be handled differently.
sns.set(style='darkgrid')
sns.countplot(x='gear_id', data=strava_data)
plt.title('Gear Id Histogram')
plt.xlabel('Gear Id')
plt.ylabel('Frequency')
plt.show()

########### PART FOUR ############
def desc_stats(data, xlabel):
    mean = round(data.mean(), 2)
    median = round(data.median(), 2)
    mode = data.mode()
    spread = round(data.var(), 2)
    sd = round(data.std())
    print(xlabel, '\nMean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0],'\nSpread: ',spread, '\nStd Dev: ', sd, '\n')
    plt.figure(figsize=(10,5))
    plt.hist(data,bins=20,color='grey')
    plt.axvline(mean,color='red',label='Mean')
    plt.axvline(median,color='yellow',label='Median')
    plt.axvline(mode[0],color='green',label='Mode')
    plt.axvline(sd,color='orange',label='Std Dev')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

desc_stats(distance_mi, 'Distance in Miles')
desc_stats(moving_time, 'Moving Time in Seconds')
desc_stats(total_elevation_gain, 'Total Elevation Gain')
desc_stats(avg_speed_mph, 'Average Speed in MPH')
desc_stats(average_watts, 'Average Power in Watts')
desc_stats(average_heartrate, 'Average Heart Rate')
desc_stats(average_temp, 'Average Temperature in Celsius')

########### PART FIVE ############
def pmf_stuff(width, x_low, x_high, third, pmf_one, pmf_two, label, y_axis_scale):
    width=width
    axis=[x_low, x_high, third, y_axis_scale]
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(pmf_one, align='right', width=width)
    thinkplot.Hist(pmf_two, align='left', width=width)
    thinkplot.Config(xlabel=label, ylabel='PMF', axis=axis)

    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([pmf_one, pmf_two])
    thinkplot.Config(xlabel=label, ylabel='PMF', axis=axis)
    thinkplot.Show()

over_one_hr = moving_time[moving_time > 3600]
less_one_hr = moving_time[moving_time <= 3600]

pmf_more = thinkstats2.Pmf(over_one_hr, label='More Than One HR')
pmf_less = thinkstats2.Pmf(less_one_hr, label="Less Than One HR")

pmf_stuff(1, 1000, 10000, 0, pmf_more, pmf_less, 'Ride Length (Min)', 0.05)

over_three_hr = moving_time[moving_time > 10800]
less_three_hr = moving_time[moving_time <= 10800]
pmf_more = thinkstats2.Pmf(over_three_hr, label="More Than Three HR")
pmf_less = thinkstats2.Pmf(less_three_hr, label='Less Than Three HR')
pmf_stuff(1, 8000, 22000, 0, pmf_more, pmf_less, 'Ride Length (Min)', 0.02)

########### PART SIX ############
cdf = thinkstats2.Cdf(moving_time, label='Moving Time')
thinkplot.Cdf(cdf)
thinkplot.Show(xlabel='Moving Time in Min', ylabel='CDF')

more_cdf = thinkstats2.Cdf(over_one_hr, label='Over Than One Hr')
less_cdf = thinkstats2.Cdf(less_one_hr, label='Less Than One Hr')
thinkplot.PrePlot(2)
thinkplot.Cdfs([more_cdf, less_cdf])
thinkplot.Show(xlabel='Moving Time (Min)', ylabel='CDF')

########### PART SEVEN ############
avg_watts = average_watts.dropna()

def MakeNormalModel(data, label):
    cdf = thinkstats2.Cdf(data, label=label)

    mean, var = thinkstats2.TrimmedMeanVar(data)
    std = np.sqrt(var)
    print('n, mean, std', len(data), mean, std)

    xmin = mean - 4 * std
    xmax = mean + 4 * std

    xs, ps = thinkstats2.RenderNormalCdf(mean, std, xmin, xmax)
    thinkplot.Plot(xs, ps, label='model', linewidth=4, color='0.8')
    thinkplot.Cdf(cdf)

# Watts
MakeNormalModel(avg_watts, 'Average Power in Watts')
thinkplot.Config(title='Power in Watts, Scale', xlabel='Watts',
                 ylabel='CDF', loc='upper right')
thinkplot.Show()

# Log Watts
log_watts = np.log10(avg_watts)
MakeNormalModel(log_watts, 'Average Power in Log Watts')
thinkplot.Config(title='Avg Watts, Log Scale', xlabel='Watts (log10 w)',
                 ylabel='CDF', loc='upper right')
thinkplot.Show()

# Distance in MI
MakeNormalModel(distance_mi, 'Distance in Miles')
thinkplot.Config(title='Distance in Miles, Scale', xlabel='Distance (MI)',
                 ylabel='CDF', loc='upper right')
thinkplot.Show()

# Log Watts
log_dist = np.log10(distance_mi)
MakeNormalModel(log_dist, 'Distance in Log Miles')
thinkplot.Config(title='Distance in Miles, Log Scale', xlabel='Distance (MI)',
                 ylabel='CDF', loc='upper right')
thinkplot.Show()

########### PART EIGHT ############
def random_sample(df, nrows, replace=False):
    indices = np.random.choice(df.index, nrows, replace=replace)
    sample = df.loc[indices]
    return sample

def plot_sample(x_data, y_data, x_label, y_label, axis):
    thinkplot.Scatter(x_data, y_data, alpha=1)
    thinkplot.Config(xlabel=x_label,
                 ylabel=y_label,
                 axis=axis,
                 legend=False)
    thinkplot.Show()

def covariance (xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov

def corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = covariance(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr

sample = random_sample(strava_data, 500)

# Speed / Power
speed, watts = sample.avg_speed_mph, sample.average_watts
plot_sample(speed, watts, 'Speed in MPH', 'Power in Watts', [5, 28, 30, 450])
# Need only records where watts is > 0
strava_data_with_watts = strava_data[
    (strava_data.average_watts > 0)]
print(corr(strava_data_with_watts.avg_speed_mph, strava_data_with_watts.average_watts))

# Speed / Temperature
speed, temp = sample.avg_speed_mph, sample.average_temp
plot_sample(speed, temp, 'Speed in MPH', 'Temperature in Celsius', [5, 25, -5, 40])
# Need only records where temperature exists
strava_data_with_temp = strava_data[
    (strava_data.average_temp.notnull())]
print(corr(strava_data_with_temp.avg_speed_mph, strava_data_with_temp.average_temp))

########### PART NINE ############
class CorrelationPermute(thinkstats2.HypothesisTest):
    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys

cleaned_data = strava_data.dropna(subset=['avg_speed_mph', 'average_watts'])
data = cleaned_data.avg_speed_mph.values, cleaned_data.average_watts.values
corr_perm = CorrelationPermute(data)
pvalue = corr_perm.PValue()
print(pvalue)
print(corr_perm.actual, corr_perm.MaxTestStat())

########### PART TEN ############
reg_formula = 'avg_speed_mph ~ average_watts'
model = smf.ols(reg_formula, data=strava_data)
results = model.fit()
print(results.summary())
print('Intercept:', results.params['Intercept'])
print('Slope:', results.params['average_watts'])
print('Slope p-value:', results.pvalues['average_watts'])
print('R-Squared:', results.rsquared)

reg_formula = 'avg_speed_mph ~ average_watts + total_elevation_gain + moving_time + average_temp'
model = smf.ols(reg_formula, data=strava_data)
results = model.fit()
print(results.summary())
print('Intercept:', results.params['Intercept'])
print('Slope:', results.params['average_watts'])
print('Slope p-value:', results.pvalues['average_watts'])
print('R-Squared:', results.rsquared)



