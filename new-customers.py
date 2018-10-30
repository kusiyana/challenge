#-------------------------------------------------------
#
# Script to get new customer sign ups with cumulative approach
# Hayden Eastwood - 19-10-2018
# Last updated:
#
#
#
# -------------------------------------------------------

from sklearn import linear_model
import pandas as pd
import calendar
import matplotlib.pyplot as plt


print " "
print " * * * * * * * * * * * * * * * * *"
print " - Totus data challenge.         -"
print " - Hayden Eastwood               - "
print " - hayden.eastwood@gmail.com.    - "
print " * * * * * * * * * * * * * * * * *"
print " "

print " -- Loading data"
with open('challenge.json', 'r') as f:
	data = pd.read_json(f)

print " -- Setting parameters"
start_year = 2008
end_year = 2018
month_range = 12

data.register_date = pd.to_datetime(data.register_date, format='%Y-%m-%d')
data.set_index('register_date', inplace=True)
data = data.sort_index()
data_date_sorted = data
month_count = 0

uniques_per_month = []
uniques_per_month_count = []
uniques_per_month_count = []

updated_set = set()

print " -- Finding cumulative users per month"
for year in range(start_year, end_year):
	usersArray = []
	if year == 2018:
		month_range = 6 # make sure we only look at last 6 months since data collection stopped mid month in following months
	for month in range(0, month_range):
		date_string_base = str(year) + '-' + str(month + 1) 
		start_date = date_string_base + '-01' 
		end_date = date_string_base + '-' + str(calendar.monthrange(year, month + 1)[1])
		print "     -- " + str(start_date) + ' - ' + str(end_date)
		usersArray.append(set(data_date_sorted[(data_date_sorted.index > start_date) & (data_date_sorted.index <= end_date)].groupby('customer_code').describe().index))
		if month_count > 0:
			updated_set = updated_set.union(*usersArray[0:month])
		uniques_per_month.append(usersArray[month] - updated_set)
		uniques_per_month_count.append(len(uniques_per_month[month_count]))
		month_count += 1

print " -- Generate graph and fit line"
cumsum = pd.DataFrame(uniques_per_month_count).cumsum()
indexes = pd.DataFrame(range(0,cumsum.shape[0]))
lm = linear_model.LinearRegression(fit_intercept=True)
model = lm.fit(indexes,cumsum[0])

sign_up_rate = model.coef_
print "     -- Customer sign up rate (regression calculated): " + str(sign_up_rate[0])

predictions = lm.predict(indexes)
plt.plot(predictions)
plt.title('Cumulative customer sign up')
plt.plot(cumsum)
plt.show()
