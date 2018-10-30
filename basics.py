#-------------------------------------------------------
#
# Script to get basic overview of data for Totvs data challenge
# Hayden Eastwood - 30-10-2018
# Last updated: 30-10-2018
# Version: 1.0
#
#
# -------------------------------------------------------
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt; plt.rcdefaults()


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

data.register_date = pd.to_datetime(data.register_date, format='%Y-%m-%d')
data.set_index('register_date', inplace=True)
data = data.sort_index()

print " -- Generating summary"

data_length = len(data) #1. Number of records: 204428
print "     -- Number of records: " + str(data_length)

start_date = data.head().index.tolist()[0] # transactions start 2008
print "     -- Start date: " + str(start_date)

end_date = data.tail().index.tolist()[0] # transactions end 2018
print "     -- End date  : " + str(end_date)

per_month = data.groupby('register_date').agg(np.sum).reset_index()
turnover_per_customer = data.groupby('customer_code').agg(np.sum).reset_index()

print "     -- Aggregate customer spending  : "
describe_stats = turnover_per_customer['total_price'].describe()
for j in range(len(describe_stats)):
	print "         --  " + describe_stats.index[j] + ": " + str(describe_stats[j])

unique_groups = len(data.groupby('group_code').nunique())
print "     -- unique groups  : " + str(unique_groups)
unique_sellers = len(data.groupby('seller_code').nunique())
print "     -- unique sellers  : " + str(unique_sellers)
unique_branches = len(data.groupby('branch_id').nunique())
print "     -- unique branches  : " + str(unique_branches)
unique_segments = len(data.groupby('segment_code').nunique())
print "     -- unique segments  : " + str(unique_segments)

plt.subplot(2, 1, 1)
plt.hist(turnover_per_customer['total_price'], bins=40,  range=(0, 2000000))
plt.title('Total spend per customer')

plt.subplot(2, 1, 2)
plt.title('Quantity bought per customer')
plt.hist(turnover_per_customer['quantity'], bins=40, range=(0, 10000))

plt.show()
