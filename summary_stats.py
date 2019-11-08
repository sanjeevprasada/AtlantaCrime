import csv
import pandas as pd
import datetime
startTime = datetime.datetime.now()

# Read the data
df = pd.read_csv("data/COBRA-2009-2018.csv")
keep_col = ['Occur Date', 'Occur Time', 'Location', 'Shift Occurence', 'UCR Literal', 'Neighborhood', 'Longitude', 'Latitude']
new_f = df[keep_col]

# new_f = new_f['Beat'].fillna(0)
# print(new_f['Occur Time'].dtype)
# new_f['Beat'] = new_f['Beat'].astype('str')
# new_f[['Beat','Useless']] = new_f['Beat'].str.split('.',expand=True)

new_f = new_f.drop(columns = ['Shift Occurence'])

# new_f[['Neighborhood']] = new_f[['Neighborhood']].fillna(value='Null')

def occurence_time(row):
	if row['Occur Time'] < 800:
		return "Morning"
	elif row['Occur Time'] < 1600:
		return "Day"
	else:
		return "Evening"

def print_month(row):
    cur_date = row['Occur Date']
    year, month, day = (int(x) for x in cur_date.split('-'))    
    month = int(month)
    return month

def day_of_week(row):
    cur_date = row['Occur Date']
    year, month, day = (int(x) for x in cur_date.split('-'))    
    dt = datetime.date(year, month, day)
    return(dt.weekday())

new_f['Day of Week'] = new_f.apply(lambda row: day_of_week (row), axis=1)
new_f['Month'] = new_f.apply(lambda row: print_month (row), axis=1)
new_f['Shift Occurence'] = new_f.apply (lambda row: occurence_time (row),axis=1)

new_f = new_f[ (new_f['Longitude'] >= -84.5) & (new_f['Longitude'] <= -84.2)]
new_f = new_f[ (new_f['Latitude'] >= 33.61) & (new_f['Latitude'] <= 33.92)]

new_f = new_f.dropna(axis=0, subset=['Neighborhood'])


n_list = new_f.Neighborhood.unique()
# print(n_list)

shifts = ('Morning', 'Day', 'Evening')
days = (0,1,2,3,4,5,6)
months = (1,2,3,4,5,6,7,8,9,10,11,12)

#category = dict((day, shift) for item in day)
#stuff = dict((months, category) for month in months)
#data_dict = dict((key, stuff) for key in n_list)
def count_occurrences(local,month,day,shift):
	cat1 = len(new_f[(new_f['Neighborhood']==local) & (new_f['Month']==month) & (new_f['Day of Week']==int(day)) & (new_f['Shift Occurence']==shift) & ((new_f['UCR Literal']=='HOMICIDE') | (new_f['UCR Literal']=='MANSLAUGHTER'))])
	cat2 = len(new_f[(new_f['Neighborhood']==local) & (new_f['Month']==month) & (new_f['Day of Week']==int(day)) & (new_f['Shift Occurence']==shift) & ((new_f['UCR Literal']=='AGG ASSAULT') | (new_f['UCR Literal']=='ROBBERY-PEDESTRIAN') | (new_f['UCR Literal']=='ROBBERY-COMMERCIAL') | (new_f['UCR Literal']=='ROBBERY-RESIDENCE'))])
	cat3 = len(new_f[(new_f['Neighborhood']==local) & (new_f['Month']==month) & (new_f['Day of Week']==int(day)) & (new_f['Shift Occurence']==shift) & ((new_f['UCR Literal']=='BURGLARY-RESIDENCE') | (new_f['UCR Literal']=='BURGLARY-NONRES') | (new_f['UCR Literal']=='AUTO THEFT'))])
	cat4 = len(new_f[(new_f['Neighborhood']==local) & (new_f['Month']==month) & (new_f['Day of Week']==int(day)) & (new_f['Shift Occurence']==shift) & ((new_f['UCR Literal']=='LARCENY-FROM VEHICLE') | (new_f['UCR Literal']=='LARCENY-NON VEHICLE'))])
	count = (cat1, cat2, cat3, cat4)
	return count

our_dict = {}
for local in n_list:
	our_dict[local] = {}

	for month in months:
		our_dict[local][month] = {}
		for day in days:
			our_dict[local][month][day] = {}
			for shift in shifts:
				our_dict[local][month][day][shift] = count_occurrences(local, month, day, shift)
				print(our_dict[local][month][day][shift])


# new_f.to_csv("cobra-summary.csv", index=False)
	
print(our_dict)
csv_file = "stats.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile)
        writer.writeheader()
        for data in our_dict:
            writer.writerow(data)
    print(datetime.datetime.now() - startTime)
except IOError:
    print("I/O error") 