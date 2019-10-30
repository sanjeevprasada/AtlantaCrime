import csv
import pandas as pd
df = pd.read_csv("data/COBRA-2009-2018.csv")
keep_col = ['Occur Date', 'Occur Time', 'Beat', 'Location', 'Shift Occurence', 'UCR #', 'Neighborhood', 'NPU', 'Longitude', 'Latitude']
new_f = df[keep_col]

# new_f = new_f['Beat'].fillna(0)
print(new_f['Occur Time'].dtype)
new_f['Beat'] = new_f['Beat'].astype('str')
new_f[['Beat','Useless']] = new_f['Beat'].str.split('.',expand=True)

new_f = new_f.drop(columns = ['Useless', 'Shift Occurence'])

new_f[['Neighborhood']] = new_f[['Neighborhood']].fillna(value='Null')

def occurence_time(row):
	if row['Occur Time'] < 800:
		return "Morning"
	elif row['Occur Time'] < 1600:
		return "Day"
	else:
		return "Evening"


new_f['Shift Occurence'] = new_f.apply (lambda row: occurence_time (row),axis=1)

new_f = new_f[ (new_f['Longitude'] >= -85) & (new_f['Longitude'] <= -84) ]



new_f.to_csv("cobra-clean.csv", index=False)