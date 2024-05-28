import requests
from bs4 import BeautifulSoup as bs
import time
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

start_time = time.time()

# Store all the pages that we are interested in the website
pages = []

# Construct URLs for each page
url_start = 'https://www.centralcharts.com/en/price-list-ranking/ALL/desc/ts_19-us-nasdaq-stocks--qc_3-previous-close-change?p='
for page_number in range(1, 5):
    url = url_start + str(page_number)
    pages.append(url)

# Contains lists that represent each row of the table
stock_list = [] 
for page in pages:
    webpage = requests.get(page)
    soup = bs(webpage.text, 'html.parser')
    
    stock_table = soup.find('table', class_='tabMini tabQuotes')
    tr_tag_list = stock_table.find_all('tr')
    
    for each_tr_tag in tr_tag_list[1:]:
        td_tag_list = each_tr_tag.find_all('td')
        
        row_values = []
        for each_td_tag in td_tag_list[0:7]:
           new_value = each_td_tag.text.strip()
           row_values.append(new_value)
        stock_list.append(row_values)
        
print(stock_list)
print('--- %s seconds ---' % (time.time() - start_time))


column_names = ['Name', 'Current Price', 'Change %', 'Open', 'High', 'Low', 'Volume']

df = pd.DataFrame(stock_list, columns = column_names)

print(df)

df.dropna(inplace = True)

#to use the Change % data I need to convert to a float variable

df['Change %'] = df['Change %'].str.rstrip('%').astype(float)

print(df.describe())

#winsorizing the dataset is important to make sure outliers are not effecting the analysis.
#For irregular stock activities, I will winsorize to conform towards normality.
# A good prescription is to winsorize rates of return around plus and minus 10- 15%, especially for samples of all CRSP stocks. There is no meaningful predictive gain using stocks’ own means and standard deviations and/or contemporaneous market rates of return; and eliminating or zeroing returns rather than winsorizing them. Winsorizing is better than not winsorizing.

from scipy.stats.mstats import winsorize
df['Volume'] = winsorize(df['Volume'], limits=[0.05, 0.05])  # Winsorize volume column

df['Change %'] = winsorize(df['Change %'], limits=[0.05, 0.05])

def clean_volume(volume):
    if isinstance(volume, str):
        return float(volume.replace(',', ''))
    else:
        return volume

# Apply the custom function to the 'Volume' column
df['Volume'] = df['Volume'].apply(clean_volume)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_columns = ['Current Price', 'Change %', 'Open', 'High', 'Low', 'Volume']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

df[numeric_columns]

df

print(df.describe())

sns.violinplot(x=df['Volume'], inner='quartile')
plt.xlabel('Volume')
plt.title('Violin Plot of Volume')
plt.show()

sns.scatterplot(x='Volume', y='Change %', data=df)
plt.xlabel('Volume')
plt.ylabel('Change %')
plt.title('Scatter Plot: Volume vs. Change %')
plt.show()

correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

plt.hist(df['Change %'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Change %')
plt.ylabel('Frequency')
plt.title('Histogram of Change %')

plt.show()

#I want to analyze the winners

df_change_percentage_sorted = df.sort_values(by='Change %', ascending=False)

Top_5_Highest = df_change_percentage_sorted.head(5)

Top_5_Lowest = df_change_percentage_sorted.tail(5)

print(Top_5_Highest['Name'])

print(Top_5_Lowest['Name'])

#Add in data to connect with the industry sectors each companies are categorized in.
#Analyze what industry had the largest movement and trading volume.
