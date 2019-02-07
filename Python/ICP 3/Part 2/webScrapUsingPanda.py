#importing panda library
import pandas as pd
url = "https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India"
#read_html to read the HTML tables
#Indentifying the table of required class using 'attrs'
df = pd.read_html(url, attrs={"class":"wikitable sortable plainrowheaders"})
#Writing the DataFrame into csv file
df[0].to_csv('output3.csv')
