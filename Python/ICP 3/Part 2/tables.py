from bs4 import BeautifulSoup
import urllib.request
import csv
import pandas as pd
# Using CSV to write to a file without using Pandas
outputFile = open('output1.csv', 'w')
file = csv.writer(outputFile)
# Writing the headers
file.writerow(["State", "Adminstrative captials", "Legislative capitals", "Judiciary capitals", "Year capital established","The Former capital"])
url = "https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India"
source_code = urllib.request.urlopen(url)
soup= BeautifulSoup(source_code,  "html.parser")
table =  soup.find("table", {"class":"wikitable sortable plainrowheaders"})
rows=table.find_all('tr')[1:]
for row in rows:
        # Printing the th tag text
        state = row.find('th').text
        print("Headers:", state)
        # Finding all td tags
        td = row.find_all('td')
        # Checking the index range with length of td and getting required data
        if 1 in range(len(td)):
            column1 = td[1].text
            print("Adminstrative Capitals:", column1)
        if 2 in range(len(td)):
            column2 = td[2].text
            print("Legislative Capitals:", column2)
        if 3 in range(len(td)):
            column3 = td[3].text
            print("Judiciary Capitals:", column3)
        if 4 in range(len(td)):
            column4 = td[4].text
            print("Year of Capital establishment:", column4)
        # Not displaying Empty cells
        if 5 in range(len(td)) and td[5].text.strip() != '':
            column5 = td[5].text
            print("Former Capital:", column5)
        # Writing to a Row in CSV file
        file.writerow([state, column1, column2, column3, column4, column5])
outputFile.close()

#loading data frame into csv file using Pandas
df = pd.read_html(url, attrs={"class":"wikitable sortable plainrowheaders"})
df[0].to_csv('output2.csv')




