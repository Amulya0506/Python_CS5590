from bs4 import BeautifulSoup
import urllib.request

url = "https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India"
source_code = urllib.request.urlopen(url)
soup= BeautifulSoup(source_code,  "html.parser")
print(soup)
print("Title of the page: %s"%(soup.title.text))
print(soup.find_all('a'))
for link in soup.find_all('a'):
    print(link.get('href'))
