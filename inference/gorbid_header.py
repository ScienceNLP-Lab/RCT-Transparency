import os

from header.outline import header_clarify
from bs4 import BeautifulSoup

files = [i for i in os.listdir('xml/')]

for i in files:
    with open('xml/'+i, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')
    title = soup.title.getText()
    print(title)

    headers_list = soup.find_all('head')
    for header in headers_list:
        classify_header = ''
        classify_header = header_clarify([[header.getText()]])
        print(classify_header)