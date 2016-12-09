'''
Created on Nov 23, 2016

@author: AA361063
'''
from bs4 import BeautifulSoup
import urllib2
import csv


count=0
with open("result.txt","r") as f_url,open("word2vec_train_data","w") as f_out:
    
    for url in f_url.readlines():
        try:
            html=urllib2.urlopen(url)
        except urllib2.HTTPError:
            print "HTTPError:",url
        soup=BeautifulSoup(html)
        
#         for section in soup.find_all("div",class_="section"):
#             
#             #section div is composed by section title and section content
#             section_title=section.contents[0].text
#             print "##title",section_title
#             section_content=section.contents[1].text
#             print "##content",section_content
#             
#             writer.writerow([title_tag,section_title,section_content])          
        
        body=soup.find("body")
        
        next_tag=body
        while next_tag is not None:
            if next_tag.name=="p" or next_tag.name=="li":
                f_out.write(next_tag.text+" ")
            
            next_tag=next_tag.next
