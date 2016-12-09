'''
Created on Nov 8, 2016

@author: AA361063
'''
from selenium import webdriver


# binary = 'C:\Users\IBM_ADMIN\phantomjs-2.1.1-windows\phantomjs-2.1.1-windows\bin\phantomjs.exe'
# browser = webdriver.PhantomJS()
# browser.get("https://www.ibm.com/support/knowledgecenter/ja/SS8NLW_11.0.1/com.ibm.discovery.es.nav.doc/explorer_analytics.htm")
# 
# 
# print browser.page_source
# browser.close()

import re
def link_generator(href_str,current_url):
    tail="?view=embed"
    head="https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1"
    
    p_dir=re.compile('\.\./com\.ibm\.discovery\.es\..*?/(.*?\.html?)')
    if p_dir.search(href_str):
        return head+href_str[2:]+tail
    
    p_html=re.compile("^[^/]+?\.html?")
    if p_html.search(href_str):
        html_file_name=re.compile("[^/]+?\.html?").findall(current_url)[0]
        return current_url[:-len(html_file_name)]+href_str+tail
    
    return None

#test
print link_generator("gomi.html","https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1/com.ibm.discovery.es.nav.doc/explorer_analytics.html")
print link_generator("../com.ibm.discovery.es.ftakashi/explorer_analytics.html","https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1/com.ibm.discovery.es.nav.doc/explorer_analytics.html")

class KcCrawler:
    
    def __init__(self,url_regular):
        self.p=re.compile(url_regular)
        self.crawled_urls=[]
        self.browser=webdriver.PhantomJS()
        self.host="https://www.ibm.com"
        
    def crawl(self,starturl):
        print "visitting",starturl
        self.browser.get(starturl)
        #print self.browser.page_source
        refs=self.p.findall(self.browser.page_source)
        for ref in refs:
            url=link_generator(ref, starturl[:-11])
            if url is None:
                continue
            
            if url not in self.crawled_urls:
                self.crawled_urls.append(url)
                self.crawl(url)


pattern='https://www\.ibm\.com/support/knowledgecenter/en/SS8NLW_11\.0\.1/com\.ibm\.discovery\.es\..*?\.html'
pattern2='/support/knowledgecenter/en/SS8NLW_11\.0\.1/com\.ibm\.discovery\.es\..*?\.html'
pattern3='href="(.*?)"'
start_url="https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1/com.ibm.discovery.es.nav.doc/explorer_analytics.html"
start_url2="https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1/com.ibm.discovery.es.in.doc/iiypofnv_install_cont.html?view=embed"

#regular expression test
p=re.compile(pattern)
print p.findall(start_url)


crawler=KcCrawler(pattern3)

seed_urls=["https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1/com.ibm.discovery.es.in.doc/iiypofnv_install_cont.html?view=embed"
           ,"https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1/com.ibm.discovery.es.nav.doc/explorer_analytics.htm?view=embed"
           ,"https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1/com.ibm.discovery.es.nav.doc/iiysaovca.htm?view=embed"
           ,"https://www.ibm.com/support/knowledgecenter/en/SS8NLW_11.0.1/com.ibm.discovery.es.ad.doc/iiysacolsrch.htm?view=embed"]

for url in seed_urls:
    crawler.crawl(url)
with open("result.txt","w") as f:
    for url in crawler.crawled_urls:
        f.write(url+"\n")

print len(crawler.crawled_urls),"urls"
print "finished"
    