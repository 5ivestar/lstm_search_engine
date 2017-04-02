import data_handler
import stemmer as stm
import csv
import codecs

kcdata_csv="data/kc_data.csv"
output_file="data/kcdata.txt"

stemmer=stm.SimpleStemmer()
with codecs.open(kcdata_csv,encoding="utf-8") as f,open(output_file,"w") as out:
    reader=csv.reader(f)
    
    #skip header
    next(reader)
    
    for row in reader:
        line=row[1]+" "+row[2]+" "+row[3]
        stemmed=stemmer.get_stemed_sentence(line)
        out.write(stemmed+"\n")
        
    
