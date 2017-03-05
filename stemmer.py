from nltk import stem
import re

class SimpleStemmer:
    def __init__(self):
        self.lemmatizer=stem.WordNetLemmatizer()
        
    def get_stemed_sentence(self,sentence):
        stemed_sentence=""
        for token in self.get_stemed_tokens(sentence):
            stemed_sentence+=token+" "
        return stemed_sentence[:-1]
        
    def get_stemed_tokens(self,sentence):
        lower=sentence.lower()
        
        #removing special character
        lower=re.compile(r"[^a-zA-z\s\n]").sub("",lower)
        
        #tokenize here(removing "\n")
        tokens=filter(lambda w: len(w) > 0, re.split(r'\s|\n',lower))
        
        stemmed_tokens=[]
        for token in tokens:
            try:
                stemmed_tokens.append(self.lemmatizer.lemmatize(token))
            except LookupError:
                continue
            
        return stemmed_tokens
#test
if __name__=="__main__":
    stemmer=SimpleStemmer()
    print(stemmer.get_stemed_sentence("Watson is$ cognitive\ncomputing systems"))
    print(stemmer.get_stemed_tokens("Watson is$ cognitive\ncomputing systems"))