#!/usr/bin/env python
# coding: utf-8

# In[14]:


from newspaper import Article
import nltk
nltk.download('all')
import string
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords


# In[4]:


url='https://www.thehindu.com/incoming/kishan-reddy-has-lunch-with-the-family-that-hosted-dr-ambedkar-in-1943/article68065398.ece'


# In[5]:


article= Article(url)


# In[7]:


article.download()
article.parse()
article.nlp()


# In[10]:


news = article.text


# In[11]:


token_sent= sent_tokenize(news)


# In[20]:


def remove_punctuation(s):
    if s not in string.punctuation:
        return True
    else:
        return False


# In[35]:


tagged=[]
for i in token_sent:
    word_list= word_tokenize(i)
    stopword_list= list(set(stopwords.words('english')))
    stemmer= LancasterStemmer()
    print(stemmer.stem(i))
    filtered_list= [i for i in word_list if i not in stopword_list]
    tagged_list= nltk.pos_tag(list(filter(remove_punctuation,filtered_list)))
    tagged.extend(tagged_list)


# In[32]:


chunked_sents= nltk.ne_chunk(tagged)


# In[34]:


for sent in chunked_sents:
    if hasattr(sent,'label'):
        name= ''.join(n[0] for n in sent)
        label= sent.label()
        print(name, label)


# In[ ]:




