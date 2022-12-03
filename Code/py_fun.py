import pandas as pd
import matplotlib.pyplot as plt
#import collections
from sklearn import preprocessing
#import plotly.io as pio
import numpy as np
from tqdm import tqdm
import chardet
#from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.express as px

from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import textblob
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import math
from nltk import bigrams,trigrams,ngrams
#from collections import Counter
from sklearn.metrics import confusion_matrix
from nltk import everygrams, word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_confusion_matrix
CA_Asian_business_review = pd.read_csv('CA_Asian_business_review.csv',low_memory=False)
CA_Asian_business_review = CA_Asian_business_review[CA_Asian_business_review['is_open'] != 0].reset_index(drop = True)




#---------------------------------word cloud ---------------------------------------#

def show_word_cloud(sentiment, business_name):
  new_df= CA_Asian_business_review[CA_Asian_business_review['sentiment'] == sentiment].reset_index(drop=True)
  new_df= new_df[new_df['name'] == business_name].reset_index(drop=True)
  w_c = ""
  for i in range(len(new_df)):
    cur_list = new_df.loc[i,'text_cleaned'].split(',')
    for word in cur_list:
      word = word[2:-1]
      w_c += "".join(word)+ " "
  print(w_c)
  wordcloud_final = WordCloud(width = 800, height = 800, background_color ='white',min_font_size = 10, collocation_threshold = 3, min_word_length=3).generate(w_c)
  
  plt.imshow(wordcloud_final)
  plt.title(str(sentiment))
  plt.axis("off")
  plt.tight_layout(pad = 0)
  plt.show()
  
  
  
  
#--------------------------------- N gram plot ---------------------------------------#
def check_len(sentiment, business_name):
  new_df= CA_Asian_business_review[CA_Asian_business_review['sentiment'] == sentiment].reset_index(drop=True)
  new_df= new_df[new_df['name'] == business_name].reset_index(drop=True)
  cur_len = len(new_df)
  return cur_len



def generate_N_grams(text,ngram):
  temp=zip(*[text[i:] for i in range(0,ngram)])
  ans=["".join(ngram) for ngram in temp]
  return ans



def n_gram_plot(num_gram,business_name,num_result,sentiment, font, rotation):
  negative_dict = defaultdict(int)
  new_df = CA_Asian_business_review[CA_Asian_business_review.sentiment == sentiment]
  new_df = new_df[new_df.name == business_name]
  for text in new_df.text_cleaned:
    cur_list = text.split(",")
    for word in generate_N_grams(cur_list,num_gram):
      word = word.replace("'",'')
      negative_dict[word] += 1
          
          
  df_negative = pd.DataFrame(sorted(negative_dict.items(), key = lambda x : x[1], reverse= True))
  

  if sentiment == 'Positive':
    color = 'green'
  elif sentiment == 'Neutral':
    color = 'yellow'
  else:
    color = 'red'

  plt.figure(figsize=(6,2),dpi = 100)
  plt.bar(df_negative[0][:num_result], df_negative[1][:num_result], color = color, width = 0.3)
  plt.xticks(rotation = rotation)
  plt.ylabel("Count")
  plt.title("Top " + str(num_result) + ' '+ sentiment + " word")
  plt.xticks(fontsize = font)
  plt.yticks(fontsize = font)
  plt.show()






#---------------------------------Food list TF-IDF ---------------------------------------#

#Generate food list
Chinese_food_list = ['Almond milk', 'Ants Climbing a Tree','Asian pear','Baby bok choy','Baijiu','Beef brisket',"Beggar's Chicken",'Bingtang hulu','Bitter melon','Bubble tea',
"Buddha's Delight","Cantonese roast duck",'Century egg', 'thousand-year egg','Cha siu', 'Cantonese roast pork','Char kway teow',
'Chicken feet','Chinese sausage','Chow mein','Chrysanthemum tea','Claypot rice','Congee','Conpoy (dried scallops)','Crab rangoon','Dan Dan noodles',
'Dragonfruit',"Dragon's Beard candy","Dried cuttlefish",'Drunken chicken','Dry-fried green beans','Egg drop soup','Egg rolls','Egg tart','Fresh bamboo shoots',
'Fortune cookies','Fried milk','Fried rice','Gai lan' ,"General Tso's Chicken",'Gobi Manchurian','Goji berries' ,'Grass jelly','Hainan chicken rice','Hand-pulled noodles','Har gau ',
'Haw flakes','Hibiscus tea','Hong Kong-style Milk Tea','Hot and sour soup','Hot Coca-Cola with Ginger','Hot Pot',
'Iron Goddess tea' ,'Tieguanyin','Jellyfish','Kosher Chinese food','Kung Pao Chicken','Lamb skewers', "yangrou chua'r","Lion's Head meatballs",
'Lomo Saltado','Longan fruit','Lychee','Macaroni in soup with Spam','Malatang','Mantou','Mapo Tofu','Mock meat',
'Mooncake', 'Nor mai gai','Pan-fried dumplings','Peking duck','Pineapple bun','Prawn crackers',"Pu'er tea",'Rambutan',
'Red bean','Red bayberry','Red cooked pork','Roast pigeon','Rose tea','Roujiamo','Scallion pancakes','Shaved ice dessert',
'Sesame chicken','Sichuan pepper','Sichuan preserved vegetable', 'zhacai','Silken tofu','Soy milk','Steamed egg custard','Stinky tofu','Sugar cane juice',
'Sweet and sour pork', 'Sweet and sour chicken','Sweet and sour shrimp','Taro','Tea eggs','Tea-smoked duck','Turnip cake (law bok gau)',
'Twice-cooked pork','Water chestnut cake', 'mati gau','Wonton noodle soup','Wood ear',
'Xiaolongbao,' 'soup dumplings','Yuanyang','Yunnan goat cheese']


Japanese_food_list = ['sushi', 'ramen', 'udon', 'matcha', 'sashimi', 'tempura', 'unagi', 'wagyu', 'kushiyaki', 'yakitori', 'takoyaki', 'tofu', 'miso', 'kaisendon', 'donburi', 'chirashizushi',
'eel', 'unagi', 'anago', 'zuke', 'tako tamago', 'fukagawa', 'fugu', 'crab', 'okonomiyaki', 'oysters', 'gyoza', 'karaage', 'Japanese curry', 'croquette', 'korokke', 'yakiniku', 'tonkatsu'
, 'miso katsu', 'motsunabe', 'jingisukan', 'basashi', 'sakuraniku', 'dashi', 'tebasaki', 'doteni', 'kishimen', 'tsukemen', 'kakuni manju', 'champon', 'chawanmushi', 'konnyaku', 'tamagoyaki',
'odon', 'nabemono', 'onigiri', 'monjayaki', 'okonomiyaki', 'TKG', 'yakiniku', 'kaiseki','mochi', 'castella', 'sake', 'shabu shabu', 'miso Soup', 'soba', 'gyudon', 'natto','kashipan', 'sukiyaki',
'mentaiko', 'nikujaga', 'edamame', 'yakisoba','wagashi']


Korean_food_list = ['bibimbap', 'gimbap', 'tteokguk', 'japchae', 'ramyun', 'naengmyeon', 'haemul pajeon', 'godeungo jorim', 'ganjang gejang', 'nakji bokkeum', 'hotteok', 'tteokbokki',
'kimchi', 'samgyetang', 'sundubu jjigae', 'gamjatang', 'budae jjigae', 'army stew', 'galbijjim', 'bulgogi', 'samgyeopsal','Korean fried chicken', 'bossam', 'soondae', 'bingsu', 'sikhye', 'banchan',
'maekju','chueotang', 'gogigui', 'Korean BBQ', 'gomtang', 'jajangmyeon', 'jjukumi', 'kalguksu', 'makgeolli', 'mandu', 'naengmyeon', 'gochujang', 'octopus', 'squid', 'nakji bokkeum', 'ojingeo bokkeum',
'jeon', 'haemul pajeon', 'sannakji', 'seolnongtang', 'beongdegi', 'bindaetteok', 'fishcake', 'eomuk', 'odeng', 'tteokbokki', 'gyeranppang', 'hotteok', 'soondae','twigim', 'tempura', 'ddongbbang',
'dondurma', 'schnee pang', 'tornado potato']



new_Chinese_food_list = []
new_Japanese_food_list = []
new_Korean_food_list = []

def init_new_list (old_list, new_list):
  for word in old_list:
    cur_word = word.split(' ')
    for i in cur_word:
        new_list.append(i)

init_new_list(Chinese_food_list, new_Chinese_food_list)
init_new_list(Japanese_food_list, new_Japanese_food_list)
init_new_list(Korean_food_list, new_Korean_food_list)



def tfidf_viz(business_name, sentiment, num_gram, top_n, font, rotation, category):
  spec_restaurant = CA_Asian_business_review[CA_Asian_business_review['name'] == business_name]
  spec_sent = spec_restaurant[spec_restaurant['sentiment'] == sentiment]
  
  tfidf = TfidfVectorizer(ngram_range=(1,num_gram))
  spec_sent['TFIDF_text'] = [''.join(map(str,l)) for l in spec_sent.text_cleaned]
  transformed = tfidf.fit_transform(pd.Series(spec_sent.TFIDF_text))
  
  temp_df = pd.DataFrame(transformed[0].T.todense(),index=tfidf.get_feature_names(), columns=["TF-IDF"])
  
  temp_df = temp_df.sort_values('TF-IDF', ascending=False)
  
  temp_df = temp_df[temp_df['TF-IDF'] > 0].reset_index()
  temp_df = temp_df.rename(columns= {'index':'text'})
  
  if category ==  'Chinese':
    cur_list =new_Chinese_food_list
  elif category == 'Japanese':
    cur_list =new_Japanese_food_list
  elif category == 'Korean':
    cur_list =new_Korean_food_list
  else:
    cur_list = new_Chinese_food_list+new_Japanese_food_list+new_Korean_food_list
  
  
  index_list = []
  for i in range(len(temp_df)):
      word = temp_df.loc[i,'text']
      cur_word = word.split(' ')
      if not any(elem in elem in cur_list for elem in cur_word):
          index_list.append(i)
  
  temp_df.drop(index_list,axis = 0, inplace=True)
  
  
  temp_df = temp_df.head(top_n)
  temp_df.plot(x = 'text', y = 'TF-IDF', kind = 'bar')
  plt.xticks(rotation = rotation,fontsize = font)
  plt.ylabel('TF-IDF Score')
  plt.show()




