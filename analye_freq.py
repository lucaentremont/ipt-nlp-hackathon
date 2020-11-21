import pandas as pd

df = pd.read_csv('clean_freqcheck.csv')
l = []
df['ServiceProcessed'] = df['ServiceProcessed'].apply(lambda x: l.append(x))
print(set(l))
'''
words = []

df['Combined_Content'] = df['Combined_Content'].apply(lambda x: [words.append(a) for a in str(x).split(' ')])

word_counter = {}
for word in words:
     if word in word_counter:
         word_counter[word] += 1
     else:
         word_counter[word] = 1 
popular_words = sorted(word_counter, key = word_counter.get, reverse = True)

top_100 = popular_words[:100]
 
print(top_100)
'''