from TokenizeSentiment import  *
from ExceptionWords import *
from Performance import *

url_pos = 'positive-words.txt'
url_neg = 'negative-words.txt'
# url_pos = r'https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/positive-words.txt'
# 
# url_neg = r'https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/negative-words.txt'

# pos_list = request.urlopen(url_pos).read().decode('utf-8')[1:]
pos_list = open(url_pos).read()
pos_list = pos_list[pos_list.find("a+"):].split("\n")

# neg_list = request.urlopen(url_neg).read().decode('ISO-8859-1')[1:]
neg_list = open(url_neg).read()
neg_list = neg_list[neg_list.find("2-faced"):].split("\n")

test = open('data/annotated_test_set.txt','r', encoding='utf8')
test_file = test.read()
test.close()
test_file[:200]

test_file = re.sub(r"{[^{}]+}", lambda x: x.group(0).replace(",", ";"), test_file)
test_file = test_file.replace(';', "%")
test_file = test_file.replace(',', ";")
test_file = test_file.replace('%', ",")
test_file = test_file.replace('{', "{'")
test_file = test_file.replace(',', ",'")
test_file = test_file.replace(':', "':")
test_file = test_file.replace("},'", "}")

# Once fixed, save and load:
text_file = open("data/annotated_test_set_corrected.csv", "w")
for row in test_file.split(",\n"):
    text_file.write(row)
    text_file.write("\n")
text_file.close()

#File opening
test = open('data/annotated_test_set_corrected.csv','r', encoding='utf8')
test_file = test.read()
test.close()

test = pd.read_csv('data/annotated_test_set_corrected.csv', delimiter = ";")
test.columns = ['review_id', 'Product', 'Sentiments_test']

df = pd.read_csv('data/Amazon_Unlocked_Mobile.csv', delimiter = ",")
n = len(df)
#Key column of the csv file
df.columns = ['Product', 'Brand', 'Price', 'Rating', 'Review', 'Votes']
df['id_col'] = range(0, n)

n_reviews = 1000 # Let's get a sample
keep = sorted(random.sample(range(1,n),n_reviews))
keep += list(set(test.review_id)) # this are the reviews annotated for test

df = df[df.id_col.isin(keep)]
n_reviews = len(df)
df['id_new_col'] = range(0, n_reviews)

df.head() # dataframe head



#tic=timeit.default_timer()

tokenized_reviews = get_tokens(df, stem = False, negation = False)
tokenized_pos = get_pos(tokenized_reviews)
tdm = get_tdm(tokenized_reviews)
vsm = normalize_tdm(tdm)
tf_idf = get_tf_idf_transform(tokenized_reviews, tdm, n_reviews)

#toc=timeit.default_timer()

#print("minutes: ", (toc - tic)/60)

def get_product_tokens(df):
    stop = set(stopwords.words('english'))
    products = []
    i = 1
    
    for product in df["Product"]:
        tokenized_product = []      

        product = product.lower() # lowercase
        
        product = re.sub(r'[^0-9A-Za-z \.]','',product)    
    
        tokens = word_tokenize(product)[:11]
        
        for token in tokens:
            # Remove stop words
            if token not in stop:
                tokenized_product.append(token)       
            
        products.append(tokenized_product)
        
        if i%100 == 0:
            print('progress: ', (i/len(df["Product"]))*100, "%")
        i = i + 1
        
    return products


def standardize_names(products_idf, colors, common_terms):
    standard_names = []
    brands = [str(x).lower() for x in set(df.Brand)]
    
    for product in products_idf:
        
        for k, v in product.items():
            # Remove color and brand words
            if k in colors or k in common_terms or k in brands:
                product[k] = 0
        
        # Grab the first 5 words with highest score
        product = sorted(product.items(), key=operator.itemgetter(1), reverse = True)[:5]
        
        standard_names.append(product)
        
        tokenized_standard_product_names = []
        
    for product in standard_names:
        product_name = []
        for word in product:
            if word[1] > 0:
                product_name.append(word[0])

        tokenized_standard_product_names.append(product_name)
    
    
        
    return tokenized_standard_product_names


def get_wordnet_pos(pos):   
    for tag in [('J','ADJ'),('V','VERB'),('N','NOUN'),('R','ADV')]:
        if pos.startswith(tag[0]):
            return getattr(wordnet,tag[1])
    else:
        return 'None'

def get_adj(review):
    with_adj = [tup for tup in review if tup[1] == 'JJ']
    return with_adj

# score for each word
def senti(synset):
    s = swn.senti_synset(synset).pos_score() - swn.senti_synset(synset).neg_score()
    if s>=0:
        return 1
    else:
        return -1
    
    
def slist(tokenized_pos):
    score = []
    for k in [i for i in to_prune if i!=0]:
        r = get_adj(tokenized_pos[k-1])
        tag = [get_wordnet_pos(tuple[1]) for tuple in r]
        synsets = [r[i][0] + '.' + tag[i] + '.01' for i in range(len(r))] 
        score.append([senti(i) for i in synsets])
    return score

def balance(score_list):
    m=-1
    for k in [i for i in to_prune if i!=0]:
        m+=1
        s = score_list[m]
        if 1 in s and -1 in s and max([s.count(1),s.count(-1)])/min([s.count(1),s.count(-1)]) <= 3:
            to_prune[k-1] = 0
    return to_prune


def average_score(score_list):
    m = -1
    for k in [i for i in to_prune if i!=0]:
        m += 1
        s = score_list[m]
        if sum(s)>=0 and (sum(s)+1)*(ratings[k-1]-2.5)<=0:
            to_prune[k-1] = 0
        elif sum(s)<0 and (sum(s)+1)*(2.5-ratings[k-1])<=0:
            to_prune[k-1] = 0
    return to_prune

# lookup_review = 1
# for val in df[df.id_new_col == lookup_review]["Review"]: 
#     print(val)
#     out = val.to_json()[:200].replace('}, {', '} {')
#     with open("NewFile.txt", "w") as f:
#         f.write(out)

# display(tokenized_reviews[lookup_review]) # tokenized reviews
# display(tokenized_pos[lookup_review]) #part of speech
# display(tdm[lookup_review]) # term document matrix(TDM)
# display(tf_idf[lookup_review]) #TD + IDF transformation



tokenized_products = get_product_tokens(df)
products_tokenized_pos = get_pos(tokenized_products)
products_tdm = get_tdm(tokenized_products)
products_tf_idf = get_tf_idf_transform(tokenized_products, products_tdm, n_reviews)
products_idf = get_idf_transform(tokenized_products, products_tdm, n_reviews)


lookup_product = 53
#visualization for analysis below
#display(df[df.id_new_col== lookup_product]["Product"])

#display(products_tokenized_pos[lookup_product])

colors = ["black", "red", "blue", "white", "gray", "green","yellow", "pink", "gold"]
common_terms = ["smarthphone", "phone", "cellphone", "retail", "warranty", 
                "silver", "bluetooth", "wifi", "wireless", "keyboard", "gps",
               "original", "unlocked", "camera", "certified", "international",
               "actory", "packaging", "us", "usa", "international", "refurbished", 
               "phones", "att", "verizon", "-", "8gb", "16gb", "32gb", "64gb", "contract"]



standard_product_names = standardize_names(products_idf, colors, common_terms)

product_tdm = get_tdm(standard_product_names)
product_vsm = normalize_tdm(product_tdm)
product_vsm[1]

similarity = product_tdm
product_names_clusters = int(round(n_reviews/2,0))

similarity_matrix = pd.DataFrame(get_similarity_matrix(similarity, standard_product_names), columns = get_all_terms(standard_product_names))

kmeans = KMeans(n_clusters=product_names_clusters, random_state=0).fit(similarity_matrix)
clusters=kmeans.labels_.tolist()

clustered_matrix = similarity_matrix.copy()
clustered_matrix['product_name_cluster'] = clusters
clustered_matrix['id_col'] = range(0, n_reviews)

# display(clustered_matrix[:5])

count_clusters = pd.DataFrame(clustered_matrix.product_name_cluster.value_counts())
# display(count_clusters[:5])

df["cluster_name"] = list(clustered_matrix.product_name_cluster)


#Clustering product name standardization
#3 ways we have
#1) manually inputted set / gazetteer with words to be removed
#2) IDF importance score and
#3) Clustering 
def create_standard_name(df):
    new_names = defaultdict(int)
    
    current_names = df.groupby('cluster_name').first().Product
    
    
    for i in set(clusters):
        cluster_name = df[df.cluster_name == i].Product.value_counts().index[0]
        new_name = []
        
        for word in cluster_name.split():
            temp_word= re.sub(r'[^0-9A-Za-z \.\-]','',word).lower()
            if temp_word not in colors and temp_word not in common_terms :
                new_name.append(word)
        new_names[i] = ' '.join(new_name)
    
    new_standard_names = []
    

    for row in df.cluster_name:
        
        new_standard_names.append(new_names[row])
    
    df["Standard_Product_Name"] = new_standard_names
    
    return df

df = create_standard_name(df)         
        
df.head()

df[["Product","Standard_Product_Name"]][df['Product'].str.contains("iPhone")][:8]

def get_all_terms_pos_dft(all_terms, terms_dft):
    all_terms_pos = nltk.pos_tag(all_terms)
    
    i = 0
    for k, v in terms_dft.items():
        all_terms_pos[i] = all_terms_pos[i] + (v,)        
        i+=1
        
    return all_terms_pos

def get_threshold_terms(all_terms_pos_dft, threshold = 20):
    threshold_terms = []
    
    for term in all_terms_pos_dft:
        if term[0] in exceptions_to_consider or (term[2] >= threshold and term[1] in ["NN", "NNS", "NNP", "NNPS"] and term[0] not in exceptions_not_to_consider):
            threshold_terms.append(term)
    
    return threshold_terms
            
all_terms = get_all_terms(tokenized_reviews)    
terms_dft = get_all_terms_dft(tokenized_reviews, all_terms)
all_terms_pos_dft = get_all_terms_pos_dft(all_terms, terms_dft)
threshold_terms = get_threshold_terms(all_terms_pos_dft, threshold = 0.01 * n_reviews)

threshold_terms[:10]

characteristics = [x[0] for x in threshold_terms]

characteristics[:10]

#take the range of reviews
to_prune = [i+1 for i in range(n_reviews)]
ratings = list(df['Rating'])


adjs = {x.name().split('.', 1)[0] for x in wn.all_synsets('a')}

def prune_adj(tokenized_pos):    
    for k in [i for i in to_prune if i!=0]:
        if not len(get_adj(tokenized_pos[k-1])) or not all(i[0] in adjs for i in get_adj(tokenized_pos[k-1])):
                to_prune[k-1] = 0
    return to_prune



to_prune = [i+1 for i in range(n_reviews)]

to_prune = prune_adj(tokenized_pos)
score_list = slist(tokenized_pos)
to_prune = balance(score_list)
to_prune = average_score(score_list)

to_keep = [i for i in to_prune if i!=0]
to_keep += list(df[df.id_col.isin(list(set(test.review_id)))].id_new_col) # this are the reviews annotated for test
to_keep = list(set(to_keep))

df_filtered = df[df.id_new_col.isin(to_keep)]
df_filtered[:3]

len(list(df_filtered[df_filtered.id_col.isin(list(set(test.review_id)))].id_new_col))

to_keep = list(df_filtered.id_new_col)

n_reviews = len(to_keep)

tokenized_reviews = get_tokens(df_filtered, stem = False, negation = False)
tokenized_pos = get_pos(tokenized_reviews)
tdm = get_tdm(tokenized_reviews)
vsm = normalize_tdm(tdm)
tf_idf = get_tf_idf_transform(tokenized_reviews, tdm, n_reviews)


similarity = vsm

similarity_matrix = pd.DataFrame(get_similarity_matrix(similarity, tokenized_reviews), columns = get_all_terms(tokenized_reviews))

similarity_matrix[:10]

kmeans = KMeans(n_clusters=int(round(math.sqrt(n_reviews),0)), random_state=0).fit(similarity_matrix)
clusters=kmeans.labels_.tolist()

clustered_matrix = similarity_matrix.copy()
clustered_matrix['cluster'] = clusters
clustered_matrix['id_col'] = to_keep

top_clusters = pd.DataFrame(clustered_matrix.cluster.value_counts())

limit = top_clusters.cluster.quantile(0.3)
cluster_filter = top_clusters[top_clusters.cluster > limit]

# display(cluster_filter)

list(cluster_filter.index)

df_filtered["cluster"] = list(clustered_matrix.cluster)
df_filtered[:3]

to_keep = list(df_filtered[df_filtered.cluster.isin(list(cluster_filter.index))].id_new_col)
to_keep += list(df[df.id_col.isin(list(set(test.review_id)))].id_new_col) # this are the reviews annotated for test
to_keep = list(set(to_keep))

df_filtered = df_filtered[df_filtered.id_new_col.isin(to_keep)]
df_filtered[:3]

def filter_with_characteristics(df_filtered, characteristics):
    tokenized_reviews = get_tokens(df_filtered, stem = False, negation = False)
    to_keep_in = []
    j = 0
    
    for i in df_filtered.id_col: 
        for token in tokenized_reviews[j]:
            if token in characteristics:
                to_keep_in.append(i)
                break
        
        j+=1
        
    return to_keep_in
                
to_keep_in = filter_with_characteristics(df_filtered, characteristics)
len(to_keep_in)

ignore_exceptions += colors


def compute_score(word, word_neg):
    if word in ignore_exceptions: 
        return 0
    
    if word in positive_exceptions:
        if word_neg.find("_NEG") == -1:
            return 1
        else:
            return -1
        
    if word in negative_exceptions:
        print(word)
        if word_neg.find("_NEG") == -1:
            return -1
        else:
            return 1
        
    word2 = ''.join([word,".a.01"])
    try:
        pos_score = swn.senti_synset(word2).pos_score()
        neg_score = swn.senti_synset(word2).neg_score()
    except:
        if word in pos_list:
            pos_score = 1
            neg_score = 0
        elif word in neg_list:
            pos_score = 0
            neg_score = 1
        else:
            return 0
    
    if pos_score > neg_score:
        if word_neg.find("_NEG") == -1:
            return 1
        else:
            
            return -1
    elif neg_score > pos_score:
        if word_neg.find("_NEG") == -1:            
            return -1
        else:
            
            return 1   
    else:
        if word in pos_list:
            return 1
        elif word in neg_list:
            return -1
        else:
            return 0

    
def extract_characteristic_opinion_words(review, review_neg, max_opinion_words = 2, max_distance = 5, use_distance = False):
    review_charactetistics_sentiment = defaultdict(list) 
    i = 0
    
    temp_review = []
    for word in review: 
        word = word + ("free",)
        temp_review.append(list(word))
            
    for i in range(len(review)):
        if review[i][0] in characteristics:
            keep_forward = True
            keep_backward = True
            opinion_words = 0
            
            for j in range(1,max_distance+1):
                
                if  i+j >= len(review):
                    keep_forward = False
                if keep_forward:
                    if  review[i+j][0] in characteristics or opinion_words >= max_opinion_words:
                        keep_forward = False

                    elif i+j < len(review) and (review[i+j][1] in ["JJ", "JJR", "JJS"] or review[i+j][0] in word_exceptions) and temp_review[i+j][2] == "free":
                        sentiment = defaultdict(int)
                        score = compute_score(review[i+j][0], review_neg[i+j][0])                   
                        if score == 0: continue

                        if use_distance:
                            distance = j
                        else:
                            distance = 1

                        sentiment[review[i+j][0]] = (score,distance)
                        review_charactetistics_sentiment[review[i][0]].append(sentiment)
                        temp_review[i+j][2] = "used"
                        opinion_words +=1
                
                
                if  i-j < 0:
                    keep_backward = False
                if keep_backward:
                    if  review[i-j][0] in characteristics or opinion_words >= max_opinion_words:
                        keep_backward = False

                    elif i-j > -1 and (review[i-j][1] in ["JJ", "JJR", "JJS"] or review[i-j][0] in word_exceptions) and temp_review[i-j][2] == "free":
                        sentiment = defaultdict(int)
                        score = compute_score(review[i-j][0], review_neg[i-j][0])         

                        if score == 0: continue

                        if use_distance:
                            distance = j
                        else:
                            distance = 1

                        sentiment[review[i-j][0]] = (score,distance)

                        review_charactetistics_sentiment[review[i][0]].append(sentiment)
                        temp_review[i-j][2] = "used"  
                        opinion_words +=1
                
                if not keep_forward and not keep_backward:
                    break
    
    return review_charactetistics_sentiment


def consolidate_score(characteristic_dict):
    num = 0
    den = 0
    
    for opinion in characteristic_dict:
        for k, v in opinion.items():
            num += v[0]/v[1]
            den += 1/v[1]

    return num/den


def compute_sentiment_scores(tokenized_pos, tokenized_pos_neg, max_distance = 5, use_distance = True):
    
    if len(tokenized_pos) != len(tokenized_pos_neg):
        return None
    
    else:
        
        reviews_sentiment_scores = []        
        
        for i in range(len(tokenized_pos)):
            review_sentiment_score = defaultdict(int)
            
            review_characteristics_opinion_words = extract_characteristic_opinion_words(tokenized_pos[i], tokenized_pos_neg[i], max_distance = max_distance, use_distance = use_distance)
            
            for k, v in review_characteristics_opinion_words.items():
                review_sentiment_score[k] = consolidate_score(v)
                
            reviews_sentiment_scores.append(review_sentiment_score)
            
        return reviews_sentiment_scores

    
def get_NN_count(tokenized_pos):
    NN_count = []
    
    for review in tokenized_pos:
        review_NN_count = 0
        for token in review: 
            if token[1] in ["NN", "NNS", "NNP"] or token[0] in characteristics:
                review_NN_count += 1
        NN_count.append(review_NN_count)
    
    return NN_count


tokenized_reviews = get_tokens(df_filtered, stem = False, negation = False)
tokenized_pos = get_pos(tokenized_reviews)

tokenized_reviews_neg = get_tokens(df_filtered, stem = False, negation = True)
tokenized_pos_neg = get_pos(tokenized_reviews_neg)

NN_count = get_NN_count(tokenized_pos)

df_filtered['new_id'] = range(0, len(df_filtered))

lookup_product_id = 7

review_characteristics_opinion_words = extract_characteristic_opinion_words(tokenized_pos[lookup_product_id], tokenized_pos_neg[lookup_product_id], max_distance = 5, use_distance = True)               

review_sentiment_scores = compute_sentiment_scores(tokenized_pos, tokenized_pos_neg, max_distance = 5, use_distance = True)
review_sentiment_scores[:6]

df_filtered["Sentiments"] = list(review_sentiment_scores)
df_filtered["NN_count"] = list(NN_count)
df_filtered[:3]

test = open('data/annotated_test_set.txt','r', encoding='utf8')
test_file = test.read()
test.close()
test_file[:200]

test_file = re.sub(r"{[^{}]+}", lambda x: x.group(0).replace(",", ";"), test_file)
test_file = test_file.replace(';', "%")
test_file = test_file.replace(',', ";")
test_file = test_file.replace('%', ",")
test_file = test_file.replace('{', "{'")
test_file = test_file.replace(',', ",'")
test_file = test_file.replace(':', "':")
test_file = test_file.replace("},'", "}")

# Once fixed, save and load:
text_file = open("data/annotated_test_set_corrected.csv", "w")
for row in test_file.split(",\n"):
    text_file.write(row)
    text_file.write("\n")
text_file.close()


test = open('data/annotated_test_set_corrected.csv','r', encoding='utf8')
test_file = test.read()
test.close()

test = pd.read_csv('data/annotated_test_set_corrected.csv', delimiter = ";")
test.columns = ['review_id', 'Product', 'Sentiments_test']


df_merge = pd.merge(df_filtered, test, left_on='id_col', right_on='review_id', how = "left")
print(df_merge[df_merge.Sentiments_test.isnull()==False])



with open('products.txt', 'w') as outfile:
    df_merge['Product_y'].to_json(outfile)

with open('sentiment.txt', 'w') as outfile:
    df_merge['Sentiments'].to_json(outfile)

with open('output.txt', 'w') as outfile:
    df_merge.to_json(outfile)

lookup = 1000
    
for val in df_merge[df_merge.id_col == lookup].Review:
    print(val)
    
df_merge[df_merge.id_col == lookup]
