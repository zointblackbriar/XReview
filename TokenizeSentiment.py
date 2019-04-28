from Include import *


#this could be used for sentence tokenization
def get_tokens(df, stem = False, negation = False):
    stemmer = PorterStemmer()
    stop = set(stopwords.words('english'))
    reviews = []    
    i = 1
    
    for review in df["Review"]:
        tokenized_review = []      

        review = str(review).lower() # lowercase
        
        # Remove every character except A-Z, a-z,space 
        # and punctuation (we'll need it for negation)
        review = re.sub(r'[^A-Za-z /.]','',review) 
        
        # mark_negation needs punctuation separated by white space.
        review = review.replace(".", " .")   
        
        tokens = word_tokenize(review)
        
        
        for token in tokens:
            # Remove single characters and stop words
            if (len(token)>1 or token == ".") and token not in stop: 
                if stem:
                    tokenized_review.append(stemmer.stem(get_synonym(token)))            
                else:
                    tokenized_review.append(get_synonym(token))
        
        if negation:
            tokenized_review = sentiment.util.mark_negation(tokenized_review)   
        
        # Now we can get rid of punctuation and also let's fix some spellings:
        tokenized_review = [correction(x) for x in tokenized_review if x != "." ]
        
            
        reviews.append(tokenized_review)
        
        if i%100 == 0:
            print('progress: ', (i/len(df["Review"]))*100, "%")
        i = i + 1
        
    return reviews
 

def get_pos(tokenized_reviews):
    tokenized_pos = []
    
    for review in tokenized_reviews:
        tokenized_pos.append(nltk.pos_tag(review))
    
    return tokenized_pos
        
    
def get_frequency(tokens):    
    term_freqs = defaultdict(int)    
    
    for token in tokens:
        term_freqs[token] += 1 
            
    return term_freqs

#tdm is tokenized word frequency for scoring
def get_tdm(tokenized_reviews):
    tdm = []
    
    for tokens in tokenized_reviews:
        tdm.append(get_frequency(tokens))
    
    return tdm

def normalize_tdm(tdm):    
    tdm_normalized = []
        
    for review in tdm:
        den = 0
        review_normalized = defaultdict(int)
        
        for k,v in review.items():
            den += v**2
        den = math.sqrt(den)
    
        for k,v in review.items():
            review_normalized[k] = v/den
        
        tdm_normalized.append(review_normalized)
        
    return tdm_normalized

def get_all_terms(tokenized_reviews):
    all_terms = []
    
    for tokens in tokenized_reviews:
        for token in tokens:
            all_terms.append(token)
            
    return(set(all_terms))
    
def get_all_terms_dft(tokenized_reviews, all_terms):
    terms_dft = defaultdict(int)  
    
    for term in all_terms: 
        for review in tokenized_reviews:
            if term in review:
                terms_dft[term] += 1
                
    return terms_dft


def get_tf_idf_transform(tokenized_reviews, tdm, n_reviews):
    tf_idf = []        
    all_terms = get_all_terms(tokenized_reviews)    
    terms_dft = get_all_terms_dft(tokenized_reviews, all_terms)
    
    for review in tdm:
        review_tf_idf = defaultdict(int)
        for k,v in review.items():
            review_tf_idf[k] = v * math.log(n_reviews / terms_dft[k], 2)
        
        tf_idf.append(review_tf_idf)     
    
    return tf_idf


def get_idf_transform(tokenized_reviews, tdm, n_reviews):
    idf = []    
    terms_dft = defaultdict(int)    
    
    all_terms = get_all_terms(tokenized_reviews)
    
    for term in all_terms: 
        for review in tokenized_reviews:
            if term in review:
                terms_dft[term] += 1
    
    for review in tdm:
        review_idf = defaultdict(int)
        for k,v in review.items():
            review_idf[k] = math.log(n_reviews / terms_dft[k], 2)
        
        idf.append(review_idf)     
    
    return idf


def correction(x):
    ok_words = ["microsd"]
    
    if x.find("_NEG") == -1 and x not in ok_words: # Don't correct if they are negated words or exceptions
        return spell(x)
    else:
        return x

def get_synonym(word):
    synonyms = [["camera","video", "display"], 
                ["phone", "cellphone", "smartphone", "phones"],
               ["setting", "settings"],
               ["feature", "features"],
               ["pictures", "photos"],
               ["speakers", "speaker"]]
    synonyms_parent = ["camera", "phone", "settings", "features", "photos", "speakers"]
    
    for i in range(len(synonyms)):
        if word in synonyms[i]:
            return synonyms_parent[i]
    
    return word


def get_similarity_matrix(similarity, tokenized_reviews):
    similarity_matrix = []
    all_terms = get_all_terms(tokenized_reviews)
    
    for review in similarity:
        similarity_matrix_row = []
        for term in all_terms:
            similarity_matrix_row.append(review[term])
            
        similarity_matrix.append(similarity_matrix_row)
            
    return similarity_matrix