from  Include import *

def compute_characteristics_extraction_performance(df_merge, pd):
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    
    for i in range(len(df_merge)):
        NN_count = df_merge.NN_count[i]
        training = df_merge.Sentiments[i]
        test = df_merge.Sentiments_test[i]
        if pd.isnull(test): continue
        TP, TN, FP, FN = characteristics_extraction_performance(NN_count, training, test)
        
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN
        
    if total_TP + total_FP == 0:
        TPR_RECALL = 0
    else:
        TPR_RECALL =  total_TP / (total_TP + total_FP)
        
    TNR_SPECIFICITY = total_TN / (total_TN + total_FN)
    F1_Score = 2* total_TP / (2*total_TP + total_FP + total_FN)
    Accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    fpr = total_FP / (total_FN + total_FP)
    
    return TPR_RECALL, TNR_SPECIFICITY, F1_Score, Accuracy, fpr

def characteristics_extraction_performance(NN_count, training, test):
    TP = 0 # True Positive
    TN = 0 # False Negative
    FP = 0 # False Positive
    FN = 0 # False Negative
    temp_test = []
    test = eval(test)
    
    for test_characteristic in test.keys():
        test_characteristic = str(test_characteristic).lower()
        test_characteristic = re.sub(r'[^A-Za-z /.]','',test_characteristic)
        temp_test.append(test_characteristic)
            
        if test_characteristic in training.keys():
            TP += 1
        else: 
            FN += 1
            
    TN = NN_count - len(training.keys()) - FN
        
    for train_characteristic in training.keys():
        if train_characteristic not in temp_test:
            FP += 1
    
    return TP, TN, FP, FN

def characteristics_sentiment_performance(training, test):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    test = eval(test)
    
    for test_characteristic, test_score in test.items():
        test_characteristic = str(test_characteristic).lower()
        test_characteristic = re.sub(r'[^A-Za-z /.]','',test_characteristic)
                        
        if test_characteristic in training.keys():
            if test_score == training[test_characteristic]:
                if test_score > 0:
                    TP += 1
                else:
                    TN += 1
            else:
                if test_score > 0:
                    FN += 1
                else:
                    FP += 1            
        else: 
            continue    

    return TP, TN, FP, FN


def compute_characteristics_sentiment_performance(df_merge, pd):
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    cases = 0
    
    for i in range(len(df_merge)):
        training = df_merge.Sentiments[i]
        test = df_merge.Sentiments_test[i]
        if pd.isnull(test): continue
        TP, TN, FP, FN = characteristics_sentiment_performance(training, test)
        if TP+ TN+ FP+ FN > 0:
            cases+=1            
        
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN
        
    if total_TP + total_FP == 0:
        TPR_RECALL = 0
    else:
        TPR_RECALL =  total_TP / (total_TP + total_FP)
        
    TNR_SPECIFICITY = total_TN / (total_TN + total_FN)
    F1_Score = 2* total_TP / (2*total_TP + total_FP + total_FN)
    Accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    try:
        fpr = total_FP / (total_FN + total_FP)
    except ZeroDivisionError:
        fpr = float('Inf')
    
    return TPR_RECALL, TNR_SPECIFICITY, F1_Score, Accuracy, cases