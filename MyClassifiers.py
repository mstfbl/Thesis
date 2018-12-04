from coling_baseline import ColingBaselineClassifier
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

def set_features1(sentence, i):
    word = sentence[i][1]
    # Set the features of the word
    features = [
        'word.lower=' + word.lower(),
        'word.length=' + str(len(word)),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.isdot=%s' % isdot(word),    
        'word.isdash=%s' % isdash(word),          
        'word.iscomma=%s' % iscomma(word),
        'postag=' + sentence[i][0],
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(sentence[i][4]))
    ]
    if i > 0:
        # Set the features of relationship with previous word.
        word1 = sentence[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.length=' + str(len(word1)),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isdigit=%s' % word1.isdigit(), 
            '-1:word.isdot=%s' % isdot(word1),    
            '-1:word.isdash=%s' % isdash(word1),  
            '-1:word.iscomma=%s' % iscomma(word1),  
            'postag=' + sentence[i-1][0],
            'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(sentence[i-1][4]))       
        ])
    else:
        features.append('Begin_Of_Sentence')
        
    if i < len(sentence)-1:
        # Set the features of relationship with next word.
        word1 = sentence[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.length=' + str(len(word1)),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:word.isdot=%s' % isdot(word1),    
            '+1:word.isdash=%s' % isdash(word1),  
            '+1:word.iscomma=%s' % iscomma(word1),
            'postag=' + sentence[i+1][0],
            'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(sentence[i+1][4])) 
        ])
    else:
        features.append('End_Of_Sentence')
                
    return features

# ----------2------------

def set_features2(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    modifierType = sentence[i][-1]
    # Set the features of the word
    features = [
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag)),
        #'modifierType=' + modifierType
    ]
    if i > 0:
        # Set the features of relationship with previous word.
        wordBefore = sentence[i-1][1]
        postagBefore = sentence[i-1][4]
        #modifierTypeBefore = sentence[i-1][-1]
        features.extend([
            '-1:word.lower=' + wordBefore.lower(),
            '-1:word.isupper=%s' % wordBefore.isupper(),
            '-1:word.istitle=%s' % wordBefore.istitle(),
            '-1postag=' + postagBefore,
            '-1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagBefore)),
            #'-1modifierType=' + modifierTypeBefore    
        ])
    else:
        features.append('B_o_S') #Beginning of Sentence
        
    if i < len(sentence)-1:
        # Set the features of relationship with next word.
        wordAfter = sentence[i+1][1]
        postagAfter = sentence[i+1][4]
        #modifierTypeAfter = sentence[i+1][-1]
        features.extend([
            '+1:word.lower=' + wordAfter.lower(),
            '+1:word.isupper=%s' % wordAfter.isupper(),
            '+1:word.istitle=%s' % wordAfter.istitle(),
            '+1postag=' + postagAfter,
            '+1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagAfter)),
            #'+1modifierType=' + modifierTypeAfter
        ])
    else:
        features.append('E_o_S') #End of Sentence
                
    return features

#-----------Honnibal---------

def set_features_honnibal(sentence, i):
    #cB = commaBefore
    word = sentence[i][1]
    postag = sentence[i][4]
    cB = "0"
    if i > 1:
        for x in range(i,1,-1):
            if sentence[x][1] == ',':
                cB = "1"
                break
    features = [
        'cB_postag=' + cB+postag
    ]
    return features

#-----------Dornescu---------

def set_features_dornescu(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    # Set the features of the word
    features = [
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag)),
    ]
    if i > 0:
        # Set the features of relationship with previous word.
        wordBefore = sentence[i-1][1]
        postagBefore = sentence[i-1][4]
        features.extend([
            '-1:word.lower=' + wordBefore.lower(),
            '-1:word.isupper=%s' % wordBefore.isupper(),
            '-1:word.istitle=%s' % wordBefore.istitle(),
            '-1postag=' + postagBefore,
            '-1postag[:2]=' + postagBefore[:2],
            '-1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagBefore)),
        ])
    else:
        features.append('B_o_S') #Beginning of Sentence
        
    if i < len(sentence)-1:
        # Set the features of relationship with next word.
        wordAfter = sentence[i+1][1]
        postagAfter = sentence[i+1][4]
        features.extend([
            '+1:word.lower=' + wordAfter.lower(),
            '+1:word.isupper=%s' % wordAfter.isupper(),
            '+1:word.istitle=%s' % wordAfter.istitle(),
            '+1postag=' + postagAfter,
            '+1postag[:2]=' + postagAfter[:2],
            '+1lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postagAfter)),
        ])
    else:
        features.append('E_o_S') #End of Sentence
                
    return features

def isdot(word):
    return True if word in '.' else False

def isdash(word):
    return True if word in '-' else False

def iscomma(word):
    return True if word in ',' else False

def ispunctuation(word): 
    return True if word in string.punctuation else False

def get_features(sent, type):
    if type == 1:
        return [set_features1(sent, i) for i in range(len(sent))]
    elif type == 2:
        return [set_features2(sent, i) for i in range(len(sent))]
    elif type == "dornescu":
        return [set_features_dornescu(sent, i) for i in range(len(sent))]
    elif type == "honnibal":
        return [set_features_honnibal(sent, i) for i in range(len(sent))]
    else:
        raise Exception

def get_labels(sent):
    return [x[-2] for x in sent]

def get_tokens(sent):
    return [x[-2] for x in sent]   

def convertPtbToLemmatizerPos(ptbPos):
    if ptbPos.startswith('VB'):
        return VERB
    if ptbPos.startswith('JJ'):
        return ADJ
    if ptbPos.startswith('RB'):
        return ADV
    return NOUN