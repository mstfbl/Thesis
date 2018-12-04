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

# ----------3------------

def baseFeature(sentence, i, prefix):
    word = sentence[i][1]
    postag = sentence[i][4]
    # Set the features of the word
    features = [prefix + feat for feat in
        [
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag))
    ]]

    return features

def mainFeature(sentence, i):
    word = sentence[i][1]
    postag = sentence[i][4]
    features = baseFeature(sentence, i, '')
    if i > 0:
        features.extend(extractWordFeatures(sentence, i - 1 , '-1:'))
    else:
        features.append('BOS=true')
        
    if i < len(sentence)-1:
        features.extend(ColingBaselineClassifier.extractWordFeatures(sentence, i + 1 , '+1:'))
    else:
        features.append('EOS=true')                    
    return features
def set_features4(sentence, i):
    def featCommaBefore(sentence, nodeIndex):
        indBefore = min(tree[nodeIndex].get_subtree()) - 1
        ret = 'commaBefore='
        if indBefore < 1:
            return ret + '0'
        return ret + '1' if [tree[indBefore].word == ','] else ret + '0'
    def prefixFeats(prefix, feats):
        return [prefix + feat for feat in feats]

    #word = sentence[i][1].lower()
    candidateType = sentence[i][10]+sentence[i][11]
    baseFeatures = mainFeature(sentence, i)
    parentInd = sentence[i][8] #maybe not 8, but 9?

    parentFeatures = prefixFeats('parent:',mainFeature(sentence, parentInd))

    #Type and (not-working)Comma Features
    newFeatres = ['type=' + candidateType]#, featCommaBefore(sentence, i),]

    #POBJ
    for i,child in enumerate([child for child in sentence[i].children if child.parent_relation == 'PMOD']):
            newFeatres.extend(prefixFeats('pobj{0}:'.format(i), ColingBaselineClassifier._extractFeatures(tree, child.id, candidateType)))

    allFeatures =  baseFeatures + parentFeatures + newFeatres

    allFeatures += ['{0}|{1}={2}|{3}'.format('type', feat.split('=')[0], candidateType, feat.split('=')[1])
                                                 for feat in allFeatures]

    print(allFeatures)
    return allFeatures

#-----------5---------

def featCommaBefore(tree, nodeIndex):
    indBefore = min(tree[nodeIndex].get_subtree()) - 1
    ret = 'commaBefore='
    if indBefore < 1:
        return ret + '0'
    return ret + '1' if [tree[indBefore].word == ','] else ret + '0'
    
def prefixFeats(prefix, feats):
    return [prefix + feat for feat in feats] 

def extractWordFeatures(tree, nodeIndex, featuresPrefix):
    word = tree[nodeIndex].word
    postag = tree[nodeIndex].pos
    return [featuresPrefix + feat for feat in
            [
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'lemma=' + ColingBaselineClassifier.lmtzr.lemmatize(word, convertPtbToLemmatizerPos(postag))
    ]]

def extractLMRFeatures(tree, nodeIndex, candidateType):
    print(nodeIndex,candidateType)
    word = tree[nodeIndex].word
    postag = tree[nodeIndex].pos
    features = extractWordFeatures(tree, nodeIndex, '')
    if nodeIndex > 0:
        features.extend(extractWordFeatures(tree, nodeIndex - 1 , '-1:'))
    else:
        features.append('BOS=true')
        
    if nodeIndex < len(tree)-1:
        features.extend(extractWordFeatures(tree, nodeIndex + 1 , '+1:'))
    else:
        features.append('EOS=true')                    
    return features

def extractExtendedFeatures(tree, nodeIndex, candidateType):
    word = tree[nodeIndex].word.lower()
    baseFeats = extractLMRFeatures(tree, nodeIndex, candidateType)
    parentInd = tree[nodeIndex].parent_id
    sent = ' '.join([tree[tok].word for tok in sorted(tree)[1:]])
    # parent feats
    parentFeats = prefixFeats('parent:', extractLMRFeatures( tree, parentInd, candidateType))
    # type and comma
    newFeats = ['type=' + candidateType, featCommaBefore(tree, nodeIndex)]
    # pobj
    for i,child in enumerate([child for child in tree[nodeIndex].children if child.parent_relation == 'PMOD']):
        newFeats.extend(prefixFeats('pobj{0}:'.format(i), extractLMRFeatures(tree, child.id, candidateType)))
    feats =  baseFeats + parentFeats + newFeats
    feats += ['{0}|{1}={2}|{3}'.format('type', feat.split('=')[0], candidateType, feat.split('=')[1]) for feat in feats]
    return feats

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
    elif type == 3:
        return [mainFeature(sent, i) for i in range(len(sent))]
    elif type == 4:
        return [set_features4(sent, i) for i in range(len(sent))]
    elif type == 5:
        return [extractExtendedFeatures(sent, i) for i in range(len(sent))]
    else:
        raise Exception

def get_labels(sent):
    return [x[-1] for x in sent]

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