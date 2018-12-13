#!/usr/bin/env python
# coding: utf-8

'''
Model to train for differentiating restrictive vs. non-restrictive

@author: Mustafa Bal
'''

import sys, time, string
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from classifier import Classifier
from coling_baseline import ColingBaselineClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from operator import itemgetter
from props.dependency_tree.tree_readers import create_dep_trees_from_stream
import SpaCyParserWrapper
import pycrfsuite

import MyClassifiers

def get_input(input_file):
    result_list = [t for t in create_dep_trees_from_stream(open(input_file),False)]
    return result_list

def extractCorpusData(trainFile):
    pass

def parseCorpusFile(corpusFile):
    ret = []
    for line in open(corpusFile):
        data = line.strip().split('\t')
        if data != ['']:
            (sentInd, nodeIndex) = map(int, data[8:10])
            val = data[1]
            candidateType = data[11]
            tags = data[13:]
            ret.append((sentInd, nodeIndex, val, candidateType, tags))
    return ret

def trainModel(data, model_file):
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params({
        'c1': 3.0,   # coefficient for L1 penalty
        'c2': 1e-20,  # coefficient for L2 penalty
#             'max_iterations': 50,  # stop earlier
    
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
            
    x_train = map(itemgetter(0), data)
    y_train = map(itemgetter(1), data)
    trainer.append(x_train, y_train)
    trainer.train(model_file)

    return trainer

def train(training_set):
    data = []
    with open(features_file, 'w') as fout:
        for object in training_set:
            for key,val in object.items():
                #val are DepTree objects
    return ""

def classify(testFile, outputFile):
    pass

def bio_classification_report(y_true, y_pred):
    '''
    Classification report for a list of BIO-encoded sequences.    
    Note that it requires scikit-learn 0.15+ (or a version from github master) to calculate averages properly!
    '''

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    
    tagset = set(lb.classes_) - {'_'} - {'POSTADJ-MOD'}

    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def main(training_file, testing_file, model_file, ft):
    
    start = time.time()
    
    # Get training and testing set of data
    training_set = get_input(training_file)
    testing_set = get_input(testing_file)

    train(training_set)

    classify()
    
    end = time.time()
    print('CRF model has been generated.')
    print('runtime:', end - start)

training_file = "corpus/train.txt"
testing_file = "corpus/test.txt"
model_file = "model.crfsuite"
features_file = "features/myFeatures.txt"

main(training_file, testing_file, model_file, 5)

