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
from props.dependency_tree.tree_readers import create_dep_trees_from_stream
import SpaCyParserWrapper
import pycrfsuite

import MyClassifiers

def get_input (input_file):
    '''
    get input file by utf8 encoding. Read the file content then return it.
    '''
    sentence = []
    result_list = []

    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            #(0)posInSentence
            #(1)spelling1, (2)spelling2, (3)spelling3
            #(4)postype1, (5)postype2
            #(6)unk1, (7)unk2
            #(8)connected1, (9)connected2,  
            #(10)candidateType1, (11)candidateType2
            #(-2)RESTR/NON-RESTR, (-1)modifierType
    
            if data == ['']: # end of sentence
                result_list.append(sentence)
                sentence = []
            else:
                tup = (data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],int(data[8]),int(data[9]),data[-2],data[-1])
                sentence.append(tup)
        f.close()
    return result_list

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
def getTrainerFeatures(ft):
    #c1: coefficient for L1 penalty
    #c2: coefficient for L2 penalty
    #'feature.possible_transitions':
    #   include transitions that are possible, but not observed
    if ft == "dornescu" or ft == "honnibal":
        return {'c1': 3.0,'c2': 1e-20,'feature.possible_transitions': True}
    elif ft == 1 or ft == 2:
        return {'c1': 0.5,'c2': 1e-3, 'max_iterations': 1000, 'feature.possible_transitions': True}
    else:
        raise Exception
def main(training_file, testing_file, model_file, ft):
    
    start = time.time()
    
    # Get training and testing set of data
    training_set = get_input(training_file)
    testing_set = get_input(testing_file)
    
    # Get features of each word on training set
    X_train = [MyClassifiers.get_features(s, ft) for s in training_set]
    y_train = [MyClassifiers.get_labels(s) for s in training_set]
    
    # Get features of each word on testing set
    X_test = [MyClassifiers.get_features(s, ft) for s in testing_set]
    y_test = [MyClassifiers.get_labels(s) for s in testing_set]

    # Create trainer model of CRF
    trainer = pycrfsuite.Trainer(verbose=False)

    trainer.set_params(getTrainerFeatures(ft))    

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    # Train the model and save the trained model into model_file
    trainer.train(model_file)
    #print ("Log of last iteration={}".format(trainer.logparser.iterations[-1]))

    # Initial tagger for prediction task
    trained_model = pycrfsuite.Tagger()
    trained_model.open(model_file) # Load the trained model.
        
    # Get prediction tag results from trained model
    y_pred = [trained_model.tag(xseq) for xseq in X_test]
    
    # Print the Precision, Recall, and F-1 score
    print(bio_classification_report(y_test, y_pred))
    
    end = time.time()
    print('CRF model has been generated.')
    print('runtime:', end - start)

training_file = "corpus/combined.txt"
testing_file = "corpus/test.txt"
model_file = "models/model.crfsuite"

main(training_file, testing_file, model_file, "honnibal")

