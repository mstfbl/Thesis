'''
Model to train for differentiating restrictive vs. non-restrictive

@author: Mustafa Bal
'''

import sys, time, string
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite

def bio_classification_report(y_true, y_pred):
    '''
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "N" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master) to calculate averages properly!
    '''
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'N'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def main(training_file, testing_file, model_file):
    
    start = time.time()
    
    # Get training and testing set of data
    training_set = get_input(training_file)
    testing_set = get_input(testing_file)
    
    # Get features of each word on training set
    X_train = [get_features(s) for s in training_set]
    y_train = [get_labels(s) for s in training_set]
    
    # Get features of each word on testing set
    X_test = [get_features(s) for s in testing_set]
    y_test = [get_labels(s) for s in testing_set]

    # Create trainer model of CRF
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 0.5,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 1000,  # stop earlier
    
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })    
    
    # Train the model and save the trained model into model_file
    trainer.train(model_file)
    print ("Log of last iteration={}".format(trainer.logparser.iterations[-1]))

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

if __name__ == '__main__':
    # Get input and output parameters
    if len(sys.argv) != 4:
        print('Usage: python ' + sys.argv[0] + ' <training data input file> <testing data input file> <model file name>')
        print('       The program requires Python 3.5.2 and scikit-learn 0.15+ to execute.')
        print('       Note that it requires')
        print('       - training data input file : Tokenized training data input file location.')     
        print('       - testing data input file = Tokenized testing data input file location.')      
        print('       - model file name = Trained CRF model file path and name.')                                          
        exit ()    

    else:           
        training_file = sys.argv[1]
        testing_file = sys.argv[2]
        model_file = sys.argv[3]

        main(training_file, testing_file, model_file)