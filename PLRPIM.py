# -*- coding: utf-8 -*-
###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn import metrics

#import tensorflow as tf
#tf.python.control_flow_ops = tf
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io

from keras.models import Sequential
from keras.layers.core import  AutoEncoder, Dropout, Activation, Flatten, Merge, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import containers
from keras import regularizers
from keras.constraints import maxnorm

def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**4
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        n=n/base
        ch3=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return  nucle_com   

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com

def translate_sequence (seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    import string
    from_list = []
    to_list = []
    for k,v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(string.maketrans(str(from_list), str(to_list)))
    #TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq

def get_4_nucleotide_composition(tris, seq, pythoncount = True):
    seq_len = len(seq)
    tri_feature = []
    
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num)/seq_len)
    else:
        k = len(tris[0])
        tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1- k):
            kmer = seq[x:x+k]
            if kmer in tris:
                ind = tris.index(kmer)
                tmp_fea[ind] = tmp_fea[ind] + 1
        tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()        
    return tri_feature

def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group) #Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index]) #K:V map, use group's first letter as represent.
        index = index + 1
    return result

def prepare_data(deepmind = False, seperate=False):
    print "loading data"
    lncRNA = pd.read_csv("zma_lncRNA.csv")
    protein = pd.read_csv("zma_rbp.csv")
    interaction = pd.read_fwf("ZMAInteraction.txt") #fwf stands for fixed width formatted lines
    
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    with open('ZMAInteraction.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label = values[1]
                name = values[0].split('_')
                protein = name[0] + '-' + name[1]
                RNA = name[0] + '-' + name[1]
                if label == 'interactive':
                    interaction_pair[(protein, RNA)] = 1
                else:
                    interaction_pair[(protein, RNA)] = 0
                index  = 0
            else:
                seq = line[:-1]
                if index == 0:
                    protein_seq_dict[protein] = seq
                else:
                    RNA_seq_dict[RNA] = seq
                index = index + 1
    #name_list = read_name_from_lncRNA_fasta('ncRNA-protein/lncRNA_RNA.fa')           
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    #tris3 = get_3_trids()
    train = []
    label = []
    chem_fea = []
    for key, val in interaction_pair.iteritems():
        protein, RNA = key[0], key[1]
        #pdb.set_trace()
        if RNA_seq_dict.has_key(RNA) and protein_seq_dict.has_key(protein): #and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
            label.append(val)
            RNA_seq = RNA_seq_dict[RNA]
            protein_seq = translate_sequence (protein_seq_dict[protein], group_dict)
            if deepmind:
                RNA_tri_fea = get_RNA_seq_concolutional_array(RNA_seq)
                protein_tri_fea = get_RNA_seq_concolutional_array(protein_seq) 
                train.append((RNA_tri_fea, protein_tri_fea))
            else:
                #pdb.set_trace()
                RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount =False)
                protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
                #RNA_tri3_fea = get_4_nucleotide_composition(tris3, RNA_seq, pythoncount =False)
                #RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                #tmp_fea = protein_fea_dict[protein] + tri_fea #+ RNA_fea_dict[RNA]
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)
                    #chem_tmp_fea = (protein_fea_dict[protein], RNA_fea_dict[RNA])
                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea
                    #chem_tmp_fea = protein_fea_dict[protein] + RNA_fea_dict[RNA] 
                train.append(tmp_fea)
                #chem_fea.append(chem_tmp_fea)
        else:
            print RNA, protein   
    
    return np.array(train), label

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
    else:
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    #pdb.set_trace()
    #pdb.set_trace()
    for val in data:
        #formated_matrix1.append(np.array([val[0]]))
        formated_matrix1.append(val[0])
        formated_matrix2.append(val[1])
        #formated_matrix1[0] = np.array([val[0]])
        #formated_matrix2.append(np.array([val[1]]))
        #formated_matrix2[0] = val[1]      
    
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_data(X, scaler=None, stand = True):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def get_blend_data(j, clf, skf, X_test, X_dev, Y_dev, blend_train, blend_test):
        print 'Training classifier [%s]' % (j)
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)
            
            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]
            
            clf.fit(X_train, Y_train)
            
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            #blend_train[cv_index, j] = clf.predict(X_cv)
            #blend_test_j[:, i] = clf.predict(X_test)
            blend_train[cv_index, j] = clf.predict_proba(X_cv)[:,1]
            blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)
    
        print 'Y_dev.shape = %s' % (Y_dev.shape)

def multiple_autoencoder_extract_feature(dataset = 'RPI-Pred'):
    X, labels = get_data(dataset)
    #X = pca_reduce_dimension(X, n_components = 300)
    y, encoder = preprocess_labels(labels)
    num_cross_val = 5
    batch_size  = 50
    nb_epoch = 20
    all_performance = []
    all_performance_rf = []
    all_performance_svm = []
    all_performance_lgb = []
    all_performance_ensemb = []
    all_performance_ae = []
    all_performance_rf_seq = []
    all_performance_chem = []
    all_performance_blend = []
    activation = 'relu' #'linear' #'relu, softmax, tanh'
    for fold in range(num_cross_val):
        train = []
        test = []
        train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        #chem_train = np.array([x for i, x in enumerate(chem_fea) if i % num_cross_val != fold])
        #chem_test = np.array([x for i, x in enumerate(chem_fea) if i % num_cross_val == fold])

        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        blend_train = np.zeros((train.shape[0], 5)) # Number of training data x Number of classifiers
        blend_test = np.zeros((test.shape[0], 5)) # Number of testing data x Number of classifiers 
        skf = list(StratifiedKFold(train_label_new, 5, shuffle=True,))  
        #pdb.set_trace()     
        class_index = 0                   
        encoders = multiple_layer_autoencoder(train, test, activation = 'sigmoid', batch_size = 100, nb_epoch = 20, last_dim = 128, weight_reg = regularizers.l2(0.01), activity_reg = regularizers.l1(0.01))
        prefilter_train = np.copy(train)
        prefilter_test = np.copy(test) 
        for ae in encoders:
            prefilter_train = ae.predict(prefilter_train)
            print prefilter_train.shape
            prefilter_test = ae.predict(prefilter_test)
            
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(prefilter_train, train_label_new)
        y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        #pdb.set_trace()
        y_pred = transfer_label_from_prob(y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_ae.append([acc, precision, sensitivity, specificity, MCC])
        print '---' * 50
        
        get_blend_data(class_index, RandomForestClassifier(n_estimators=50), skf, prefilter_test, prefilter_train, np.array(train_label_new), blend_train, blend_test)
        
        
        prefilter_train, prefilter_test = autoencoder_fine_tuning(encoders, train, train_label, test, 100, 100)
        #prefilter_train, new_scaler = preprocess_data(prefilter_train)
        #prefilter_test, new_scaler = preprocess_data(prefilter_test, scaler = new_scaler)
        #prefilter_train = np.concatenate((prefilter_train, chem_train), axis = 1)
        #prefilter_test = np.concatenate((prefilter_test, chem_test), axis = 1)        
        print 'using SVM after sequence autoencoder'
        class_index = class_index + 1
        parameters = {'kernel': ['rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
        svr = svm.SVC(probability = True)
        clf = grid_search.GridSearchCV(svr, parameters, cv=3)        
        #clf = RandomForestClassifier(n_estimators=50)
	#seed = 7
	#clf = GradientBoostingClassifier(n_estimators=50, random_state=seed)
        clf.fit(prefilter_train, train_label_new)
        y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        #pdb.set_trace()
        y_pred = transfer_label_from_prob(y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance.append([acc, precision, sensitivity, specificity, MCC])
        print '---' * 50
        
        get_blend_data(class_index, grid_search.GridSearchCV(svr, parameters, cv=3), skf, prefilter_test, prefilter_train, np.array(train_label_new), blend_train, blend_test)
        

        print 'using SVM using only sequence feature'
        class_index = class_index + 1
        parameters = {'kernel': ['rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
        svr = svm.SVC(probability = True)
        clf = grid_search.GridSearchCV(svr, parameters, cv=3)  
        clf.fit(train, train_label_new)
        y_pred_rf_prob = clf.predict_proba(test)[:,1]
        y_pred_rf = transfer_label_from_prob(y_pred_rf_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_rf,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_rf_seq.append([acc, precision, sensitivity, specificity, MCC]) 
        print '---' * 50
        get_blend_data(class_index, grid_search.GridSearchCV(svr, parameters, cv=3) , skf, test, train, np.array(train_label_new), blend_train, blend_test)
        
        #roc = metrics.roc_auc_score(y_valid, valid_preds)
        # Start blending!
        bclf = LogisticRegression()
        bclf.fit(blend_train, train_label_new)
        Y_test_predict = bclf.predict(blend_test)
        print 'blend result'
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), Y_test_predict,  real_labels)
        print acc, precision, sensitivity, specificity, MCC   
        all_performance_blend.append([acc, precision, sensitivity, specificity, MCC])     
        print '---' * 50
        
        '''
        print 'ensemble deep learning and rf'
        ensemb_prob = get_preds( y_pred_prob, y_pred_rf_prob, ae_y_pred, [0.3, 0.30, 0.4])       
        ensemb_label = transfer_label_from_prob(ensemb_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), ensemb_label,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_ensemb.append([acc, precision, sensitivity, specificity, MCC]) 
        print '---' * 50
        '''
    print 'in summary'
    print 'mean performance of chem autoencoder without fine tunning'
    print np.mean(np.array(all_performance_ae), axis=0)  
    print '---' * 50
    print 'mean performance of sequence autoencoder'
    print np.mean(np.array(all_performance), axis=0)
    print '---' * 50   
    print 'mean performance of only chem using RF'
    print np.mean(np.array(all_performance_svm), axis=0)
    print '---' * 50     
    #print 'mean performance of only chem using SVM'
    #print np.mean(np.array(all_performance_svm), axis=0)
    #print '---' * 50     
    print 'mean performance of blend fusion'
    print np.mean(np.array(all_performance_blend), axis=0) 
    print '---' * 50 
     
def multiple_layer_autoencoder(X_train, X_test, activation = 'linear', batch_size = 100, nb_epoch = 20, last_dim = 64):
    nb_hidden_layers = [X_train.shape[1], 256, 128, last_dim]
    X_train_tmp = np.copy(X_train)
    #X_test_tmp = np.copy(X_test)
    encoders = []
    for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
        print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
        # Create AE and training
        ae = Sequential()
        encoder = containers.Sequential([Dense(n_in, n_out, activation=activation)])
        decoder = containers.Sequential([Dense(n_out, n_in, activation=activation)])
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                           output_reconstruction=False))
        ae.add(Dropout(0.5))
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        ae.compile(loss='mean_squared_error', optimizer='adam')#'rmsprop')
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, callbacks = [EarlyStopping(monitor='val_acc', patience=2)])
        # Store trainined weight and update training data
        #encoders.append(ae.layers[0].encoder)
        encoders.append(ae)
        X_train_tmp = ae.predict(X_train_tmp)
        print X_train_tmp.shape
        #X_test_tmp = ae.predict(X_test_tmp)
        
    #return encoders, X_train_tmp, X_test_tmp
    return encoders
      

def autoencoder_two_subnetwork_fine_tuning(X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = None, batch_size =100, nb_epoch = 20):
    print 'autoencode learning'
    last_dim = 64
    encoders1 = multiple_layer_autoencoder(X_train1, X_test1, activation = 'sigmoid', batch_size = batch_size, nb_epoch = nb_epoch, last_dim = last_dim)
    encoders2 = multiple_layer_autoencoder(X_train2, X_test2, activation = 'sigmoid', batch_size = batch_size, nb_epoch = nb_epoch, last_dim = last_dim)
    #pdb.set_trace()
    
    X_train1_tmp_bef = np.copy(X_train1)
    X_test1_tmp_bef = np.copy(X_test1) 
    for ae in encoders1:
        X_train1_tmp_bef = ae.predict(X_train1_tmp_bef)
        print X_train1_tmp_bef.shape
        X_test1_tmp_bef = ae.predict(X_test1_tmp_bef)
    
    X_train2_tmp_bef = np.copy(X_train2)
    X_test2_tmp_bef = np.copy(X_test2) 
    for ae in encoders2:
        X_train2_tmp_bef = ae.predict(X_train2_tmp_bef)
        print X_train2_tmp_bef.shape
        X_test2_tmp_bef = ae.predict(X_test2_tmp_bef)
        
    prefilter_train_bef = np.concatenate((X_train1_tmp_bef, X_train2_tmp_bef), axis = 1)
    prefilter_test_bef = np.concatenate((X_test1_tmp_bef, X_test2_tmp_bef), axis = 1)
        
    print 'fine tunning'
    print 'number of layers:', len(encoders1)
    sec_num_hidden = last_dim
    model1 = Sequential()
    ind = 0
    for encoder in encoders1:
        model1.add(encoder.layers[0].encoder)
        if ind != len(encoders1)  - 1 :
            model1.add(Dropout(0.5)) 
            ind = ind + 1
    model1.add(PReLU((sec_num_hidden,)))
    model1.add(BatchNormalization((sec_num_hidden,)))
    model1.add(Dropout(0.5))
    

    model2 = Sequential()
    ind = 0
    for encoder in encoders2:
        model2.add(encoder.layers[0].encoder)
        if ind != len(encoders2)  - 1 :
            model2.add(Dropout(0.5)) 
            ind = ind + 1
    model2.add(PReLU((sec_num_hidden,)))
    model2.add(BatchNormalization((sec_num_hidden,)))   
    model2.add(Dropout(0.5))     
         
    model = Sequential()
    model.add(Merge([model1, model2], mode='concat'))
    total_hid = sec_num_hidden + sec_num_hidden
    
    model.add(Dense(total_hid, 2))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))
    #model.get_config(verbose=0)
    sgd = SGD(lr=3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd) #'rmsprop')
    model.fit([X_train1, X_train2], Y_train, batch_size=100, nb_epoch=30, verbose=0, callbacks = [EarlyStopping(monitor='val_acc', patience=2)])
    #config = autoencoder.get_config(verbose=1)
    #autoencoder = model_from_config(config)
    #pdb.set_trace()
    X_train1_tmp = np.copy(X_train1)
    X_test1_tmp = np.copy(X_test1)  
    ae=model.layers[0].layers[0]  
    ae.compile(loss='mean_squared_error', optimizer='adam')
    X_train1_tmp = ae.predict(X_train1_tmp)
    X_test1_tmp = ae.predict(X_test1_tmp)

    X_train2_tmp = np.copy(X_train2)
    X_test2_tmp = np.copy(X_test2)  
    ae=model.layers[0].layers[1]  
    ae.compile(loss='mean_squared_error', optimizer='adam')
    X_train2_tmp = ae.predict(X_train2_tmp)
    X_test2_tmp = ae.predict(X_test2_tmp)
    
    prefilter_train = np.concatenate((X_train1_tmp, X_train2_tmp), axis = 1)
    prefilter_test = np.concatenate((X_test1_tmp, X_test2_tmp), axis = 1)
    return prefilter_train, prefilter_test, prefilter_train_bef, prefilter_test_bef
    #return model
def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )

def DeepPLRPIM():
    X, labels = prepare_data(seperate = True)
    
    X_data1, X_data2 = transfer_array_format(X)
    print X_data1.shape, X_data2.shape
    X_data1, scaler1 = preprocess_data(X_data1)
    X_data2, scaler2 = preprocess_data(X_data2)
    y, encoder = preprocess_labels(labels)
    
    num_cross_val = 5
    all_performance_bef = []
    all_performance_lgb = []
    all_performance_rf = []
    all_performance_blend1 = []
    all_performance_blend = []
    all_performance_ensemb = []
    all_labels = []
    all_prob = {}
    num_classifier = 5
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_prob[4] = []
    all_prob[5] = []
    all_averrage = []
    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
  
          
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        
        blend_train = np.zeros((train1.shape[0], num_classifier)) # Number of training data x Number of classifiers
        blend_test = np.zeros((test1.shape[0], num_classifier)) # Number of testing data x Number of classifiers 
        skf = list(StratifiedKFold(train_label_new, num_classifier))  
        class_index = 0
        prefilter_train, prefilter_test, prefilter_train_bef, prefilter_test_bef = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)
        #X_train1_tmp, X_test1_tmp, X_train2_tmp, X_test2_tmp, model = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)
        #model = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)
        #model = merge_seperate_network(train1, train2, train_label)
        #proba = model.predict_proba([test1, test2])[:1]
        
        
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
                
        all_labels = all_labels + real_labels
        #prefilter_train, new_scaler = preprocess_data(prefilter_train, stand =False)
        #prefilter_test, new_scaler = preprocess_data(prefilter_test, scaler = new_scaler, stand = False)
        '''
        prefilter_train1 = xgb.DMatrix( prefilter_train, label=train_label_new)
        evallist  = [(prefilter_train1, 'train')]
        num_round = 10
        clf = xgb.train( plst, prefilter_train1, num_round, evallist )
        prefilter_test1 = xgb.DMatrix( prefilter_test)
        ae_y_pred_prob = clf.predict(prefilter_test1)
        '''
        tmp_aver = [0] * len(real_labels)
        print 'deep autoencoder'
        #parameters = {'kernel': ['linear','poly', 'rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
        #svr = svm.SVC(probability = True)
        #clf1 = grid_search.GridSearchCV(svr, parameters, cv=3)
        clf = RandomForestClassifier(n_estimators=50)
	#eclf = VotingClassifier(estimators=[('svm', clf1),('rf', clf2)], voting='soft')  
        clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        tmp_aver = [val1 + val2/3 for val1, val2 in zip(ae_y_pred_prob, tmp_aver)]
        proba = transfer_label_from_prob(ae_y_pred_prob)
        #pdb.set_trace()            
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
	auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        #all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
        get_blend_data(class_index, RandomForestClassifier(n_estimators=50), skf, prefilter_test, prefilter_train, np.array(train_label_new), blend_train, blend_test)
        
        print 'deep autoencoder without fine tunning'
        class_index = class_index + 1
	clf = RandomForestClassifier(n_estimators=50)  
        clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob_bef = clf.predict_proba(prefilter_test_bef)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob_bef]
        tmp_aver = [val1 + val2/3 for val1, val2 in zip(ae_y_pred_prob_bef, tmp_aver)]
        proba = transfer_label_from_prob(ae_y_pred_prob_bef)            
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance_bef.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])

	print 'RPISeq_RF'
        class_index = class_index + 1
	clf = RandomForestClassifier(n_estimators=50)
        clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        proba = transfer_label_from_prob(ae_y_pred_prob)

        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print "RF :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
	all_performance_rf.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
        get_blend_data(class_index, RandomForestClassifier(n_estimators=70), skf, prefilter_test, prefilter_train, np.array(train_label_new), blend_train, blend_test) 	
	              
        class_index = class_index + 1
	bclf = RandomForestClassifier(n_estimators=50)
        bclf.fit(blend_train, train_label_new)
        stack_y_prob = bclf.predict_proba(blend_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in stack_y_prob]
        Y_test_predict = bclf.predict(blend_test)
        print 'RPI-SAN stacked ensembling'
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), Y_test_predict,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score  
        all_performance_blend1.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])     

        class_index = class_index + 1
        bclf = LogisticRegression()
        bclf.fit(blend_train, train_label_new)
        stack_y_prob = bclf.predict_proba(blend_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in stack_y_prob]
        Y_test_predict = bclf.predict(blend_test)
        print 'stacked ensembling'
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), Y_test_predict,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score  
        all_performance_blend.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])     

	print 'Ensemble Classifiers raw feature'
	class_index = class_index + 1
        prefilter_train = np.concatenate((train1, train2), axis = 1)
        prefilter_test = np.concatenate((test1, test2), axis = 1)

	clf2 = RandomForestClassifier(n_estimators=50)
	clf3 = XGBClassifier()
	clf5 = GradientBoostingClassifier(n_estimators=70, random_state=7)
	lgmodel = lgb.LGBMClassifier(boosting_type='gbdt',
        objective=None, 
        learning_rate=0.5,
        colsample_bytree=1,
        subsample=1,
        random_state=None,
        n_estimators=100,
        num_leaves=50, max_depth=5, num_class=None)
	eclf = VotingClassifier(estimators=[('rf', clf2),('lgb',lgmodel)], voting='soft')
	eclf.fit(prefilter_train, train_label_new)
	ae_y_pred_prob = eclf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        proba = transfer_label_from_prob(ae_y_pred_prob)
                 
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print "Ens :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance_ensemb.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
        print '---' * 50
        
    print 'mean performance of PLRPIM'
    print np.mean(np.array(all_performance_ensemb), axis=0)
    print '---' * 50 
    print 'mean performance of RPISeq_RF'
    print np.mean(np.array(all_performance_rf), axis=0)
    print '---' * 50   
    print 'mean performance of RPI-SAN'
    print np.mean(np.array(all_performance_blend1), axis=0)
    print '---' * 50
    print 'mean performance of stacked ensembling'
    print np.mean(np.array(all_performance_blend), axis=0)
    print '---' * 50
    
    fileObject = open('PLRPIMresultListAUC_integrate_add_lncRNA_protein.txt', 'w')
    for i in all_performance_ensemb:
        k=' '.join([str(j) for j in i])
	fileObject.write(k+"\n")
    fileObject.write('\n')
    for i in all_performance_rf:
        k=' '.join([str(j) for j in i])
        fileObject.write(k+"\n")
    fileObject.write('\n')
    for i in all_performance_blend1:
        k=' '.join([str(j) for j in i])
        fileObject.write(k+"\n")
    for i in all_performance_blend: 
        k=' '.join([str(j) for j in i])
        fileObject.write(k+"\n")

    fileObject.close()
    Figure = plt.figure()
    plot_roc_curve(all_labels, all_prob[3], 'RPI-SAN')
    plot_roc_curve(all_labels, all_prob[2], 'RPISeq-RF')
    plot_roc_curve(all_labels, all_prob[4], 'IPMiner')
    plot_roc_curve(all_labels, all_prob[5], 'PLRPIM')  
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    #plt.savefig(save_fig_dir + selected + '_' + class_type + '.png') 
    plt.show() 
   

def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label


if __name__=="__main__":
    DeepPLRPIM()
