'''
Created on Dec 15, 2016

Multiple Myeloma progression predictor.

@author: Inwoo Chung(gutomitai@gmail.com)
'''

import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import nan

# Constants.
NUM_GENES = 18898
ES_LAYERS = [NUM_GENES, 2000, 2000, 200, 200, 100, 1]

GENE_EXPRESSION_FILE_NAME = 'expressions_FINAL.csv'
GENE_CVR_FILE_NAME = 'copynumber_FINAL.csv'
GENE_MUTATIONS_FILE_NAME = 'mutations_FINAL.csv'
GROUND_TRUTH_FILE_NAME = 'groundtruth_FINAL.csv'

class MultipleMyelomaInfo:
    '''
        Multiple myeloma data handler.
    '''
    
    def __init__(self, dataPath):
        '''
            Constructor.
            @param dataPath: Directory including data files. 
        '''
        
        # Load and parse each data.
        self.geneExps = pd.read_csv(dataPath + "/" + GENE_EXPRESSION_FILE_NAME)
        self.geneCNVs = pd.read_csv(dataPath + "/" + GENE_CVR_FILE_NAME)
        self.geneMutations = pd.read_csv(dataPath + "/" + GENE_MUTATIONS_FILE_NAME)
        self.groundTruth = pd.read_csv(dataPath + "/" + GROUND_TRUTH_FILE_NAME)
        
    def getData(self):
        '''
            Get pre-processed gene expression, copy number variation, mutation, ground truth data.
        '''
        
        # Preprocess data.
        geneExpsM = np.asarray(self.geneExps)
        geneCNVsM = np.asarray(self.geneCNVs)
        geneMutationsM = np.asarray(self.geneMutations)
        groundTruthM = np.asarray(self.groundTruth)
        
        # Outliers.
        
        # Missing factors.
        # Fill average values.
        self.fillAverageValues(geneExpsM, geneCNVsM, geneMutationsM, groundTruthM)
        
        return (geneExpsM, geneCNVsM, geneMutationsM, groundTruthM[0,:], groundTruthM[1,:])
        
    def fillAverageValues(self, geneExpsM, geneCNVsM, geneMutationsM, groundTruthM):
        '''
            Fill average values for each factor.
        '''
        
        # Get average values for each factor.
        geneExpsAvgVec = list()
        geneCNVsAvgVec = list()
        geneMutationsAvgVec = list()
        groundTruthAvgVec = list()
             
        for key in self.geneExps.keys():
            geneExpsAvgVec.append(self.geneExps[key].mean())
            geneCNVsAvgVec.append(self.geneCNVs[key].mean())
            geneMutationsAvgVec.append(self.geneMutationsVec[key].mean())
        
        for key in self.groundTruth.keys():
            groundTruthAvgVec.append(self.groundTruth[key].mean())
        
        # Fill average values for missing factors.    
        numSamples = geneExpsM.shape[0]
        
        for geneNum in range(NUM_GENES):
            for sampleNum in range(numSamples):
                if (geneExpsM[sampleNum, geneNum] != nan): # nan != nan is True.
                    geneExpsM[sampleNum, geneNum] = geneExpsAvgVec[geneNum]
                if (geneCNVsM[sampleNum, geneNum] != nan):
                    geneCNVsM[sampleNum, geneNum] = geneCNVsAvgVec[geneNum]
                if (geneMutationsM[sampleNum, geneNum] != nan):
                    geneMutationsM[sampleNum, geneNum] = geneMutationsAvgVec[geneNum]
        
        for v in range(len(self.groundTruth.keys())):
            for sampleNum in range(numSamples):
                if (groundTruthM[sampleNum, v] != nan):
                    groundTruthM[sampleNum, v] = groundTruthAvgVec[v]
        
# Create a multiple Myeloma progression prediction model.
# Load data.
MMInfo = MultipleMyelomaInfo('./')
fMMData = MMInfo.getData()

# Declare affecting factors and a target feature.
geneExps = tf.placeholder(tf.float32, shape=[None, NUM_GENES], name='geneExps')
geneCNVs = tf.placeholder(tf.float32, shape=[None, NUM_GENES], name='geneCNVs')
geneMuts = tf.placeholder(tf.float32, shape=[None, NUM_GENES], name='geneMuts')
surTP = tf.placeholder(tf.float32, shape=[None, 1], name='surTP')
surTO = tf.placeholder(tf.float32, shape=[None, 1], name='surTO')

# Create a neural network feed forward model for the relationship between gene 
# expression and survival time.
# Layers about gene expression to affecting gene expression combination.
esAct = None
surTOEst = None

for i in range(len(ES_LAYERS) - 1):
    
    # Weight and bias.
    esW = tf.Variable(tf.zeros([ES_LAYERS[i], ES_LAYERS[i + 1]]))
    esB = tf.Variable(tf.zeros([ES_LAYERS[i + 1]]))
    
    if (i == 0):
        esAct = tf.nn.relu(tf.matmul(geneExps, esW) + esB)
        continue
    elif (i < len(ES_LAYERS) - 2):
        esAct = tf.nn.relu(tf.matmul(esAct, esW) + esB)
        continue
    
    surTOEst = tf.matmul(esAct, esW) + esB
    
# Loss function.
loss = tf.reduce_mean(tf.square(surTO - surTOEst))

# Train the model.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Variable initializer.
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    geneExpsBatch, groundTruthBatch = tf.train.shuffle_batch([tf.constant(fMMData[0]), tf.constant(fMMData[4])] \
                                                             , batch_size=332 \
                                                             , capacity=10000 \
                                                             , min_after_dequeue=5000 \
                                                             , num_threads=4 \
                                                             , enqueue_many=True \
                                                             , allow_smaller_final_batch=True)
    sess.run(train_step, feed_dict={geneExps: geneExpsBatch, surTO: groundTruthBatch})

# Evaluate the MM prediction model.
    
        
        
    
     