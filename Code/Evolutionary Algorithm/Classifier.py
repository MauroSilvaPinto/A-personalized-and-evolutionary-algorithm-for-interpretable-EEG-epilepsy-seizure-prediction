"""

Classifier class.
In this class, we available all machine learning part:
    data balancing, classifier training, standardization,
    calculate inter-ictal period duration
    calculate FPR, seizure sensitivity, sample sensititivty
    and sample specificity


"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from Filter import Filter


# in the class Classifier, one finds the methods referring to the time-series
# machine-learning procedure
class Classifier:
    
    # replace NaN by Zeros, this is due to the possibility of having NaN features
    # as in find peaks where there were no peaks found
    def replaceNanByZeros(data):
        data[np.isnan(data)]=0
        return data
    
    # the same happens in the division, where one must replace Zeros by Ones
    # for not having NaN or Inf values
    def replaceZerosByOnes(data):
        data[data==0]=1
        return data
    
    # applies zscoring to the training data
    # returns the features zscored, as well as the mean and standard deviation
    # for each feature to then apply in the validation and test phase
    def zScoreTrainingData(features):
        standard_deviation=np.std(features,axis=0)
        mean=np.mean(features,axis=0)
        
        standard_deviation=Classifier.replaceZerosByOnes(standard_deviation)
        features=Classifier.applyPreprocess(features,mean,standard_deviation)
        
        return [features, mean, standard_deviation]
    
    
    # preprocesses the features by replacing NaN by zeros and by z-scoring the 
    # features
    def preProcessTrainingData(features):
        features=Classifier.replaceNanByZeros(features) 
        return Classifier.zScoreTrainingData(features)
    
    # balances the data in terms of labels with the chosen method:
    #   oversampling: duplicates randomly samples from the less representative
    # class until class balance is achieved
    #
    #   undersampling: removes randomly samples from the most representative class
    # until class balance is achieved
    def balanceData(features,labels,method):
        if method=="over":
            ros = RandomOverSampler()
        elif method=="under":
            ros=RandomUnderSampler()
        
        return ros.fit_resample(features, labels)
    
    def computeBalancedClassWeights(labels):
        from sklearn.utils import class_weight
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels),
                                                 labels)
        
        return class_weights
    
    def computeSampleWeights(labels,class_weights):
        sample_weights=np.zeros(len(labels))
        
        sample_weights[np.where(labels==0)[0]]=class_weights[0]
        sample_weights[np.where(labels==1)[0]]=class_weights[1]
       
        return sample_weights
        
    # first: balances the data with the balanceData function
    # second: trains a classifier
    # note: there are several options that are commented, in this case it is being
    #       used an svm
    def trainClassifier(features,labels,method):
        #features, labels = Classifier.balanceData(features,labels,method)
       
        return Classifier.trainLogisticRegression(features,labels)
#        return Classifier.trainLDA(features,labels)
#        return Classifier.trainKNN(features,labels,7)
#        return Classifier.trainLinearRegression(features,labels)
#        return Classifier.trainRandomForest(features,labels)
#        return Classifier.trainSVM(features,labels,'rbf')
#        return Classifier.trainAdaBoost(features,labels)
#        return Classifier.trainNaiveBayes(features,labels)
#        return Classifier.trainDecisionTree(features,labels)
    
    # returns the classification prediction labels of the
    # the provided features with the provided already trained classifier
    def classify(classifier,features):
        return classifier.predict(features)
    
    # returns the classification prediction scores of the provided features
    # of the provided trained classifier
    def getScores(classifier,features):
        return classifier.predict_proba(features)[:,1]      
  
    # trains a decision tree classifier
    def trainDecisionTree(features,labels):
#        class_weights=Classifier.computeBalancedClassWeights(labels)
#        sample_weights=Classifier.computeSampleWeights(labels,class_weights)
        clf = tree.DecisionTreeClassifier(class_weight='balanced')
        return clf.fit(features, labels)
    
    # trains a logistic regression classifier
    def trainLogisticRegression(features,labels):
        class_weights=Classifier.computeBalancedClassWeights(labels)
        sample_weights=Classifier.computeSampleWeights(labels,class_weights)
        logreg = LogisticRegression()
        return logreg.fit(features, labels, sample_weight=sample_weights)
        
     
    # trains a naive bayes classifier
    def trainNaiveBayes(features,labels):
        class_weights=Classifier.computeBalancedClassWeights(labels)
        sample_weights=Classifier.computeSampleWeights(labels,class_weights)
        gnb = GaussianNB()
        return gnb.fit(features,labels,sample_weight=sample_weights)
    
    def trainAdaBoost(features,labels):
        class_weights=Classifier.computeBalancedClassWeights(labels)
        sample_weights=Classifier.computeSampleWeights(labels,class_weights)

        ada=AdaBoostClassifier(n_estimators=100, random_state=0)
        return ada.fit(features,labels, sample_weight=sample_weights)
    
    # trains a KNN classifier with K neighbors
    def trainKNN(features,labels,k):
        features, labels = Classifier.balanceData(features,labels,'over')
        
        KNN = KNeighborsClassifier(n_neighbors=k)
        return KNN.fit(features, labels) 
    
    # trains a linear regression classifier    
    def trainLinearRegression(features,labels):
        class_weights=Classifier.computeBalancedClassWeights(labels)
        sample_weights=Classifier.computeSampleWeights(labels,class_weights)
        
        regr = linear_model.LinearRegression()
        return regr.fit(features, labels,sample_weight=sample_weights)
    
    
    # trains a linear discriminant analysis classifier
    def trainLDA(features,labels):
        features, labels = Classifier.balanceData(features,labels,'over')
        
        LDA = LinearDiscriminantAnalysis()
        return LDA.fit(features,labels)
    
    # trains a random forest classifier with 100 trees
    def trainRandomForest(features,labels):
        class_weights=Classifier.computeBalancedClassWeights(labels)
        sample_weights=Classifier.computeSampleWeights(labels,class_weights)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                            random_state=0)
        return clf.fit(features,labels,sample_weight=sample_weights)  
    
    # trains an SVM
    def trainSVM(features,labels,kernel):
        class_weights=Classifier.computeBalancedClassWeights(labels)
        sample_weights=Classifier.computeSampleWeights(labels,class_weights)
        
        clf = svm.SVC(gamma='scale',kernel=kernel)
        return clf.fit(features, labels,sample_weight=sample_weights)  
    
    # applies the preprocess for the validation and test data
    #   zscores the features with the mean and standard deviation obtained from
    #   training
    def applyPreprocess(features,mean,standard_deviation):
        features=Classifier.replaceNanByZeros(features)
        return np.divide(np.subtract(features,mean),standard_deviation) 
    
    
    # calculates sensitivity from a provided confusion_matrix
    def sensitivity(confusion_matrix):
        return (confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1]))
    
    # calculates specificity from a provided confusion matrix
    def specificity(confusion_matrix):
        return (confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1]))
    
    # calculates the confusion matrix with the labels and predicted ones
    def confusionMatrix(labels,predicted):
        return confusion_matrix(labels,predicted,labels=[1,0])
    
    # calculates the Area Under the Curve of a ROC curve with the labels and the
    # predicted scores
    def calculateAUC(labels,scores):
        return roc_auc_score(labels, scores)
    
    # calculates hte false positive rate per hour
    #
    #   - obtains the interictal indexes of the labels
    #   -  with these, obtains the total length of the interictal period through
    # the use of the interictal indexes
    #   - calculates the number of alarms triggered
    #   - returns the division of the number of triggered alarms by the number of hours
    def falsePositiveRateHour(labels, predicted,sop):
        interictal_indexes=np.where(np.array(labels)==0)[-1]
        time_in_hours=Filter.convertMinutesInHours(len(interictal_indexes)*Filter.getStepFilterInMinutesInterictal())
        number_of_alarms=Filter.calculateNumberOfFalseAlarms(labels,predicted)
        return number_of_alarms/(time_in_hours-Classifier.LostRefractoryTime(labels,predicted,sop))
    
    def getInterIctalTotalPeriod(labels,predicted,sop):
        interictal_indexes=np.where(np.array(labels)==0)[-1]
        time_in_hours=Filter.convertMinutesInHours(len(interictal_indexes)*Filter.getStepFilterInMinutesInterictal())
        return time_in_hours;
    
    # calculates hte false positive rate per hour
    #
    #   - obtains the interictal indexes of the labels
    #   -  with these, obtains the total length of the interictal period through
    # the use of the interictal indexes
    #   - calculates the number of alarms triggered
    #   - returns the division of the number of triggered alarms by the number of hours
    def falsePositiveRateHourNoRefractoryPeriod(labels, predicted,sop):
        interictal_indexes=np.where(np.array(labels)==0)[-1]
        time_in_hours=Filter.convertMinutesInHours(len(interictal_indexes)*Filter.getStepFilterInMinutesInterictal())
        number_of_alarms=Filter.calculateNumberOfFalseAlarms(labels,predicted)
        return number_of_alarms/(time_in_hours)

    # the refractory time discount is done differently (i have to account that the pre-ictal can be anywhere and
    #not at the end of the seizure)
    def falsePosititiveRateHourSurrogate(labels, predicted,sop):
        interictal_indexes=np.where(np.array(labels)==0)[-1]
        time_in_hours=Filter.convertMinutesInHours(len(interictal_indexes)*Filter.getStepFilterInMinutesInterictal())
        number_of_alarms=Filter.calculateNumberOfFalseAlarms(labels,predicted)
        
        return number_of_alarms/(time_in_hours-Classifier.LostRefractoryTimeSurrogate(labels,predicted,sop))
        
        
    
#    def falsePositiveRateHourSurrogate(labels,predicted):
#        number_of_alarms=0
#        for i in range(0,len(labels)):
#            if labels[i]==0 and predicted[i]==1:
#                number_of_alarms=number_of_alarms+1
#        
#        interictal_indexes=np.where(np.array(labels)==0)[-1]
#        time_in_hours=Filter.convertMinutesInHours(len(interictal_indexes)*Filter.getStepFilterInMinutesInterictal())
#        
#        return number_of_alarms/time_in_hours
    
    
    def timeUnderFalseAlarm(labels, predicted):
        interictal_indexes=np.where(np.array(labels)==0)[-1]
        return np.sum(predicted[interictal_indexes])/len(interictal_indexes)
    
    # it aint simply removing all false alarms and multiplying by SOP+SPH
    # why? what if it was the last inter-ictal sample? why will i remove
    # inter-ictal that will no more exists? the classifier will not fire an
    # alarm anyway, of course    
    def LostRefractoryTime(labels,predicted,sop):
        indexes_ending_interictal=np.where(np.diff(labels)==1)[0]+1
#        indexes_ending_preictal=np.where(np.diff(labels)==(-1))[0]+1
        
        time_removed=0
        for i in range (1,len(labels)):
            if predicted[i]==1 and labels[i]==0:
                distance_to_pre_ictal=abs(indexes_ending_interictal[0]-i);
                
                if distance_to_pre_ictal>(Filter.getStepFilterInMinutesInterictal()*sop+Filter.getStepFilterInMinutesInterictal()*10):
                    time_removed=time_removed+Filter.getStepFilterInMinutesInterictal()*sop+Filter.getStepFilterInMinutesInterictal()*10
                else:
                    time_removed=time_removed+distance_to_pre_ictal*Filter.getStepFilterInMinutesInterictal();
        return time_removed/60   
        
        
    
    
    # it aint simply removing all false alarms and multiplying by SOP+SPH
    # why? what if it was the last inter-ictal sample? why will i remove
    # inter-ictal that will no more exists? the classifier will not fire an
    # alarm anyway, of course. and what if it is the last sample? will i remove
    # data that does not exist?
    def LostRefractoryTimeSurrogate(labels,predicted,sop):
        indexes_ending_interictal=np.where(np.diff(labels)==1)[0]+1
#        indexes_ending_preictal=np.where(np.diff(labels)==(-1))[0]+1
        time_removed=0
        for i in range (1,len(labels)):
            if predicted[i]==1 and labels[i]==0:
                distance_to_pre_ictal=indexes_ending_interictal[0]-i;
                if distance_to_pre_ictal>0:     
                    if abs(distance_to_pre_ictal)>(Filter.getStepFilterInMinutesInterictal()*sop+Filter.getStepFilterInMinutesInterictal()*10):
                        time_removed=time_removed+Filter.getStepFilterInMinutesInterictal()*sop+Filter.getStepFilterInMinutesInterictal()*10
                    else:
                        time_removed=time_removed+abs(distance_to_pre_ictal*Filter.getStepFilterInMinutesInterictal());
                else:
                    distance_to_end=abs(len(labels)-i)
                    if distance_to_end>(Filter.getStepFilterInMinutesInterictal()*sop+Filter.getStepFilterInMinutesInterictal()*10):
                        time_removed=time_removed+Filter.getStepFilterInMinutesInterictal()*sop+Filter.getStepFilterInMinutesInterictal()*10
                    else:
                        time_removed=time_removed+distance_to_end*Filter.getStepFilterInMinutesInterictal();
        
        return time_removed/60   
            
          
        
        
       
                                      
            
       