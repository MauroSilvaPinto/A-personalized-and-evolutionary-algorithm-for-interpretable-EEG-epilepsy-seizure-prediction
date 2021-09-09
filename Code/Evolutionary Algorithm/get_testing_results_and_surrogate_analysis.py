'''

A code to get the testing results and statistical validation (surrogate analysis)
'''

# %% Imports

import os

from Database import Database
from Population import Population
import warnings
import numpy as np
from Filter import Filter
import copy
import scipy
from scipy import stats
from f_RandPredictd import f_RandPredictd

# %% To you, my dearest user, fill this

patient_number=["53402"]


####

# clear the screen
os.system('clear')
# ignores warnings
warnings.filterwarnings("ignore")

my_database_patient_list=[53402]
sop_period=40

mean_correct_seizures=[]
mean_fprs=[]
mean_fps=[]
mean_fitnesses=[]
mean_sensibilities=[]


for patient in patient_number:
    

    # %% Loading the Database and Prunning it
    # go back to data folder
    os.chdir("..")
    os.chdir("..")
    os.chdir("Data")
    path_data=os.getcwd()
    
    path_stored_files=os.getcwd()+'/Evolutionary_executions/sop_min_40'
    os.chdir("Preprocessed_data")
    path=os.getcwd()
    
    
    os.chdir(path)
    filenames=os.listdir()       
       
    # Load Database
    my_database=Database(path)
    
    my_database.eliminateSeizuresWithLessThan(230)
    my_database.eliminatePatientsWithLowSeizureNumber(5)
   
    my_database.eliminateAllPatientsExcept(int(patient))
    my_database.eliminateSeizuresWithLessThan(230)
    path_stored_files=(path_stored_files+"/patient_"+patient)
    os.chdir(path_stored_files)
    
    # %%  Parameters of the population
    number_of_features=1
    population_size=1
    
    # %% Initialization of the population
    
    # Initialize the population
    my_population=Population(population_size,number_of_features,path,my_database)
    
    # %% Loading the Database and Prunning it
    os.chdir(path_stored_files)
    
    population=100
    method="B"
    mutation="1.0"
    recombination="0.8"
    kind="one_by_one"
    
    os.chdir(path_stored_files)    
    
    print("__Patient__")
    print(patient_number)
    print("")

    for sop_period in range(sop_period,sop_period+1):
                
        correct_seizures=[]
        fps=[]
        
        correct_seizures_surrogate=[]
        fps_surrogate=[]
        correct_seizures_surrogate_std=[]
        
    
        for run in range(1,31):
            os.chdir(path_stored_files)
            print("execution "+str(run)+"...")
            
            population=100
            method="B"
            mutation="1.0"
            recombination="0.8"
            kind="one_by_one"
                
            
            run_1_filter=np.load('patient_'+str(patient)+'_method_'+
                     method+'_population_'+str(population)+
                     "_mutation_"+mutation+"_recombination_"+recombination+
                     '_run_' + str(run)+'_best_individual.npy')
            run_1_filter_phenotype=copy.deepcopy(run_1_filter)
            run_1_filter_phenotype=my_population.getCurrentFilterFromLoadedPhenotype(run_1_filter_phenotype)
            
            
            
            
            metrics=my_database.applyFilterToDatabaseWithPast(run_1_filter_phenotype,
                                                                                           "test_data",
                                                                                           "sensibility_fpr_hour",
                                                                                           "all_patients",
                                                                                           0)
            
            
            correct_seizures.append(metrics[0][5])
            fps.append(metrics[0][3])
            
            
            
            
            number_surrogates=10;    
            metrics_surrogate=my_database.surrogateAnalysis(run_1_filter_phenotype,
                                                                                           "test_data",
                                                                                           "sensibility_fpr_hour",
                                                                                           "all_patients",
                                                                                           0,number_surrogates)
            
            
            correct_seizures_surrogate.append(metrics_surrogate[0][5])
            fps_surrogate.append(metrics_surrogate[0][3])
            correct_seizures_surrogate_std.append(metrics_surrogate[0][9])
            

        
        
        print("")
        print("Seizure Sensibility: "+str(round(np.mean(correct_seizures)/metrics[0][1],2))+
              " +/- "+str(round(np.std(correct_seizures)/metrics[0][1],2)))        
        print("FPR/h: "+str(round(np.mean(fps),2))+
              " +/- "+str(round(np.std(fps),2)))
        print("")
        print("______Statistical Validation____________________")
        print("Surrogate Analysis")
        print("Seizure Sensibility: "+str(round((np.mean(correct_seizures_surrogate)/number_surrogates)/metrics[0][1],2))+
              " +/- "+str(round((np.std(correct_seizures_surrogate)/number_surrogates)/metrics[0][1],2)))        
        print("FPR/h: "+str(round(np.mean(fps_surrogate),2))+
              " +/- "+str(round(np.std(fps_surrogate),2)))
        
        
        
