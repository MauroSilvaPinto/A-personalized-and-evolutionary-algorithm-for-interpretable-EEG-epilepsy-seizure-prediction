"""
Script for plotting the graphs from the phenotype study from the paper.
To work, this script has to be executed inside the Evolutionary Algorithm folder,
along with the Analyze class.

"""


# %% Imports

import os

from Database import Database
from Population import Population
import warnings
import numpy as np
import os
from Analyze import Analyze
import matplotlib.pyplot as plt
from Filter import Filter
import copy
import scipy
from scipy import stats

# %% System settings

# clear the screen
os.system('clear')
# ignores warnings
warnings.filterwarnings("ignore")

# %% Loading the Database and Prunning it
# go back to data folder
os.chdir("..")
os.chdir("..")
os.chdir("Data")
path_data=os.getcwd()
os.chdir("Preprocessed_data")
path=os.getcwd()

# Save the run files in the data folder
path_stored_files=path_data
os.chdir(path)
filenames=os.listdir()        

# Load Database
my_database=Database(path)

my_database.eliminateSeizuresWithLessThan(230)
my_database.eliminatePatientsWithLowSeizureNumber(5)


# Print the number of seizures for each patient
#my_database.printSeizuresFromAllPatients()

# Make a list of all patients
my_database_patient_list=['53402']


#%% Analysis of all the best individuals

filters=[]
predictive_powers=[]
weights_patient=[]

for patients in range(0,len(my_database_patient_list)):
    

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
    
    patient=my_database_patient_list[patients]
       
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
    method="A"
    mutation="1.0"
    recombination="0.0"
    kind="one_by_one"
    
    os.chdir(path_stored_files)
        
    
    for forget_this in range(0,1):
    
        for run in range(1,31):
            os.chdir(path_stored_files)
            
            population=100
            method="B"
            mutation="1.0"
            recombination="0.8"
            kind="one_by_one"
                
        
            run_filter=np.load('patient_'+str(patient)+'_method_'+
                    method+'_population_'+str(population)+
                    "_mutation_"+mutation+"_recombination_"+recombination+
                    '_run_' + str(run)+'_best_individual.npy')
            run_filter_phenotype=copy.deepcopy(run_filter)
            run_filter_phenotype=my_population.getCurrentFilterFromLoadedPhenotype(run_filter_phenotype)
            
            
            # Calculating predictive power for validation and test data
            predictive_powa=my_database.calculateFilterPredictivePowerWithPast(run_filter_phenotype,
                                                                      "validation_data",
                                                                      0)*0.60
                                                                               
            predictive_powa=predictive_powa+my_database.calculateFilterPredictivePowerWithPast(run_filter_phenotype,
                                                                      "test_data",
                                                                      0)*0.40
            predictive_powa=predictive_powa/np.sum(predictive_powa)
             
            # Adding to the list of filters and predictive powers
            filters.append(run_filter)
            predictive_powers.append(predictive_powa)
            
            
# Interpretability Analysis
    
## Electrode Plot
electrodes_list, presence, predictive_power=Analyze.calculatePredictivePowerElectrode(filters, predictive_powers)
    
any_electrode_presence=presence
any_electrode_power=predictive_power

## Number of Different Electrodes
different_electrodes_list, presence=Analyze.calculateDifferentElectrodesPresence(filters,predictive_powers)
    
any_electrode_number_presence=presence
any_electrode_number_power=predictive_power
    

## Lobe Plot
lobe_list, presence, predictive_power=Analyze.calculatePredictivePowerLobe(filters,predictive_powers)
any_lobe_presence=presence
any_lobe_power=predictive_power
    
lobe_list=["Central","Frontal","Pre-Frontal","Occipital", "Parietal","Temporal"]
    
 ## Number of Different Lobes
different_lobes_list, presence=Analyze.calculateDifferentLobesPresence(filters,predictive_powers)  
any_lobe_number_presence=presence
    

## Hemisphere Plot
hemisphere_list, presence, predictive_power=Analyze.calculatePredictivePowerHemisphere(filters,predictive_powers)

any_hemisphere_presence=presence
any_hemisphere_power=predictive_power
hemisphere_list=["Left","Central","Right"]    
    
## Number of Different Hemispheres
different_hemispheres_list, presence=Analyze.calculateDifferentHemispheresPresence(filters, predictive_powers)  
any_hemisphere_number_presence=presence

## Number of Different Characteristics
different_characteristics_list, presence=Analyze.calculateDifferentCharacteristicsPresence(filters, predictive_powers)  
any_characteristic_number_presence=presence

## Number of Different Characteristics
different_operators_list, presence=Analyze.calculateDifferentOperatorsPresence(filters, predictive_powers)  
any_operator_number_presence=presence

# Characteristic plot
characteristic_list, presence, predictive_power=Analyze.calculatePredictivePowerCharacteristic(filters,predictive_powers)
any_characteristic_presence=presence
any_characteristic_power=predictive_power
    
characteristic_list[0]='Mean \n Frequency'
characteristic_list[1]='Band \n Power'
characteristic_list[2]='Medium \n Intensity \n Normalized'
characteristic_list[3]='Medium \n Intensity \n Unormalized'
characteristic_list[4]='Variance'
characteristic_list[5]='Delta \n Band'
characteristic_list[6]='Theta \n Band'
characteristic_list[7]='Alpha \n Band'
characteristic_list[8]='Betha \n Band'
characteristic_list[9]='Gamma 1 \n Sub-Band'
characteristic_list[10]='Gamma 2 \n Sub-Band'
characteristic_list[11]='Gamma 3 \n Sub-Band'


## Characteristics Types
characteristic_type_list, presence, predictive_power=Analyze.calculatePredictivePowerCharacteristicType(filters,predictive_powers)
any_characteristic_type_presence=presence
any_characteristic_type_power=predictive_power

characteristic_type_list=["Non-Rhythmic","Rhythmic"]

# QUantity of Different Types of Activity
different_characteristic_types_list, presence=Analyze.calculateDifferentCharacteristicTypesPresence(filters,predictive_powers)
any_characteristic_type_number_presence=presence


## Mathematical Operators
operators_list,presence,predictive_power=Analyze.calculatePredictivePowerOperators(filters,predictive_powers)
any_operator_presence=presence
any_operator_power=predictive_power

## Window Length
windowlengths_list, presence, predictive_power=Analyze.calculatePredictivePowerWindowLength(filters,predictive_powers)
any_window_presence=presence
any_window_power=predictive_power

## Quantity of Different Simultaneous Multiscales
different_windows_list, presence=Analyze.calculateDifferentWindowsPresence(filters,predictive_powers)
any_windows_number_presence=presence

## Chronology
chronology_list, presence, predictive_power=Analyze.calculatePredictivePowerWindowChronology(filters,predictive_powers)
any_chronology_presence=presence
any_chronology_power=predictive_power   

## Quantity of Different Simultaneous Events
different_chronology_list, presence=Analyze.calculateDifferentChronologyPresence(filters,predictive_powers)
any_chronology_number_presence=presence
any_chronology_number_power=predictive_power


######### Electrodes ####################
 
any_electrode_presence=np.array(any_electrode_presence[0:19])
any_electrode_power=np.array(any_electrode_power[0:19])

any_electrode_presence=any_electrode_presence.tolist()
any_electrode_presence.append(any_electrode_presence[0])

any_electrode_power=any_electrode_power.tolist()
any_electrode_power.append(any_electrode_power[0])


######### Lobes ####################
 
any_lobe_presence=np.array(any_lobe_presence[0:6])
any_lobe_power=np.array(any_lobe_power[0:6])

any_lobe_presence=any_lobe_presence.tolist()
any_lobe_presence.append(any_lobe_presence[0])

any_lobe_power=any_lobe_power.tolist()
any_lobe_power.append(any_lobe_power[0])

######### Hemispheres ####################
 
any_hemisphere_presence=np.array(any_hemisphere_presence[0:3])
any_hemisphere_power=np.array(any_hemisphere_power[0:3])

any_hemisphere_presence=any_hemisphere_presence.tolist()
any_hemisphere_presence.append(any_hemisphere_presence[0])

any_hemisphere_power=any_hemisphere_power.tolist()
any_hemisphere_power.append(any_hemisphere_power[0])

######### Characteristics ####################
 
any_characteristic_presence=np.array(any_characteristic_presence[0:12])
any_characteristic_power=np.array(any_characteristic_power[0:12])
  
any_characteristic_presence=any_characteristic_presence.tolist()
any_characteristic_presence.append(any_characteristic_presence[0])

any_characteristic_power=any_characteristic_power.tolist()
any_characteristic_power.append(any_characteristic_power[0])

######### Operators ####################
 
any_operator_presence=np.array(any_operator_presence[0:12])
any_operator_power=np.array(any_operator_power[0:12])

any_operator_presence=any_operator_presence.tolist()
any_operator_presence.append(any_operator_presence[0])

any_operator_power=any_operator_power.tolist()
any_operator_power.append(any_operator_power[0]) 

######### Window Scales ####################
 
any_window_presence=np.array(any_window_presence[0:6])
any_window_power=np.array(any_window_power[0:6])  

any_window_presence=any_window_presence.tolist()
any_window_presence.append(any_window_presence[0])

any_window_power=any_window_power.tolist()
any_window_power.append(any_window_power[0])
 

######### Chronology Moments ####################
 
any_chronology_presence=np.array(any_chronology_presence[0:5])
any_chronology_power=np.array(any_chronology_power[0:5])
 
any_chronology_presence=any_chronology_presence.tolist()
any_chronology_presence.append(any_chronology_presence[0])

any_chronology_power=any_chronology_power.tolist()
any_chronology_power.append(any_chronology_power[0])


################ Characteristic Histogram ##################

presence=any_characteristic_presence[0:12]
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)
presence=presence+any_characteristic_type_presence[0:2].tolist()
presence.append(0)
presence=presence+np.zeros(2).tolist()
presence.append(0)
presence=presence+any_operator_presence[0:4]
presence.append(0)
presence=presence+[0,0,0,0]

presence_number=np.zeros(12).tolist()
presence_number.append(0)
presence_number=presence_number+any_characteristic_number_presence[0:5].tolist()
presence_number.append(0)
presence_number=presence_number+np.zeros(2).tolist()
presence_number.append(0)
presence_number=presence_number+any_characteristic_type_number_presence[0:2].tolist()
presence_number.append(0)
presence_number=presence_number+[0,0,0,0]
presence_number.append(0)
presence_number=presence_number+any_operator_number_presence[0:4].tolist()



power=any_characteristic_power[0:12]
power.append(0)
power=power+[0,0,0,0,0]
power.append(0)
power=power+any_characteristic_type_power[0:6].tolist()
power.append(0)
power=power+[0,0]
power.append(0)
power=power+any_operator_power[0:4]
power.append(0)
power=power+[0,0,0,0]

def autolabel(rects, xpos='center',font_size=None):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = round(rect.get_height(),2)
        if height >0 and font_size is None:
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom', fontsize='x-small')
        elif height > 0:
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom', fontsize=font_size)

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    
    code from user "cheersmate" from stackoverflow, which can be found in:
    https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
    
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh+0.01)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
    

men_means = presence
women_means = power
ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.60  # the width of the bars


fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(ind , men_means, width,
                label='Presence (0-1)',alpha=0.80)
rects2 = ax.bar(ind, women_means, width,
                label='Predictive Power (0-1)',alpha=0.80)

rects3 = ax.bar(ind, presence_number, width,
                label='Different Elements Presence (0-1)',alpha=0.80)

labels=["Mean. Norm. Freq.","Average Power","Medium Intensity \n Norm.","Medium Intensity \n Not Norm.",
        "Variance", "Delta Band" ,"Theta Band", "Alpha Band", "Betha Band",
        "Gamma-1 Sub-Band","Gamma-2 Sub-Band","Gamma-3 Sub-Band"]
labels.append("")
labels=labels+['1','2','3','4','5']
labels.append("")
labels=labels+["Non-Band Wave","Band-Wave"]
labels.append("")
labels=labels+['1','2']
labels.append("")
labels=labels+["Median","Mean","Variance","Integral"]
labels.append("")
labels=labels+['1','2','3','4']

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('$Pp(gene_{value})$ and $Presence(gene_{value})$ (0-1)')
ax.set_title('Phenotype Feature Study for Patient 53402')
ax.set_xticks(ind)
ax.set_xticklabels(labels,fontsize='x-small',
        rotation=80)
ax.legend(bbox_to_anchor=(0.35, 0.73),fontsize=8)
plt.grid(color='k', alpha=0.10, linestyle='-', linewidth=1)


autolabel(rects1, "center",5)
autolabel(rects3, "center",5)
#autolabel(rects2, "center")


bars=np.arange(len(presence))
heights=presence

barplot_annotate_brackets(0, 11, "Features", bars, heights, dh=0.30,fs=8)

barplot_annotate_brackets(13, 17, "#Different \n Features", bars, heights, dh=0.64,fs=8)

barplot_annotate_brackets(19, 20, "Group of Features", bars, heights, dh=0.27,fs=8)

barplot_annotate_brackets(22, 23, "#Different \n Feature Groups", bars, heights, dh=0.77,fs=8)

barplot_annotate_brackets(25, 28, "Mathematical Operators", bars, heights, dh=0.05,fs=8)

barplot_annotate_brackets(30, 33, "#Different \n Mathematical Operators", bars, heights, dh=0.50,fs=8)

fig.tight_layout()


plt.show()


################ Spatio Temporal ##################

# temporal
presence=any_window_presence[0:5]
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)

presence_number=np.zeros(5).tolist()
presence_number.append(0)
presence_number=presence_number+any_windows_number_presence[0:5].tolist()
presence_number.append(0)

power=any_window_power[0:5]
power.append(0)
power=power+[0,0,0,0,0]
power.append(0)

presence=presence+any_chronology_presence[0:5]
presence.append(0)
presence=presence+np.zeros(5).tolist()

presence_number=presence_number+np.zeros(5).tolist()
presence_number.append(0)
presence_number=presence_number+any_chronology_number_presence[0:5].tolist()

power=power+any_chronology_power[0:5]
power.append(0)
power=power+[0,0,0,0,0]

labels=windowlengths_list[0:5].tolist()
labels.append("")
labels=labels+different_windows_list[0:5].tolist()
labels.append("")

labels=labels+chronology_list[0:5].tolist()
labels.append("")
labels=labels+different_chronology_list[0:5].tolist()


# electrodes
presence.append(0)
presence=presence+any_electrode_presence[0:19]
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)
presence=presence+any_lobe_presence[0:6]
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)
presence=presence+any_hemisphere_presence[0:3]
presence.append(0)
presence=presence+np.zeros(3).tolist()

presence_number.append(0)
presence_number=presence_number+np.zeros(19).tolist()
presence_number.append(0)
presence_number=presence_number+any_electrode_number_presence[0:5].tolist()
presence_number.append(0)
presence_number=presence_number+np.zeros(6).tolist()
presence_number.append(0)
presence_number=presence_number+any_lobe_number_presence[0:5].tolist()
presence_number.append(0)
presence_number=presence_number+np.zeros(3).tolist()
presence_number.append(0)
presence_number=presence_number+any_hemisphere_number_presence[0:3].tolist()

power.append(0)
power=power+any_electrode_power[0:19]
power.append(0)
power=power+[0,0,0,0,0]
power.append(0)
power=power+any_lobe_power[0:6]
power.append(0)
power=power+[0,0,0,0,0]
power.append(0)
power=power+any_hemisphere_power[0:3]
power.append(0)
power=power+[0,0,0]

labels.append("")
labels=labels+electrodes_list[0:19]
labels.append("")
labels=labels+['1','2','3','4','5']
labels.append("")
labels=labels+lobe_list
labels.append("")
labels=labels+['1','2','3','4','5']
labels.append("")
labels=labels+hemisphere_list
labels.append("")
labels=labels+['1','2','3']



men_means = presence
women_means = power
ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.60  # the width of the bars


fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(ind , men_means, width,
                label='Presence (0-1)',alpha=0.80)
rects2 = ax.bar(ind, women_means, width,
                label='Predictive Power (0-1)',alpha=0.80)

rects3 = ax.bar(ind, presence_number, width,
                label='Different Elements Presence (0-1)',alpha=0.80)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('$Pp(gene_{value})$ and $Presence(gene_{value})$ (0-1)')
ax.set_title('Phenotype Temporal and Spatial Study for Patient 53402')
ax.set_xticks(ind)
ax.set_xticklabels(labels,fontsize='x-small',rotation=70)
ax.legend(bbox_to_anchor=(0.35, 0.73),loc=3,fontsize=8)
plt.grid(color='k', alpha=0.10, linestyle='-', linewidth=1)


autolabel(rects1, "center",5)
autolabel(rects3, "center",5)
#autolabel(rects2, "center")

fig.tight_layout()
    
    
bars=np.arange(len(presence))
heights=presence

size_font=8

barplot_annotate_brackets(0, 4, "Window \n Length", bars, heights, dh=0.40,fs=size_font)

barplot_annotate_brackets(6, 10, "#Window \n Lengths", bars, heights, dh=0.47,fs=size_font)

barplot_annotate_brackets(12, 16, "Instants", bars, heights, dh=0.05,fs=size_font)

barplot_annotate_brackets(18, 22, "#Instants", bars, heights, dh=0.37,fs=size_font)

barplot_annotate_brackets(24, 42, "Electrode", bars, heights, dh=0.15,fs=size_font)

barplot_annotate_brackets(44, 48, "#Electrodes", bars, heights, dh=0.55,fs=size_font)

barplot_annotate_brackets(50, 55, "Lobe", bars, heights, dh=0.15,fs=size_font)

barplot_annotate_brackets(57, 61, "#Lobes", bars, heights, dh=0.43,fs=size_font)

barplot_annotate_brackets(63, 65, "Hemisphere", bars, heights, dh=0.05,fs=size_font)

barplot_annotate_brackets(67, 69, "#Hemispheres", bars, heights, dh=0.45,fs=size_font)

plt.show()





