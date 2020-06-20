# minority-type-imbalanced

## Command Line Arguments  
input file  
label column name  
k value (for kNN creation)  
using header  
save path  
file description  
read/write (read or write, for storing/retrieving the kNN DataFrame)  
read/write path  
currently unused  
sampling methods (all remaining parameters chosen from: none, undersample, oversample, smote, smotePlus)  
  
  
### Example:  
~/covtype/covtype10k.csv label 5 yes ~/test/results covtype10k write ~/covtype10k/minorityDF 1 none,smote  

## The base class:  
edu.vcu.sleeman.Classifier  

## Datasets  

Experiements have been performed using the following datasets:  
  
### From the UCI Machine Learning Repository:  
https://archive.ics.uci.edu/ml/datasets/covertype  
https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT  
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.  
  
### Traffic Violations:   
https://catalog.data.gov/dataset/traffic-violations-56dda  
  
### SEER
SEER: https://seer.cancer.gov/  
Cancer type predictions were performed on data the SEER cancer registry. Using that data will require requesting access from SEER and agreeing to their terms of use.  
  
### Intel Sensors:  
data/sensors.csv  
The original source (http://db.csail.mit.edu/labdata/labdata.html) from the Intel Berkely Research Lab appears to be offline so the version of the data used has been saved in the data directory of this repository.  

