# Handwritten-digits-Classification
## Overview
Hand written digit recognition using Logistic Regression, SVM, Deep Neural Network (Lenet-5 architecture) for MINST dataset.

## To run the code
### For Deep Neural Net
```bash
python3 main.py --Method Lenet
```
-  Parameters  
    - Method - Classifiers. *Default :- 'Lenet'*

### For Logistic Regression
```bash
python3 main.py --Method LR --DimRed LDA 
```
-  Parameters  
    - Method - Classifiers. *Default :- 'Lenet'*
    - DimRed - Dimensionality Reduction technique. Option : 'PCA/LDA' *Default :- 'PCA'* 

### For SVM
```bash
python3 main.py --Method SVM --DimRed LDA --Kernel Polynomial
```
-  Parameters  
    - Method - Classifiers. *Default :- 'Lenet'*
    - DimRed - Dimensionality Reduction technique. Option : 'PCA/LDA' *Default :- 'PCA'* 
    - DimRed - kernel for Kernel SVM. Option : 'Polynomial/RBF' *Default :- 'Linear'* 

## Results
### LeNet-5 
