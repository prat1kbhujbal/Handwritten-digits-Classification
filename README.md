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
### Logistic Regression

PCA | LDA 
:-:|:-:|
![env](./Results/lr_pca_acc.png) | ![env](./Results/lr_lda_acc.png) 
![env](./Results/lr_pca_cm.png) | ![env](./Results/lr_lda_cm.png) 

### LeNet-5 
Accuracy | Confusion Matrix
:-:|:-:
![env](./Results/lenet.png) | ![env](./Results/cm_lenet.png) 

### SVM

Kernel | PCA | LDA
:-:|:-:|:-:
| Linear     |![env](./Results/SVM_PCA_lin.png) | ![env](./Results/SVM_LDA_lin.png) 
| Polynomial |![env](./Results/SVM_PCA_poly.png) | ![env](./Results/SVM_LDA_poly.png) 
| RBF        |![env](./Results/SVM_PCA_rbf.png) | ![env](./Results/SVM_LDA_rbf.png) 

