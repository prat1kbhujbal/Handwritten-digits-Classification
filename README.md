# Handwritten-digits-Classification
## Overview
Hand written digit recognition using Logistic Regression, kernel SVM, Deep Neural Network (Lenet-5 architecture) with PCA/LDA dimensionality reduction for MINST dataset.

## To run the code
### Deep Learning
```bash
python3 main.py --Method Lenet
```
-  Parameters  
    - Method - Classifiers. *Default :- 'Lenet'*

### Logistic Regression
```bash
python3 main.py --Method LR --DimRed LDA 
```
-  Parameters  
    - Method - Classifiers. *Default :- 'Lenet'*
    - DimRed - Dimensionality Reduction technique. Option : 'PCA/LDA' *Default :- 'PCA'* 

### SVM
```bash
python3 main.py --Method SVM --DimRed LDA --Kernel Polynomial
```
-  Parameters  
    - Method - Classifiers. *Default :- 'Lenet'*
    - DimRed - Dimensionality Reduction technique. Option : 'PCA/LDA' *Default :- 'PCA'* 
    - DimRed - kernel for Kernel SVM. Option : 'Polynomial/RBF' *Default :- 'Linear'* 

## Results
### LeNet-5 
Accuracy | Confusion Matrix
:-:|:-:
![env](./Results/lenet.png) | ![env](./Results/cm_lenet.png) 

### Logistic Regression

Dim Red. | Accuracy | Confusion
:-:|:-:|:-:
| PCA |![env](./Results/lr_acc_loss_pca.png) | ![env](./Results/lr_pca_cm.png) 
| LDA |![env](./Results/lr_acc_loss_lda.png) | ![env](./Results/lr_lda_cm.png) 

### SVM

Kernel | PCA | LDA
:-:|:-:|:-:
| Linear     |![env](./Results/SVM_PCA_lin.png) | ![env](./Results/SVM_LDA_lin.png) 
| Polynomial |![env](./Results/SVM_PCA_poly.png) | ![env](./Results/SVM_LDA_poly.png) 
| RBF        |![env](./Results/SVM_PCA_rbf.png) | ![env](./Results/SVM_LDA_rbf.png) 

