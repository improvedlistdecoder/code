# Improving the List Decoding Version of the Cyclically Equivariant Neural Decoder

An implementation of described in "Improving the List Decoding Version of the Cyclically Equivariant Neural Decoder" (Pytorch implementation) 

## Abstract

The cyclically equivariant neural decoder was recently proposed in [Chen-Ye, International Conference on Machine Learning, 2021] to decode cyclic codes. In the same paper,a list decoding procedure was also introduced for two widely used classes of cyclic codes—BCH codes and punctured ReedMuller (RM) codes. While the list decoding procedure significantly improves the Frame Error Rate (FER) of the cyclically equivariant neural decoder, the Bit Error Rate (BER) of the list decoding procedure is even worse than the unique decoding algorithm when the list size is small. In this paper, we propose an improved version of the list decoding algorithm for BCH codes and punctured RM codes. Our new proposal significantly reduces the BER while maintaining the same (in some cases even smaller) FER. More specifically, our new decoder provides up to 2dB gain over the previous list decoder when measured by BER, and the running time of our new decoder is 15% smaller. 

## Step1: Install the requirements
- Pytorch == 1.6.0

- Python3 (Recommend Anaconda)

- Matlab 2018b with Communications Toolbox 7.0


```bash
pip install -r requirements.txt
```

## Step2: Produce parity check matrix and generator matrix
For BCH code, run ```GenAndPar.m```   
For punctured RM code, run ```GenPolyRM.m```  
This gives you the coefficients of generator polynomial and parity check polynomial. For example, running ```GenAndPar.m``` with parameters n=63 and k=45 gives you

```
 n = 63 
 k = 45 
 Generator matrix row: 
1 1 1 1 0 0 1 1 0 1 0 0 0 0 0 1 1 1 1 
 Parity matrix row: 
1 1 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 0 1 0 0 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 1 0 1 1 1 1 0 0 1 1 
```

Another example: running ```GenPolyRM.m``` with parameters m=6,r=3 gives you

```
Punctured Reed-Muller codes parameters are 
 m = 6 
 r = 3 
 n = 63 
 k = 42 
 Generator matrix row: 
1 1 0 1 0 0 0 1 1 1 1 1 1 0 0 1 1 0 1 0 0 1 
 Parity matrix row: 
1 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 0 0 0 0 1 0 1 1 1 
```

After obtaining these coefficients, we use ```util.translation.py``` to produce the parity check matrix and the generator matrix. See ```BCH_H_63_45.txt``` for an example of parity check matrix of BCH(63,45) and ```BCH_G_63_45.txt``` for an example of generator matrix of BCH(63,45).Then put them in data file.

## Step3:Produce the permutations used in list decoding
- Set the parameter m in ```GFpermutation.m``` (Example: for BCH(63,45), pick m=6 because 2^6=64).  
- Run ```GFpermutation.m``` in Matlab with Communications Toolbox and we get the ```GFpermutation.txt```
- Put the path in config to get the permutation matrix used in ```train.py```. Example: For BCH(63,45), put the path of ```GFpermutation.txt``` in ```config.bch6345_p=4.yaml```

## Step4: Model training
1. Set the config file
   - bch6345_p=4 is the config name for BCH(63,45) and p is the number of permutations.You can change it to other names for other codes in the config file,such as bch6324_p=16,bch12764_p=128,rm12764_p=128
   - Hyperparameters such as the learning rate and batch size, you can set in the config file

2. Run ```python -m app.train bch6345_p=4``` to train the model
    - Train with GPU. For 2 hours of training with 1080Ti 11GB GPU ,around epoch 11 you should get the model ```BCH_63_45_11.pth``` in save file.
    - The training results are in the log file.

## Step5: Model testing
1. Set the config file
    Put the path of test model name (```./save/BCH_63_45/p=4/BCH_63_45_11.pth```) in the config (```config.bch6345_p=4.txt```). 
2. Run ```python -m app.test bch6345_p=4``` to get the Bit Error Rate results and Frame Error Rate results in the log file.

Reproduce the performance of the model in our paper for BCH(63,45) when p=4:

| SNR | BER       | FER       | BER/FER |
|-----|-----------|-----------|---------|
| 4   | 0.0040096 | 0.0712600 | 0.0562  |
| 5   | 0.0004878 | 0.0104000 | 0.0469  |
| 6   | 0.0000204 | 0.0006100 | 0.0334  |

