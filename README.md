# Exploring Cost-Effective Alternatives to Depth of Anesthesia Monitoring
 
## Author
**Name:** Hao-Tien Yu, Guan-Yu Chen, Chien-Yi Wen  
**Student Number:** 110030038, 110012013, 111090001

## Abstract
This study explores three methods for predicting the depth of anesthesia: Analysis of Variance (ANOVA), Machine Learning (ML), and ML with additional parameters. Data preprocessing involved using ASR to remove noise, splitting data into 5-second samples, and extracting features: Spectrum-bis, FFT-EEG, and SampEn-EEG. ANOVA identified significant differences in EEG data segments. ML models, including SVM, Random Forest, and Gradient Boosting Machine (GBM), were used for prediction. Results showed that both Random Forest and GBM accurately predicted BIS values, with GBM demonstrating smaller MAPE and better performance. Using more parameters did not significantly improve accuracy, as features like spectrum-bis provided limited predictive power, and raw data showed the lowest accuracy. Overfitting was noted due to an excess of normal range data, clustering predictions in the safe range. This research contributes to developing efficient and economical DoA monitoring techniques.

## Introduction
This project explores three ML-based/Statistic methods for predicting the depth of anesthesia using EEG signals: Analysis of Variance (ANOVA), Machine Learning, and Machine Learning with additional parameters. The goal is to evaluate and compare these methods to figure out a cost-effective alternatives to DoA Monitoring.

## Model Framework

### Preprocessing
![image](https://hackmd.io/_uploads/rkn8FQJBR.png)

#### Data Cleansing
Because our raw EEG data only has one channel, we found that ICA could not be performed. Although ASR can be executed, it can only remove signals outside the threshold.



#### Cut data:
Since the bis data samples every 5 seconds, we have to make sure our 128-sampling-rate EEG aligns to it. So we split the original data and take every 5s of bis/EEG as a testcase.

#### Feature Extraction
We extract 3 type of features in total:
* **Spectrum-bis**: This data is derived from transforming the original "bis" variable into deviations from the 40-60 range. Define 40-60 as 0, 60-70 as +1(each subsequent interval upward adds +1), 30-40 as -1(each subsequent interval downward subtracts -1). Therefore, there are a total of 9 labels ranging from -4 to +4.
* **FFT-EEG**: This is derived from using FFT to transform every EEG in testcase into Frequency domain.
* **SampEn-EEG**: We use sample enthropy as a feature for every EEG in testcase to consider its uncertainty.


### Prediction

#### Machine Learning
![image](https://hackmd.io/_uploads/HyphoN1H0.png)
- **Input/Output Mechanisms:** Uses FFT-EEG from case1~case18 as training dataset and case19~21 as validation dataset, providing *"MAPE(Mean Absolute Percentage Error) as output"*.
- **Machine Learning Models Utilized:** SVM, Random Forest, and Gradient Boosting Machine.
- **Validation:** Uses case22~24 as test dataset to calculate the MAPE.

#### Machine Learning with additional features (SampEn, FFT)
![image](https://hackmd.io/_uploads/ryZjSUJBC.png)
- **Input/Output Mechanisms:** Using 80% of FFT-EEG, Raw-EEG, and SampEn-EEG (a total of 6 permutations) to predict Spectrum-bis (because we found that predicting the original bis had poor accuracy), and outputting the *"True-bis vs. Predicted-bis plot against the Sample index"* and the *"confusion matrix"*.
- **Machine Learning Models Utilized:** Random Forest only.
- **Validation:** Using 20% of Spectrum-bis to validate the accuracy.

## Addition
### ANOVA
- **Input/Output Mechanisms:** The model uses Raw-EEG as input and provides ANOVA statistical metrics such as F-values and P-values as outputs, indicating the significance of differences between experimental conditions.
- **Analysis Method:** The model applies ANOVA (Analysis of Variance) to determine significant differences between different conditions. This statistical method tests whether there are any statistically significant differences between the means of independent (unrelated) groups. By comparing the variances within and between groups, ANOVA helps in identifying whether the observed differences are greater than what might be expected due to random variation alone.


## Usage
1. **Environment and Dependencies:**
   - Python 3.8 or later / matlab R2023
   - Required libraries: please refer to requirements.txt
2. In *Machine Learning with additional features* Section, we use a 40-cores cluster to execute model training.
3. Instruction to execute the code:
    * Machine Learning:
        - Open your matlab.
        - Change your directory in the .m file
        - Execute
    * Machine Learning:
        - Open your IDE
        - Use `python XXX.py` to execute it. You don't need to change the directroy in the program
    * ANOVA
        - Open your IDE
        - Use `python ANOVA_case1.py` to execute it. You can change to any case from case1~case24 in the directory


## Results
- **Machine Learning** 
    - Random forest and gradient boosting machine both show generally correct trends in predicting BIS values. However, the gradient boosting machine has larger fluctuations in the peak regions, closer to the true BIS values, resulting in a smaller MAPE. We think the gradient boosting machine should be used to predict BIS values, as a smaller MAPE indicates better model performance.
- **Machine Learning (More Parameters)** 
    - We found that simply using bis data resulted in low accuracy, with less than 20%. Even when using spectrum-bis blurred values, the accuracy was only around 40% to 50%, indicating that these features may not provide meaningful predictions. Comparatively, although more features can improve accuracy, when comparing on a level basis (e.g., using only SampEn/Raw/FFT or combining two of them), using Raw data had the lowest accuracy among the three features. Additionally, we observed in the confusion matrix that the predicted values were mostly concentrated in the -1 to 1 range (safe range ±10). We attribute this to an overfitting model due to an excess of data in the normal range, causing the predictions to cluster in this area.
- **ANOVA**
    - High F-values and Low P-values: These results indicate that there are significant differences between BIS and EEG data segments. This suggests that the BIS and EEG data in these segments do not originate from the same distribution, reflecting different states or conditions.
    - Similar State Identification: Segments with lower F-values show smaller differences, indicating they might reflect similar or identical conditions.
    - The ANOVA analysis in this study was instrumental in determining whether there were significant differences between segments of BIS and EEG data. By using ANOVA, we were able to identify which segments reflect different states and which segments might represent similar conditions. This deeper understanding of the data can help in further studies and analyses of BIS and EEG characteristics.


## References
* Liu, Q., Ma, L., Fan, S. Z., Abbod, M. F., & Shieh, J. S. (2018). Sample entropy analysis for the estimating depth of anaesthesia through human EEG signal at different levels of unconsciousness during surgeries. PeerJ, 6, e4817. https://doi.org/10.7717/peerj.4817
* Akhtar, M. T., Mitsuhashi, W., & James, C. J. (2012). Employing spatially constrained ICA and wavelet denoising, for automatic removal of artifacts from multichannel EEG data. Signal Processing, 92(2), 401-416. https://doi.org/10.1016/j.sigpro.2011.08.005
