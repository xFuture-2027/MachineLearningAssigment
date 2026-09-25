# Likely questions and model answers

STUDY ONLY. "Preparedness" (6.5 marks) is judged on whether you can explain your own work confidently. Each presenter should own the questions for their half, but both should know the headline numbers.

## Presenter A – data and method

**Why this dataset?**
It is a public, labelled, well-documented UCI benchmark on a real security problem, with 11,055 records and no missing values. It differs from the Telco data we used in Assessment 2, and its errors (missed phishing vs false alarm) are easy to interpret.

**What does a value like URL_Length = 1 mean?**
It is a category, not a measurement: +1 means "legitimate-like" (a short URL), 0 suspicious, −1 phishing-like. That is why we one-hot encode instead of treating the codes as numbers.

**Why did you group repeated rows? Isn't that unusual?**
Only 5,785 of 11,055 rows are unique. With a random split the same pattern can be in training and test, so the model is marked on answers it memorised – data leakage (Kaufman et al., 2012). Grouping keeps each pattern on one side, so our test score reflects unseen patterns.

**Why keep the 64 conflicting patterns?**
Removing them would make the task artificially easier and inflate scores. Real data has label noise, so we kept it and report it as a limitation.

**Why F1 for model selection instead of accuracy?**
F1 balances precision (trustworthy alerts) and recall (phishing caught). The majority baseline gets 55.7% accuracy while catching zero phishing, which shows accuracy alone can mislead.

**How did you avoid leakage in feature selection?**
Mutual-information selection and all encoding are inside the scikit-learn Pipeline, so they are re-fitted on each training fold only. The test set is never used for any choice.

**Why these extra models?**
The brief allows "any other algorithms as you see fit" and suggests LSTM/RNN. Logistic Regression is a linear baseline, k-NN a similarity method, SVM a kernel method and Gradient Boosting the other main tree ensemble, so every major family is represented.

**How does the LSTM use tabular data?**
Each of the 30 features becomes a token (feature × value = 90 possible tokens) with a learned 16-dimensional embedding; a bidirectional LSTM reads the 30 tokens and the average hidden state feeds a sigmoid output. Early stopping on the grouped validation fold picks the number of epochs.

## Presenter B – results and conclusions

**Why did Random Forest win?**
Averaging 150 decorrelated trees reduces variance while still capturing non-linear interactions between categorical indicators such as SSL state and link destinations. It had the best CV F1 (94.58%) and the best test F1 (95.91%).

**Gradient Boosting almost tied in CV. Why not choose it?**
Our rule was fixed in advance: highest mean CV F1 wins. The forest was 0.01 points ahead, so it was selected before the test. On the test set the forest was also significantly better (McNemar p < 0.001).

**Are the differences statistically significant?**
Yes on this test set: exact McNemar tests with Holm correction give p < 0.01 against all ten rivals. Caveats: confidence intervals of close rivals overlap, repeated rows mean the test rows are not fully independent, and it is one split of one dataset.

**What does the 95% confidence interval mean?**
We resampled whole input-pattern groups of the test set 1,000 times and recomputed F1. 95% of those F1 values fell between 94.3% and 97.3%.

**Why didn't deep learning win?**
There are only 30 categorical inputs with no natural order, so an LSTM has no sequence structure to exploit, and tree ensembles capture threshold-style interactions directly. This matches Grinsztajn et al. (2022). The deeper ANN did improve on the required ANN (94.27% vs 93.57% F1) and gave the best precision (97.18%).

**What is permutation importance?**
Shuffle one feature on the test set and measure how much F1 drops. SSLfinal_State (−13.6 points) and URL_of_Anchor (−11.8) matter most. It shows association with predictions, not causation.

**Why did the top-10 model do worse?**
Even weak features add information: F1 fell 2.40 points and false alarms doubled (38 → 78).

**Is this model ready to deploy?**
No. The data are from 2015, features were pre-extracted, and there is no domain or time separation. Next steps: recent independent data, a cost-based decision threshold and live feature extraction.

**What would you do differently?**
Repeated outer cross-validation for more robust estimates, larger hyperparameter searches, probability calibration, and cost-sensitive threshold tuning.

## Numbers to know by heart

| Fact | Value |
|---|---|
| Records / features | 11,055 / 30 |
| Phishing share | 44.3% (4,898) |
| Unique patterns / repeats / conflicting | 5,785 / 5,270 / 64 |
| Development / test rows | 8,845 / 2,210 |
| RF CV F1 | 94.58% (GB 94.57%) |
| RF test accuracy / F1 / AUC | 96.38% / 95.91% / 0.993 |
| RF missed / false alarms | 42 / 38 |
| Deep ANN / LSTM / RNN test F1 | 94.27% / 93.03% / 92.82% |
| Top-10 subset F1 | 93.50% (−2.40 pp) |
