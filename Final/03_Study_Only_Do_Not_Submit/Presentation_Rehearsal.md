# Presentation rehearsal script – Presenter A and Presenter B

STUDY ONLY – do not submit. This matches the 20-slide deck in 01_Submission (the same text is in each slide's speaker notes). Replace [Presenter A name] / [Presenter B name] with your real names.

| | Presenter A | Presenter B |
|---|---|---|
| Slides | 1–9 | 10–18 (19–20 backup) |
| Target time | ≈ 5 min 40 s | ≈ 5 min 45 s |
| Focus | Why, what data, how we prepared and evaluated it | What we found, how sure we are, what it means |

Total ≈ 11 min 25 s: leaves 3.5 minutes of buffer under the 15-minute group limit and keeps each person under 6 minutes.

## Delivery tips (Delivery = 10 of 30 marks)

- Speak slower than feels natural (~130 words per minute). Pause after every big number.
- Point at the chart/table element you are describing. Look at the camera, not the notes.
- Handover line (end of slide 9) and thank-you line (start of slide 10) are scripted — rehearse them together.
- Rehearse with a timer at least three times; record once and listen back for volume and filler words.
- Know the three headline numbers without notes: 95.91% F1, 42 missed / 38 false alarms, 94.58% CV F1.

## Slide 1 — Title – Phishing Website Detection

**PRESENTER A (about 25 s)**

Hello everyone. We are Group [No.] from [Campus]. I am [Presenter A name], and my teammate is [Presenter B name].

Our project is phishing website detection. We compared machine learning and deep learning models on 11,055 website records from the UCI Machine Learning Repository to see which model separates phishing websites from legitimate ones most reliably.

I will cover the motivation, the dataset, our preprocessing and our evaluation design. [Presenter B name] will then present the results, the testing and our conclusion.

(Cover artwork is an AI-generated conceptual illustration.)

## Slide 2 — Roadmap and who presents what

**PRESENTER A (about 20 s)**

Here is our roadmap. I will take slides 3 to 9: why phishing matters, the data, how we prepared it and how we designed a fair evaluation, and the eleven models we trained.

[Presenter B name] takes slides 10 to 18: which model won in cross-validation, how every model performed on unseen test data, whether the differences are statistically significant, what the deep learning models taught us, and our conclusion.

## Slide 3 — Why phishing detection?

**PRESENTER A (about 40 s)**

Why phishing? The Anti-Phishing Working Group observed about 3.8 million phishing attacks in 2025, with more than 850 thousand in the last quarter alone. At that scale, manual blocklists cannot keep up, so we want models that learn from the characteristics of a website.

The two possible mistakes are not equal. Missing a phishing site can lead to stolen credentials, while a false alarm blocks a legitimate site. That is why we report precision, recall and F1, not just accuracy.

Our research question is: which machine learning or deep learning model best separates phishing from legitimate websites when it is evaluated fairly on patterns it has never seen?

## Slide 4 — Dataset: UCI Phishing Websites

**PRESENTER A (about 40 s)**

Our dataset is the UCI Phishing Websites dataset by Mohammad and McCluskey. It has 11,055 website records and 30 input features with no missing values.

The features were engineered by the dataset authors and fall into four groups: address-bar features such as whether the site has a valid HTTPS certificate; abnormal-content features such as where links and forms point; HTML and JavaScript tricks such as hidden iframes; and domain-reputation features such as domain age and web traffic.

Each feature is a category: minus one looks phishing-like, zero is suspicious, and plus one looks legitimate. The target is phishing or legitimate; we made phishing the positive class, so all precision, recall and F1 values measure phishing detection. About 44 percent of records are phishing, so the classes are reasonably balanced.

## Slide 5 — Exploratory analysis: two indicators stand out

**PRESENTER A (about 45 s)**

Before modelling we explored the data. On the left, the classes are moderately balanced: 56 percent legitimate and 44 percent phishing. That means accuracy is meaningful, but we still used phishing F1 to choose models because it balances missed attacks and false alarms.

On the right is mutual information, which measures how much knowing a feature reduces uncertainty about the label. We calculated it on the development data only, so the test set was never used.

Two indicators dominate: SSLfinal_State, which describes the HTTPS certificate, and URL_of_Anchor, the share of links pointing to other domains. After those, the signal drops sharply, so a good model has to combine many weaker indicators. We also checked a correlation heat map, which is in our results folder.

## Slide 6 — Data quality: repeated inputs need careful splitting

**PRESENTER A (about 40 s)**

Our data audit found something important. Of 11,055 rows, only 5,785 have a unique combination of feature values. The other 5,270 rows repeat a pattern that already exists, and 64 patterns even appear with both labels.

Why does this matter? With an ordinary random split, the same pattern can land in both training and test sets. The model is then tested on examples it has effectively memorised, and the scores look better than they really are. This is data leakage.

Our fix was to hash every pattern into a group and keep each group entirely on one side of every split. We kept the conflicting records rather than deleting them to make the scores look better.

## Slide 7 — Preprocessing and feature engineering

**PRESENTER A (about 40 s)**

Here is our preprocessing pipeline. First, we mapped the target so phishing is class 1. Second, we grouped identical rows, as I just explained.

Third, encoding. The feature values are categories, so for the tree, forest, neural networks, logistic regression, k-nearest neighbours and SVM we one-hot encoded them, turning 30 features into 90 binary columns. Naive Bayes and gradient boosting take the category codes directly, and the LSTM and RNN turn each feature value into a token with a learned embedding.

Fourth, as a feature-engineering extension, we tested keeping only the top 10 features by mutual information, selected inside each training fold.

Every step is inside a scikit-learn pipeline, so nothing is learned from validation or test data. No scaling or imputation was needed because the inputs are already bounded and complete.

## Slide 8 — Evaluation design: choose first, test once

**PRESENTER A (about 40 s)**

Our evaluation design follows one rule: choose first, then test once.

Step one, we reserved 20 percent of the data, 2,210 records, as a test set, keeping groups intact.

Step two, we tuned every model on the remaining 8,845 records with three grouped cross-validation folds. Every model saw exactly the same folds, and we tuned for phishing F1.

Step three, the model with the highest mean cross-validation F1 was selected before we looked at any test result.

Step four, we evaluated every model once on the test set with accuracy, precision, recall, F1, ROC AUC and the confusion matrix, and added a bootstrap confidence interval and McNemar significance tests. The code checks automatically that no pattern crosses a boundary.

## Slide 9 — Eleven models, three families

**PRESENTER A (about 50 s)**

We trained eleven models in three families.

The four required models: a decision tree, which learns if-then splits; a random forest of 150 trees that vote; categorical naive Bayes, which combines per-feature probabilities assuming independence; and a multilayer neural network with two hidden layers.

To make the study comprehensive we added four more algorithms: logistic regression as a linear baseline, k-nearest neighbours, a support vector machine with an RBF kernel, and gradient boosting.

For deep learning we added a deeper three-layer network, and, as the brief suggests, an LSTM and a simple RNN built in PyTorch, which read the 30 indicators as a sequence of tokens.

We also include a majority-class baseline. Every model was tuned on the same grouped folds with fixed random seeds.

Now I will hand over to [Presenter B name], who will show you how these models performed.

## Slide 10 — Cross-validation selected the Random Forest

**PRESENTER B (about 40 s)**

Thank you, [Presenter A name]. I will start with model selection.

This chart shows mean F1 across the three grouped development folds. Note that the axis starts at 85 percent, so the differences look larger than they are.

The random forest had the highest cross-validation F1, 94.58 percent, so it became our primary model, and this decision was fixed before we opened the test set. Gradient boosting was effectively tied, only 0.01 points behind, and the deep ANN, SVM and LSTM were close. Naive Bayes and the single decision tree were weakest at around 91.7 percent.

The forest scored 98.8 percent on its training folds, so there is a normal gap between training and validation, which we monitor rather than hide.

## Slide 11 — Test performance: 12 models, same 2,210 records

**PRESENTER B (about 50 s)**

Here is every model on the same 2,210 unseen test records. The random forest also leads on the test set, with 96.38 percent accuracy, 95.71 percent recall and 95.91 percent F1. Its 95 percent confidence interval for F1 runs from about 94.3 to 97.3 percent.

The deep ANN had the highest precision, 97.18 percent, meaning its alerts are the most trustworthy, but it caught fewer phishing sites. Gradient boosting tied the forest on ROC AUC at 0.993.

Every learned model beats the majority baseline, which always predicts legitimate. It gets 55.7 percent accuracy but catches zero phishing. That is exactly why accuracy alone is not enough.

## Slide 12 — ROC curves: ranking quality across all thresholds

**PRESENTER B (about 30 s)**

ROC curves show the trade-off between catching phishing and raising false alarms at every possible threshold. We zoomed into the top-left corner because that is where the models differ.

The random forest, shown in the thick teal line, and gradient boosting both reach an AUC of 0.993. Every learned model is above 0.95, far from the 0.5 of random guessing.

AUC measures ranking, not a final decision. A real system would pick a threshold based on how costly a missed phishing site is compared with a false alarm.

## Slide 13 — The forest’s 80 errors still matter

**PRESENTER B (about 40 s)**

High scores can hide individual errors, so here is the forest's confusion matrix. It caught 937 of 979 phishing sites and missed 42. It wrongly flagged 38 of 1,231 legitimate sites. So recall is 95.7 percent and precision is 96.1 percent.

In security terms, about 4 percent of phishing sites would slip through, and about 3 percent of legitimate sites would be blocked.

On the right you can see that models make different kinds of mistakes. The deep ANN raises the fewest false alarms, only 26, but it misses 83 phishing sites, twice as many as the forest. For a security tool where a missed attack is costly, the forest's balance is preferable.

## Slide 14 — Is the forest really better? Statistical testing

**PRESENTER B (about 40 s)**

Are these differences real or just luck? We tested this in two ways.

First, a bootstrap confidence interval: we resampled whole groups of the test set a thousand times. The forest's F1 interval is about 94.3 to 97.3 percent.

Second, McNemar's test, which looks only at the records where two models disagree. For example, against the SVM, the forest was right and the SVM wrong on 43 records, but the reverse happened only 21 times. After a Holm correction for eleven comparisons, the forest is significantly better than every other model at the 1 percent level.

We are careful with this claim: the intervals of close rivals overlap, repeated rows are not fully independent, and this is one split of one dataset.

## Slide 15 — Deep learning: what we learned

**PRESENTER B (about 40 s)**

What did deep learning add? The deeper three-layer network improved on the required two-layer ANN, from 93.6 to 94.3 percent F1. The bidirectional LSTM and RNN, built in PyTorch, reached about 93 percent. None of them beat the random forest.

Our saved loss curves show the networks did converge, so this is not simply under-training.

The reason is the data: 30 categorical indicators with no natural order. An LSTM is designed for sequences such as text or time series, so here it has no temporal pattern to exploit, while trees capture threshold-style interactions directly. This matches published research showing tree ensembles often outperform deep networks on medium-sized tabular data.

## Slide 16 — What drives the forest’s decisions?

**PRESENTER B (about 45 s)**

To explain the forest, we used permutation importance: we shuffle one indicator and measure how much test F1 drops. The same two indicators from the exploratory analysis dominate: SSLfinal_State and URL_of_Anchor each cost more than 11 F1 points when shuffled. Prefix-suffix in the domain, web traffic and links in tags follow.

We then tested a simpler model using only the top ten features by mutual information. F1 fell by 2.4 points, and false alarms doubled from 38 to 78. So although most signal comes from a few indicators, the remaining features still add useful information. Importance here shows association, not causation.

## Slide 17 — Conclusion, limitations and future work

**PRESENTER B (about 45 s)**

To conclude: our answer to the research question is that, on this benchmark, a random forest separates phishing from legitimate websites most reliably. It was selected by cross-validation, reached 95.9 percent F1 and 96.4 percent accuracy on unseen data, and was significantly better than all ten rival models.

Deeper networks, the LSTM and the RNN did not beat tree ensembles on this tabular data, and although two indicators drive most decisions, all 30 features add value.

The limitations are important: the data are from 2015, features were pre-extracted, we could not separate by domain or time, and we used one outer split.

For future work, we would test on recent independent data, choose a threshold based on error costs, and extract features live from URLs.

## Slide 18 — References

**PRESENTER B (about 15 s)**

These are our references, including the dataset, the phishing trends report, and the methods we used. All code, results and figures are reproducible from our project folder. Thank you for listening. We are happy to take questions.

## Slide 19 — Backup: hyperparameter search

**BACKUP – use only if asked. GridSearchCV refits the best configuration on all development rows. Seeds fixed at 565. For the recurrent models, early stopping on each grouped validation fold picked the epoch count; the final model is refit on all development rows for the median epoch count. Because the epoch count was chosen on validation data, the recurrent CV F1 is slightly optimistic.**



## Slide 20 — Backup: metric formulas with the forest’s numbers

**BACKUP – use only if asked about metrics or encoding. F1 penalises a low value in either precision or recall. AUC does not certify calibrated probabilities.**


