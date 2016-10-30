# Udacity DAND - Intro to Machine Learning
Submission by Giacomo Sarchioni

## Introduction
This README file aims at guiding the reader in the understanding of my 
submission for the **Intro to Machine Learning Project**
of *Udacity Data Analyst Nanodegree*.

In this document I am going to provide:
* instructions on how to navigate through this repository;
* answers to the *free-response* questions asked [here](https://docs.google.com/document/d/1NDgi1PrNJP7WTbfSUuRUnz8yzs5nGVTSzpO7oeNTEWA/pub?embedded=true).

## Instructions
This repository is built as follows:
* *data*, i.e. a folder with original dataset, dumped dataset and classifier;
* *eda*, i.e. a folder with the EDA analysis I have performed on the dataset;
* *notebooks*, i.e. a folder with some Jupyter notebooks. The *Classification Full* notebook is very comprehnsive and covers all the steps I took while building the classifier;
* *scripts*, i.e. main scripts;
* *tools*, i.e. supporting scripts.


## Free-response questions
### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it

The goal of this project is that of building a model that classifies an Enron employee as a *Person of Interest* (`poi`), i.e. a person who was involved in the Enron accounting scandal. The classification is binary,
i.e. a person is either a poi or is not.
<br>In this context, a ML algorithm is a great tool to build a predictive model, withouth being constrained by any parametric model. ML also allows to evaluate the performance/quality of the predictions and, most importanly, to optimise the model on specific performance metrics.
<br>The dataset is a Python dictionary, where for every 
key (i.e. the name of an employee) there is a dictionary of values,
including financial and message-related variables. The original 
dataset is made of 146 observations, of which **only 18** are
actual poi. The dataset, therefore, is very unbalanaced.
<br>There are 14 financial variables (e.g. salary, bonust, etc.), 6 messages variables (e.g. number of emails sent, number of emails received, etc.) and 1 labelling feature (poi or non poi).
<br>
<br>
As I showed in the `Outlier Identification.html` file (available in the folder *notebook*), I have identified six outliers which I have 
completely removed from the final dataset. They are:

| Name                          | Rational for removal                                                        |
|-------------------------------|---------------------------------------------------------------|
| TOTAL                         | This observation is actually the sum of all the previous ones |
| SHAPIRO RICHARD S             | Incredibly high number of messages sent                       |
| KAMINSKI WINCENTY J           | Incredibly high number of messages received                   |
| KEAN STEVEN J                 | Incredibly high number of messages received                   |
| LOCKHART EUGENE E             | Apart from the name, there is no data for this person.        |
| THE TRAVEL AGENCY IN THE PARK | It doesn't seem an Enron employee :)                          |

<br>

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not?*

The variables I ended up using are:


| Variable                       | New | Definition                                                                                                                                |
|--------------------------------|-----|-------------------------------------------------------------------------------------------------------------------------------------------|
| `poi`                          | No  |  Boolean, 0 if not POI, 1 if POI                                                                                                          |
| `to_messages`                  | No  | Float, number of messages received                                                                                                        |
| `expenses`                    | No  | Float, amount of expenses (USD)                                                                                                           |
| `from_poi_to_this_person`      | No  | Float, number of messages received from POI                                                                                               |
| `shared_with_poi_ratio`        | Yes | Float, number of messages received and shared with at least one POI as a proportion of total messages received                            |
| `shared_receipt_with_poi`      | No  | Float, number of messages received and shared with at least one POI                                                                       |
| `other`                        | No  | Float, other financial variable (USD)                                                                                                     |
| `to_poi_ratio`                 | Yes | Float, number of messages sent to POI as a proportion of total messages sent                                                              |
| `bonus`                        | No  | Float, bonus (USD)                                                                                                                        |
| `total_stock_value`            | No  | Float, total stock value (USD)                                                                                                            |
| `restricted_stock`             | No  | Float, restricted stock value (USD)                                                                                                       |
| `salary`                       | No  | Float, salary (USD)                                                                                                                       |
| `sqrt_wealth`                  | Yes | Float, sqrt transformation of wealth, i.e, the sum of salary, total_payments, bonus, total_stock_value, expenses, other, restricted_stock |
| `total_payments`               | No  | Float, total payments (USD)                                                                                                               |
| `exercised_stock_options`      | No  | Float, exercised stock options (USD)                                                                                                      |
| `sqrt_exercised_stock_options` | Yes | Float, sqrt transformation of exercised stock options                                                                                     |

I started by creating some new variables (see table above and Classification notebook).
<br>One of the variable is what I call `wealth`, which is simply the sum of most financial variables. I noticed, in fact, that financial features are in general quite correlated so I wanted to try a new feature which is just the sum of them.
<br>For messages variables, I created a series of ratios which, in my intention, should normalise these features. For example, the `to_poi_ratio` is the ratio between the absolute number of emails sent to poi and the total number of email sent. In this way, observations become comparable.
<br>
<br>
I only keep variables for which the percentage of missing values (i.e. `NaN`) is below 50% - to do that, I have actually built a function called `extract_fields_for_ml` in `dict_parser.py` module called `extract_fields_for_ml`.
<br>
I did an ANOVA test on all the original features vs the label (i.e. poi or not). I did it using `SelectKBest` (kind of a shortcut) and kept variables whose p-value is below 5% (this is the list at the beginning of this paragraph).
<br>
Since I have used algorithms like SVM, I have scaled all the features using `MinMaxScaler`. Scaling allows me to remove any influence due to values which are represented in different scale (e.g. wealth can reach millions of USD, while a percentage will have a much narrower range of values).

<br>

### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

I tried a series of algorithms, including SVC, Logistic Regression, Decision Tree, K Nearest Neighbors, Ball Tree, Random Forest, etc. (a full list is available in the *Classification Full* notebook). The process I have used is the following:
* **optimise** the algorithms by using GridSearchCV. Since I wanted my algorithm to have good precision and recall, I am optimising on *f1* score. Please note that I was doing optimisation on a 10-fold cross validation Stratified Shuffle Split (I wanted to perform optimisaton not just once).
* **evaluate** the algorithms using a 1,000-fold cross validation Stratified Shuffled Split.

There are three major things I would like to highlight here:
* for all the algorithms, I have run an all-feature version (i.e. I am using all the features I have selected) and a pca-one (where the number of components is chosen in the optimisation process).
* for all the appropriate algorithms, I have set up the *class_weight* parameter equal to `balance` so that the fact that only 18 observations are true POIs (out of 140 total samples, after having removed outliers) was taken into account.
* my main script is optimising and evaluating the algorithms on `True` values of the label *poi*, i.e. optimisations and metrics are calculated so that the prediction power of true POIs is maximised. In the testing script provided, however, metrics are calculated globally, e.g. precision and accuracy are calculated on all predicted values.
<br>
I still prefer my original optimisation and evaluation process, but for the purpose of this exercise, I was then using global optimisation and evaluation. In my code, this is reflected in the two modules `optimiser.py` and `evaluate.py`. In the first one, I am setting the scoring parameter equal to *f1_micro* (i.e. the global score), while on the second I am setting a custom parameter called *tester* equal to `True` (in this case, the evaluation is the same as the one provided in `tester.py`).

<br>
The table below reports the global metrics for the algorithms I have used (optimisation on *f1*). The algorithm I ended up using is the XXX, with the following parameters:

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?
Tuning the parameter of an algorithm is equivalent to an optimisation process. Algorithms have different parameters (e.g. *C* in svm and Logistic Regression, *class_weight* in most of them, etc.) whivh may have an impact on the performance of the algorithms themselves. If you have an objective in mind for your algorithm (e.g. having an algorithm as accurate as possible), you may want to tune your algorithm's parameter so that your objective is achieved (or at least maximised). If you don't tune your parameter well, you might end up with an algorithm which is "making too many mistakes" or it might be not reobust enough for generalisation (i.e. it might be overfitted). The parameter *C* of svm and Logistic Regression is a classic example of a parameter that does have an impact on the overfitting of an algorithm.

In my script, I tuned my parameters using the module `GridSearchCV`. This module tries a series of values for the given parameters of an algorithm and returns the best set of parameters' value given an optimisation objective. As said in the previous paragraph, I am running this optimisation process with the objective to maximise the *f1* score. I also cross-validated the values of my parameter by setting the `cv` parameter equal to 10-fold Stratified Shuffled Split.

### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?
Validation is the process to test how good an algorithm reacts to a series of data it is not been trained on. In other words, if I want to test my algorithm on new values, I am using validation to check how good it performs. The classic mistake of doing validation in the wrong way is **overfitting**.
<br>
In general, a machine learning algorithm is trained on a set of samples called *train* data and it is validated on an another set, called *test* (or validation) data. If I pass to the algorithm just this single training data, the machine will only learn from that and it may not take into account other important observations, thus becoming overfitted to those specific values.
<br>To overcome this problem, and check if the algorithm is good at generalising, it is wise to use **cross-validation**, i.e. a process through which multiple train/test datasets are passed to the algorithm so that we can assess its average performance on multiple situations.
<br>
<br>
In my analysis, following the example provided in `tester.py`, I am using Stratified Shuffled Split (1,000 folds) to perform cross-validation. In essence, this creates 1,000 different train/test sets which are used to train and validate, respectively, the algorithms.

### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

### Links
MD table generator: http://www.tablesgenerator.com/markdown_tables 
