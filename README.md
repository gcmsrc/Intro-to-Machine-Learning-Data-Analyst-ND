# Udacity DAND - Intro to Machine Learning
Submission by Giacomo Sarchioni

## Introduction
This README file aims at guiding the reader in the understanding of my 
submission for the Intro to Machine Learning Project
of Udacity Data Analyst Nanodegree.

In this document I am going to provide:
* answers to the *free-response* questions asked [here](https://docs.google.com/document/d/1NDgi1PrNJP7WTbfSUuRUnz8yzs5nGVTSzpO7oeNTEWA/pub?embedded=true);
* instructions on how to navigate through this repository.

## Free-response questions

> *Summarize for us the goal of this project and how machine learning 
is useful in trying to accomplish it.*

The goal of this project is that of building a classification model that is able to classify an Enron employee
as a *Person of Interest* (POI), i.e. a person who was involved in the 
Enron accounting scandal. The classification is binary,
i.e. a person is either a POI or is not. In this context, an ML algorithm is a great tool to build a predictive model, withouth being constrained to any parametric models. ML also allows me to evaluate the performance/quality of the predictions and, most importanly, to optimise the model given some metrics I would like to maximise.
<br>The dataset is the form of a Python dictionary, where for every 
key (i.e. the name of an employee) there is a dictionary of values,
including financial and message-related variables. The original 
dataset is made of 146 observations, of which **only 18** are
actual POIs. The dataset, therefore, is very unbalanaced.
<br>The original dataset has 14 financial variables (e.g. salary, bonust, etc.), 6 messages variables (e.g. number of emails sent, number of emails received, etc.) and 1 labelling features (POI or non POI).
<br>
<br>
As I showed in the *Outlier Identificatio* file (availabel as Jupyter
notebook or html), I have identified six outliers which I have 
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
<br>

> *What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not?*
### Links
MD table generator: http://www.tablesgenerator.com/markdown_tables 
