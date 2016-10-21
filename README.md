# Udacity DAND - Intro to Machine Learning
Submission by Giacomo Sarchioni

## Introduction
This README file aims at guiding the reader in the understanding of my submission for the Intro to Machine Learning Project
of Udacity Data Analyst Nanodegree.

In this document I am going to provide:
* answers to the *free-response* questions asked [here](https://docs.google.com/document/d/1NDgi1PrNJP7WTbfSUuRUnz8yzs5nGVTSzpO7oeNTEWA/pub?embedded=true);
* instructions on how to navigate through this repository.

## Free-response questions

> *Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it.*

The goal of this project is that of building a Machine Learning (ML) algorithm that is able to classify an Enron employee
as a *Person of Interest* (POI), i.e. a person who was involved in the Enron accounting scandal. The classification is binary,
i.e. a person is either a POI or is not.
<br>The dataset is the form of a Python dictionary, where for every key (i.e. the name of an employee) there is a dictionary of values,
including financial and message-related variables. The original dataset is made of 146 observations, of which **only 18** are
actual POIs. The dataset, therefore, is very unbalanaced. 
