# Atlanta Crime Mapping for CS7641 - Group 17
#### Abdurrahmane Rikli, Gabriel Leventhal-Douglas, Kevin Tynes, Aayush Dubey, and Sanjeev Prasada

![Atlanta Pic](http://media.bizj.us/view/img/6139341/atlanta-skyline*750xx3684-2070-0-28.jpg)

## Motivation
In Atlanta, crime is an ever-present concern. With almost 30 thousand crimes a year and a 61% crime rate per capita,
Atlanta is one of the 3% most dangerous cities in the United States [1]. With such issues, the police force cannot deal
with crime on a case-by-case basis. They need to be directed to crime-heavy areas preemptively. All of this can be
achieved using a good enough predictive model. We reviewed literature of machine learning crime prediction methods
using spatial [5, 3] and temporal [2] data in conjunction with crime-type. We will build upon this prior work by applying
these methods to Atlanta crime data and improving predictive model efficiency.


## Dataset (Needs description of features, accessability, etc.)
After analyzing the Atlanta PD Crime dataset from 2009-2018, the most popular crimes in descending order are larceny from vehicle, larceny non vehicle, burglary at residence, and automobile theft.

### Original Dataset

 Report Number | Occur Date | Occur Time | Neighborhood |   UCR Literal       | Latitude | Longitude 
 ------------- |:----------:|:----------:|:------------:|:-------------------:|:--------:|:---------:
 090010930     | 2009-01-01 |    1145    | Greenbriar   | LARCENY NON-VEHICLE | 33.69    | -84.49    
 090011083     | 2009-01-01 |    1330    | Downtown     | LARCENY NON-VEHICLE | 33.75    | -84.39    
 090011208     | 2009-01-01 |    1500    | Adamsville   | LARCENY NON-VEHICLE | 33.76    | -84.50    
 ...     | ... |    ...    | ...   | ... | ...    | ...   

### Crime Score preprocessed dataset



## Approach
It is important to cluster based on location and time, as they are relevant features of a crime’s occurrence and are useful
for a police force’s patrol. Hence, the mean shift algorithm would be useful as one of the unsupervised learning methods
to explore, in addition to k-means clustering. As for supervising learning techniques, decision trees have been used as a
means of classification [2, 4]. Utilizing the severity of a crime would serve beneficial to the analysis. Assuming the
decision tree works well, then a random forest algorithm can supplement crime analysis even further. Lastly, we can
explore the accuracy given by a Naive Bayes Classifier. All algorithms need at least half of the available training data
in order to build a successful set of clusters or prediction model to suffice for trends for crimes in future years, and
to provide a police department with the necessary information on how they should run their patrol. Throughout our
modeling and data pre-processing, we expect to use primarily Python, along with a few Python packages: sci-kit learn,
sci-py, pandas, and numpy. Given access to a computer provided by the class, we will be able to efficiently run our
model using parallelization in Python and/or PySpark.


## Unsupervised Methods



## Supervised Methods



## Discussion 
True crime prediction entails a complex set of variables that may not be publicly available for intrepid data scientists.
Socioeconomic factors may be difficult to aggregate, while psychological motivators are highly abstract. Identification
of crime hotspots allows law enforcement agencies to allocate police routes and other crime inhibiting factors, such as
CCTV cameras, lights or neighborhood watches, more effectively [3]. Crime inciters, such as gang territories, bars, and
construction sites can be monitored more frequently. We will be using metrics such as F1 score, accuracy, and/or loss to
evaluate our model then continue hyper-tuning parameters.


## References 
[1] Schiller, Andrew. "Atlanta, GA Crime Rates." NeighborhoodScout. NeighborhoodScout, 10 June 2019. Web. 30
Sept. 2019. </br>

[2] Mcclendon, Lawrence, and Natarajan Meghanathan. "Using Machine Learning Algorithms to Analyze Crime
Data." Machine Learning and Applications: An International Journal 2.1 (2015): 1-12. Print. </br>

[3] Lin, Ying-Lung, Meng-Feng Yen, and Liang-Chih Yu. "Grid-Based Crime Prediction Using Geographical Features."
ISPRS International Journal of Geo-Information 7.8 (2018): 298. Print. </br>

[4] Kim, Suhong, Param Joshi, Parminder Singh Kalsi, and Pooya Taheri. "Crime Analysis Through Machine Learning."
2018 IEEE 9th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON)
(2018): n. pag. Print. </br>

[5] Bappee, Fateha Khanam, Amílcar Soares Júnior, and Stan Matwin. "Predicting Crime Using Spatial Features."
Advances in Artificial Intelligence Lecture Notes in Computer Science (2018): 367-73. Print.

