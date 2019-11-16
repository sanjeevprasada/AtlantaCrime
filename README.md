# Atlanta Crime Mapping for CS7641 - Group 17
#### Abdurrahmane Rikli, Gabriel Leventhal-Douglas, Kevin Tynes, Aayush Dubey, and Sanjeev Prasada

![Atlanta Skyline](http://media.bizj.us/view/img/6139341/atlanta-skyline*750xx3684-2070-0-28.jpg)
# Introduction
## Motivation
In Atlanta, the overall crime rate is 108% higher than the national average. Crime is an ever-present concern. With almost 30 thousand crimes a year and a 61% crime rate per capita, Atlanta is one of the 3% most dangerous cities in the United States [1]. With such issues, the police force cannot deal with crime on a case-by-case basis. They need to be directed to crime-heavy areas preemptively. Sufficient patrols in crime-heavy areas can be
achieved using a prediction model to estimate the areas with the most severe crimes. We reviewed literature of machine learning crime prediction methods using spatial [5, 3] and temporal [2] data in conjunction with crime-type. We will build upon this prior work by applying these methods to Atlanta crime data and improving predictive model efficiency.

#### What have people already done?
References are important. We may need to reference our proposal here and summarize our literature review. Will work on this tonight after dinner (post 9pm).

#### What are you trying to do to tackle with your project motivation or problem?
Help police offers re-allocate and distribute their patrols to areas of higher need. (Figure out sexier way to say help the cops patrol, mention our Machine Learning algorithms for clout).


## Dataset (Needs description of features, accessability, etc.)
After analyzing the Atlanta PD Crime dataset from 2009-2018, the most popular crimes in descending order are larceny from vehicle, larceny non vehicle, burglary at residence, and automobile theft. Our dataset is record-based; each row in the dataset represents one crime and the features of that crime (represented below in tables). We have a total of 20 features per record and a total of 317,905 records of crime within the Perimeter of Atlanta. There were a few columns we had to remove due a large number of null's and a few rows were removed based on inconsistency of data.


### Original Dataset

 Report Number | Occur Date | Occur Time | Neighborhood |   UCR Literal       | Latitude | Longitude 
 ------------- |:----------:|:----------:|:------------:|:-------------------:|:--------:|:---------:
 090010930     | 2009-01-01 |    1145    | Greenbriar   | LARCENY NON-VEHICLE | 33.69    | -84.49    
 090011083     | 2009-01-01 |    1330    | Downtown     | LARCENY NON-VEHICLE | 33.75    | -84.39    
 090011208     | 2009-01-01 |    1500    | Adamsville   | LARCENY NON-VEHICLE | 33.76    | -84.50    
 ...           |    ...     | ...        | ...          | ...                 | ...      | ...

### Supervised algorithms preprocessed dataset

Occur Date  | Neighborhood|   UCR Literal       | Latitude | Longitude  | Shift Occurrence 
 -----------|:-----------:|:-------------------:|:--------:|:----------:|:----------------:
 2009-01-01 | Greenbriar  | LARCENY NON-VEHICLE | 33.69    | -84.49     |  Day
 2009-01-01 | Downtown    | LARCENY NON-VEHICLE | 33.75    | -84.39     |  Day
 2009-01-01 | Adamsville  | LARCENY NON-VEHICLE | 33.76    | -84.50     |  Day
 ...        | ...         |    ...              | ...      | ...        |  ...  

### Unsupervised algorithms

Year  | Month |  Day  | Day of Week  | Category 1 | Category 2 | Category 3 | Category 4 
 -----|:-----:|:-----:|:------------:|:----------:|:----------:|:----------:|:----------:
 2009 | 1     |   1   | 3            | 0          |  15        |    58      |  48
 2009 | 1     |   2   | 4            | 0          |  15        |    46      |  73
 2009 | 1     |   3   | 5            | 1          |  21        |    37      |  56
 ...  | ...   | ...   | ...          | ...        |  ...       |    ...     |  ...       

## Approach
It is important to cluster based on location and time, as they are relevant features of a crime’s occurrence and are useful for a police force’s patrol. Hence, the mean shift algorithm would be useful as one of the unsupervised learning methods to explore, in addition to k-means clustering. To finishAs for supervising learning techniques, decision trees have been used as a means of classification [2, 4]. Utilizing the severity of a crime served beneficial to the analysis. The decision tree works well, so naturally a random forest algorithm will supplement crime analysis further. Lastly, we can explore the accuracy given by a Naive Bayes Classifier. All algorithms need at least half of the available training data in order to build a successful set of clusters or prediction model to suffice for trends for crimes in future years, and to provide a police department with the necessary information on how they should run their patrol. Throughout our modeling and data pre-processing, we used primarily Python, along with a few Python packages: sci-kit learn, sci-py, pandas, and numpy.

#### What is new in our approach?
We generated a crime score for each neighborhood for each day. This is our "secret sauce". We believe that in order to label locations as __hotspots__, we needed to aggregate a score including heavy weightage for the most severe crimes.

## Visualization
Crime instensities across the city limits of Atlanta. 

![Atlanta all categories visualization](images/visualization/Crime_intensities.png?raw=true)


![Category 1](images/visualization/Crime_intensities_category_1.png?raw=true)

![Category 2](images/visualization/Crime_intensities_category_2.png?raw=true)

![Category 3](images/visualization/Crime_intensities_category_3.png?raw=true)

![Category 4](images/visualization/Crime_intensities_category_4.png?raw=true)



+ shapefile was imported in Python to read the shapefile data
+ pandas was to read the csv data
+ seaborn was used for coloring
+ matplotlib was to display the data (both pandas and shapefile)
+ need one paragraph explaining the legend and the conclusions we could make from the visualizations.

## Unsupervised Methods
Our tech stack for the unsupervised methods were sklearn in Python. First, we plotted the DBSCAN function and a corresponding elbow plot to __________ and optimize the ___________ and we conducted this method on k=3 to k=100. 
+ __Comment__ about what we learned through DBSCAN and drove the decision to also create __**DBSCAN Method 2** spatial representation__. 



+ __Comment__ about what DBSCAN reduced set told us about our data and what the reduced set aimed to do.

+ Mean shift is our next algorithm of choice. Mean shift results can vary as the bandwidth (radius) parameter is adjusted.

![Mean Shift](link)

We conducted PCA on the cobra-clean2009.csv and cobra-clean2019.csv dataset and received similar results. This shows that an almost identical amount of variance can be explained from the same components across different years. This shows the explained variance ratio is non-random and we have a justified reason to be using those Components.

![PCA explained ratio 2009](link)

![PCA explained ratio 2019](link)

## Supervised Methods
Our tech stack for the supervised methods were sklearn in Python. Some initial preprocessing is done with the data before the entered into the model. We utilize 10% of the data for testing, and 90% for training. This is the first time we use the Crime Score. We created this metric after obtaining domain knowledge of severity in crimes. Understanding the judicial system's consequences for certain crimes, we were able to manufacture a crime score for each neighborhood to took the severity of the crime into account. This is unique part of our project that aims to help map the toughest crime hotspots to police officers. 



#### Metrics & Results
1. Decision Tree
2. Random Forest
3. Naive-Bayes Classifier
4. Support Vector Machine
5. Logistic Regression

![Metric 1](link)
![Metric 2](link)
![Metric 3](link)
![Metric 4](link)
![Metric 5](link)


## Discussion 
True crime prediction entails a complex set of variables that may not be publicly available for intrepid data scientists. Socioeconomic factors may be difficult to aggregate, while psychological motivators are highly abstract. Identification of crime hotspots allows law enforcement agencies to allocate police routes and other crime inhibiting factors, such as CCTV cameras, lights or neighborhood watches, more effectively [3]. Crime inciters, such as gang territories, bars, and construction sites can be monitored more frequently. 

We evaluated our approach using accuracy, precision, and recall for classification of categories. In regression of crime scores, our metric is RMSE.

## Conclusion

#### Major Achievement
Our major achievement was our supervised model using crime score. Classification methods and metrics that are above the lower bound that we set for. 
#### Future work
Without question, our methodology could be improved. Given more time and resources, we would plan to merge our dataset with other datasets regarding Atlanta location specifics. Giving neighborhoods more features and more variability would only help our model learn and raise our accuracy. Another thing we would may be consider would be implementing similar prediction using deep learning neural networks.

Although we had access to 300,000+ rows of data, access to more data would make our model more robust.

## References 
[1] Schiller, Andrew. "Atlanta, GA Crime Rates & Statistics." NeighborhoodScout. NeighborhoodScout, 10 June 2019. Web. 30
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

