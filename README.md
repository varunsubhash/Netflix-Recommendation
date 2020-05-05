# Netflix-Recommendation

<h1>1. Business Problem </h1>
<h2> 1.1 Problem Description </h2>
<p>
Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: CinematchSM. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix use those predictions to make personal movie recommendations based on each customer’s unique tastes. And while <b>Cinematch</b> is doing pretty well, it can always be made better.
</p>
<p>Now there are a lot of interesting alternative approaches to how Cinematch works that netflix haven’t tried. Some are described in the literature, some aren’t. We’re curious whether any of these can beat Cinematch by making better predictions. Because, frankly, if there is a much better approach it could make a big difference to our customers and our business.</p>
<p> Credits: https://www.netflixprize.com/rules.html </p>

<h2> 1.2 Problem Statement </h2>

<p>
Netflix provided a lot of anonymous rating data, and a prediction accuracy bar that is  better than what Cinematch can do on the same training data set. (Accuracy is a measurement of how closely predicted ratings of movies match subsequent actual ratings.) 
</p>

<h2>1.4 Real world/Business Objectives and constraints  </h2>

Objectives:
1. Predict the rating that a user would give to a movie that he ahs not yet rated.
2. Minimize the difference between predicted and actual rating (RMSE and MAPE)
<br>

Constraints:
1. Some form of interpretability.

<h1> 2. Machine Learning Problem </h1>

<h2>2.1 Data </h2>

<h3> 2.1.1 Data Overview </h3>

<p> Get the data from : https://www.kaggle.com/netflix-inc/netflix-prize-data/data </p>
<p> Data files : 
<ul> 
<li> combined_data_1.txt </li>
<li> combined_data_2.txt </li>
<li> combined_data_3.txt </li>
<li> combined_data_4.txt </li>
<li> movie_titles.csv </li>
</ul>
<pre>  
The first line of each file [combined_data_1.txt, combined_data_2.txt, combined_data_3.txt, combined_data_4.txt] contains the movie id followed by a colon. Each subsequent line in the file corresponds to a rating from a customer and its date in the following format:

CustomerID,Rating,Date

MovieIDs range from 1 to 17770 sequentially.
CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
Ratings are on a five star (integral) scale from 1 to 5.
Dates have the format YYYY-MM-DD.
</pre>

<h3> 2.1.2 Example Data point </h3>

<pre>
1:
1488844,3,2005-09-06
822109,5,2005-05-13
885013,4,2005-10-19
30878,4,2005-12-26
823519,3,2004-05-03
893988,3,2005-11-17
124105,4,2004-08-05
1248029,3,2004-04-22
1842128,4,2004-05-09
2238063,3,2005-05-11
1503895,4,2005-05-19
2207774,5,2005-06-06
2590061,3,2004-08-12
2442,3,2004-04-14
543865,4,2004-05-28
1209119,4,2004-03-23
804919,4,2004-06-10
1086807,3,2004-12-28
1711859,4,2005-05-08
372233,5,2005-11-23
1080361,3,2005-03-28
1245640,3,2005-12-19
558634,4,2004-12-14
2165002,4,2004-04-06
1181550,3,2004-02-01
1227322,4,2004-02-06
427928,4,2004-02-26
814701,5,2005-09-29
808731,4,2005-10-31
662870,5,2005-08-24
337541,5,2005-03-23
786312,3,2004-11-16
1133214,4,2004-03-07
1537427,4,2004-03-29
1209954,5,2005-05-09
2381599,3,2005-09-12
525356,2,2004-07-11
1910569,4,2004-04-12
2263586,4,2004-08-20
2421815,2,2004-02-26
1009622,1,2005-01-19
1481961,2,2005-05-24
401047,4,2005-06-03
2179073,3,2004-08-29
1434636,3,2004-05-01
93986,5,2005-10-06
1308744,5,2005-10-29
2647871,4,2005-12-30
1905581,5,2005-08-16
2508819,3,2004-05-18
1578279,1,2005-05-19
1159695,4,2005-02-15
2588432,3,2005-03-31
2423091,3,2005-09-12
470232,4,2004-04-08
2148699,2,2004-06-05
1342007,3,2004-07-16
466135,4,2004-07-13
2472440,3,2005-08-13
1283744,3,2004-04-17
1927580,4,2004-11-08
716874,5,2005-05-06
4326,4,2005-10-29
</pre>

<h2>2.2 Mapping the real world problem to a Machine Learning Problem </h2>

<h3> 2.2.1 Type of Machine Learning Problem </h3>

<pre>
For a given movie and user we need to predict the rating would be given by him/her to the movie. 
The given problem is a Recommendation problem 
It can also seen as a Regression problem 
</pre>

<h3> 2.2.2 Performance metric </h3>

<ul>
<li> Mean Absolute Percentage Error: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error </li>
<li> Root Mean Square Error: https://en.wikipedia.org/wiki/Root-mean-square_deviation </li>
</ul>

<h3> 2.2.3 Machine Learning Objective and Constraints </h3>

1. Minimize RMSE.</br>
2. Try to provide some interpretability.</br>

<h1> 3. Exploratory Data Analysis </h1>

<h2> 3.1 Preprocessing</h2> 

- Here we are converting / merging whole data to required format: u_i, m_j, r_ij</h3></br>

- We checking for NaN values and make sure all are deleted.</br>

-  Removing Duplicates is also done.</br>

<h3>3.1.4 Basic Statistics (#Ratings, #Users, and #Movies)</h3>

Total no of ratings : 100480507</br>
Total No of Users   : 480189</br>
Total No of movies  : 17770</br>

<h2>3.2 Spliting data into Train and Test(80:20) </h2>

<h3>3.2.1 Basic Statistics in Train data (#Ratings, #Users, and #Movies)</h3>

Total no of ratings : 80384405</br>
Total No of Users   : 405041</br>
Total No of movies  : 17424</br>

<h3>3.2.2 Basic Statistics in Test data (#Ratings, #Users, and #Movies)</h3>

Total no of ratings : 20096102</br>
Total No of Users   : 349312</br>
Total No of movies  : 17757</br>

 <h2> 3.3 Exploratory Data Analysis on Train data </h2>
 
 <h3> 3.3.1 Distribution of ratings over training set.</h3></br>
 
   ![](Capture_1.PNG)
   
 - We add a new column (week day) to the data set for analysis 'day_of_week'.</br>
 
 <h3> 3.3.2 Number of Ratings per a month </h3></br>
 
  ![](Capture_2.PNG)
  
  <h3> 3.3.3 Analysis on the Ratings given by user </h3></br>
  
 - Below are the ratings the top number of movies rated by a particular user.</br>
 
 user
305344     17112
2439493    15896
387418     15402
1639792     9767
1461435     9447
 
 - The CDF and PDF are plotted below for the number of ratings per user.</br>
 
 - When we use the describe function we get the below results.</br>
 
 count    405041.000000</br>
mean        198.459921</br>
std         290.793238</br>
min           1.000000</br>
25%          34.000000</br>
50%          89.000000</br>
75%         245.000000</br>
max       17112.000000</br>
 
 - Looks like we may have to inspect the percentile to see how many values are abnormally high.</br>
 - On plotting value at quantiles vs no of ratings by users, we get the below plot. </br>
 
 ![](Capture_2.PNG)
