# King-County-Houses-Business-Insights
Data Visualization app for business propositions and insights for a real estate business model in King County.

# 1. Business Problem.
KC Houses is a real estate company based on King County, which is searching for a way to automatize commercial opportunities analyses, aiming to enable making decisions faster, with a smaller work force. 
KC Houses CEO also wants to receive ten business insights for his next strategic board meeting, which also should contain graphics for data visualization and further consultations.

# 2. Business Assumptions.	
•	Due to bad weather conditions and harder transportation, colder seasons tend to rise slightly house prices.
•	KC Houses is a company specialized in mid to high end real estate proprieties, so it will only consider buying houses in good conditions

# 3. Solution Strategy.
3.1. Which are the most profitable houses to buy?
To help us find this answer, first we have to categorize the houses in our dataset, so we can have even comparisons based in size, location and condition.
After said categorization, we can compare if each house is cheaper than the average in the same location and with the same size and condition category.
This will create a new dataset, returning only entries with greater potential for profits and providing us with data to create a scatter map, which pinpoints each of these houses with geolocation, in a cluster-responsive map.

![image](https://user-images.githubusercontent.com/99055161/180351615-91bd52ce-dc63-4afe-80dd-44378e4aa1d6.png)
 
3.2. When is the best time to sell a house?
Assuming weather conditions affect house pricing, the creation of a function that calculates the estimate price difference between seasons was issued, helping us measure such impact in the sales of all houses owned by the company.

![image](https://user-images.githubusercontent.com/99055161/180351631-54ba7fee-e7d4-4eba-b861-b3d0525ae09d.png)
 
3.3 Hypothesis testing for business insights with data visualization. 
First, we start by making 10 business assumptions based on previous exploratory researches and personal experience. 
Then, we proceed to making comparisons between two elements of our dataset, expecting to find answers based on data analysis. 
After that, the data returned by said analysis was used to create a graph for each hypothesis best suited for better data visualization.

Hypothesis list:

•	Houses with waterfront are 30% more expensive

•	Houses that were built before 1955 are 50% cheaper in average

•	Houses with a basement are 40% larger

•	By average values, house prices rise 10% YOY

•	The average MOM increase in house prices with three or more bathrooms is 5%

•	Houses with one room represent 15% of all houses built after the year of 2010

•	Bigger houses are 50% more expensive than smaller ones

•	Houses in good condition are 30% more expensive

•	Houses with big plots are 20% more expensive

•	Houses renovated within the last 5 years are more likely to receive a grade over 10

# 4.Top 3 Data Insights.
•	Houses with waterfront are 30% more expensive

![image](https://user-images.githubusercontent.com/99055161/180351664-05e9f327-bc0b-4552-a12a-556c5cf004f5.png)

False. As observed, results show us that houses with ocean view tend to be almost 70% more expensive.
•	Houses in good condition are 30% more expensive

![image](https://user-images.githubusercontent.com/99055161/180351684-e3c073ae-9d89-4e11-9f0f-24f838baf852.png)

False. Houses in the highest conservation condition tend to be only 13% more expensive.
•	Houses renovated within the last 5 years are more likely to receive a grade over 10

![image](https://user-images.githubusercontent.com/99055161/180351705-f41e19fe-d3bc-4477-b214-b37eb41f10f8.png)

False. This dataset uses data from years 2014-2015, and as unlikely as it seems, houses renovated after the year of 2010 do not tend to receive a better construction level and design grade.

# 5.Business Results.
Business-wise, our most reliable source of profit in this analysis is the “Most profitable houses to buy” algorithm and its map.

![image](https://user-images.githubusercontent.com/99055161/180351735-992b22c1-00d3-44a2-815f-921cef554af8.png)

The estimate profit of buying and selling all the 239 houses from the dataset provided by this feature is of U$21.743.018,15. It’s important to remind that these houses are the best in their regions and have a price below average.
The “Best time to sell” feature helps the company calculate the impact of seasons when selling houses, with values that can change between 10% and 30%.
If all the houses in the dataset were owned by KC Houses and sold in the best time possible, which is a most unlikely condition, the profit estimate would be of U$3.117.757.632,40.

# 6.Conclusions.
This dataset provides us with plenty of data to reach important conclusions on the nature and shifts of the Real Estate market in King County.
However, such can be extended to any other country. Using data analysis automation, we are able to gather important information in an almost instant timeframe, which can help executives and software to make important decisions, resulting in great profit for the company.
