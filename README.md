# machine-learning-text-it-service-desk-tickets
A text analysis study on classifying IT ticket attributes from text data. The project aims to identify if the ticket triaging process can be automated by using text analysis and machine learning to predict the attributes of an IT ticket.   

# 1. Executive Summary
Automating the categorization and prioritization process of IT incidents and requests through Machine Learning.   

- The goal of IT Service Management is the effective resolution of IT incidents and request to reduce interruptions to daily business operations
- IT service teams are expected to deliver with a high level of service as technologies are constantly changing within an organization
- However, processing IT tickets is a highly manual process, that is dependent on the quality of the processing pipeline. 

# 2. Problem 
- 30% -40% of IT Tickets are not routed to the correct team
- The list of technologies serviced by IT is constantly growing creating more uses cases that need to be supported
- The Service Desk often forward tickets to incorrect teams due knoweledge gaps imperfect change management 

# 3. Data
##### Data Acquisition 
 - Data was acquired from karolzak's GitHub site
 - The data set contained text data and nominal ticket attributes which had been cleaned and anonymized by karolzak
 - The text variables in the dataset were "title" and "body"
 - The nominal variables were "ticket_type", "category", "sub_category_1", "sub_category_2", "business_service", "urgency", "impact"
 - In this project variables focused on where "body", "category", "urgency", "impact"
 ## 3.1 Data Summary
 - JMP software was used to create distribution of the data to identify the frequencies between the classes.
 - Below I could see the data was highly unbalanced. Showing low distribution between certain classes for all variables.
 
 ![image](https://user-images.githubusercontent.com/99374452/159406153-226c89b6-4b7a-4640-bdb1-b1b084929e2d.png)
 
 - In the category variable large unbalanced with category 4 having 25,000 more tickets than the next class. 
 - In machine learning unbalanced data can be problematic, because its predictions will be biased to classes with the most data.
 - Two techniques used into this project to mitigate the balance issue are over-sampling and undersampling. 
 ## 3.2 Preprocessing
 #### Significant Categories
 - Prior to using any of the mentioned sampling techniques, the categories variable was reviewed to identify what were the most significant classes.
 - Class with frequencies greater than 1% of the data were recognized as significant.
 - With other class recognized as insignificant and therefore exclude from the training and validation sets.  
 
 ![image](https://user-images.githubusercontent.com/99374452/159411675-eef8359b-69af-4757-99ca-2122bc597869.png)
 
 ![image](https://user-images.githubusercontent.com/99374452/159416638-f1c34ef0-6731-4262-9b74-af582fe78dca.png)

 - This approach made practical sense as well, because most tickets within the IT Service Desk fall within few of many common categories. 
 - As these tickets are common and the process of triaging them is repeatible, this is the optimal zone for automation. 
 - Tickets with insignificant categories, are uncommon and can be remain manually managed.

 #### Oversampling / Under-sampling
 To balanced the data oversampling and under-sampling techniques.
- ##### Oversampling
  -  When oversampling, I simply create random duplicates of observations in the classes that have the lowest sample count.
  -  This essentially fattens the data, creating more total observations to training our model.
  -  However, the major weakness in this model is that it is likely to cause overfitting.
  
  ![image](https://user-images.githubusercontent.com/99374452/159418392-b3bccc46-f0c6-478b-961e-7d004ce1499a.png)
  
  -  After oversampling the dataset expanded from ~ 48,000 to ~ 136,000 observations.
  -  The large dataset initally made oversampling a more attractive option to train the model.

- ##### Under-sampling
  -  With under-sampling, random samples are extracted from classes with more sample count to match the sample count of the classes with the lowest observations.
  -  This technique shrinks the data, which can be an issue as potentially valuable data is left out of the model. 
  -  
  ![image](https://user-images.githubusercontent.com/99374452/159421609-330f797e-0798-4e76-bead-468240a6efff.png)
  
  -  After under-sampling, the data shrunk to a total of 3060 observations.

# 4. Modeling
## 4.1 Multinomial Naive Bayes Classification Model
#### Model Overview
Classification algorithm which analyzes the probability of unique words to an outcome. It is based of Baye's theorem - "The probability of an event is based on prior knowledge  of conditions related to that event”
 -  A simplified example:
   - Email: ”Dinner Money Money”
   - What is the probabilty of the above email being spam? 
   - It is the product of the probabilities of the individual words relating to a spam email.
   <img width="200" alt="image" src="https://user-images.githubusercontent.com/99374452/160127132-9af3c838-4f88-46e2-8697-7bef5ab80f4f.png">

#### TFIDF - Text Frequency - Inverse Document Frequency
 - To model the text data, I need to get the frequencies of words in each ticket, how often they appear in the whole dataset. 
<img width="1391" alt="image" src="https://user-images.githubusercontent.com/99374452/160130078-1b3b6ffd-c757-43f3-a4ce-d3080f2b3296.png">

 - I used the TFIDF Vectorizer to fromt the Sci-kit Learn package to perfrom this step.
#### Hyperparameter Tuning

- Applying a 5 fold validaton to reduce potential overfitting especially when oversampling the text data
- I used GridSearchCV to identify the best hyper parameters for the model. 
- The top five parameters combinations can be seen below:

<img width="668" alt="image" src="https://user-images.githubusercontent.com/99374452/160131297-9a39dc28-fdc9-4434-b804-4bb67edb427f.png">

- The best model required a combination of using just unigrams, no weighting of words using IDFs, using an alpha value of 1 to address the zero probability issue and using the same probabilties in the train fit. 

#### Results
- ##### Oversampling:
<img width="1275" alt="image" src="https://user-images.githubusercontent.com/99374452/160199055-3fd84ec2-c62b-44a8-b765-bde6ae22913f.png">

  - The model predicted well for classes 7 and 11, which was concerning because those were classes with the least amount of data and need to be oversampled the most. 
  - It essentially meant the model predicted well for duplicate text.  
  - Indicating bias and confidence in the accuracy should  be low.  
  - The true accuracy of the model  would be < 55%
    
- ##### Undersampling:
<img width="1274" alt="image" src="https://user-images.githubusercontent.com/99374452/160199497-de0c93bd-f59d-45a2-b279-2c8828818489.png">

  - The undersample set perfromed much better than the oversample set.
  - Intuitively,  made more sense as the undersample data set contain more raw data to work with although there was a limited amout of it. 
  - This model was more reliable, because it predicted similarly between classes. 
  - With overal  accuracy at 75% percent this was an overall better model

## 4.2 Bert Classification Model
#### Model Overview
  - BERT is a natural language processing trained by Google that creates numerical vector respresentations of natural lanaguage to be used in modeling. After processing the text data a neural network can be fit to classify the data. 
  - The size of the vector otherwise the hidden size is configurable. With the common size being 768. To reduce the computation demand of a 768 vector for each ticket a BERT hidden size of 128 was used in this project. 
#### Results
Understanding the inaccuracy of the oversampled data, the under-sampled data was only used for the BERT model. 

<img width="1274" alt="image" src="https://user-images.githubusercontent.com/99374452/160199380-f4248f6b-5dcc-4244-bbc4-854aaaeb6e9a.png">

# 5. Conclusions
# 5. Code Highlights

- text analysis Advanced analytics package was used to 

  -  







