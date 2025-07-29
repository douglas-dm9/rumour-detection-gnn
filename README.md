# Paper code of "Early rumor detection: Evaluating the effectiveness of graph neural networks"


#### 1. JSON to CSV files.
Notebook to create csv dataset from JSON raw files ```Dataset creation from files.ipynb```. The output, as example is the  ```charliehebdo-all-rnr-threads.csv```, where each line is an reaction with all source tweet features and the reply features.

#### 2. EDA
Exploratory Data Analysis to understand data distribuition and differences between rumours and not rumours propagation
* ```eda.ipynb```

#### 3. Pre processing and Feature Engineering

The notebook ```Pre Processing - Feature Engineering.ipynb``` show the process of NLP cleaning to final generate the word embeddings for the text of source tweets and reply text. It also includes some feature creation like time difference between the interation and the source tweet, number of replies and the time of first reply

#### 4. Time filter module

4 Python classes was created to evaluate the algorithms performance as time progresses and the number of posts/interactions increases. The classes were developed to filter the test data every 10 minutes, provided there is a date and time column. With a fixed training set, the model is trained before the first inference, and every 10 minutes until the last interaction in the test set, new inferences are made, evaluating the model's performance as new posts/interactions emerge. The model is also updated every 10 minutes, as interactions on posts in the training set may have new interactions as time passes.
The inference step records important classification metrics such as precision, AUC, recall, and f1-score, which are recorded using the MLflow library. This way, the metrics for each experiment are organized, recorded, and can later be easily compared and queried in the MLflow database or through its interface.

* **Load_Rumours_Dataset_filtering_since_first_post**: Class created to work with a single event (test and train from same context) and tabular data.
* **Hetero_Data_Processor_Filter_on_Test_since_first_post** Class created to work with a single event (test and train from same context) and graph  data.
* **Hetero_Data_Processor_Transfer_Learning**: Class created to work with two events (test and train from two different contexts) and graph data (Transfer Learning approach).
* **Load_Rumours_Dataset_filtering_since_first_post_Transfer_Learning**:  Class created to work with two events (test and train from two different contexts) and tabular data (Transfer Learning approach).

The classes are stored in the ```utils.py``` file

#### 5. Experiments Single Event (Using the data of Charlie Hebdo attack)

* ```HAN Charlie Hebdo.ipynb```
* ```GAT Charlie Hebdo.ipynb```
* ```Random Forest Charlie Hebdo.ipynb```
* ```Light Gbm Charlie Hebdo.ipynb```
* ```LSTM Charlie Hebdo.ipynb```

#### 6. Experiments using two events (Transfer learning)

* ```Han Transfer Learning.ipynb```
* ```GAT Transfer Learning.ipynb```
* ```Random Transfer Learning.ipynb```
* ```Light Transfer Learning.ipynb```
* ```LSTM Transfer Learning.ipynb```

#### 7. Summary of all experiments

Plotly charts with the summary of all experiments, using the data stored in the mlflow database

* ```Summary experiments Single Event.ipynb```
* ```Summary Transfer Learning.ipynb```
