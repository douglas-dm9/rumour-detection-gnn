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

#### 5. Experiments Single Events

* ```HAN.ipynb```
* ```GAT.ipynb```
* ```Random Forest.ipynb```
* ```Light GBM.ipynb```
* ```LSTM.ipynb```

#### 6. Experiments using two events (Transfer learning)

* ```HAN TF.ipynb```
* ```GAT TF.ipynb```
* ```Random Forest TF.ipynb```
* ```Light GBM TF.ipynb```
* ```LSTM TF.ipynb```

#### 7. Summary of single event experiments

Plotly charts with the summary of all experiments, using the data stored in the mlflow database

* ```Results Charlie Hebdo.ipynb```
* ```Results Ferguson.ipynb```
* ```Results Ottawa Shooting.ipynb```
* ```Results Sydney Siege.ipynb```
*  ```Results German Wings Crash.ipynb```
 
#### 8. Summary of two events experiments (Transfer learning)

Plotly charts with the summary of all experiments, using the data stored in the mlflow database

* ```ResultsTransfer Learning.ipynb```

#### Conclusions

The graph neural networks (GNNs) delivered stronger overall performance, showing a more balanced trade-off between precision and recall across all single-event experiments. This held both over time and at time 0 metrics, when posts are first published.

In the transfer-learning experiments, where the full dataset from one event plus 30% of another event (for fine-tuning) were used to train the models, GNNs were outperformed by the LSTM and boosting models. This outcome was expected due the fact that at time 0 in a transfer-learning setup, there are few or no connected nodes available for the graph structure, because the two events do not share the same users in the PHEME dataset. Even so, once time progresses and the graph becomes more populated, the GNNs are able to maintain consistent performance.

One clear advantage of the GNNs in the results, compared with the other models, was the performance improvement in events with high class imbalance, at least within the datasets available for this study.

In summary, traditional boosting algorithms showed very low variability across the rumour-spread timeline, making them strong candidates for baseline models. LSTMs can also be competitive when the temporal dimension is available, though their training complexity and metrics variability should be considered. GNN models, especially those designed for heterogeneous graphs, can be highly effective. By leveraging propagation and structural information, they offer an additional mechanism to handle extremely imbalanced datasets an fight rumour spread early.
