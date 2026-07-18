# Paper code of "Early rumor detection: Evaluating the effectiveness of graph neural networks"

### Link to master's thesis
[Early rumor detection: Evaluating the effectiveness of graph neural networks](https://repositorio.ufu.br/handle/123456789/47362)

### Paper publication
Coming soon .. 

1. **Prepare raw data:** An extraction module was developed to retrieve key features from CSV or JSON files.

2. **NLP and word embeddings:** If textual information is available, standard NLP pre-processing techniques are applied. Word embeddings are then created using a pre-trained *embeddings* generator model, ensuring that the numerical representation of text captures both semantic and syntactic relationships.

3. **Data split and feature engineering:** The dataset is divided into training, validation, and test sets using a chronological split rather than a random one. This is essential to avoid data leakage and to better simulate real-world conditions, where predictions must be made using only past data. Once the dataset is prepared, if a Graph Neural Network (GNN) will be used for training, the tabular data is converted into a PyTorch Geometric dataset.

4. **Model training:** The selected algorithm, whether a traditional machine learning model or a GNN, is trained on the tabular or graph-structured data. Classification metrics are recorded for the training set.

5. **Time Filter Module:** A custom Python class was developed to evaluate model performance over time as the number of tweets or interactions increases. This framework filters the test data at predefined time intervals (using a timestamp column) or evaluates predictions each time a new post is published, which is the focus of the experiments in this work. For graph-based algorithms, the graph structure is updated as time progresses through the test dataset. Likewise, for tabular algorithms, the model is retrained to update dynamic ("live") features, such as the number of replies, which evolves over time.

6. **Model inference:** In the final stage, the model generates predictions, and key classification metrics are recorded using the MLflow tracking library.

The classes are stored in the ```utils.py``` file

#### 5. Experiments Single Events

* ```HAN.ipynb```
* ```GAT.ipynb```
* ```Random Forest.ipynb```
* ```Light GBM.ipynb```
* ```LSTM.ipynb```


#### 7. Summary of event experiments

Plotly charts with the summary of all experiments, using the data stored in the mlflow database

* ```Results Charlie Hebdo.ipynb```
* ```Results Ferguson.ipynb```
* ```Results Ottawa Shooting.ipynb```
* ```Results Sydney Siege.ipynb```
*  ```Results German Wings Crash.ipynb```
 

#### Data set stats

### Description of the Events

| **Event** | **Threads** | **Train rumour rate** | **Test rumour rate** |
|------------|------------:|----------------------:|---------------------:|
| Charlie Hebdo | 2,079 | 15.3 | 38.76 |
| Sydney Siege | 1,221 | 47.2 | 31.8 |
| Ferguson | 1,143 | 16.1 | 47.2 |
| Ottawa Shooting | 890 | 53.0 | 53.9 |
| Germanwings Crash | 469 | 42.5 | 67.8 |
| Putin Missing | 238 | 56.3 | 53.6 |


### Results -  Sydney Siege

| **Model** | **Recall** | **Precision** | **F1** | **FPR** |
|------------|-----------:|--------------:|--------:|---------:|
| HAN | 0.364 | 0.500 | 0.421 | 0.131 |
| LGBM | 0.366 | 0.259 | 0.304 | 0.382 |
| LSTM | 0.348 | 0.244 | 0.287 | 0.389 |
| RF | 0.347 | 0.225 | 0.273 | 0.391 |
| GAT | 0.345 | 0.215 | 0.265 | 0.426 |

### Results -Ferguson

| **Model** | **Recall** | **Precision** | **F1** | **FPR** |
|------------|-----------:|--------------:|--------:|---------:|
| LGBM | 0.538 | 0.453 | 0.492 | 0.417 |
| RF | 0.523 | 0.449 | 0.483 | 0.413 |
| GAT | 0.524 | 0.441 | 0.479 | 0.424 |
| LSTM | 0.517 | 0.435 | 0.473 | 0.425 |
| HAN | 0.112 | 0.667 | 0.192 | 0.028 |

## Results - Charlie Hebdo

| **Model** | **Recall** | **Precision** | **F1** | **FPR** |
|------------|-----------:|--------------:|--------:|---------:|
| LSTM | 0.495 | 0.416 | 0.452 | 0.354 |
| RF | 0.476 | 0.406 | 0.438 | 0.345 |
| GAT | 0.469 | 0.401 | 0.432 | 0.354 |
| LGBM | 0.422 | 0.387 | 0.404 | 0.326 |
| HAN | 0.193 | 0.511 | 0.280 | 0.080 |

### Results - Ottawa Shooting

| **Model** | **Recall** | **Precision** | **F1** | **FPR** |
|------------|-----------:|--------------:|--------:|---------:|
| HAN | 0.430 | 0.879 | 0.577 | 0.032 |
| LSTM | 0.570 | 0.538 | 0.554 | 0.377 |
| LGBM | 0.563 | 0.528 | 0.545 | 0.382 |
| GAT | 0.570 | 0.513 | 0.540 | 0.415 |
| RF | 0.547 | 0.516 | 0.531 | 0.380 |

### Results -Germanwings Crash

| **Model** | **Recall** | **Precision** | **F1** | **FPR** |
|------------|-----------:|--------------:|--------:|---------:|
| LGBM | 0.593 | 0.676 | 0.632 | 0.319 |
| GAT | 0.605 | 0.645 | 0.624 | 0.380 |
| LSTM | 0.580 | 0.662 | 0.618 | 0.329 |
| RF | 0.556 | 0.670 | 0.608 | 0.301 |
| HAN | 0.284 | 0.853 | 0.426 | 0.039 |

### Results - Putin Missing

| **Model** | **Recall** | **Precision** | **F1** | **FPR** |
|------------|-----------:|--------------:|--------:|---------:|
| LSTM | 0.786 | 0.688 | 0.733 | 0.349 |
| GAT | 0.714 | 0.577 | 0.638 | 0.478 |
| LGBM | 0.619 | 0.542 | 0.578 | 0.440 |
| RF | 0.613 | 0.543 | 0.576 | 0.410 |
| HAN | 0.381 | 0.727 | 0.500 | 0.079 |


#### 8 Conclusions
The experimental results indicate that no model consistently outperforms the others across all PHEME events at time zero. Traditional classifiers, particularly LSTM and LightGBM, achieve the highest F1 scores in four of the six events, while GAT does not outperform the non-graph baselines despite its ability to exploit relational information. This suggests that, when only the source post is available, the graph structure is too sparse to provide meaningful contextual signals, limiting the effectiveness of graph-based message passing. In contrast, HAN exhibits a markedly different behavior, consistently achieving high precision and very low false positive rates at the expense of substantially lower recall, indicating a conservative prediction strategy. Furthermore, considerable variation in rumor prevalence between the training and test sets—such as Charlie Hebdo (15.3% vs. 38.8%) and Ferguson (16.1% vs. 47.2%)—likely introduces distribution shift, contributing to performance variability across events. Overall, these findings suggest that textual content remains the dominant source of predictive information at the earliest stage of rumor propagation, while the advantages of graph neural networks are likely to emerge only as richer conversational and propagation structures become available over time.
