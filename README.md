# rumour-detection-pheme
Atualização em andamento

#### 1. JSON to CSV files.
Notebook to create csv dataset from JSON raw files ```Dataset creation from files.ipynb```. The output, as example is the  ```charliehebdo-all-rnr-threads.csv```, where each line is an reaction with all source tweet features and the reply features.

#### 2. EDA
Exploratory Data Analysis to understand data distribuition and differences between rumours and not rumours propagation

#### 3. Pre processing and Feature Engineering

The notebook ```Pre Processing - Feature Engineering.ipynb``` show the process of NLP cleaning to final generate the word embeddings for the text of source tweets and reply text. It also includes some feature creation like time difference between the interation and the source tweet, number of replies and the time of first reply
