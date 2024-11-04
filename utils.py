import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

class LoadRumoursDataset:
    def __init__(self, file_path_replies, file_path_posts, time_cut):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.df_replies = None
        self.df_posts = None
        self.df_final = None

    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        post_features = ['followers','favorite_count','retweet_count','verified','rumour','id','embeddings_avg']
        
        reply_features = ['reply_followers','reply_user_id','reply_verified','time_diff','reply_embeddings_avg','id']

        filtered_replies = self.df_replies[reply_features][self.df_replies.time_diff < self.time_cut]
        grouped_replies = filtered_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        self.df_posts = self.df_posts[post_features]
        self.df_final = self.df_posts.merge(grouped_replies, on="id", how="left")
        self.df_final['replies'] = self.df_final['replies'].fillna(0)
        self.df_final['first_time_diff'] = self.df_final['first_time_diff'].fillna(0)
        self.df_final = self.df_final.drop(columns=['id'])

        
        # Initialize the Robust Scaler
        scaler = RobustScaler()
        
        # Assuming data is a DataFrame containing your dataset
        scaled_features = ['followers', 'favorite_count', 'retweet_count', 'first_time_diff']
        # Convert the scaled features back to a DataFrame
        scaled_data = pd.DataFrame(scaler.fit_transform(self.df_final [scaled_features]),columns=scaled_features)    
        
        self.df_final [scaled_features] = scaled_data


        # One-hot encoding
        self.df_final ['verified'] = self.df_final ['verified'].astype('str').str.\
                     replace(' ', '').replace('True', '1').replace('False', '0')\
                     .astype('int64')
        
        self.df_final  = pd.concat([self.df_final , pd.get_dummies(\
                                  self.df_final ["verified"],dtype=int)], axis=1, join='inner')
        self.df_final .drop(["verified"], axis=1, inplace=True)
        self.df_final .rename(columns={1:'verified',0:'no_verified'},inplace=True)

    def get_final_dataframe(self):
        return self.df_final



class HeteroDataProcessor:
    def __init__(self, file_path_replies, file_path_posts, time_cut=15):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.df_replies = None
        self.df_posts = None
        self.post_map = None
        self.reply_user_map = None

    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        # Filter and group replies
        self.df_replies = self.df_replies[reply_features][self.df_replies.time_diff < self.time_cut]
        grouped_replies = self.df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge posts and replies
        self.df_posts = self.df_posts[post_features].merge(grouped_replies, on="id", how="left")
        self.df_posts['replies'] = self.df_posts['replies'].fillna(0)
        self.df_posts['first_time_diff'] = self.df_posts['first_time_diff'].fillna(0)

        # One-hot encoding for verified columns
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        self.df_replies['reply_verified'] = self.df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_replies = pd.concat([self.df_replies, pd.get_dummies(self.df_replies["reply_verified"], dtype=int)], axis=1)
        self.df_replies.drop(["reply_verified"], axis=1, inplace=True)
        self.df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        # Mapping post ids
        self.post_map = {value: i for i, value in enumerate(self.df_posts['id'].unique())}
        self.df_replies["id"] = self.df_replies['id'].map(self.post_map).astype(int)

        # Mapping reply user ids
        self.reply_user_map = {value: i for i, value in enumerate(self.df_replies['reply_user_id'].unique())}
        self.df_replies["reply_user_id"] = self.df_replies["reply_user_id"].map(self.reply_user_map)

    def create_features(self):
        post_features = self.df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]


        # Initialize the Robust Scaler
        scaler = RobustScaler()
        
        # Assuming data is a DataFrame containing your dataset
        scaled_features = scaler.fit_transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        
        # Convert the scaled features back to a DataFrame
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        
        # Add the binary features back to the scaled data
        scaled_data['no_verified'] = post_features['no_verified']
        scaled_data['verified'] = post_features['verified']
        post_features = scaled_data

        post_embeddings = np.array(self.df_posts['embeddings_avg'].tolist())
        #post_features = self.scaler.fit_transform(post_features)
        x1 = np.concatenate((post_features, post_embeddings), axis=1)

        scaler = RobustScaler()
        reply_features = self.df_replies[["reply_followers", "reply_no_verified", "reply_verified","time_diff"]]
        reply_features[['reply_followers','time_diff']] = scaler.fit_transform(reply_features[['reply_followers','time_diff']])

        reply_embeddings = np.array(self.df_replies['reply_embeddings_avg'].tolist())
        #reply_features = self.scaler.transform(reply_features)
        x2 = np.concatenate((reply_features, reply_embeddings), axis=1)

        return x1, x2

    def create_heterodata(self, x1, x2):
        y = self.df_posts['rumour'].to_numpy()
        edge_index = self.df_replies[["id", "reply_user_id"]].values.transpose()

        num_rows = x1.shape[0]
        indices = np.arange(num_rows)
        np.random.shuffle(indices)
        train_end = int(0.70 * num_rows)
        val_end = train_end + int(0.15 * num_rows)
        train_indices, val_indices, test_indices = indices[:train_end], indices[train_end:val_end], indices[val_end:]

        train_mask = np.zeros(num_rows, dtype=bool)
        val_mask = np.zeros(num_rows, dtype=bool)
        test_mask = np.zeros(num_rows, dtype=bool)
        train_mask[train_indices], val_mask[val_indices], test_mask[test_indices] = True, True, True

        data = HeteroData()
        data['id'].x = torch.tensor(x1, dtype=torch.float32)
        data['id'].y =  torch.from_numpy(y)
        data['id'].train_mask = torch.tensor(train_mask)
        data['id'].val_mask = torch.tensor(val_mask) 
        data['id'].test_mask = torch.tensor(test_mask)
        data['reply_user_id'].x = torch.tensor(x2, dtype=torch.float32)
        data['id', 'retweet', 'reply_user_id'].edge_index = torch.from_numpy(edge_index.reshape(2,len(x2)))
        data = T.ToUndirected()(data)

        return data

    def process(self):
        self.load_data()
        self.process_data()
        x1, x2 = self.create_features()
        return self.create_heterodata(x1, x2)



class LoadRumoursDatasetFilterNode:
    def __init__(self, file_path_replies, file_path_posts, time_cut):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.df_replies = None
        self.df_posts = None
        self.df_final = None

    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        post_features = ['followers','favorite_count','retweet_count','verified','rumour','id','embeddings_avg']
        
        
        self.df_replies['min_since_fst_post'] = round((self.df_replies['time'] - self.df_replies['time'].min())\
                        .dt.total_seconds() / 60,2)
        
        reply_features = ['reply_followers','reply_user_id','reply_verified','time_diff','reply_embeddings_avg',\
                          'min_since_fst_post','id','time']

        filtered_replies = self.df_replies[reply_features][(self.df_replies.time_diff <= self.time_cut)&\
                                                           (self.df_replies.min_since_fst_post <= self.time_cut)]
        
        grouped_replies = filtered_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        self.df_posts = self.df_posts[post_features]
        self.df_final = self.df_posts.merge(grouped_replies, on="id", how="inner")
        self.df_final['replies'] = self.df_final['replies'].fillna(0)
        self.df_final['first_time_diff'] = self.df_final['first_time_diff'].fillna(0)
        self.df_final = self.df_final.drop(columns=['id'])

        
        # Initialize the Robust Scaler
        scaler = RobustScaler()
        
        # Assuming data is a DataFrame containing your dataset
        scaled_features = ['followers', 'favorite_count', 'retweet_count', 'first_time_diff']
        # Convert the scaled features back to a DataFrame
        scaled_data = pd.DataFrame(scaler.fit_transform(self.df_final [scaled_features]),columns=scaled_features)    
        
        self.df_final [scaled_features] = scaled_data


        # One-hot encoding
        self.df_final ['verified'] = self.df_final ['verified'].astype('str').str.\
                     replace(' ', '').replace('True', '1').replace('False', '0')\
                     .astype('int64')
        
        self.df_final  = pd.concat([self.df_final , pd.get_dummies(\
                                  self.df_final ["verified"],dtype=int)], axis=1, join='inner')
        self.df_final .drop(["verified"], axis=1, inplace=True)
        self.df_final .rename(columns={1:'verified',0:'no_verified'},inplace=True)

    def get_final_dataframe(self):
        return self.df_final

class HeteroDataProcessorFilterNode:
    def __init__(self, file_path_replies, file_path_posts, time_cut=15):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.df_replies = None
        self.df_posts = None
        self.post_map = None
        self.reply_user_map = None

    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        # Filter and group replies
        self.df_replies['min_since_fst_post'] = round((self.df_replies['time'] - self.df_replies['time'].min())\
                        .dt.total_seconds() / 60,2)
        
        self.df_replies = self.df_replies[reply_features][(self.df_replies.time_diff <= self.time_cut)&\
                                                           (self.df_replies.min_since_fst_post <= self.time_cut)]
        grouped_replies = self.df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge posts and replies
        self.df_posts = self.df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        self.df_posts['replies'] = self.df_posts['replies'].fillna(0)
        self.df_posts['first_time_diff'] = self.df_posts['first_time_diff'].fillna(0)

        # One-hot encoding for verified columns
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        self.df_replies['reply_verified'] = self.df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_replies = pd.concat([self.df_replies, pd.get_dummies(self.df_replies["reply_verified"], dtype=int)], axis=1)
        self.df_replies.drop(["reply_verified"], axis=1, inplace=True)
        self.df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        # Mapping post ids
        self.post_map = {value: i for i, value in enumerate(self.df_posts['id'].unique())}
        self.df_replies["id"] = self.df_replies['id'].map(self.post_map).astype(int)

        # Mapping reply user ids
        self.reply_user_map = {value: i for i, value in enumerate(self.df_replies['reply_user_id'].unique())}
        self.df_replies["reply_user_id"] = self.df_replies["reply_user_id"].map(self.reply_user_map)

    def create_features(self):
        post_features = self.df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]


        # Initialize the Robust Scaler
        scaler = RobustScaler()
        
        # Assuming data is a DataFrame containing your dataset
        scaled_features = scaler.fit_transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        
        # Convert the scaled features back to a DataFrame
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        
        # Add the binary features back to the scaled data
        scaled_data['no_verified'] = post_features['no_verified']
        scaled_data['verified'] = post_features['verified']
        post_features = scaled_data

        post_embeddings = np.array(self.df_posts['embeddings_avg'].tolist())
        #post_features = self.scaler.fit_transform(post_features)
        x1 = np.concatenate((post_features, post_embeddings), axis=1)

        scaler = RobustScaler()
        reply_features = self.df_replies[["reply_followers", "reply_no_verified", "reply_verified","time_diff"]]
        reply_features[['reply_followers','time_diff']] = scaler.fit_transform(reply_features[['reply_followers','time_diff']])

        reply_embeddings = np.array(self.df_replies['reply_embeddings_avg'].tolist())
        #reply_features = self.scaler.transform(reply_features)
        x2 = np.concatenate((reply_features, reply_embeddings), axis=1)

        return x1, x2

    def create_heterodata(self, x1, x2):
        y = self.df_posts['rumour'].to_numpy()
        edge_index = self.df_replies[["id", "reply_user_id"]].values.transpose()

        num_rows = x1.shape[0]
        indices = np.arange(num_rows)
        np.random.shuffle(indices)
        train_end = int(0.70 * num_rows)
        val_end = train_end + int(0.15 * num_rows)
        train_indices, val_indices, test_indices = indices[:train_end], indices[train_end:val_end], indices[val_end:]

        train_mask = np.zeros(num_rows, dtype=bool)
        val_mask = np.zeros(num_rows, dtype=bool)
        test_mask = np.zeros(num_rows, dtype=bool)
        train_mask[train_indices], val_mask[val_indices], test_mask[test_indices] = True, True, True

        data = HeteroData()
        data['id'].x = torch.tensor(x1, dtype=torch.float32)
        data['id'].y =  torch.from_numpy(y)
        data['id'].train_mask = torch.tensor(train_mask)
        data['id'].val_mask = torch.tensor(val_mask) 
        data['id'].test_mask = torch.tensor(test_mask)
        data['reply_user_id'].x = torch.tensor(x2, dtype=torch.float32)
        data['id', 'retweet', 'reply_user_id'].edge_index = torch.from_numpy(edge_index.reshape(2,len(x2)))
        data = T.ToUndirected()(data)

        return data

    def process(self):
        self.load_data()
        self.process_data()
        x1, x2 = self.create_features()
        return self.create_heterodata(x1, x2)



class HeteroDataProcessorFilterNodeonTest:
    def __init__(self, file_path_replies, file_path_posts, time_cut=15):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.df_replies = None
        self.df_posts = None

    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        # Define post and reply features
        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        # Filter and group replies
        self.df_replies['min_since_fst_post'] = round(
            (self.df_replies['time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
        grouped_replies = self.df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge posts and replies
        self.df_posts = self.df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        self.df_posts['replies'] = self.df_posts['replies'].fillna(0)
        self.df_posts['first_time_diff'] = self.df_posts['first_time_diff'].fillna(0)

        # One-hot encoding for verified columns
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        self.df_replies['reply_verified'] = self.df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_replies = pd.concat([self.df_replies, pd.get_dummies(self.df_replies["reply_verified"], dtype=int)], axis=1)
        self.df_replies.drop(["reply_verified"], axis=1, inplace=True)
        self.df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        # Train/test split
        train, not_train = train_test_split(self.df_posts, test_size=0.3, random_state=42, stratify=self.df_posts['rumour'])
        val, test = train_test_split(not_train, test_size=0.5, random_state=42, stratify=not_train['rumour'])

        # Post features processing
        post_features = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        scaler_posts = RobustScaler()
        scaled_features =scaler_posts.fit_transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        scaled_data['no_verified'] = np.array(train['no_verified'])
        scaled_data['verified'] = np.array(train['verified'])
        post_features = scaled_data
        post_embeddings = np.array(train['embeddings_avg'].tolist())
        x1 = np.concatenate((post_features, post_embeddings), axis=1)

        # Reply features processing
        scaler_replies = RobustScaler()
        reply_features = self.df_replies[self.df_replies.id.isin(np.array(train.id))][["reply_followers", "reply_no_verified", "reply_verified", "time_diff"]]
        reply_features[['reply_followers', 'time_diff']] = scaler_replies.fit_transform(reply_features[['reply_followers', 'time_diff']])
        reply_embeddings = np.array(self.df_replies[self.df_replies.id.isin(np.array(train.id))]['reply_embeddings_avg'].tolist())
        x2 = np.concatenate((reply_features, reply_embeddings), axis=1)

        # Test/validation data preparation
        test_val_df_replies = pd.read_pickle(self.file_path_replies)
        test_val_df_posts = pd.read_pickle(self.file_path_posts)
        test_val_df_posts = test_val_df_posts[~test_val_df_posts.id.isin(train.id)]
        test_val_df_replies = test_val_df_replies[~test_val_df_replies.id.isin(train.id)]

        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        test_val_reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']
        
        test_val_df_replies['min_since_fst_post'] = round((test_val_df_replies['time'] - test_val_df_replies['time'].min())\
        .dt.total_seconds() / 60,2)
        
        test_val_df_replies = test_val_df_replies[test_val_reply_features][(test_val_df_replies.time_diff <= self.time_cut)&\
           (test_val_df_replies.min_since_fst_post <= self.time_cut)]
        grouped_replies = test_val_df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()
        
        test_val_df_posts = test_val_df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        test_val_df_posts['replies'] = test_val_df_posts['replies'].fillna(0)
        test_val_df_posts['first_time_diff'] = test_val_df_posts['first_time_diff'].fillna(0)
        
        # One-hot encoding for verified columns
        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)
        
        test_val_df_replies['reply_verified'] = test_val_df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_replies = pd.concat([test_val_df_replies, pd.get_dummies(test_val_df_replies["reply_verified"], dtype=int)], axis=1)
        test_val_df_replies.drop(["reply_verified"], axis=1, inplace=True)
        test_val_df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)
        
        test_val_df_posts = test_val_df_posts.merge(pd.concat([val,test])[['id']].reset_index(),on='id',how='left')
        test_val_df_posts.set_index('index',drop=True,inplace=True)
        
        post_features = test_val_df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        
        
        scaled_features = scaler_posts.transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        
        # Convert the scaled features back to a DataFrame
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        
        # Add the binary features back to the scaled data
        scaled_data['no_verified'] = np.array(post_features['no_verified'])
        scaled_data['verified'] = np.array(post_features['verified'])
        post_features = scaled_data
        
        post_embeddings = np.array(test_val_df_posts['embeddings_avg'].tolist())
        #post_features = scaler.fit_transform(post_features)
        x3 = np.concatenate((post_features, post_embeddings), axis=1)
        
        test_val_reply_features =  test_val_df_replies[["reply_followers", "reply_no_verified","reply_verified","time_diff"]]
        test_val_reply_features[['reply_followers','time_diff']] = scaler_replies.transform(test_val_reply_features[['reply_followers','time_diff']])
        
        test_val_reply_embeddings = np.array(test_val_df_replies['reply_embeddings_avg'].tolist())
        x4 = np.concatenate((test_val_reply_features, test_val_reply_embeddings), axis=1)
        
        
        # Mapping post ids
        post_map = {value: i for i, value in enumerate(pd.concat([train[['id']],test_val_df_posts[['id']]])['id'].unique())}
        df_replies_edges = pd.concat([self.df_replies[self.df_replies.id.isin(np.array(train.id))][["id", "reply_user_id"]],\
                                test_val_df_replies[["id", "reply_user_id"]]])
        
        df_replies_edges["id"] = df_replies_edges['id'].map(post_map).astype(int)
        
        # Mapping reply user ids
        reply_user_map = {value: i for i, value in enumerate(df_replies_edges['reply_user_id'].unique())}
        df_replies_edges["reply_user_id"] = df_replies_edges["reply_user_id"].map(reply_user_map)

        return train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4


    def create_heterodata(self,train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4):

        y = pd.concat([train['rumour'],test_val_df_posts['rumour']]).to_numpy()
        edge_index = df_replies_edges.values.transpose()
        x = np.concatenate((x1,x3))
        x_reply = np.concatenate((x2,x4))
            
        num_rows = x.shape[0]
        train_mask = np.zeros(num_rows, dtype=bool)
        val_mask = np.zeros(num_rows, dtype=bool)
        test_mask = np.zeros(num_rows, dtype=bool)
        train_mask[:-x3.shape[0]]=True
        val_mask[-x3.shape[0]:-int(x3.shape[0]/2)]=True
        test_mask[-int(x3.shape[0]/2):]=True
            
        data = HeteroData()
        data['id'].x = torch.tensor(x, dtype=torch.float32)
        data['id'].y =  torch.from_numpy(y)
        data['id'].train_mask = torch.tensor(train_mask)
        data['id'].val_mask = torch.tensor(val_mask) 
        data['id'].test_mask = torch.tensor(test_mask)
        data['reply_user_id'].x = torch.tensor(x_reply, dtype=torch.float32)
        data['id', 'retweet', 'reply_user_id'].edge_index = torch.from_numpy(edge_index.reshape(2,len(x_reply)))
        data = T.ToUndirected()(data)
        
        return data

    def process(self):
            
        self.load_data()
        train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4 = self.process_data()
        return self.create_heterodata(train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4)





class HeteroDataProcessorFilterNodeonTestV2:
    #Different time cuts 
    def __init__(self, file_path_replies, file_path_posts, time_cut_replies=15,time_cut_posts=15):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut_replies = time_cut_replies
        self.time_cut_posts = time_cut_posts
        self.df_replies = None
        self.df_posts = None

    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        # Define post and reply features
        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        # Filter and group replies
        self.df_replies['min_since_fst_post'] = round(
            (self.df_replies['time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
        grouped_replies = self.df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge posts and replies
        self.df_posts = self.df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        self.df_posts['replies'] = self.df_posts['replies'].fillna(0)
        self.df_posts['first_time_diff'] = self.df_posts['first_time_diff'].fillna(0)

        # One-hot encoding for verified columns
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        self.df_replies['reply_verified'] = self.df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_replies = pd.concat([self.df_replies, pd.get_dummies(self.df_replies["reply_verified"], dtype=int)], axis=1)
        self.df_replies.drop(["reply_verified"], axis=1, inplace=True)
        self.df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        # Train/test split
        train, not_train = train_test_split(self.df_posts, test_size=0.3, random_state=42, stratify=self.df_posts['rumour'])
        val, test = train_test_split(not_train, test_size=0.5, random_state=42, stratify=not_train['rumour'])

        # Post features processing
        post_features = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        scaler_posts = RobustScaler()
        scaled_features =scaler_posts.fit_transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        scaled_data['no_verified'] = np.array(train['no_verified'])
        scaled_data['verified'] = np.array(train['verified'])
        post_features = scaled_data
        post_embeddings = np.array(train['embeddings_avg'].tolist())
        x1 = np.concatenate((post_features, post_embeddings), axis=1)

        # Reply features processing
        scaler_replies = RobustScaler()
        reply_features = self.df_replies[self.df_replies.id.isin(np.array(train.id))][["reply_followers", "reply_no_verified", "reply_verified", "time_diff"]]
        reply_features[['reply_followers', 'time_diff']] = scaler_replies.fit_transform(reply_features[['reply_followers', 'time_diff']])
        reply_embeddings = np.array(self.df_replies[self.df_replies.id.isin(np.array(train.id))]['reply_embeddings_avg'].tolist())
        x2 = np.concatenate((reply_features, reply_embeddings), axis=1)

        # Test/validation data preparation
        test_val_df_replies = pd.read_pickle(self.file_path_replies)
        test_val_df_posts = pd.read_pickle(self.file_path_posts)
        test_val_df_posts = test_val_df_posts[~test_val_df_posts.id.isin(train.id)]
        test_val_df_replies = test_val_df_replies[~test_val_df_replies.id.isin(train.id)]

        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        test_val_reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']
        
        test_val_df_replies['min_since_fst_post'] = round((test_val_df_replies['time'] - test_val_df_replies['time'].min())\
        .dt.total_seconds() / 60,2)
        
        test_val_df_replies = test_val_df_replies[test_val_reply_features][(test_val_df_replies.time_diff <= self.time_cut_replies)&\
           (test_val_df_replies.min_since_fst_post <= self.time_cut_posts)]
        grouped_replies = test_val_df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()
        
        test_val_df_posts = test_val_df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        test_val_df_posts['replies'] = test_val_df_posts['replies'].fillna(0)
        test_val_df_posts['first_time_diff'] = test_val_df_posts['first_time_diff'].fillna(0)
        
        # One-hot encoding for verified columns
        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)
        
        test_val_df_replies['reply_verified'] = test_val_df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_replies = pd.concat([test_val_df_replies, pd.get_dummies(test_val_df_replies["reply_verified"], dtype=int)], axis=1)
        test_val_df_replies.drop(["reply_verified"], axis=1, inplace=True)
        test_val_df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)
        
        test_val_df_posts = test_val_df_posts.merge(pd.concat([val,test])[['id']].reset_index(),on='id',how='left')
        test_val_df_posts.set_index('index',drop=True,inplace=True)
        
        post_features = test_val_df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        
        
        scaled_features = scaler_posts.transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        
        # Convert the scaled features back to a DataFrame
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        
        # Add the binary features back to the scaled data
        scaled_data['no_verified'] = np.array(post_features['no_verified'])
        scaled_data['verified'] = np.array(post_features['verified'])
        post_features = scaled_data
        
        post_embeddings = np.array(test_val_df_posts['embeddings_avg'].tolist())
        #post_features = scaler.fit_transform(post_features)
        x3 = np.concatenate((post_features, post_embeddings), axis=1)
        
        test_val_reply_features =  test_val_df_replies[["reply_followers", "reply_no_verified","reply_verified","time_diff"]]
        test_val_reply_features[['reply_followers','time_diff']] = scaler_replies.transform(test_val_reply_features[['reply_followers','time_diff']])
        
        test_val_reply_embeddings = np.array(test_val_df_replies['reply_embeddings_avg'].tolist())
        x4 = np.concatenate((test_val_reply_features, test_val_reply_embeddings), axis=1)
        
        
        # Mapping post ids
        post_map = {value: i for i, value in enumerate(pd.concat([train[['id']],test_val_df_posts[['id']]])['id'].unique())}
        df_replies_edges = pd.concat([self.df_replies[self.df_replies.id.isin(np.array(train.id))][["id", "reply_user_id"]],\
                                test_val_df_replies[["id", "reply_user_id"]]])
        
        df_replies_edges["id"] = df_replies_edges['id'].map(post_map).astype(int)
        
        # Mapping reply user ids
        reply_user_map = {value: i for i, value in enumerate(df_replies_edges['reply_user_id'].unique())}
        df_replies_edges["reply_user_id"] = df_replies_edges["reply_user_id"].map(reply_user_map)

        return train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4


    def create_heterodata(self,train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4):

        y = pd.concat([train['rumour'],test_val_df_posts['rumour']]).to_numpy()
        edge_index = df_replies_edges.values.transpose()
        x = np.concatenate((x1,x3))
        x_reply = np.concatenate((x2,x4))
            
        num_rows = x.shape[0]
        train_mask = np.zeros(num_rows, dtype=bool)
        val_mask = np.zeros(num_rows, dtype=bool)
        test_mask = np.zeros(num_rows, dtype=bool)
        train_mask[:-x3.shape[0]]=True
        val_mask[-x3.shape[0]:-int(x3.shape[0]/2)]=True
        test_mask[-int(x3.shape[0]/2):]=True
            
        data = HeteroData()
        data['id'].x = torch.tensor(x, dtype=torch.float32)
        data['id'].y =  torch.from_numpy(y)
        data['id'].train_mask = torch.tensor(train_mask)
        data['id'].val_mask = torch.tensor(val_mask) 
        data['id'].test_mask = torch.tensor(test_mask)
        data['reply_user_id'].x = torch.tensor(x_reply, dtype=torch.float32)
        data['id', 'retweet', 'reply_user_id'].edge_index = torch.from_numpy(edge_index.reshape(2,len(x_reply)))
        data = T.ToUndirected()(data)
        
        return data

    def process(self):
            
        self.load_data()
        train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4 = self.process_data()
        return self.create_heterodata(train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4)
    


    


class LoadRumoursDatasetFilterNodeonTestV2:
    def __init__(self, file_path_replies, file_path_posts, time_cut_replies=80, time_cut_posts=15):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut_replies = time_cut_replies
        self.time_cut_posts = time_cut_posts
        self.scaler_posts = RobustScaler()
        
    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)
        
    def process_data(self):
        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']
        
        # Compute minutes since first post
        self.df_replies['min_since_fst_post'] = round(
            (self.df_replies['time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)

        # Group replies
        grouped_replies = self.df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge posts and replies
        self.df_posts = self.df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        self.df_posts['replies'] = self.df_posts['replies'].fillna(0)
        self.df_posts['first_time_diff'] = self.df_posts['first_time_diff'].fillna(0)

        # One-hot encode 'verified' column
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)
        
        train, not_train = train_test_split(self.df_posts, test_size=0.3, random_state=42, stratify=self.df_posts['rumour'])
        val, test = train_test_split(not_train, test_size=0.5, random_state=42, stratify=not_train['rumour'])
        
        post_features = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", 
                               "rumour", "embeddings_avg", "replies", "first_time_diff"]]
        scaled_features = self.scaler_posts.fit_transform(post_features[['followers', 'favorite_count', 
                                                                         'retweet_count', 'first_time_diff', 'replies']])
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 
                                                             'first_time_diff', 'replies'])
        scaled_data['no_verified'] = np.array(train['no_verified'])
        scaled_data['verified'] = np.array(train['verified'])
        scaled_data['embeddings_avg'] = np.array(train['embeddings_avg'])
        scaled_data['rumour'] = np.array(train['rumour'])
        
        self.train_dataset = scaled_data

        test_val_df_replies = pd.read_pickle(self.file_path_replies)
        test_val_df_posts = pd.read_pickle(self.file_path_posts)
        test_val_df_posts = test_val_df_posts[~test_val_df_posts.id.isin(train.id)]
        test_val_df_replies = test_val_df_replies[~test_val_df_replies.id.isin(train.id)]

        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        test_val_reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        test_val_df_replies['min_since_fst_post'] = round((test_val_df_replies['time'] - test_val_df_replies['time'].min())
                                                          .dt.total_seconds() / 60, 2)

        test_val_df_replies = test_val_df_replies[test_val_reply_features][(test_val_df_replies.time_diff <= self.time_cut_replies) &
                                                                          (test_val_df_replies.min_since_fst_post <= self.time_cut_posts)]

        grouped_replies = test_val_df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        test_val_df_posts = test_val_df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        test_val_df_posts['replies'] = test_val_df_posts['replies'].fillna(0)
        test_val_df_posts['first_time_diff'] = test_val_df_posts['first_time_diff'].fillna(0)

        # One-hot encoding for verified columns
        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        test_val_df_posts = test_val_df_posts.merge(pd.concat([val, train])[['id']].reset_index(), on='id', how='left')
        test_val_df_posts.set_index('index', drop=True, inplace=True)

        post_features = test_val_df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified", 
                                           'replies', "first_time_diff", "embeddings_avg", "rumour"]]

        scaled_features = self.scaler_posts.transform(post_features[['followers', 'favorite_count', 
                                                                     'retweet_count', 'first_time_diff', 'replies']])

        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 
                                                             'first_time_diff', 'replies'])

        scaled_data['no_verified'] = np.array(post_features['no_verified'])
        scaled_data['verified'] = np.array(post_features['verified'])
        scaled_data['embeddings_avg'] = np.array(post_features['embeddings_avg'])
        scaled_data['rumour'] = np.array(post_features['rumour'])

        self.test_dataset = scaled_data
        
    def get_final_dataframes(self):
        return self.train_dataset, self.test_dataset


class Load_Rumours_Dataset_filtering_since_first_post:
    def __init__(self, file_path_replies, file_path_posts, time_cut):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut=time_cut
        self.scaler_posts = RobustScaler()
        
    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)
        
    def process_data(self):
        features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']
        
        # Compute minutes since first post
        self.df_replies['min_since_fst_post'] = round(
            (self.df_replies['time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
        
        self.df_replies['reply_min_since_fst_post'] = round(
            (self.df_replies['reply_time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
        

        # Group replies
        grouped_replies = self.df_replies.groupby(['id','min_since_fst_post']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge posts and replies
        self.df_posts = self.df_posts[features].merge(grouped_replies, on="id", how="inner")
        self.df_posts['replies'] = self.df_posts['replies'].fillna(0)
        self.df_posts['first_time_diff'] = self.df_posts['first_time_diff'].fillna(0)
        self.df_posts['min_since_fst_post'] = self.df_posts['min_since_fst_post'].fillna(0)

        # One-hot encode 'verified' column
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)
        
        train, not_train = train_test_split(self.df_posts, test_size=0.3, random_state=42, stratify=self.df_posts['rumour'])
        val, test = train_test_split(not_train, test_size=0.5, random_state=42, stratify=not_train['rumour'])
        
        post_features = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", 
                               "rumour", "embeddings_avg", "replies", "first_time_diff","min_since_fst_post"]]
        
        scaled_features = self.scaler_posts.fit_transform(post_features[['followers', 'favorite_count','retweet_count', \
                                                                         'first_time_diff', 'replies','min_since_fst_post']])
        
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count','retweet_count', 'first_time_diff',\
                                                             'replies','min_since_fst_post'])
        scaled_data['no_verified'] = np.array(train['no_verified'])
        scaled_data['verified'] = np.array(train['verified'])
        scaled_data['embeddings_avg'] = np.array(train['embeddings_avg'])
        scaled_data['rumour'] = np.array(train['rumour'])
        
        self.train_dataset = scaled_data

        test_val_df_replies = pd.read_pickle(self.file_path_replies)
        test_val_df_posts = pd.read_pickle(self.file_path_posts)
        test_val_df_posts = test_val_df_posts[~test_val_df_posts.id.isin(train.id)]
        test_val_df_replies = test_val_df_replies[~test_val_df_replies.id.isin(train.id)]

        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        test_val_reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', \
                                   'reply_embeddings_avg', 'id']

        test_val_df_replies['min_since_fst_post'] = round((test_val_df_replies['time'] - test_val_df_replies['time'].min())
                                                          .dt.total_seconds() / 60, 2)
        
        test_val_df_replies['reply_min_since_fst_post'] = round((test_val_df_replies['reply_time']\
                            - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)

        test_val_df_replies = test_val_df_replies[(test_val_df_replies.reply_min_since_fst_post <= self.time_cut) &
                                                (test_val_df_replies.min_since_fst_post <= self.time_cut)]

        grouped_replies = test_val_df_replies.groupby(['id','min_since_fst_post']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        test_val_df_posts = test_val_df_posts[features].merge(grouped_replies, on="id", how="inner")
        test_val_df_posts['replies'] = test_val_df_posts['replies'].fillna(0)
        test_val_df_posts['first_time_diff'] = test_val_df_posts['first_time_diff'].fillna(0)
        test_val_df_posts['min_since_fst_post'] = test_val_df_posts['min_since_fst_post'].fillna(0)

        # One-hot encoding for verified columns
        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str)\
                                        .replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        test_val_df_posts = test_val_df_posts.merge(pd.concat([val, train])[['id']].reset_index(), on='id', how='left')
        test_val_df_posts.set_index('index', drop=True, inplace=True)
        
        

        post_features = test_val_df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified", 
                                           'replies', "first_time_diff", "embeddings_avg", "rumour","min_since_fst_post"]]

        scaled_features = self.scaler_posts.transform(post_features[['followers', 'favorite_count','retweet_count',\
                                                                     'first_time_diff','replies','min_since_fst_post']])

        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count','retweet_count',\
                                                             'first_time_diff', 'replies','min_since_fst_post'])

        scaled_data['no_verified'] = np.array(post_features['no_verified'])
        scaled_data['verified'] = np.array(post_features['verified'])
        scaled_data['embeddings_avg'] = np.array(post_features['embeddings_avg'])
        scaled_data['rumour'] = np.array(post_features['rumour'])

        self.test_dataset = scaled_data
        
    def get_final_dataframes(self):
        return self.train_dataset, self.test_dataset
    






    
class Hetero_Data_Processor_Filter_on_Test_since_first_post:
    def __init__(self, file_path_replies, file_path_posts, time_cut=15):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.df_replies = None
        self.df_posts = None

    def load_data(self):
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        # Define post and reply features
        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        # Filter and group replies
        self.df_replies['min_since_fst_post'] = round(
            (self.df_replies['time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
        
        self.df_replies['reply_min_since_fst_post'] = round(
            (self.df_replies['reply_time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
            
            
        grouped_replies = self.df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge posts and replies
        self.df_posts = self.df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        self.df_posts['replies'] = self.df_posts['replies'].fillna(0)
        self.df_posts['first_time_diff'] = self.df_posts['first_time_diff'].fillna(0)

        # One-hot encoding for verified columns
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        self.df_replies['reply_verified'] = self.df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_replies = pd.concat([self.df_replies, pd.get_dummies(self.df_replies["reply_verified"], dtype=int)], axis=1)
        self.df_replies.drop(["reply_verified"], axis=1, inplace=True)
        self.df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        # Train/test split
        train, not_train = train_test_split(self.df_posts, test_size=0.3, random_state=42, stratify=self.df_posts['rumour'])
        val, test = train_test_split(not_train, test_size=0.5, random_state=42, stratify=not_train['rumour'])

        # Post features processing
        post_features = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        scaler_posts = RobustScaler()
        scaled_features =scaler_posts.fit_transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        scaled_data['no_verified'] = np.array(train['no_verified'])
        scaled_data['verified'] = np.array(train['verified'])
        post_features = scaled_data
        post_embeddings = np.array(train['embeddings_avg'].tolist())
        x1 = np.concatenate((post_features, post_embeddings), axis=1)

        # Reply features processing
        scaler_replies = RobustScaler()
        reply_features = self.df_replies[self.df_replies.id.isin(np.array(train.id))][["reply_followers", "reply_no_verified", "reply_verified", "time_diff"]]
        reply_features[['reply_followers', 'time_diff']] = scaler_replies.fit_transform(reply_features[['reply_followers', 'time_diff']])
        reply_embeddings = np.array(self.df_replies[self.df_replies.id.isin(np.array(train.id))]['reply_embeddings_avg'].tolist())
        x2 = np.concatenate((reply_features, reply_embeddings), axis=1)

        # Test/validation data preparation
        test_val_df_replies = pd.read_pickle(self.file_path_replies)
        test_val_df_posts = pd.read_pickle(self.file_path_posts)
        test_val_df_posts = test_val_df_posts[~test_val_df_posts.id.isin(train.id)]
        test_val_df_replies = test_val_df_replies[~test_val_df_replies.id.isin(train.id)]

        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        test_val_reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']
        
        test_val_df_replies['min_since_fst_post'] = round((test_val_df_replies['time'] - test_val_df_replies['time'].min())\
        .dt.total_seconds() / 60,2)
    
        
        test_val_df_replies['reply_min_since_fst_post'] = round((test_val_df_replies['reply_time'] - test_val_df_replies['time'].min())\
        .dt.total_seconds() / 60,2)
        
        test_val_df_replies = test_val_df_replies[test_val_reply_features][(\
                            test_val_df_replies.reply_min_since_fst_post <= self.time_cut)&\
                            (test_val_df_replies.min_since_fst_post <= self.time_cut)]
            
        grouped_replies = test_val_df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()
        
        test_val_df_posts = test_val_df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        test_val_df_posts['replies'] = test_val_df_posts['replies'].fillna(0)
        test_val_df_posts['first_time_diff'] = test_val_df_posts['first_time_diff'].fillna(0)
        
        # One-hot encoding for verified columns
        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)
        
        test_val_df_replies['reply_verified'] = test_val_df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_replies = pd.concat([test_val_df_replies, pd.get_dummies(test_val_df_replies["reply_verified"], dtype=int)], axis=1)
        test_val_df_replies.drop(["reply_verified"], axis=1, inplace=True)
        test_val_df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)
        
        test_val_df_posts = test_val_df_posts.merge(pd.concat([val,test])[['id']].reset_index(),on='id',how='left')
        test_val_df_posts.set_index('index',drop=True,inplace=True)
        
        post_features = test_val_df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified",\
                                           "first_time_diff"]]
        
        
        scaled_features = scaler_posts.transform(post_features[['followers', 'favorite_count', 'retweet_count', \
                                                                'first_time_diff']])
        
        # Convert the scaled features back to a DataFrame
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        
        # Add the binary features back to the scaled data
        scaled_data['no_verified'] = np.array(post_features['no_verified'])
        scaled_data['verified'] = np.array(post_features['verified'])
        post_features = scaled_data
        
        post_embeddings = np.array(test_val_df_posts['embeddings_avg'].tolist())
        #post_features = scaler.fit_transform(post_features)
        x3 = np.concatenate((post_features, post_embeddings), axis=1)
        
        test_val_reply_features =  test_val_df_replies[["reply_followers", "reply_no_verified","reply_verified","time_diff"]]
        test_val_reply_features[['reply_followers','time_diff']] = scaler_replies.transform(test_val_reply_features[['reply_followers','time_diff']])
        
        test_val_reply_embeddings = np.array(test_val_df_replies['reply_embeddings_avg'].tolist())
        x4 = np.concatenate((test_val_reply_features, test_val_reply_embeddings), axis=1)
        
        
        # Mapping post ids
        post_map = {value: i for i, value in enumerate(pd.concat([train[['id']],test_val_df_posts[['id']]])['id'].unique())}
        df_replies_edges = pd.concat([self.df_replies[self.df_replies.id.isin(np.array(train.id))][["id", "reply_user_id"]],\
                                test_val_df_replies[["id", "reply_user_id"]]])
        
        df_replies_edges["id"] = df_replies_edges['id'].map(post_map).astype(int)
        
        # Mapping reply user ids
        reply_user_map = {value: i for i, value in enumerate(df_replies_edges['reply_user_id'].unique())}
        df_replies_edges["reply_user_id"] = df_replies_edges["reply_user_id"].map(reply_user_map)

        return train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4


    def create_heterodata(self,train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4):

        y = pd.concat([train['rumour'],test_val_df_posts['rumour']]).to_numpy()
        edge_index = df_replies_edges.values.transpose()
        x = np.concatenate((x1,x3))
        x_reply = np.concatenate((x2,x4))
            
        num_rows = x.shape[0]
        train_mask = np.zeros(num_rows, dtype=bool)
        val_mask = np.zeros(num_rows, dtype=bool)
        test_mask = np.zeros(num_rows, dtype=bool)
        train_mask[:-x3.shape[0]]=True
        val_mask[-x3.shape[0]:-int(x3.shape[0]/2)]=True
        test_mask[-int(x3.shape[0]/2):]=True
            
        data = HeteroData()
        data['id'].x = torch.tensor(x, dtype=torch.float32)
        data['id'].y =  torch.from_numpy(y)
        data['id'].train_mask = torch.tensor(train_mask)
        data['id'].val_mask = torch.tensor(val_mask) 
        data['id'].test_mask = torch.tensor(test_mask)
        data['reply_user_id'].x = torch.tensor(x_reply, dtype=torch.float32)
        data['id', 'retweet', 'reply_user_id'].edge_index = torch.from_numpy(edge_index.reshape(2,len(x_reply)))
        data = T.ToUndirected()(data)
        
        return data

    def process(self):
            
        self.load_data()
        train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4 = self.process_data()
        return self.create_heterodata(train,test_val_df_posts,df_replies_edges,x1,x2,x3,x4)