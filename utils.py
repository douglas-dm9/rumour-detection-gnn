import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
import pickle

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
        post_features = ['followers','favorite_count','retweet_count','verified','rumour','id','embeddings_avg','text']
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
        self.df_replies["reply_user_id"] = self.df_replies["reply_user_id"].map(self.reply_user_map).astype(int)

    def create_features(self):
        post_features = self.df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        post_embeddings = np.array(self.df_posts['embeddings_avg'].tolist())
        #post_features = self.scaler.fit_transform(post_features)
        x1 = np.concatenate((post_features, post_embeddings), axis=1)

        reply_features = self.df_replies[["reply_followers", "reply_no_verified", "reply_verified", "time_diff"]]
        reply_embeddings = np.array(self.df_replies['reply_embeddings_avg'].tolist())
        #reply_features = self.scaler.transform(reply_features)
        x2 = np.concatenate((reply_features, reply_embeddings), axis=1)

        return x1, x2

    def create_heterodata(self, x1, x2):
        y = self.df_posts['rumour'].to_numpy()
        edge_index = self.df_replies[["id", "reply_user_id"]].values.T

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
        data['twitter_id'].x = torch.tensor(x1, dtype=torch.float32)
        data['twitter_id'].y = torch.tensor(y, dtype=torch.float32)
        data['twitter_id'].train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data['twitter_id'].val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data['twitter_id'].test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data['reply_user_id'].x = torch.tensor(x2, dtype=torch.float32)
        data['twitter_id', 'retweet', 'reply_user_id'].edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = T.ToUndirected()(data)

        return data

    def process(self):
        self.load_data()
        self.process_data()
        x1, x2 = self.create_features()
        return self.create_heterodata(x1, x2)