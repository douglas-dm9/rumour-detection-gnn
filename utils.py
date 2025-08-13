import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split



class Load_Rumours_Dataset_filtering_since_first_post:
    """
    Class for loading, preprocessing, and splitting a rumour detection dataset 
    based on reply posts and their timing, filtering data up to a specified cutoff 
    since the first post.

    Attributes:
        file_path_replies (str): Path to the pickled replies DataFrame.
        file_path_posts (str): Path to the pickled posts DataFrame.
        time_cut (float): Time cutoff in minutes for filtering replies/posts.
        scaler_posts (RobustScaler): Scaler used to normalize numerical post features.
    """

    def __init__(self, file_path_replies, file_path_posts, time_cut):
        """
        Initializes the data loader with file paths and time cutoff.

        Args:
            file_path_replies (str): Path to the replies pickle file.
            file_path_posts (str): Path to the posts pickle file.
            time_cut (float): Time window in minutes since the first post to include replies.
        """
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.scaler_posts = RobustScaler()

    def load_data(self):
        """
        Loads the pickled replies and posts datasets into memory.
        """
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        """
        Processes the loaded datasets by:
            - Filtering replies within the specified time cutoff.
            - Computing features such as number of replies and time since post.
            - One-hot encoding 'verified' column.
            - Splitting the data into train, validation, and test sets.
            - Scaling numerical features.
            - Returning processed and scaled train and test DataFrames.
        """
        features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']

        # Compute time since first post for replies
        self.df_replies['min_since_fst_post'] = round(
            (self.df_replies['time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
        self.df_replies['reply_min_since_fst_post'] = round(
            (self.df_replies['reply_time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)

        # Aggregate replies by post
        grouped_replies = self.df_replies.groupby(['id', 'min_since_fst_post']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge with post features
        self.df_posts = self.df_posts[features].merge(grouped_replies, on="id", how="inner").fillna(0)

        # One-hot encode verification status
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        # Train/validation/test split
        #train, not_train = train_test_split(self.df_posts, test_size=0.3, random_state=42, stratify=self.df_posts['rumour'])
        #val, test = train_test_split(not_train, test_size=0.5, random_state=42, stratify=not_train['rumour'])

        df_post_time = self.df_replies[['time','id']].drop_duplicates().sort_values(by='time')
        train = self.df_posts[self.df_posts.id.isin(df_post_time.iloc[0:int(0.7*df_post_time.shape[0])].id)]
        not_train = self.df_posts[~self.df_posts.id.isin(train.id)]
        val = not_train.iloc[0:int(0.5*not_train.shape[0])]
        test = not_train.iloc[int(0.5*not_train.shape[0]):]
        #train, not_train =self.df_posts.iloc[0:int(0.7*df_post_time.shape[0])], self.df_posts.iloc[int(0.7*df_post_time.shape[0]):]
        #val, test = not_train.iloc[0:int(0.5*not_train.shape[0])],not_train.iloc[int(0.5*not_train.shape[0]):]

        # Scale selected numeric features for training
        train_features = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", 
                                "rumour", "embeddings_avg", "replies", "first_time_diff", "min_since_fst_post"]]
        scaled_numeric = self.scaler_posts.fit_transform(train_features[['followers', 'favorite_count',
                                                                          'retweet_count', 'first_time_diff', 'replies']])
        scaled_data = pd.DataFrame(scaled_numeric, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff', 'replies'])
        scaled_data['no_verified'] = train['no_verified'].values
        scaled_data['verified'] = train['verified'].values
        scaled_data['embeddings_avg'] = train['embeddings_avg'].values
        scaled_data['rumour'] = train['rumour'].values
        scaled_data['min_since_fst_post'] = train['min_since_fst_post'].values

        self.train_dataset = scaled_data

        # Reload and filter posts/replies for test/val based on time_cut
        test_val_df_replies = pd.read_pickle(self.file_path_replies)
        test_val_df_posts = pd.read_pickle(self.file_path_posts)

        test_val_df_posts = test_val_df_posts[~test_val_df_posts.id.isin(train.id)]
        test_val_df_replies = test_val_df_replies[~test_val_df_replies.id.isin(train.id)]

        test_val_df_replies['min_since_fst_post'] = round(
            (test_val_df_replies['time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)
        test_val_df_replies['reply_min_since_fst_post'] = round(
            (test_val_df_replies['reply_time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)

        # Filter by time_cut
        test_val_df_replies = test_val_df_replies[
            (test_val_df_replies.reply_min_since_fst_post <= self.time_cut) &
            (test_val_df_replies.min_since_fst_post <= self.time_cut)]

        # Group replies again
        grouped_replies = test_val_df_replies.groupby(['id', 'min_since_fst_post']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge with posts and apply same transformations
        test_val_df_posts = test_val_df_posts[features].merge(grouped_replies, on="id", how="inner").fillna(0)

        test_val_df_posts.sort_values(by='min_since_fst_post',ascending=True,inplace=True)

        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        # Reindex based on existing validation and training IDs
        test_val_df_posts = test_val_df_posts.merge(pd.concat([val, train])[['id']].reset_index(), on='id', how='left')
        test_val_df_posts.set_index('index', drop=True, inplace=True)

        post_features = test_val_df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified",
                                           'replies', "first_time_diff", "embeddings_avg", "rumour", "min_since_fst_post"]]

        scaled_numeric = self.scaler_posts.transform(post_features[['followers', 'favorite_count',
                                                                    'retweet_count', 'first_time_diff', 'replies']])
        scaled_data = pd.DataFrame(scaled_numeric, columns=['followers', 'favorite_count', 'retweet_count',
                                                            'first_time_diff', 'replies'])
        scaled_data['no_verified'] = post_features['no_verified'].values
        scaled_data['verified'] = post_features['verified'].values
        scaled_data['embeddings_avg'] = post_features['embeddings_avg'].values
        scaled_data['rumour'] = post_features['rumour'].values
        scaled_data['min_since_fst_post'] = post_features['min_since_fst_post'].values

        self.test_dataset = scaled_data

    def get_final_dataframes(self):
        """
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The processed training and test datasets.
        """
        return self.train_dataset, self.test_dataset


class Load_Rumours_Dataset_filtering_since_first_post_Transfer_Learning:
    """
    This class loads, preprocesses, and transforms rumor detection data for transfer learning.
    It combines a portion of the test dataset into the training set, processes reply-post
    interactions, performs feature engineering (e.g., reply timing), and scales the features.
    """

    def __init__(self, train_dataset, test_dataset, time_cut=24*3*60, test_size=0.7):
        """
        Initialize dataset paths and default parameters.

        Parameters:
        - train_dataset (str): Name of the training dataset (used for file paths).
        - test_dataset (str): Name of the test dataset (used for file paths).
        - time_cut (int): Maximum allowed time since the first post (in minutes).
        - test_size (float): Proportion of test data used as test set.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.time_cut = time_cut
        self.scaler_posts = RobustScaler()
        self.test_size = test_size

        # Set file paths for replies and posts
        self.file_path_replies_train = f"replies_{self.train_dataset}.pkl"
        self.file_path_posts_train = f"posts_{self.train_dataset}.pkl"
        self.file_path_replies_test = f"replies_{self.test_dataset}.pkl"
        self.file_path_posts_test = f"posts_{self.test_dataset}.pkl"

    def load_data(self):
        """
        Load data from pickle files for both training and testing sets.
        """
        self.df_replies_train = pd.read_pickle(self.file_path_replies_train)
        self.df_posts_train = pd.read_pickle(self.file_path_posts_train)
        self.df_replies = pd.read_pickle(self.file_path_replies_test)
        self.df_posts = pd.read_pickle(self.file_path_posts_test)

    def process_data(self):
        """
        Processes the dataset by merging posts and replies, generating features,
        filtering by time_cut, scaling, and combining datasets.
        """

        # Merge replies time info into posts, and sort by time
        self.df_posts = self.df_posts.merge(
            self.df_replies[['id', 'time']].drop_duplicates(), on='id', how='left'
        ).sort_values(by='time', ascending=True)

        # Split test posts into test and transfer parts
        df_posts_test = self.df_posts[int(len(self.df_posts) * (1 - self.test_size)):]
        df_posts_concat = self.df_posts[:int(len(self.df_posts) * (1 - self.test_size))]

        # Split replies accordingly
        df_replies_concat = self.df_replies[self.df_replies.id.isin(df_posts_concat.id)]
        df_replies_test = self.df_replies[~self.df_replies.id.isin(df_posts_concat.id)]

        # Concatenate partial test data into training
        df_posts_train = pd.concat([self.df_posts_train, df_posts_concat])
        df_replies_train = pd.concat([self.df_replies_train, df_replies_concat])

        # Relevant features
        features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']

        # Compute time since first post (in minutes)
        df_replies_train['min_since_fst_post'] = round(
            (df_replies_train['time'] - df_replies_train['time'].min()).dt.total_seconds() / 60, 2
        )
        df_replies_test['min_since_fst_post'] = round(
            (df_replies_test['time'] - df_replies_test['time'].min()).dt.total_seconds() / 60, 2
        )

        # Compute time difference between reply and first post
        df_replies_train['reply_min_since_fst_post'] = round(
            (df_replies_train['reply_time'] - df_replies_train['time'].min()).dt.total_seconds() / 60, 2
        )
        df_replies_test['reply_min_since_fst_post'] = round(
            (df_replies_test['reply_time'] - df_replies_test['time'].min()).dt.total_seconds() / 60, 2
        )

        # Group reply features for training data
        grouped_replies = df_replies_train.groupby(['id', 'min_since_fst_post']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge reply info with posts (training)
        df_posts_train = df_posts_train[features].merge(grouped_replies, on="id", how="inner")
        df_posts_train.fillna(0, inplace=True)

        # One-hot encode 'verified'
        df_posts_train['verified'] = df_posts_train['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        df_posts_train = pd.concat([df_posts_train, pd.get_dummies(df_posts_train["verified"], dtype=int)], axis=1)
        df_posts_train.drop(["verified"], axis=1, inplace=True)
        df_posts_train.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        # Scale numeric features (training)
        post_features = df_posts_train[[
            "followers", "favorite_count", "retweet_count", "no_verified",
            "verified", "rumour", "embeddings_avg", "replies", "first_time_diff", "min_since_fst_post"
        ]]
        scaled_features = self.scaler_posts.fit_transform(
            post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff', 'replies']]
        )

        scaled_data = pd.DataFrame(scaled_features, columns=[
            'followers', 'favorite_count', 'retweet_count', 'first_time_diff', 'replies'
        ])
        scaled_data['no_verified'] = post_features['no_verified'].values
        scaled_data['verified'] = post_features['verified'].values
        scaled_data['embeddings_avg'] = post_features['embeddings_avg'].values
        scaled_data['rumour'] = post_features['rumour'].values
        scaled_data['min_since_fst_post'] = post_features['min_since_fst_post'].values

        self.train = scaled_data

        # Process test/validation data
        test_val_df_posts = df_posts_test
        test_val_df_replies = df_replies_test

        # Filter replies by time cutoff
        test_val_df_replies['min_since_fst_post'] = round(
            (test_val_df_replies['time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2
        )
        test_val_df_replies['reply_min_since_fst_post'] = round(
            (test_val_df_replies['reply_time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2
        )
        test_val_df_replies = test_val_df_replies[
            (test_val_df_replies.reply_min_since_fst_post <= self.time_cut) &
            (test_val_df_replies.min_since_fst_post <= self.time_cut)
        ]

        # Group reply stats (test)
        grouped_replies = test_val_df_replies.groupby(['id', 'min_since_fst_post']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge replies with test posts
        test_val_df_posts = test_val_df_posts[features].merge(grouped_replies, on="id", how="inner")
        test_val_df_posts.fillna(0, inplace=True)

        # One-hot encode 'verified'
        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        # Preserve original index
        test_val_df_posts = test_val_df_posts.merge(df_posts_test[['id']].reset_index(), on='id', how='left')
        test_val_df_posts.set_index('index', drop=True, inplace=True)

        # Log class distribution
        print(test_val_df_posts['rumour'].value_counts())

        # Scale test features
        post_features = test_val_df_posts[[
            "followers", "favorite_count", "retweet_count", "no_verified", "verified",
            "replies", "first_time_diff", "embeddings_avg", "rumour", "min_since_fst_post"
        ]]
        scaled_features = self.scaler_posts.transform(
            post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff', 'replies']]
        )

        scaled_data = pd.DataFrame(scaled_features, columns=[
            'followers', 'favorite_count', 'retweet_count', 'first_time_diff', 'replies'
        ])
        scaled_data['no_verified'] = post_features['no_verified'].values
        scaled_data['verified'] = post_features['verified'].values
        scaled_data['embeddings_avg'] = post_features['embeddings_avg'].values
        scaled_data['rumour'] = post_features['rumour'].values
        scaled_data['min_since_fst_post'] = post_features['min_since_fst_post'].values

        self.test = scaled_data

    def get_final_dataframes(self):
        """
        Returns the processed training and test DataFrames.

        Returns:
        - train (pd.DataFrame): Scaled training data with features and labels.
        - test (pd.DataFrame): Scaled test/validation data with features and labels.
        """
        return self.train, self.test

class Hetero_Data_Processor_Filter_on_Test_since_first_post:
    """
    This class processes a rumor detection dataset composed of social media posts and replies,
    transforming it into a format suitable for training heterogeneous graph neural networks.
    It applies temporal filtering on replies based on a time cutoff relative to the first post.

    Attributes:
        file_path_replies (str): Path to the pickle file containing the replies data.
        file_path_posts (str): Path to the pickle file containing the original posts.
        time_cut (int): Time cutoff in minutes to filter replies relative to the first post.
        df_replies (pd.DataFrame): Loaded replies dataframe.
        df_posts (pd.DataFrame): Loaded posts dataframe.
    """
    
    def __init__(self, file_path_replies, file_path_posts, time_cut=15):
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.df_replies = None
        self.df_posts = None

    def load_data(self):
        """
        Load the replies and posts datasets from pickle files.
        """
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        """
        Preprocess the data:
        - Aggregates reply statistics
        - Filters replies based on `time_cut`
        - Encodes categorical variables
        - Scales numerical features
        - Splits posts into train/validation/test
        - Builds post and reply feature arrays

        Returns:
            tuple: train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4
        """
        # Select features
        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        # Compute time from first post
        self.df_replies['min_since_fst_post'] = round(
            (self.df_replies['time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
        self.df_replies['reply_min_since_fst_post'] = round(
            (self.df_replies['reply_time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)

        # Aggregate reply info by post
        grouped_replies = self.df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge aggregated replies into posts
        self.df_posts = self.df_posts[post_features].merge(grouped_replies, on="id", how="inner").fillna(0)

        # One-hot encode 'verified' column in posts and replies
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1).drop(["verified"], axis=1)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        self.df_replies['reply_verified'] = self.df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_replies = pd.concat([self.df_replies, pd.get_dummies(self.df_replies["reply_verified"], dtype=int)], axis=1).drop(["reply_verified"], axis=1)
        self.df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        # Split posts into train/validation/test
        #train, not_train = train_test_split(self.df_posts, test_size=0.3, random_state=42, stratify=self.df_posts['rumour'])
        #val, test = train_test_split(not_train, test_size=0.5, random_state=42, stratify=not_train['rumour'])

        df_post_time = self.df_replies[['time','id']].drop_duplicates().sort_values(by='time')

        train = self.df_posts[self.df_posts.id.isin(df_post_time.iloc[0:int(0.7*df_post_time.shape[0])].id)]
        not_train =  self.df_posts[~self.df_posts.id.isin(train.id)]
        val = not_train.iloc[0:int(0.5*not_train.shape[0])]
        test = not_train.iloc[int(0.5*not_train.shape[0]):]

        # Process training post features
        post_features_train = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        scaler_posts = RobustScaler()
        scaled_features = scaler_posts.fit_transform(post_features_train[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        scaled_data['no_verified'] = np.array(train['no_verified'])
        scaled_data['verified'] = np.array(train['verified'])
        x1 = np.concatenate((scaled_data, np.array(train['embeddings_avg'].tolist())), axis=1)

        # Process training reply features
        scaler_replies = RobustScaler()
        reply_subset = self.df_replies[self.df_replies.id.isin(train.id)][["reply_followers", "reply_no_verified", "reply_verified", "time_diff"]]
        reply_subset[['reply_followers', 'time_diff']] = scaler_replies.fit_transform(reply_subset[['reply_followers', 'time_diff']])
        reply_embeddings = np.array(self.df_replies[self.df_replies.id.isin(train.id)]['reply_embeddings_avg'].tolist())
        x2 = np.concatenate((reply_subset, reply_embeddings), axis=1)

        # Reload and filter replies for test/val
        test_val_df_replies = pd.read_pickle(self.file_path_replies)
        test_val_df_posts = pd.read_pickle(self.file_path_posts)
        test_val_df_posts = test_val_df_posts[~test_val_df_posts.id.isin(train.id)]
        test_val_df_replies = test_val_df_replies[~test_val_df_replies.id.isin(train.id)]

        test_val_df_posts = test_val_df_posts.merge(df_post_time,how='inner',on='id')
        test_val_df_posts.sort_values(by='time',ascending=True,inplace=True)
        test_val_df_posts.drop(columns='time',inplace=True)

        test_val_df_replies.sort_values(by='time',ascending=True,inplace=True)
    
        test_val_df_replies['min_since_fst_post'] = round((test_val_df_replies['time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)
        test_val_df_replies['reply_min_since_fst_post'] = round((test_val_df_replies['reply_time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)
        
        test_val_df_replies = test_val_df_replies[test_val_reply_features := reply_features][
            (test_val_df_replies.reply_min_since_fst_post <= self.time_cut) &
            (test_val_df_replies.min_since_fst_post <= self.time_cut)
        ]

        # Merge replies into posts
        grouped_replies = test_val_df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        test_val_df_posts = test_val_df_posts[post_features].merge(grouped_replies, on="id", how="inner").fillna(0)
      

        # One-hot encode verified columns
        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1).drop(["verified"], axis=1)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        test_val_df_replies['reply_verified'] = test_val_df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)

    
        # Create dummy columns ensuring both 0 and 1 are present
        dummies = pd.get_dummies(test_val_df_replies["reply_verified"], dtype=int)
        
        # Add missing columns if necessary
        for col in [0, 1]:
            if col not in dummies.columns:
                dummies[col] = 0
        
        # Concatenate and rename
        test_val_df_replies = pd.concat([test_val_df_replies.drop(["reply_verified"], axis=1), dummies], axis=1)
        test_val_df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)
        
                
        #test_val_df_replies = pd.concat([test_val_df_replies, pd.get_dummies(test_val_df_replies["reply_verified"], dtype=int)], axis=1).drop(["reply_verified"], axis=1)
        #test_val_df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        # Recover val/test label
        test_val_df_posts = test_val_df_posts.merge(pd.concat([val, test])[['id']].reset_index(), on='id', how='left')
        test_val_df_posts.set_index('index', drop=True, inplace=True)

        
        test_val_df_posts =test_val_df_posts.merge(df_post_time, on="id", how="inner")

        test_val_df_posts.sort_values(by='time',ascending=True,inplace=True)
        test_val_df_posts.drop(columns=['time'],inplace=True)


        

        # Scale features
        scaled_features = scaler_posts.transform(test_val_df_posts[["followers", "favorite_count", "retweet_count", "first_time_diff"]])
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        scaled_data['no_verified'] = np.array(test_val_df_posts['no_verified'])
        scaled_data['verified'] = np.array(test_val_df_posts['verified'])
        x3 = np.concatenate((scaled_data, np.array(test_val_df_posts['embeddings_avg'].tolist())), axis=1)

        # Reply test/val features
        test_val_reply_features = test_val_df_replies[["reply_followers", "reply_no_verified", "reply_verified", "time_diff"]]
        test_val_reply_features[['reply_followers', 'time_diff']] = scaler_replies.transform(test_val_reply_features[['reply_followers', 'time_diff']])
        test_val_reply_embeddings = np.array(test_val_df_replies['reply_embeddings_avg'].tolist())
        x4 = np.concatenate((test_val_reply_features, test_val_reply_embeddings), axis=1)

        # Map nodes to indices
        post_map = {value: i for i, value in enumerate(pd.concat([train[['id']], test_val_df_posts[['id']]])['id'].unique())}
        df_replies_edges = pd.concat([
            self.df_replies[self.df_replies.id.isin(train.id)][["id", "reply_user_id"]],
            test_val_df_replies[["id", "reply_user_id"]]
        ])
        df_replies_edges["id"] = df_replies_edges['id'].map(post_map).astype(int)
        reply_user_map = {value: i for i, value in enumerate(df_replies_edges['reply_user_id'].unique())}
        df_replies_edges["reply_user_id"] = df_replies_edges["reply_user_id"].map(reply_user_map)

        return train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4

    def create_heterodata(self, train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4):
        """
        Construct a HeteroData object for a heterogeneous GNN model.

        Returns:
            HeteroData: PyTorch Geometric object with node/edge types and masks.
        """
        y = pd.concat([train['rumour'], test_val_df_posts['rumour']]).to_numpy()
        edge_index = df_replies_edges.values.transpose()
        x = np.concatenate((x1, x3))
        x_reply = np.concatenate((x2, x4))

        num_rows = x.shape[0]
        train_mask = np.zeros(num_rows, dtype=bool)
        val_mask = np.zeros(num_rows, dtype=bool)
        test_mask = np.zeros(num_rows, dtype=bool)
        train_mask[:-x3.shape[0]] = True
        val_mask[-x3.shape[0]:-int(x3.shape[0] / 2)] = True
        test_mask[-int(x3.shape[0] / 2):] = True

        data = HeteroData()
        data['id'].x = torch.tensor(x, dtype=torch.float32)
        data['id'].y = torch.from_numpy(y)
        data['id'].train_mask = torch.tensor(train_mask)
        data['id'].val_mask = torch.tensor(val_mask)
        data['id'].test_mask = torch.tensor(test_mask)
        data['reply_user_id'].x = torch.tensor(x_reply, dtype=torch.float32)
        data['id', 'retweet', 'reply_user_id'].edge_index = torch.from_numpy(edge_index.reshape(2, len(x_reply)))
        data = T.ToUndirected()(data)

        return data

    def process(self):
        """
        Pipeline wrapper: load, preprocess and return the final heterogenous graph.

        Returns:
            HeteroData: Fully processed heterogenous dataset ready for GNN training.
        """
        self.load_data()
        train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4 = self.process_data()
        return self.create_heterodata(train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4)




class Hetero_Data_Processor_Transfer_Learning:
    """
    Class for loading, processing, and constructing a heterogeneous graph for rumor detection
    using transfer learning between two datasets (train and test).

    It handles:
    - Feature engineering for posts and replies
    - Merging and filtering based on time
    - Scaling and encoding features
    - Generating graph-ready input for PyTorch Geometric HeteroData format
    """

    def __init__(self, train_dataset, test_dataset, time_cut=24*3*60, test_size=0.3):
        """
        Initialize the processor with dataset names and preprocessing parameters.

        Args:
            train_dataset (str): Name of the training dataset.
            test_dataset (str): Name of the testing dataset.
            time_cut (int): Time cutoff in minutes to filter replies (default: 3 days).
            test_size (float): Proportion of the test set split from the test dataset (default: 0.3).
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.time_cut = time_cut
        self.test_size = test_size

        self.file_path_replies_train = f"replies_{self.train_dataset}.pkl"
        self.file_path_posts_train = f"posts_{self.train_dataset}.pkl"
        self.file_path_replies_test = f"replies_{self.test_dataset}.pkl"
        self.file_path_posts_test = f"posts_{self.test_dataset}.pkl"

    def load_data(self):
        """
        Loads pickled DataFrames for replies and posts for both training and test datasets.
        """
        self.df_replies_train = pd.read_pickle(self.file_path_replies_train)
        self.df_posts_train = pd.read_pickle(self.file_path_posts_train)
        self.df_replies = pd.read_pickle(self.file_path_replies_test)
        self.df_posts = pd.read_pickle(self.file_path_posts_test)

    def process_data(self):
        """
        Processes the post and reply data to generate features for graph input.

        Returns:
            tuple: Processed training/test post/reply features, edge list, and final numpy arrays (x1, x2, x3, x4).
        """
        # Merge reply time with post data and sort chronologically
        self.df_posts = self.df_posts.merge(
            self.df_replies[['id', 'time']].drop_duplicates(),
            on='id', how='left'
        ).sort_values(by='time', ascending=True)

        # Split test data into test and val sets
        df_posts_test = self.df_posts[int(len(self.df_posts)*(1 - self.test_size)):]
        df_posts_concat = self.df_posts[:int(len(self.df_posts)*(1 - self.test_size))]

        # Filter corresponding replies
        df_replies_concat = self.df_replies[self.df_replies.id.isin(df_posts_concat.id)]
        df_replies_test = self.df_replies[~self.df_replies.id.isin(df_posts_concat.id)]

        # Combine with training set
        df_posts_train = pd.concat([self.df_posts_train, df_posts_concat])
        df_replies_train = pd.concat([self.df_replies_train, df_replies_concat])

        # Feature selection
        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        # Add time difference features
        df_replies_train['min_since_fst_post'] = round(
            (df_replies_train['time'] - df_replies_train['time'].min()).dt.total_seconds() / 60, 2)
        df_replies_train['reply_min_since_fst_post'] = round(
            (df_replies_train['reply_time'] - df_replies_train['time'].min()).dt.total_seconds() / 60, 2)

        # Group replies per post
        grouped_replies = df_replies_train.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge aggregated replies with posts
        df_posts_train = df_posts_train[post_features].merge(grouped_replies, on="id", how="inner")
        df_posts_train['replies'] = df_posts_train['replies'].fillna(0)
        df_posts_train['first_time_diff'] = df_posts_train['first_time_diff'].fillna(0)

        # Encode 'verified' as binary one-hot
        df_posts_train['verified'] = df_posts_train['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        df_posts_train = pd.concat([df_posts_train, pd.get_dummies(df_posts_train["verified"], dtype=int)], axis=1)
        df_posts_train.drop(["verified"], axis=1, inplace=True)
        df_posts_train.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        df_replies_train['reply_verified'] = df_replies_train['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        df_replies_train = pd.concat([df_replies_train, pd.get_dummies(df_replies_train["reply_verified"], dtype=int)], axis=1)
        df_replies_train.drop(["reply_verified"], axis=1, inplace=True)
        df_replies_train.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        train = df_posts_train

        # Scale post features
        post_features = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        scaler_posts = RobustScaler()
        scaled_features = scaler_posts.fit_transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        scaled_data['no_verified'] = train['no_verified'].values
        scaled_data['verified'] = train['verified'].values
        post_features = scaled_data
        post_embeddings = np.array(train['embeddings_avg'].tolist())
        x1 = np.concatenate((post_features, post_embeddings), axis=1)

        # Scale reply features
        scaler_replies = RobustScaler()
        reply_features = df_replies_train[df_replies_train.id.isin(train.id)][["reply_followers", "reply_no_verified", "reply_verified", "time_diff"]]
        reply_features[['reply_followers', 'time_diff']] = scaler_replies.fit_transform(reply_features[['reply_followers', 'time_diff']])
        reply_embeddings = np.array(df_replies_train[df_replies_train.id.isin(train.id)]['reply_embeddings_avg'].tolist())
        x2 = np.concatenate((reply_features, reply_embeddings), axis=1)

        # Process test/val data
        test_val_df_replies = df_replies_test
        test_val_df_posts = df_posts_test

        post_features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']
        test_val_reply_features = ['reply_followers', 'reply_user_id', 'reply_verified', 'time_diff', 'reply_embeddings_avg', 'id']

        test_val_df_replies['min_since_fst_post'] = round(
            (test_val_df_replies['time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)
        test_val_df_replies['reply_min_since_fst_post'] = round(
            (test_val_df_replies['reply_time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)

        # Filter by time cutoff
        test_val_df_replies = test_val_df_replies[test_val_reply_features][
            (test_val_df_replies.reply_min_since_fst_post <= self.time_cut) &
            (test_val_df_replies.min_since_fst_post <= self.time_cut)
        ]

        grouped_replies = test_val_df_replies.groupby(['id']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        test_val_df_posts = test_val_df_posts[post_features].merge(grouped_replies, on="id", how="inner")
        print(test_val_df_posts['rumour'].value_counts())
        test_val_df_posts['replies'] = test_val_df_posts['replies'].fillna(0)
        test_val_df_posts['first_time_diff'] = test_val_df_posts['first_time_diff'].fillna(0)

        # Encode verified flags
        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        if 'verified' not in test_val_df_posts.columns:
            test_val_df_replies['verified'] = 0
        elif 'no_verified' not in test_val_df_posts.columns:
            test_val_df_replies['no_verified'] = 0

        test_val_df_replies['reply_verified'] = test_val_df_replies['reply_verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_replies = pd.concat([test_val_df_replies, pd.get_dummies(test_val_df_replies["reply_verified"], dtype=int)], axis=1)
        test_val_df_replies.drop(["reply_verified"], axis=1, inplace=True)
        test_val_df_replies.rename(columns={1: 'reply_verified', 0: 'reply_no_verified'}, inplace=True)

        if 'reply_no_verified' not in test_val_df_replies.columns:
            test_val_df_replies['reply_no_verified'] = 0
        elif 'reply_verified' not in test_val_df_replies.columns:
            test_val_df_replies['reply_verified'] = 0

        post_features = test_val_df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified", "first_time_diff"]]
        scaled_features = scaler_posts.transform(post_features[['followers', 'favorite_count', 'retweet_count', 'first_time_diff']])
        scaled_data = pd.DataFrame(scaled_features, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff'])
        scaled_data['no_verified'] = post_features['no_verified'].values
        scaled_data['verified'] = post_features['verified'].values
        post_features = scaled_data
        post_embeddings = np.array(test_val_df_posts['embeddings_avg'].tolist())
        x3 = np.concatenate((post_features, post_embeddings), axis=1)

        test_val_reply_features = test_val_df_replies[["reply_followers", "reply_no_verified", "reply_verified", "time_diff"]]
        test_val_reply_features[['reply_followers', 'time_diff']] = scaler_replies.transform(
            test_val_reply_features[['reply_followers', 'time_diff']]
        )
        test_val_reply_embeddings = np.array(test_val_df_replies['reply_embeddings_avg'].tolist())
        x4 = np.concatenate((test_val_reply_features, test_val_reply_embeddings), axis=1)

        # Create edge index mappings
        post_map = {value: i for i, value in enumerate(pd.concat([train[['id']], test_val_df_posts[['id']]])['id'].unique())}
        df_replies_edges = pd.concat([
            df_replies_train[df_replies_train.id.isin(train.id)][["id", "reply_user_id"]],
            test_val_df_replies[["id", "reply_user_id"]]
        ])
        df_replies_edges["id"] = df_replies_edges['id'].map(post_map).astype(int)

        reply_user_map = {value: i for i, value in enumerate(df_replies_edges['reply_user_id'].unique())}
        df_replies_edges["reply_user_id"] = df_replies_edges["reply_user_id"].map(reply_user_map)

        return train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4

    def create_heterodata(self, train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4):
        """
        Builds the HeteroData graph from preprocessed post/reply features and edge mappings.

        Returns:
            torch_geometric.data.HeteroData: A heterogeneous graph with post and reply user nodes.
        """
        y = pd.concat([train['rumour'], test_val_df_posts['rumour']]).to_numpy()
        edge_index = df_replies_edges.values.transpose()
        x = np.concatenate((x1, x3))
        x_reply = np.concatenate((x2, x4))

        num_rows = x.shape[0]
        train_mask = np.zeros(num_rows, dtype=bool)
        val_mask = np.zeros(num_rows, dtype=bool)
        test_mask = np.zeros(num_rows, dtype=bool)
        train_mask[:-x3.shape[0]] = True
        val_mask[-x3.shape[0]:-int(x3.shape[0]/2)] = True
        test_mask[-int(x3.shape[0]/2):] = True

        data = HeteroData()
        data['id'].x = torch.tensor(x, dtype=torch.float32)
        data['id'].y = torch.from_numpy(y)
        data['id'].train_mask = torch.tensor(train_mask)
        data['id'].val_mask = torch.tensor(val_mask)
        data['id'].test_mask = torch.tensor(test_mask)
        data['reply_user_id'].x = torch.tensor(x_reply, dtype=torch.float32)
        data['id', 'retweet', 'reply_user_id'].edge_index = torch.from_numpy(edge_index.reshape(2, len(x_reply)))
        data = T.ToUndirected()(data)

        return data

    def process(self):
        """
        Full pipeline to process data and return a graph for model training.

        Returns:
            HeteroData: Graph with node features, labels, edge_index, and masks.
        """
        self.load_data()
        train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4 = self.process_data()
        return self.create_heterodata(train, test_val_df_posts, df_replies_edges, x1, x2, x3, x4)

