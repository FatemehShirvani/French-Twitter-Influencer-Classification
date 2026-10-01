
# # 1. Data cleaning
# Data handling libraries
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
import emoji
import re

# Natural Language Processing (NLP) libraries
from nltk.corpus import stopwords

# Scikit-learn modeling libraries
from sklearn.dummy import DummyClassifier # For baseline model
from sklearn.feature_extraction.text import TfidfVectorizer # To convert text to numbers
from sklearn.linear_model import LogisticRegression # The classifier model
from sklearn.metrics import accuracy_score, classification_report # For evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # For splitting and validating
from sklearn.pipeline import Pipeline # To chain processing steps
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# %%
def clean_dataframe(df):
    """
    Cleans a pandas DataFrame for data classification.
    
    Performs the following IN-PLACE:
    1. Replaces '.' with '_' in column names. XXXXXXX
    2. Identifies unhashable columns (lists/dicts) and EXCLUDES them from:
       - Duplicate detection
       - Constant/Unique column removal
       - String stripping
    3. Removes duplicate rows (based on hashable columns only).
    4. Removes constant columns (hashable only).
    5. Removes unique columns (hashable only). XXXXXXX
    6. Fills NaN in numeric columns with 0.
    7. Fills NaN in string columns with "".
    8. Strips whitespace from string columns.
    Args:
        df (pd.DataFrame): The raw dataframe (Modified in-place).
        
    Returns:
        pd.DataFrame: The cleaned dataframe (reference to the same object).
    """
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")

    # --- 0. Reform Column Names --- Replace '.' with '_'
    #df.columns = df.columns.str.replace('.', '_', regex=False)

    # --- 0.5 Identify Hashable vs Unhashable Columns ---
    hashable_cols = []
    unhashable_cols = []
    for col in df.columns:
        try:
            pd.unique(df[col])
            hashable_cols.append(col)
        except TypeError:
            unhashable_cols.append(col)
    if unhashable_cols:
        print(f"Ignoring unhashable columns (lists/dicts) for structure checks: {unhashable_cols}")
    
    # --- 2. Remove Constant Columns ---
    cols_to_drop = [col for col in hashable_cols if df[col].nunique() <= 1]
    df.drop(columns=cols_to_drop, inplace=True)    
    hashable_cols = [c for c in hashable_cols if c not in cols_to_drop]
    if cols_to_drop: print(f"Dropped constant columns: {cols_to_drop}")

    # --- 3. Convert booleans as int ---
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"Converting boolean columns to integers: {bool_cols.tolist()}")
        df[bool_cols] = df[bool_cols].astype(int)

    # --- 4. Handle Numeric NaNs ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # --- 5. Handle String/Object NaNs ---
    object_cols = df.select_dtypes(include=['object', 'string']).columns    
    # Intersect object columns with hashable columns to exclude lists
    string_cols_safe = [c for c in object_cols if c in hashable_cols]
    df[string_cols_safe] = df[string_cols_safe].fillna("")

    # --- 6. Strip Whitespace ---
    for col in string_cols_safe:
        # Verify it is actually string/object before stripping
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(str).str.strip()

    print(f"Final shape: {df.shape}")
    return df

# %%
def advanced_clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\.\S+", " __URL__ ", text)
    text = re.sub(r"@\w+", " __USER__ ", text)
    text = re.sub(r"[^\w\s_]", " ", text) 
    return text

def calculate_age_vectorized(series):
    # Convert text to datetime objects (invalid dates become NaT)
    dt = pd.to_datetime(series, format="%a %b %d %H:%M:%S %z %Y", errors='coerce')
    # Define "now" (UTC)
    now = pd.Timestamp.now(tz='UTC')
    # FILL THE HOLES: Replace NaT (missing dates) with 'now'
    # If created_at == now, then age will be 0. This is perfect for your "Observer" tweets.
    dt = dt.fillna(now)
    # Calculate age in years for the whole column at once
    return (now - dt).dt.days / 365.0

def safe_len(x):
    """Returns len(x) or 0 if x is NaN."""
    return len(x) if isinstance(x, list) else 0

# %%
def extract_features_vectorized(df):

    # ---------------------
    # A. USER FIELDS
    # ---------------------
   
    df["has_description"] = (df["user.description"] != "").astype(int)
    df["has_location"] = (df["user.location"] != "").astype(int)
    df["user_has_url"] = ((df["user.url"].notna()) & (df["user.url"] != "")).astype(int)
    
    df["user.description"] =  df["user.description"].apply(advanced_clean_text)
    df["full_text"] = (df["extended_tweet.full_text"].fillna(df["text"]).fillna("")).apply(advanced_clean_text)
    df["text_length"] = df["full_text"].str.len()
    df["uppercase_ratio"] = df["full_text"].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    df["is_reply"] = (df["in_reply_to_status_id"] > 0).astype(int)
    df["num_hashtags"] = (df["extended_tweet.entities.hashtags"].fillna(df["entities.hashtags"]).fillna("[]").apply(lambda x: safe_len(x) if isinstance(x, list) else 0))
    df["num_user_mentions"] = (df["extended_tweet.entities.user_mentions"].fillna(df["entities.user_mentions"]).fillna("[]").apply(lambda x: safe_len(x) if isinstance(x, list) else 0))
    df["num_urls"] = (df["extended_tweet.entities.urls"].fillna(df["entities.urls"]).fillna("[]").apply(lambda x: safe_len(x) if isinstance(x, list) else 0))
    df["num_media"] = (df["extended_tweet.entities.media"].fillna(df["entities.media"]).fillna("[]").apply(lambda x: safe_len(x) if isinstance(x, list) else 0))

    df["user_account_age_years"] = calculate_age_vectorized(df["user.created_at"])
    df["favorites_per_tweet"] = df["user.favourites_count"] / (df["user.statuses_count"] + 1)
    df["tweets_per_year"] = df["user.statuses_count"] / (df["user_account_age_years"] + 0.1)

    # ---------------------
    # B. QUOTED USER FIELDS
    # ---------------------

    df["quoted_status.has_description"] = (df["quoted_status.user.description"] != "").astype(int)
    df["quoted_status.has_location"] = (df["quoted_status.user.location"] != "").astype(int)
    df["quoted_status.has_url"] = ((df["quoted_status.user.url"].notna()) & (df["quoted_status.user.url"] != "")).astype(int)

    df["quoted_status.full_text"] = (df["quoted_status.extended_tweet.full_text"].fillna(df["quoted_status.text"]).fillna("")).apply(advanced_clean_text)
    df["quoted_status.text_length"] = df["quoted_status.full_text"].str.len()
    df["quoted_status.uppercase_ratio"] = df["quoted_status.full_text"].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    df["quoted_status.is_reply"] = (df["quoted_status.in_reply_to_status_id"]> 0).astype(int)
    df["quoted_status.num_hashtags"] = (df["quoted_status.extended_tweet.entities.hashtags"].fillna(df["quoted_status.entities.hashtags"]).fillna("[]").apply(lambda x: safe_len(x) if isinstance(x, list) else 0))
    df["quoted_status.num_user_mentions"] = (df["quoted_status.extended_tweet.entities.user_mentions"].fillna(df["quoted_status.entities.user_mentions"]).fillna("[]").apply(lambda x: safe_len(x) if isinstance(x, list) else 0))
    df["quoted_status.num_urls"] = (df["quoted_status.extended_tweet.entities.urls"].fillna(df["quoted_status.entities.urls"]).fillna("[]").apply(lambda x: safe_len(x) if isinstance(x, list) else 0))
    df["quoted_status.num_media"] = (df["quoted_status.extended_tweet.entities.media"].fillna(df["quoted_status.entities.media"]).fillna("[]").apply(lambda x: safe_len(x) if isinstance(x, list) else 0))

    df["quoted_status.user_account_age_years"] = calculate_age_vectorized(df["quoted_status.user.created_at"])
    df["quoted_status.favorites_per_tweet"] = df["quoted_status.user.favourites_count"] / (df["quoted_status.user.statuses_count"] + 1)
    df["quoted_status.tweets_per_year"] = df["quoted_status.user.statuses_count"] / (df["quoted_status.user_account_age_years"] + 0.1)

    df["quoted_status.follower_following_ratio"] = (df["quoted_status.user.followers_count"] + 1) / (df["quoted_status.user.friends_count"] + 1)
    df["quoted_status.favorites_per_tweet"] = df["quoted_status.user.favourites_count"] / (df["quoted_status.user.statuses_count"] + 1)
    df["quoted_status.followers_per_tweet"] = df["quoted_status.user.followers_count"] / (df["quoted_status.user.statuses_count"] + 1)
    df["quoted_status.engagement_score"] = (df["quoted_status.user.followers_count"] * 0.3 + df["quoted_status.user.friends_count"] * 0.2 + df["quoted_status.user.statuses_count"] * 0.1)


# %%
train_data = pd.read_json('train.jsonl', lines=True) #training data
train_data = json_normalize(train_data.to_dict(orient='records'))

kaggle_data = pd.read_json('kaggle_test.jsonl', lines=True) #test data
kaggle_data = json_normalize(kaggle_data.to_dict(orient='records'))

y_train = train_data['label']
train_data = train_data.drop('label', axis=1)

clean_dataframe(train_data)
clean_dataframe(kaggle_data)
extract_features_vectorized(train_data)
extract_features_vectorized(kaggle_data)

t_train = np.load('./train_embeddings_finetuned.npy')
t_test  = np.load('./test_embeddings_finetuned.npy')
train_scores = np.load('./train_scores_finetuned.npy')
test_scores  = np.load('./test_scores_finetuned.npy')

# %%

# ============================================================================
# Build User Lookups with IDs and build new features
# ============================================================================

def extract_user_id_from_banner(banner_url):
    """Extract user ID from profile_banner_url"""
    if pd.isna(banner_url) or banner_url == '' or banner_url is None:
        return None
    try:
        parts = str(banner_url).split('/')
        if len(parts) >= 2:
            user_id = parts[-2]
            if user_id.isdigit():
                return user_id
    except:
        pass
    return None


def build_user_lookups(train_df, test_df):
    """Build lookup tables and assign final_user_id"""
    print("STEP 2: Building User IDs with Challenge ID Fallback")
    
    # Extract user IDs from main tweets
    print("\nExtracting user IDs from profile_banner_url...")
    train_df['extracted_user_id'] = train_df['user.profile_banner_url'].apply(extract_user_id_from_banner)
    test_df['extracted_user_id'] = test_df['user.profile_banner_url'].apply(extract_user_id_from_banner)
    
    # Use challenge_id as fallback for missing user_ids
    print("\nUsing challenge_id as fallback for missing user_ids...")
    train_df['final_user_id'] = train_df['extracted_user_id'].fillna('challenge_' + train_df['challenge_id'].astype(str))
    test_df['final_user_id'] = test_df['extracted_user_id'].fillna('challenge_' + test_df['challenge_id'].astype(str))

    # Build user_id -> followers_count from quoted_status
    print("\nCollecting followers_count from quoted_status...")
    user_to_followers = {}
    
    for _, row in train_df.iterrows():
        quoted_uid = row.get('quoted_status.user.id_str')
        quoted_followers = row.get('quoted_status.user.followers_count')
        if pd.notna(quoted_uid) and pd.notna(quoted_followers):
            user_to_followers[str(quoted_uid)] = int(quoted_followers)
    
    for _, row in test_df.iterrows():
        quoted_uid = row.get('quoted_status.user.id_str')
        quoted_followers = row.get('quoted_status.user.followers_count')
        if pd.notna(quoted_uid) and pd.notna(quoted_followers):
            user_to_followers[str(quoted_uid)] = int(quoted_followers)
    
    print(f"Users with followers_count from quoted_status: {len(user_to_followers)}")
    
    return user_to_followers, train_df, test_df


def add_additional_features_based_on_IDs(df, user_to_followers) :
    df['as_followers_info'] = df['extracted_user_id'].apply(
        lambda x: 1 if x and x in user_to_followers else 0
    )
    df['followers_count'] = df['extracted_user_id'].apply(
        lambda x: user_to_followers.get(x, 0) if x else 0
    )
    df['log_followers'] = np.log1p(df['followers_count'])
    df['followers_above_1000'] = (df['followers_count'] >= 1000).astype(int)
    df['followers_above_5000'] = (df['followers_count'] >= 5000).astype(int)

# %%
user_to_followers, train_data, kaggle_data = build_user_lookups(train_data, kaggle_data)
add_additional_features_based_on_IDs(train_data,user_to_followers)
add_additional_features_based_on_IDs(kaggle_data,user_to_followers)

# %%
import pandas as pd
import numpy as np

def aggregate_user_features_final(data_df, embeddings, scores, labels=None, is_train=True):

    df = data_df.copy()

    df['bert_vec'] = list(embeddings)
    df['scores'] = scores
    
    # Attach Labels if this is training data
    if is_train:
        if labels is None:
            raise ValueError("You must provide 'labels' (y_train) when is_train=True")
        df['label'] = labels
        print("-> Labels attached.")

    # --- 2. Define Features to Aggregate ---
    metadata_cols = [
        col for col in df.select_dtypes(include=np.number).columns.tolist() 
        if 'id' not in col.lower() and col != 'label'
    ]
    
    # Define aggregation dictionary
    agg_dict = {}
    
    # Strategy for metadata: Mean, Sum, Max
    for col in metadata_cols:
        agg_dict[col] = ['mean', 'sum', 'max']
        
    # Strategy for tweet count
    # ASSUMPTION: 'user_id' is the column name. Change if it is 'final_user_id'
    user_id_col = 'final_user_id' 
    agg_dict[user_id_col] = 'count'
    
    # Strategy for Label: Take 'first' (assuming a user is either valid or spam, not both)
    if is_train:
        agg_dict['label'] = 'first'

    print(f"-> Aggregating {len(metadata_cols)} metadata features...")

    # --- 3. Perform Aggregation (Metadata + Label) ---
    
    # Group by user and apply aggregations
    user_agg = df.groupby(user_id_col).agg(agg_dict)
    
    # Flatten the hierarchical column index (e.g., ('text_len', 'mean') -> 'text_len_mean')
    new_columns = []
    for col in user_agg.columns:
        if isinstance(col, tuple):
            if col[0] == 'label': 
                new_columns.append('label') # Keep label simple
            elif col[0] == user_id_col and col[1] == 'count':
                new_columns.append('num_tweets')
            else:
                new_columns.append(f"{col[0]}_{col[1]}") # e.g. feature_mean
        else:
            new_columns.append(col)
            
    user_agg.columns = new_columns
    user_agg = user_agg.reset_index()

    # --- 4. Process Embeddings (Separate Step) ---
    
    print("-> Aggregating and Flattening BERT Embeddings (this may take a moment)...")
    
    # Function to average embeddings per user
    # Stack them to create a 2D array (N_users x 768)
    embedding_series = df.groupby(user_id_col)['bert_vec'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    
    # Convert the Series of arrays into a proper DataFrame with separate columns
    # This creates columns: emb_0, emb_1, ... emb_767
    embedding_df = pd.DataFrame(embedding_series.tolist(), index=embedding_series.index)
    embedding_df.columns = [f'emb_{i}' for i in range(embedding_df.shape[1])]
    
    # --- 5. Final Merge ---
    
    final_df = user_agg.merge(embedding_df, on=user_id_col, how='left')
    
    print(f"-> DONE. Final shape: {final_df.shape}")
    return final_df

# %%
user_train_df = aggregate_user_features_final(
    data_df=train_data,      
    embeddings=t_train,      
    labels=y_train,          
    scores= train_scores,
    is_train=True
)

user_test_df = aggregate_user_features_final(
    data_df=kaggle_data, 
    embeddings=t_test,       
    labels=None,            
    scores= test_scores,
    is_train=False
)

# %%
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ==========================================
# 1. SELECT NUMERIC FEATURES (NO IDs)
# ==========================================
def get_clean_features(df):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Remove any column containing 'id' or 'ID'
    features = [c for c in numeric_cols if 'id' not in c.lower()]
    return features

# Get feature list from training data (excluding label)
feature_cols = get_clean_features(user_train_df)
if 'label' in feature_cols: 
    feature_cols.remove('label')
print(f"Selected {len(feature_cols)} numeric features (IDs removed).")

# Prepare X and y
y_train = user_train_df['label']
X_train = user_train_df[feature_cols]
X_test  = user_test_df[feature_cols]

# ==========================================
# 2. XGBOOST SETUP & CV
# ==========================================
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    colsample_bytree=0.5,
    tree_method='hist',
    random_state=42
)

print("\nRunning 5-Fold Cross-Validation...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)

print(f"CV Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)")

# ==========================================
# 3. TRAIN & PREDICT
# ==========================================
print("\nTraining final model...")
model.fit(X_train, y_train)

print("Predicting on Test set...")
user_preds = model.predict(X_test)

# ==========================================
# 4. GENERATE SUBMISSION
# ==========================================
# 1. Map predictions back to User ID
user_pred_df = pd.DataFrame({
    'final_user_id': user_test_df['final_user_id'], # Ensure this matches your ID col name
    'prediction': user_preds
})



# %%
user_pred_df = user_pred_df.rename(columns={'user_id': 'final_user_id'})

# %%
# 2. Merge back to original Tweet-level data to get Challenge IDs
final_submission = kaggle_data.merge(user_pred_df, on='final_user_id', how='left')

# 3. Save
submission = pd.DataFrame({
    'ID': final_submission['challenge_id'],
    'Prediction': final_submission['prediction'].fillna(0).astype(int)
})

submission.to_csv('submission.csv', index=False)
print("Done. Saved to 'submission.csv'")

# %%


# %%


# %%


# %%



