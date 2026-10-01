
import torch
import numpy as np
import pandas as pd
import re
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate
from peft import LoraConfig, get_peft_model
import json
from pandas import json_normalize
import os
import shutil

#GPU check
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU acceleration.")
else:
    device = torch.device("cpu")
    print("WARNING: CUDA not available. Running on CPU (will be very very veryyyyyyyyy slow).")

# %% [markdown]
# ### Loading data csv

# %%
# if on google colab decomment theses codes
#from google.colab import drive
#drive.mount('/content/drive')
#DATA_PATH = "/content/drive/My Drive/ML_PROJECT"

#Path to the position of train.jsonl and kaggle.jsonl
DATA_PATH = "."

# %%
# --- CLEANING FUNCTION ---
def clean_tweet(text):
    """Cleans text by removing technical noise while preserving semantic signals."""
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\.\S+|https\S+", "[URL]", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_data = pd.read_json(f"{DATA_PATH}/train.jsonl", lines=True) #training data
train_data = json_normalize(train_data.to_dict(orient='records'))
train_data["full_text"] = (train_data["extended_tweet.full_text"].fillna(train_data["text"]).fillna("")).apply(clean_tweet)
train_data = train_data[['full_text', 'label']] #for memory

# %%
data_dict = {
    "text": train_data['full_text'].tolist(),
    "label": train_data['label'].tolist()
}
full_dataset = Dataset.from_dict(data_dict)

dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
hf_train = dataset_split["train"]
hf_val = dataset_split["test"]

# %%
model_name = 'camembert-base'
tokenizer = CamembertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256 #we chose this 
    )

print("Tokenizing datasets...")
tokenized_train = hf_train.map(tokenize_function, batched=True)
tokenized_val = hf_val.map(tokenize_function, batched=True)

# %%
# --- MODEL SETUP ---
num_labels = train_data['label'].nunique()
model = CamembertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# --- LoRA CONFIGURATION (PEFT) ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"], # Common target for sequence classification
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Move the model to the GPU after applying LoRA
model.to(device)

# --- METRICS ---
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

# %%
# --- TRAINING ARGUMENTS (OPTIMIZED FOR COLAB GPU) ---
training_args = TrainingArguments(
    output_dir="./camembert_lora_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    #fp16=True, # Recommended for speed on NVIDIA GPUs
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    dataloader_num_workers=2,
)

# --- TRAINER & RUN ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

print("Starting optimized fine-tuning on Colab GPU...")
def main():
    # --- TRAINER & RUN ---
    trainer.train()
if __name__ == "__main__":
    main()

# %%
# 1. Define the final output path in your Drive
DRIVE_PATH = DATA_PATH + '/CamemBERT_FineTuned_Final'
os.makedirs(DRIVE_PATH, exist_ok=True)
# 2. Save the adapter weights (LoRA)
# For PEFT models, we save the LoRA adapters first
model.save_pretrained(DRIVE_PATH)
# 3. Save the base model's config and the tokenizer
# The tokenizer is essential for inference
tokenizer.save_pretrained(DRIVE_PATH)
# The full model can be reconstructed by loading the base model and applying the adapters
print(f"Model (LoRA adapters) and tokenizer saved permanently to: {DRIVE_PATH}")


# %% [markdown]
# ### Getting The Embeddings

# %%

# %%
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from peft import PeftModel
import json
from pandas import json_normalize

# %%
#from google.colab import drive
#drive.mount('/content/drive')
#DATA_PATH = "/content/drive/My Drive/ML_PROJECT"
DATA_PATH = "."

# %%
train_data = pd.read_json(f"{DATA_PATH}/train.jsonl", lines=True)
train_data = json_normalize(train_data.to_dict(orient='records'))
train_data["full_text"] = (train_data["extended_tweet.full_text"].fillna(train_data["text"]).fillna("")).apply(clean_tweet)

kaggle_test = pd.read_json(f"{DATA_PATH}/kaggle_test.jsonl", lines=True)
kaggle_test = json_normalize(kaggle_test.to_dict(orient='records'))
kaggle_test["full_text"] = (kaggle_test["extended_tweet.full_text"].fillna(kaggle_test["text"]).fillna("")).apply(clean_tweet)

# %%
# --- 1. SETUP & MOUNT DRIVE ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Warning: Running on CPU. This will be slow.")


# --- 2. CONFIGURATION (EDIT THESE PATHS) ---
MODEL_PATH = f"{DATA_PATH}/CamemBERT_FineTuned_Final"
# Path where you want to save the new embeddings
OUTPUT_PATH = f"{DATA_PATH}/test_embeddings_finetuned.npy"

# --- 3. LOAD & MERGE MODEL ---
print("Loading tokenizer and model...")
try:
    # A. Load Tokenizer (Use standard base tokenizer to avoid file errors)
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # B. Load Base Model
    # We load it as a classifier because that's how it was fine-tuned
    base_model = CamembertForSequenceClassification.from_pretrained(
        "camembert-base",
        num_labels=2
    )

    # C. Load LoRA Adapters
    peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    # D. Merge Weights (Optimized for Inference)
    # This merges the LoRA layers into the base model for faster speed
    model = peft_model.merge_and_unload()

    model.to(device)
    model.eval()
    print("✅ Model successfully loaded and merged!")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Tip: Check that MODEL_PATH points to the unzipped folder containing adapter_model.bin")
    raise e

# --- 4. EMBEDDING FUNCTION ---
def get_embeddings(text_list, batch_size=32):
    """
    Generates embeddings on Colab GPU.
    """
    all_embeddings = []

    # Ensure input is a list
    if not isinstance(text_list, list): text_list = list(text_list)

    print(f"Generating embeddings for {len(text_list)} tweets...")

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            # 1. Batching
            batch_texts = text_list[i : i + batch_size]
            batch_texts = [str(t) for t in batch_texts] # Ensure strings

            # 2. Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256, # 256 is the sweet spot we discussed
                return_tensors='pt'
            ).to(device)

            # 3. Forward Pass
            # output_hidden_states=True gives us the embeddings
            output = model(**inputs, output_hidden_states=True)

            # 4. Extract [CLS] Token
            # Last layer (-1), First token (0, [CLS]), All dimensions (:)
            embeddings = output.hidden_states[-1][:, 0, :]

            # 5. Store on CPU
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

# --- 5. EXECUTION ---

# Run embedding generation
# You can use a larger batch size on Colab T4 GPUs (e.g., 64 or 128)
final_embeddings = get_embeddings(kaggle_test["full_text"].tolist(), batch_size=64)

print(f"Embeddings generated! Shape: {final_embeddings.shape}")

# Save to Drive
np.save(OUTPUT_PATH, final_embeddings)
print(f"Saved to: {OUTPUT_PATH}")

# %%
# --- 1. SETUP & MOUNT DRIVE ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Warning: Running on CPU. This will be slow.")


# --- 2. CONFIGURATION (EDIT THESE PATHS) ---
MODEL_PATH = f"{DATA_PATH}/CamemBERT_FineTuned_Final"
# Path where you want to save the new embeddings
OUTPUT_PATH = f"{DATA_PATH}/train_embeddings_finetuned.npy"

# --- 3. LOAD & MERGE MODEL ---
print("Loading tokenizer and model...")
try:
    # A. Load Tokenizer (Use standard base tokenizer to avoid file errors)
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    # B. Load Base Model
    base_model = CamembertForSequenceClassification.from_pretrained(
        "camembert-base",
        num_labels=2
    )

    # C. Load LoRA Adapters
    peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    # D. Merge Weights (Optimized for Inference)
    # This merges the LoRA layers into the base model for faster speed
    model = peft_model.merge_and_unload()

    model.to(device)
    model.eval()
    print("Model successfully loaded and merged!")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Tip: Check that MODEL_PATH points to the unzipped folder containing adapter_model.bin")
    raise e

# --- 4. EMBEDDING FUNCTION ---
def get_embeddings(text_list, batch_size=32):
    """
    Generates embeddings on Colab GPU.
    """
    all_embeddings = []

    # Ensure input is a list
    if not isinstance(text_list, list): text_list = list(text_list)

    print(f"Generating embeddings for {len(text_list)} tweets...")

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            # 1. Batching
            batch_texts = text_list[i : i + batch_size]
            batch_texts = [str(t) for t in batch_texts] # Ensure strings

            # 2. Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256, # 256 is the sweet spot we discussed
                return_tensors='pt'
            ).to(device)

            # 3. Forward Pass
            # output_hidden_states=True gives us the embeddings
            output = model(**inputs, output_hidden_states=True)

            # 4. Extract [CLS] Token Last layer (-1), First token (0, [CLS]), All dimensions (:)
            embeddings = output.hidden_states[-1][:, 0, :]

            # 5. Store on CPU
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

# --- 5. EXECUTION ---

# Run embedding generation
final_embeddings = get_embeddings(train_data["full_text"].tolist(), batch_size=64)

print(f"Embeddings generated! Shape: {final_embeddings.shape}")

# Save to Drive
np.save(OUTPUT_PATH, final_embeddings)
print(f"Saved to: {OUTPUT_PATH}")

# %% [markdown]
# ### Generating Scores

# %%
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from peft import PeftModel

# --- 1. SETUP & MOUNT DRIVE ---
# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Warning: Running on CPU. This will be slow.")

# --- 2. CONFIGURATION (EDIT THESE PATHS) ---
MODEL_PATH = f"{DATA_PATH}/CamemBERT_FineTuned_Final"

# Saving as SCORES, not embeddings
OUTPUT_PATH = f"{DATA_PATH}/train_scores_finetuned.npy"

# --- 3. LOAD & MERGE MODEL ---
print("Loading tokenizer and model...")

try:
    # A. Load Tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # B. Load Base Model
    base_model = CamembertForSequenceClassification.from_pretrained(
        "camembert-base",
        num_labels=2
    )

    # C. Load LoRA Adapters
    peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    # D. Merge Weights
    model = peft_model.merge_and_unload()

    model.to(device)
    model.eval()
    print("Model successfully loaded and merged!")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Tip: Check that MODEL_PATH points to the unzipped folder containing adapter_model.bin")
    raise e

# --- 4. SCORE FUNCTION (MODIFIED) ---
def get_scores(text_list, batch_size=32):
    """
    Generates Probability Scores (0 to 1) instead of Embeddings.
    """
    all_scores = []

    # Ensure input is a list
    if not isinstance(text_list, list):
        text_list = list(text_list)

    print(f"Generating scores for {len(text_list)} tweets...")

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            # 1. Batching
            batch_texts = text_list[i : i + batch_size]
            batch_texts = [str(t) for t in batch_texts]

            # 2. Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)

            # 3. Forward Pass
            # We remove output_hidden_states=True because we want the LOGITS (answers)
            output = model(**inputs)

            # 4. Convert Logits to Probabilities
            # Apply Softmax to get 0-1 range
            # dim=1 is the class dimension
            probs = torch.nn.functional.softmax(output.logits, dim=1)

            # 5. Extract Probability of Class 1 (Disaster)
            # We assume Index 1 is the "Positive/Disaster" class.
            # This gives you a single number per tweet representing confidence.
            class_1_score = probs[:, 1]

            # 6. Store on CPU
            all_scores.append(class_1_score.cpu().numpy())

    return np.concatenate(all_scores)

# --- 5. EXECUTION ---

# Run score generation
final_scores = get_scores(train_data["full_text"].tolist(), batch_size=64)

print(f"Scores generated! Shape: {final_scores.shape}")

np.save(OUTPUT_PATH, final_scores)
print(f"Saved scores to: {OUTPUT_PATH}")

# %%
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from peft import PeftModel

# --- 1. SETUP & MOUNT DRIVE ---
# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Warning: Running on CPU. This will be slow.")

# --- 2. CONFIGURATION (EDIT THESE PATHS) ---
MODEL_PATH = f"{DATA_PATH}/CamemBERT_FineTuned_Final"

# Saving as SCORES, not embeddings
OUTPUT_PATH = f"{DATA_PATH}/test_scores_finetuned.npy"

# --- 3. LOAD & MERGE MODEL ---
print("Loading tokenizer and model...")

try:
    # A. Load Tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # B. Load Base Model
    base_model = CamembertForSequenceClassification.from_pretrained(
        "camembert-base",
        num_labels=2
    )

    # C. Load LoRA Adapters
    peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    # D. Merge Weights
    model = peft_model.merge_and_unload()

    model.to(device)
    model.eval()
    print("Model successfully loaded and merged!")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Tip: Check that MODEL_PATH points to the unzipped folder containing adapter_model.bin")
    raise e

# --- 4. SCORE FUNCTION (MODIFIED) ---
def get_scores(text_list, batch_size=32):
    """
    Generates Probability Scores (0 to 1) instead of Embeddings.
    """
    all_scores = []

    # Ensure input is a list
    if not isinstance(text_list, list):
        text_list = list(text_list)

    print(f"Generating scores for {len(text_list)} tweets...")

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            # 1. Batching
            batch_texts = text_list[i : i + batch_size]
            batch_texts = [str(t) for t in batch_texts]

            # 2. Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)

            # 3. Forward Pass
            # We remove output_hidden_states=True because we want the LOGITS (answers)
            output = model(**inputs)

            # 4. Convert Logits to Probabilities
            # Apply Softmax to get 0-1 range
            # dim=1 is the class dimension
            probs = torch.nn.functional.softmax(output.logits, dim=1)

            # 5. Extract Probability of Class 1 (Disaster)
            # We assume Index 1 is the "Positive/Disaster" class.
            # This gives you a single number per tweet representing confidence.
            class_1_score = probs[:, 1]

            # 6. Store on CPU
            all_scores.append(class_1_score.cpu().numpy())

    return np.concatenate(all_scores)

# --- 5. EXECUTION ---

# Run score generation
final_scores = get_scores(kaggle_test["full_text"].tolist(), batch_size=64)

print(f"Scores generated! Shape: {final_scores.shape}")

np.save(OUTPUT_PATH, final_scores)
print(f"Saved scores to: {OUTPUT_PATH}")


