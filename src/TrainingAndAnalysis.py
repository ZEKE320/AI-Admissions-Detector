# %% [markdown]
# # AI/Human Text Classification
#
# This project focuses on developing models to distinguish between AI-generated and human-written texts.

# %%
# Import libraries
import re
import string
from typing import Optional

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertModel,
)
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.distilbert.modeling_distilbert import DistilBertModel
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer

nlp = spacy.load("en_core_web_sm")

# %% [markdown]
# ## Sample Texts
#
# Below are examples of texts used in our analysis, including AI-generated content and a mix of human and AI-assisted writing.

# %%
# Examples
test_text_AI = """During my undergraduate studies, I gained a solid foundation in programming languages such as Python and R, as well as experience working with SQL databases. I also had the opportunity to apply these skills in a real-world setting during an internship as a data analyst at a healthcare company. In this role, I was responsible for collecting, cleaning, and analyzing large datasets to provide insights into patient outcomes and healthcare costs.
Through the Master's in Data Science program at Fordham University, I aim to further develop my expertise in data science and analytics, with a focus on machine learning and predictive modeling. I am particularly interested in courses that cover topics such as deep learning, natural language processing, and data visualization. I am confident that this program will provide me with the skills and knowledge necessary to make valuable contributions to the field of data science.
Furthermore, I am impressed with the collaborative and interdisciplinary nature of the program, and I am excited about the opportunity to work with fellow students and faculty members from diverse backgrounds and fields. I am also attracted to the program's emphasis on practical, hands-on learning, which I believe will prepare me well for a career in data science.
Thank you for considering my application. I am excited about the prospect of joining the Fordham University community and contributing to the vibrant academic and research environment of the Master's in Data Science program.
Sincerely,"""

test_text_mixed_Human_AI = """Though I was barely exposed to programming while doing my B.A. in Economics, my eagerness to face new challenges led me to learn many computer languages like R and SQL. Ultimately, I decided to pursue a master's in Data Science at Fordham University, where I am fortunate to hold a Graduate Assistant position in Fordham's Computer and Information Sciences (CIS) department. Some projects have exposed me to the undertakings of high-level research and the massive amount of objectives and tasks necessary to transform a research proposal into a ranked article publication. But, most importantly, I have comprehended the role researchers play in advancing the computer science field and the gratification that yields, having contributed to it. As a result, I am driven to pursue a Ph.D. in Computer Science at NYU Tandon School of Engineering to broaden my knowledge and become an established researcher.
In addition to my academic pursuits, I have also gained valuable experience through internships and extracurricular activities. During my undergraduate years, I interned at a non-profit organization where I assisted in developing a database management system. This experience allowed me to apply my programming skills in a real-world setting and reinforced my interest in pursuing a career in technology.
Furthermore, I have been an active member of the Computer Science Club, where I have participated in various coding competitions and hackathons. These experiences have not only honed my technical skills but also taught me the importance of teamwork and collaboration in solving complex problems. I have also volunteered as a tutor, helping students with programming and data analysis. This experience has allowed me to share my knowledge with others and has reinforced my desire to pursue a career in academia.
As I embark on the next stage of my academic journey, I am eager to continue building on my experiences and knowledge. I am confident that pursuing a Ph.D. in Computer Science at NYU Tandon School of Engineering will provide me with the tools and resources necessary to achieve my academic and professional goals. I look forward to contributing to the vibrant research community at NYU and making meaningful contributions to the field of Computer Science."""

# %% [markdown]
# ## Text Preprocessing
#
# We define a preprocessing function to clean and format text for classification, ensuring consistency by removing unnecessary characters and patterns.


# %%
def text_cleaning(text):
    """This function takes a string as input and returns a formatted version of the string.

    The function replaces specific substrings in the input string with empty strings,
    converts the string to lowercase, removes any leading or trailing whitespace,
    and removes any punctuation from the string.
    """
    text = nlp(text)
    text = " ".join(
        [token.text for token in text if token.ent_type_ not in ["PERSON", "DATE"]]
    )

    pattern1 = r"f\d+"
    pattern2 = r"\b[A-Za-z]+\d+\b"
    pattern3 = r"\[(.*?)\]"

    text = re.sub(pattern1, "", text)
    text = re.sub(pattern2, "", text)
    text = re.sub(pattern3, "", text)

    return (
        text.replace("REDACTED", "")
        .lower()
        .replace("  ", " ")
        .replace("[Name]", "")
        .replace("[your name]", "")
        .replace("\n your name", "")
        .replace("dear admissions committee,", "")
        .replace("sincerely,", "")
        .replace("[university's name]", "fordham")
        .replace("dear sir/madam,", "")
        .replace("– statement of intent  ", "")
        .replace(
            "program: master of science in data analytics  name of applicant:    ", ""
        )
        .replace("data analytics", "data science")
        .replace("| \u200b", "")
        .replace("m.s. in data science at lincoln center  ", "")
        .translate(str.maketrans("", "", string.punctuation))
        .strip()
        .lstrip()
    )


# %% [markdown]
# ## Data Preparation
#
# In this section, we load and preprocess the data for our analysis.
#
# ### Data Loading and Labeling

# %%
# Read data
df_HumanGenerated = pd.read_csv(
    "/kaggle/input/capstoneresearchds/df_real.csv", dtype=str
)
df_AIGeneratedSOI = pd.read_csv(
    "/kaggle/input/capstoneresearchds/GeneratedSOIs.csv", dtype=str
)
df_AIGeneratedLOR = pd.read_csv(
    "/kaggle/input/capstoneresearchds/Generated_LORs.csv", dtype=str
)

# Label data: {Human:0, AI: 1}
df_HumanGenerated["Target"] = 0
df_AIGeneratedSOI["Target"] = 1
df_AIGeneratedLOR["Target"] = 1
df_AIGeneratedSOI["TypeDoc"] = "SOI"
df_AIGeneratedLOR["TypeDoc"] = "LOR"
df_AIGeneratedSOI.rename(
    {df_AIGeneratedSOI.columns[0]: "ID", df_AIGeneratedSOI.columns[2]: "Text"},
    axis=1,
    inplace=True,
)
df_AIGeneratedSOI["ID"] = df_AIGeneratedSOI.ID.map(lambda x: "0" + str(x))
df_AIGeneratedLOR.rename({df_AIGeneratedLOR.columns[0]: "Text"}, axis=1, inplace=True)

# Get only relevant attributes from the original datasources
cols = ["ID", "Text", "TypeDoc", "Target"]
df_HumanGenerated = df_HumanGenerated[cols]
df_AIGeneratedSOI = df_AIGeneratedSOI[cols]

# Union both human-written and AI generated datasets and shuffle
df = pd.concat([df_HumanGenerated.dropna(axis=0), df_AIGeneratedSOI])[
    ["Text", "TypeDoc", "Target"]
]
df = pd.concat([df, df_AIGeneratedLOR])
df.reset_index(drop=True, inplace=True)
df = df.sample(len(df))

# %% [markdown]
# ### Feature Engineering


# %%
# Feature engineering - For further modeling experiment
def AvgSentence(text):
    plist = text.split("\n")
    return np.mean([p.count(".") for p in plist])


df["CountParagraphs"] = df.Text.map(lambda x: x.count("\n"))
df["SumSentences"] = df.Text.map(lambda x: x.count(". "))
df["AvgSentenceByParagraphs"] = df.Text.map(AvgSentence)

df["Text"] = df["Text"].map(lambda x: text_cleaning(x))
df = df.drop_duplicates()

# %% [markdown]
# ### Data Split
#
# We divide the data into training and test sets to evaluate our models.

# %%
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("Target", axis=1), df.Target, train_size=0.80
)
train_sentences = X_train.Text.to_list()
test_sentences = X_test.Text.to_list()

test_sentences_LOR_ids = X_test[X_test["TypeDoc"] == "LOR"].index
test_sentences_SOI_ids = X_test[X_test["TypeDoc"] == "SOI"].index
test_sentences_LOR = X_test.loc[test_sentences_LOR_ids].Text.to_list()
y_LORs = y_test.loc[test_sentences_LOR_ids]
test_sentences_SOI = X_test.loc[test_sentences_SOI_ids].Text.to_list()
y_SOIs = y_test.loc[test_sentences_SOI_ids]

# %% [markdown]
# ## Baseline Models
#
# We start with baseline models using traditional machine learning techniques, such as logistic regression and naive Bayes, combined with TF-IDF vectorization.

# %%
## Logistic Regression + TF-IDF Pipeline
model_lr = Pipeline([("tf-idf", TfidfVectorizer()), ("clf", LogisticRegression())])
model_lr.fit(X=train_sentences, y=y_train)

## Naive Bayes + TF-IDF Pipeline
model_nb = Pipeline([("tf-idf", TfidfVectorizer()), ("clf", MultinomialNB())])
model_nb.fit(X=train_sentences, y=y_train)

print("TF-IDF + LR")
print(f"Accuracy: {model_lr.score(test_sentences, y_test)}\n")
print(classification_report(y_test, model_lr.predict(test_sentences)), "\n")

print("TF-IDF + NB")
print(f"Accuracy: {model_nb.score(test_sentences, y_test)}\n")
print(classification_report(y_test, model_nb.predict(test_sentences)))

# Save baseline models
# dump(model_lr, "baseline_model_lr.joblib");
# dump(model_nb, "baseline_model_nb.joblib");

# %% [markdown]
# ## Transformer-based Models
#
# We then explore more advanced models based on Transformers.
#
# ### Early Stopping

# %%
# Transformers-based models


# earlystopping callback
class EarlyStopping:
    def __init__(self, patience: int = 2, delta: float = 0.000001) -> None:
        self.patience: int = patience
        self.counter: int = 0
        self.best_score: float | None = None
        self.delta: float = delta

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            self.save_checkpoint(val_loss, model)
            self.best_score = score
            self.counter = 0
        return False

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        torch.save(model.state_dict(), "checkpoint.pt")
        print(
            f"Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ..."
        )


# %% [markdown]
# ### DistilBERT Model Implementation
#
# Here, we implement DistilBERT, a streamlined version of the BERT model.


# %%
## DistilBert
class TextDataset(Dataset):
    def __init__(self, encodings: dict[str, list[int]], labels: np.ndarray) -> None:
        self.encodings: dict[str, list[int]] = encodings
        self.labels: np.ndarray = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# Encode the text data
train_encodings = tokenizer(
    train_sentences, padding="max_length", max_length=512, truncation=True
)
val_encodings = tokenizer(
    test_sentences, padding="max_length", max_length=512, truncation=True
)

# Include additional features
columns = X_train.drop(["Text"], axis=1).columns.to_list()

for column in columns:
    train_encodings[column] = X_train[column].to_list()
    val_encodings[column] = X_test[column].to_list()

train_dataset = TextDataset(train_encodings, y_train.values)
val_dataset = TextDataset(val_encodings, y_test.values)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class TransformerBasedModelDistilBert(nn.Module):
    def __init__(self) -> None:
        super(TransformerBasedModelDistilBert, self).__init__()
        self.bert: DistilBertModel = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.dropout: nn.Dropout = nn.Dropout(0.55)
        self.fc: nn.Linear = nn.Linear(768, 2)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


# %% [markdown]
# ### DistilBERT Model Training

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerBasedModelDistilBert().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
early_stopping = EarlyStopping()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_total = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        train_loss += loss
        train_total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        correct = 0
        test_loss = 0
        total = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            test_loss += loss_fn(logits, labels)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    early_stop = early_stopping(test_loss / total, model)

    if early_stop:
        print("Early stopping")
        break

    print(
        f"Epoch {epoch + 1} out of {num_epochs} | Train Loss: {train_loss / train_total:.6f} | Test Accuracy: {correct / total:.6f}"
    )

# %% [markdown]
# ### DistilBERT Model Saving

# %%
### best weights
model.load_state_dict(torch.load("checkpoint.pt"))


### Save as HuggingFace Model
class MyConfigDistil(PretrainedConfig):
    model_type: str = "distilbert"

    def __init__(self, final_dropout: float = 0.55, **kwargs) -> None:
        super().__init__(**kwargs)
        self.final_dropout: float = final_dropout


class MyHFModel_DistilBased(PreTrainedModel):
    config_class: type[MyConfigDistil] = MyConfigDistil

    def __init__(self, config: MyConfigDistil) -> None:
        super().__init__(config)
        self.config: MyConfigDistil = config
        self.model: nn.Module = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape)

        return self.model(input_ids=input_ids, attention_mask=attention_mask)


config = MyConfigDistil(0.55)
Custom_HF_Model = MyHFModel_DistilBased(config)

Custom_HF_Model.save_pretrained("HF_DistilBertBasedModelAppDocs")

# %% [markdown]
# ### BERT Model Implementation
#
# Next, we implement the standard BERT model.

# %%
##########
# Bert
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encode the text data
train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
val_encodings = tokenizer(test_sentences, truncation=True, padding=True)

# Include additional features
columns = X_train.drop(["Text"], axis=1).columns.to_list()

for column in columns:
    train_encodings[column] = X_train[column].to_list()
    val_encodings[column] = X_test[column].to_list()

train_dataset = TextDataset(train_encodings, y_train.values)
val_dataset = TextDataset(val_encodings, y_test.values)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class TransformerBasedModelBert(nn.Module):
    def __init__(self) -> None:
        super(TransformerBasedModelBert, self).__init__()
        self.bert: BertModel = BertModel.from_pretrained("bert-base-uncased")
        self.dropout: nn.Dropout = nn.Dropout(0.55)
        self.fc: nn.Linear = nn.Linear(768, 2)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


# %% [markdown]
# ### BERT Model Training

# %%
model = TransformerBasedModelBert().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
early_stopping = EarlyStopping()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_total = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        train_loss += loss
        train_total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        correct = 0
        test_loss = 0
        total = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            test_loss += loss_fn(logits, labels)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    early_stop = early_stopping(test_loss / total, model)

    if early_stop:
        print("Early stopping")
        break

    print(
        f"Epoch {epoch + 1} out of {num_epochs} | Train Loss: {train_loss / train_total:.6f} | Test Accuracy: {correct / total:.6f}"
    )

# %% [markdown]
# ### BERT Model Saving

# %%
### best weights
model.load_state_dict(torch.load("checkpoint.pt"))


class MyConfigBert(PretrainedConfig):
    model_type: str = "bert"

    def __init__(self, final_dropout: float = 0.55, **kwargs) -> None:
        super().__init__(**kwargs)
        self.final_dropout: float = final_dropout


class MyHFModel_BertBased(PreTrainedModel):
    config_class: type[MyConfigBert] = MyConfigBert

    def __init__(self, config: MyConfigBert) -> None:
        super().__init__(config)
        self.config: MyConfigBert = config
        self.model: nn.Module = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


config = MyConfigBert(0.55)
Custom_HF_Model = MyHFModel_BertBased(config)

Custom_HF_Model.save_pretrained("HF_BertBasedModelAppDocs")

# %% [markdown]
# ## Extended Experiments with Additional Data
#
# We expand our training dataset by incorporating Wikipedia data to enhance model performance.

# %%
## Training with Academic App docs + Wiki
df_wiki = pd.read_csv("GPT-wiki-intro.csv").sample(30000)
originals = pd.DataFrame(df_wiki["wiki_intro"]).rename({"wiki_intro": "Text"}, axis=1)
originals["Target"] = 0

generated = pd.DataFrame(df_wiki["generated_intro"]).rename(
    {"generated_intro": "Text"}, axis=1
)
generated["Target"] = 1

Wiki = pd.concat([originals, generated], ignore_index=True)
Wiki = Wiki.sample(len(Wiki))

df_larger = pd.concat([df[["Text", "Target"]], Wiki], ignore_index=True)
df_larger = df_larger.sample(len(df_larger))
df_larger["Text"] = df_larger["Text"].map(lambda x: text_cleaning(x))
df_larger = df_larger.drop_duplicates()
df_larger.reset_index(drop=True, inplace=True)

# %% [markdown]
# ### Baseline Models with Extended Data

# %%
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df_larger.drop("Target", axis=1), df_larger.Target, train_size=0.80
)
train_sentences = X_train.Text.to_list()
test_sentences = X_test.Text.to_list()
# Baseline models
model_lr = Pipeline([("tf-idf", TfidfVectorizer()), ("clf", LogisticRegression())])
model_lr.fit(X=train_sentences, y=y_train)

model_nb = Pipeline([("tf-idf", TfidfVectorizer()), ("clf", MultinomialNB())])
model_nb.fit(X=train_sentences, y=y_train)

print("TF-IDF + LR")
print(f"Accuracy: {model_lr.score(test_sentences, y_test)}\n")
print(classification_report(y_test, model_lr.predict(test_sentences)), "\n")

print("TF-IDF + NB")
print(f"Accuracy: {model_nb.score(test_sentences, y_test)}\n")
print(classification_report(y_test, model_nb.predict(test_sentences)))

# %% [markdown]
# ### Saving Extended Baseline Models

# %%
# Save baseline model
dump(model_lr, "baseline_model_lr2.joblib")
dump(model_nb, "baseline_model_nb2.joblib")

# %% [markdown]
# ### DistilBERT Model with Extended Data

# %%
# Transformers-based models

## DistilBert

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# Encode the text data
train_encodings = tokenizer(
    train_sentences, padding="max_length", max_length=512, truncation=True
)
val_encodings = tokenizer(
    test_sentences, padding="max_length", max_length=512, truncation=True
)

# Include additional features
columns = X_train.drop(["Text"], axis=1).columns.to_list()

for column in columns:
    train_encodings[column] = X_train[column].to_list()
    val_encodings[column] = X_test[column].to_list()

train_dataset = TextDataset(train_encodings, y_train.values)
val_dataset = TextDataset(val_encodings, y_test.values)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = TransformerBasedModelDistilBert().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
early_stopping = EarlyStopping()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_total = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        train_loss += loss
        train_total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        correct = 0
        test_loss = 0
        total = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            test_loss += loss_fn(logits, labels)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    early_stop = early_stopping(test_loss / total, model)

    if early_stop:
        print("Early stopping")
        break

    print(
        f"Epoch {epoch + 1} out of {num_epochs} | Train Loss: {train_loss / train_total:.6f} | Test Accuracy: {correct / total:.6f}"
    )

# %% [markdown]
# ### Saving Extended DistilBERT Model

# %%
### best weights
model.load_state_dict(torch.load("checkpoint.pt"))

### Save as HuggingFace Model
config = MyConfigDistil(0.55)
Custom_HF_Model = MyHFModel_DistilBased(config)

Custom_HF_Model.save_pretrained("HF_DistilBertBasedModelAppDocs2")

# %% [markdown]
# ### BERT Model with Extended Data

# %%
##########
# Bert
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encode the text data
train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
val_encodings = tokenizer(test_sentences, truncation=True, padding=True)

# Include additional features
columns = X_train.drop(["Text"], axis=1).columns.to_list()

for column in columns:
    train_encodings[column] = X_train[column].to_list()
    val_encodings[column] = X_test[column].to_list()

train_dataset = TextDataset(train_encodings, y_train.values)
val_dataset = TextDataset(val_encodings, y_test.values)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = TransformerBasedModelBert().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
early_stopping = EarlyStopping()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_total = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        train_loss += loss
        train_total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        correct = 0
        test_loss = 0
        total = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            test_loss += loss_fn(logits, labels)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    early_stop = early_stopping(test_loss / total, model)

    if early_stop:
        print("Early stopping")
        break

    print(
        f"Epoch {epoch + 1} out of {num_epochs} | Train Loss: {train_loss / train_total:.6f} | Test Accuracy: {correct / total:.6f}"
    )

# %% [markdown]
# ### Saving Extended BERT Model

# %%
### best weights
model.load_state_dict(torch.load("checkpoint.pt"))

config = MyConfigBert(0.55)
Custom_HF_Model = MyHFModel_BertBased(config)

Custom_HF_Model.save_pretrained("HF_BertBasedModelAppDocs2")
