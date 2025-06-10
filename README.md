
# 🤖 Multilingual Chatbot with NLTK & TensorFlow

This project helps you build a fully functional chatbot using real conversation data such as the [Twitter Customer Support Corpus (TWCS)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). It includes all the tools to convert large-scale tweet data into a chatbot-friendly `intents.json` format and train a neural network using NLTK and TensorFlow.

---

## 📁 Project Structure Overview

| File / Folder              | Description                                                           |
| -------------------------- | --------------------------------------------------------------------- |
| `prepare_data.py`          | Converts conversation CSV (like `twcs.csv`) to `intents.json` format  |
| `intents.json`             | Structured conversational data used to train the chatbot              |
| `preprocess.py`            | Prepares the dataset, tokenizes/lemmatizes text, builds training data |
| `train.py`                 | Builds and trains the neural network model using TensorFlow           |
| `chat.py`                  | Launches an interactive chatbot terminal for testing                  |
| `words.pkl`, `classes.pkl` | Saved preprocessed vocabulary and labels used by the model            |
| `chatbot_model.h5`         | Trained Keras model                                                   |
| `twcs.csv`                 | (Optional) Raw dataset — can be swapped with any similar CSV file     |

---

## 🔧 Libraries to Install

Before running anything, install the required Python packages:

```bash
pip install pandas numpy nltk tensorflow
```

Also, make sure to download the necessary NLTK data files:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🚀 How to Use the Project

### ✅ Step-by-Step Guide

---

### 1. **Prepare the Data (Optional if you already have `intents.json`)**

If you're using a raw conversation dataset like `twcs.csv`, run:

```bash
python prepare_data.py
```

This script:

* Loads your CSV
* Links tweet-response pairs
* Saves them as an `intents.json` file ready for chatbot training

📌 **Note:** You can use **any dataset** by replacing the `twcs.csv` file and changing the filename in `prepare_data.py`. Just ensure the file has the following columns:

* `tweet_id`
* `text`
* `in_response_to_tweet_id`
* `response_tweet_id`

💡 **Already have `intents.json`?**
You can **skip this step** completely and go straight to preprocessing.

---

### 2. **Preprocess the Data**

Prepare the training data by running:

```bash
python preprocess.py
```

This script:

* Loads `intents.json`
* Tokenizes and lemmatizes the text
* Builds a bag-of-words model
* Saves `words.pkl`, `classes.pkl`, and `training_data.pkl`

---

### 3. **Train the Model**

Train the chatbot neural network:

```bash
python train.py
```

This script:

* Loads the preprocessed training data
* Creates and trains a model with `tensorflow.keras`
* Saves the trained model to `chatbot_model.h5`

---

### 4. **Chat with the Bot**

Run the chatbot in terminal mode:

```bash
python chat.py
```

You can now talk to the bot using real language, and it will classify your message into an intent and reply from its learned response pool.

---

## 🔁 Using Another Dataset or Custom `intents.json`

### 🔹 Option A: New Dataset (CSV)

Just replace `twcs.csv` with your own conversation-style dataset and run `prepare_data.py` again.

Update this line in `prepare_data.py`:

```python
df = pd.read_csv('your_file.csv')
```

### 🔹 Option B: Custom `intents.json`

Already have a ready-to-use `intents.json` file?

➡️ Skip `prepare_data.py`
Just place your file in the project root and proceed with:

```bash
python preprocess.py
python train.py
python chat.py
```

---

## 🔍 Sample `intents.json` Format

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hi there!", "Hello!", "How can I help you?"]
    },
    ...
  ]
}
```

---

## 📦 Output Files After Full Run

| File                   | Purpose                                         |
| ---------------------- | ----------------------------------------------- |
| `intents.json`         | Structured intent dataset                       |
| `intents_grouped.json` | (optional, for future) Enhanced grouped intents |
| `words.pkl`            | Vocabulary used for training                    |
| `classes.pkl`          | Tags (intents) used in classification           |
| `chatbot_model.h5`     | Trained chatbot model                           |

---

## 📌 Notes & Best Practices

* The bot works best with **grouped, generalized intents** (e.g., "reset password", "account locked").
* Ensure enough **diverse patterns** per intent during data preparation.
* You can manually edit `intents.json` to improve clarity or combine similar intents.

---

## 🛠️ Future Enhancements

* Add support for multilingual pattern generation
* Visual intent clustering with embeddings
* Voice input/output modules

---

## 👩‍💻 Made For

This project was created as part of a larger AI platform project aimed at helping visually impaired users through voice-based intelligent systems.

---
