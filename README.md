# nltk_tensorflow_chatbot
This project processes the Twitter Customer Support (TWCS) dataset to generate structured chatbot training data in the intents.json format. It is designed to work with NLTK + TensorFlow chatbot architectures.

**📌 Objective**
To convert real customer support conversations from the TWCS dataset into a structured format usable by a chatbot. This enables the creation of more contextually relevant, real-world conversational agents.

**🧾 Dataset**
Source File: twcs.csv
Size: ~1 million rows
_Important Columns:_
tweet_id: Unique identifier for each tweet
text: Content of the tweet
in_response_to_tweet_id: Tweet this one is responding to
response_tweet_id: Tweet(s) responding to this one

**⚙️ How It Works**
Tweet Mapping: Maps each tweet_id to its corresponding text for quick lookup.
Intent Identification:
Uses keyword-based matching to group tweets into logical intents like password_reset, order_status, etc.
Pattern/Response Building:
For each tweet, identifies which tweet it replies to (patterns) and which replies to it (responses).
JSON Generation:
Outputs data in intents_grouped.json following this format:
```json
{
  "tag": "order_status",
  "patterns": ["Where is my order?", "I haven't received my shipment."],
  "responses": ["Let me check that for you.", "Can you share your order ID?"]
}
```

**📦 Output**
intents_grouped.json: The final file used to train your chatbot.

Intents are grouped meaningfully and cleaned to avoid too-specific or noisy entries.

**✅ Requirements**
Python 3.7+

pandas

🚀 How to Run
```bash
python prepare_data.py
```
This will:

Load and process twcs.csv

Create and save intents.json

📊 Progress Tracking
The script shows a progress bar during processing to track intent generation in real time.

🧠 Use Case
This file can now be used with chatbot training scripts in TensorFlow/Keras using a bag-of-words or embedding-based architecture.

✨ Credits
This project was built as part of a natural language processing pipeline for creating domain-specific chatbots based on real customer interaction data.
