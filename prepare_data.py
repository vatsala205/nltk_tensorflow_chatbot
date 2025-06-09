import pandas as pd
import json

# Load the CSV file
df = pd.read_csv('sample.csv')

# Create a dictionary to get tweet text by tweet_id (int)
tweet_text_by_id = {}
for _, row in df.iterrows():
    tweet_text_by_id[int(row['tweet_id'])] = row['text']

# Helper function to parse comma-separated IDs safely
def parse_ids(ids):
    result = []
    if pd.isna(ids):
        return result
    for x in str(ids).split(','):
        x = x.strip()
        if x and x.lower() != 'nan':
            try:
                result.append(int(float(x)))  # convert '3.0' -> 3 safely
            except ValueError:
                pass
    return result

# Map keywords to intent tags - add your own keywords here
keywords_to_tags = {
    'password': 'password_reset',
    'forgot': 'password_reset',
    'reset': 'password_reset',
    'login': 'account_help',
    'account': 'account_help',
    'order': 'order_status',
    'shipment': 'order_status',
    'track': 'order_status',
    'payment': 'payment_issue',
    'pay': 'payment_issue',
    'technical': 'technical_issue',
    'error': 'technical_issue',
    'help': 'general_help',
    'support': 'general_help',
    # Add more as you find in your data
}

def assign_tag(text):
    text_lower = text.lower()
    for kw, tag in keywords_to_tags.items():
        if kw in text_lower:
            return tag
    return None

# Initialize containers for patterns and responses per intent
tag_patterns = {}
tag_responses = {}

total_rows = len(df)
for idx, row in enumerate(df.itertuples(), 1):
    tweet_id = int(row.tweet_id)
    text = row.text
    in_response_ids = parse_ids(getattr(row, 'in_response_to_tweet_id', ''))
    response_ids = parse_ids(getattr(row, 'response_tweet_id', ''))

    # Assign an intent tag based on keywords in the tweet text
    tag = assign_tag(text)
    if not tag:
        continue  # Skip tweets that don't match any keyword

    # Initialize dict keys if not present
    if tag not in tag_patterns:
        tag_patterns[tag] = set()
    if tag not in tag_responses:
        tag_responses[tag] = set()

    # This tweet is considered a pattern example for the intent
    tag_patterns[tag].add(text)

    # Add the texts of tweets that respond to this tweet as responses
    for rid in response_ids:
        if rid in tweet_text_by_id:
            tag_responses[tag].add(tweet_text_by_id[rid])

    # Optionally, also add texts of tweets this tweet responds to as patterns
    for pid in in_response_ids:
        if pid in tweet_text_by_id:
            tag_patterns[tag].add(tweet_text_by_id[pid])

    # Print progress every 10,000 rows
    if idx % 10000 == 0 or idx == total_rows:
        print(f"Processed {idx}/{total_rows} rows ({(idx/total_rows)*100:.2f}%)")

# Build final intents structure
intents = {"intents": []}

for tag in tag_patterns:
    patterns = list(tag_patterns[tag])
    responses = list(tag_responses[tag])

    # If no responses found, add a default fallback response
    if not responses:
        responses = ["I'm here to help with your queries. Could you please provide more details?"]

    intent = {
        "tag": tag,
        "patterns": patterns[:50],    # limit to first 50 patterns to keep file size reasonable
        "responses": responses[:50]   # limit responses too
    }
    intents["intents"].append(intent)

# Save to JSON file
with open('intents.json', 'w', encoding='utf-8') as f:
    json.dump(intents, f, indent=4, ensure_ascii=False)

print(f"Created intents.json with {len(intents['intents'])} intents.")
