import random
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

TARGET_PER_INTENT = 4000   # 5 intents × 4000 = 20000
OUTPUT_FILE = "synthetic_customer_support_dataset.csv"

random.seed(42)

INTENTS = ["Billing", "Refund", "Technical", "Complaint", "Product Inquiry"]
SENTIMENTS = ["Negative", "Neutral", "Positive"]

# ==========================================
# ENTITY POOLS (EXPANDED)
# ==========================================

products = [
    "mobile plan", "broadband", "fiber connection", "router",
    "SIM card", "data pack", "WiFi extender", "5G service",
    "international roaming", "family plan"
]

regions = [
    "US", "UK", "India", "Canada", "Australia",
    "Germany", "Singapore", "UAE", "France", "South Africa"
]

amounts = [
    "$10", "$20", "$30", "$40", "$50", "$75",
    "$100", "$120", "$150", "$200"
]

features = [
    "data rollover", "unlimited calls", "international roaming",
    "5G access", "premium support", "family sharing",
    "priority service", "cloud storage"
]

channels = [
    "customer support", "live chat", "email support",
    "call center", "mobile app", "service portal"
]

billing_issues = [
    "overcharged", "incorrectly billed", "charged twice",
    "charged extra", "given wrong invoice", "tax miscalculated"
]

technical_issues = [
    "slow internet", "network outage", "connection drops",
    "no signal", "unstable connection", "router malfunction",
    "DNS error", "packet loss issue"
]

complaints_list = [
    "poor service", "long waiting time",
    "unhelpful support", "delayed response",
    "rude behavior", "lack of transparency"
]

inquiries = [
    "pricing details", "plan benefits", "data limits",
    "upgrade options", "contract terms",
    "international coverage", "device compatibility"
]

# ==========================================
# TEMPLATE GENERATOR FUNCTION
# ==========================================

def generate_text(intent, sentiment):
    product = random.choice(products)
    region = random.choice(regions)
    amount = random.choice(amounts)
    feature = random.choice(features)
    channel = random.choice(channels)
    billing_issue = random.choice(billing_issues)
    tech_issue = random.choice(technical_issues)
    complaint = random.choice(complaints_list)
    inquiry = random.choice(inquiries)

    if intent == "Billing":
        base = f"I was {billing_issue} {amount} for my {product} in {region} via {channel}."
    elif intent == "Refund":
        base = f"I requested a refund of {amount} for my {product} purchased in {region}."
    elif intent == "Technical":
        base = f"I am experiencing {tech_issue} with my {product} in {region}."
    elif intent == "Complaint":
        base = f"I want to report {complaint} regarding {product} support in {region}."
    elif intent == "Product Inquiry":
        base = f"I would like information about {inquiry} and {feature} for the {product}."
    else:
        base = "Customer inquiry."

    if sentiment == "Negative":
        prefix = random.choice([
            "I am extremely frustrated.",
            "This is unacceptable.",
            "I am very disappointed.",
            "This situation is frustrating.",
            "I am unhappy with the service."
        ])
    elif sentiment == "Neutral":
        prefix = random.choice([
            "I would like clarification.",
            "Please provide details.",
            "Kindly assist me.",
            "I need some information.",
            "Could you help me?"
        ])
    else:
        prefix = random.choice([
            "Thank you for your assistance.",
            "I appreciate your help.",
            "Glad this was resolved.",
            "Thanks for the quick support.",
            "I am satisfied with the service."
        ])

    return prefix + " " + base

# ==========================================
# DATA GENERATION
# ==========================================

data = []
unique_texts = set()

for intent in INTENTS:
    count = 0
    while count < TARGET_PER_INTENT:
        sentiment = random.choice(SENTIMENTS)
        text = generate_text(intent, sentiment)

        if text not in unique_texts:
            unique_texts.add(text)
            data.append({
                "text": text,
                "intent": intent,
                "sentiment": sentiment
            })
            count += 1

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(data)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset Generated Successfully")
print("Total Rows:", len(df))
print("Unique Texts:", df["text"].nunique())
print("\nIntent Distribution:\n", df["intent"].value_counts())
print("\nSentiment Distribution:\n", df["sentiment"].value_counts())
