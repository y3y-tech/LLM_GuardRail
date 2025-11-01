import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from tqdm import tqdm

# 1. Load your CSV with latin-1 encoding
df = pd.read_csv("./Financial_Phrase_Bank/all-data.csv", encoding='latin-1')

# Handle different CSV formats
print(f"Columns in CSV: {df.columns.tolist()}")

# If the data is in format "sentence@label" or needs processing
if len(df.columns) == 1:
    df[['Text', 'Category']] = df.iloc[:, 0].str.split('@', expand=True)
elif 'sentence' in df.columns:
    df = df.rename(columns={'sentence': 'Text', 'label': 'Category'})

# Clean up category labels
df['Category'] = df['Category'].str.strip().str.lower()

# Take top 20 for testing
df = df.head(20)
print(f"\nLoaded {len(df)} financial news sentences")
print(f"Categories: {df['Category'].value_counts().to_dict()}")

# 2. Load Qwen3 model and tokenizer
print("\n🔄 Loading Qwen3-4B-Instruct model...")
model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print(f"✅ Model loaded on: {model.device}")

# 3. Define sentiment categories
candidate_labels = ['positive', 'negative', 'neutral']

# 4. Create the classification function
def classify_with_qwen3(text):
    """
    Use Qwen3 to classify financial news sentiment.
    """
    
    prompt = f"""Classify the sentiment of this financial news sentence as positive, negative, or neutral.

Consider from an investor's perspective:
- Positive: Good news that may increase stock price (growth, profit, expansion, success)
- Negative: Bad news that may decrease stock price (losses, layoffs, decline, problems)
- Neutral: Factual statements without clear positive or negative implications

Categories: positive, negative, neutral

Financial News: {text}

Sentiment:"""

    # Prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    # Conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the new tokens (not the prompt)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # Clean up response - extract sentiment (case-insensitive matching)
    response_lower = response.lower()
    for label in candidate_labels:
        if label in response_lower:
            return label, 1.0
    
    # If no match found, return the raw response
    print(f"⚠️  Couldn't parse response for text: '{response}' - defaulting to neutral")
    return "neutral", 0.5

# 5. Test on all 20 samples
print("\n🧪 Testing on 20 financial news sentences...\n")
print("="*120)

correct = 0
predictions = []

for idx, row in df.iterrows():
    text = row['Text']
    actual_sentiment = row['Category']
    
    predicted_sentiment, confidence = classify_with_qwen3(text)
    
    is_correct = predicted_sentiment == actual_sentiment
    if is_correct:
        correct += 1
    
    match = "✅" if is_correct else "❌"
    
    predictions.append({
        'Text': text,
        'actual': actual_sentiment,
        'predicted': predicted_sentiment,
        'confidence': confidence,
        'correct': is_correct
    })
    
    print(f"{match} #{idx+1}")
    print(f"   Actual:    {actual_sentiment.upper():8s}")
    print(f"   Predicted: {predicted_sentiment.upper():8s}")
    print(f"   Text: {text[:100]}...")
    print("-"*120)

# 6. Results
results_df = pd.DataFrame(predictions)
accuracy = correct / len(df)

print(f"\n{'='*80}")
print(f"📈 OVERALL ACCURACY: {accuracy:.1%} ({correct}/{len(df)})")
print(f"{'='*80}")

# Show accuracy by sentiment
print("\n📊 Accuracy by actual sentiment:")
for sentiment in ['positive', 'negative', 'neutral']:
    if sentiment in df['Category'].values:
        sent_df = results_df[results_df['actual'] == sentiment]
        sent_accuracy = sent_df['correct'].mean()
        sent_correct = sent_df['correct'].sum()
        sent_total = len(sent_df)
        print(f"  {sentiment.capitalize():10s}: {sent_accuracy:.1%} ({sent_correct}/{sent_total})")

# Confusion matrix
print("\n📊 Confusion Matrix:")
print("="*80)
confusion = pd.crosstab(
    results_df['actual'], 
    results_df['predicted'], 
    rownames=['Actual'], 
    colnames=['Predicted'],
    margins=True
)
print(confusion)

# Show misclassifications
print("\n❌ Misclassified Examples:")
print("="*80)
misclassified = results_df[~results_df['correct']]
if len(misclassified) > 0:
    for i, row in misclassified.iterrows():
        print(f"\n{row['actual'].upper()} → {row['predicted'].upper()}")
        print(f"Text: {row['Text']}")
else:
    print("🎉 Perfect classification! No errors!")

# Save results
import os
os.makedirs('./Result', exist_ok=True)
results_df.to_csv('./Result/qwen3_financial_phrasebank_results.csv', index=False)
print("\n✅ Results saved to './Result/qwen3_financial_phrasebank_results.csv'")

# Clean up GPU memory
del model
del tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n🎉 Classification complete!")
print("\n💡 Next step: Fine-tune Qwen3 on Financial PhraseBank to improve accuracy!")