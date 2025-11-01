import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from tqdm import tqdm

# 1. Load your CSV
df = pd.read_csv('./Data/hardest_tags_simplified.csv')

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

# 3. Define your categories
candidate_labels = [
    'Revenue',
    'Cost', 
    'Expenses',
    'Assets',
    'Liabilities',
    'Equity',
    'Cash Flow',
    'Income',
    'Other'
]

# 4. Create the classification function
def classify_with_qwen3(tag, notes=""):
    """
    Use Qwen3 to classify a financial tag into one of the predefined categories.
    """
    # Include notes if available to provide context
    context = f"\nAdditional context: {notes}" if notes and pd.notna(notes) else ""
    
    prompt = f"""Classify this XBRL financial tag into ONE category:

Categories:
- Revenue
- Cost
- Expenses
- Assets
- Liabilities
- Equity
- Cash Flow
- Income
- Other

Tag: {tag}{context}

Category:"""

    # Prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=50,  # Limited tokens for category name only
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the new tokens (not the prompt)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # Clean up response - extract category name (case-insensitive matching)
    response_lower = response.lower()
    for label in candidate_labels:
        if label.lower() in response_lower:
            return label, 1.0
    
    # If no match found, return the raw response as Other
    print(f"⚠️  Couldn't parse response for {tag}: '{response}' - defaulting to Other")
    return "Other", 0.5

# 5. Test on a sample first (to see how it works)
sample_tags = df.head(20)

print("\n🧪 Testing on 20 sample tags...\n")
for idx, row in sample_tags.iterrows():
    tag = row['XBRL_Tag']
    notes = row['Notes'] if 'Notes' in row else ""
    predicted, confidence = classify_with_qwen3(tag, notes)
    actual = row['Category']
    
    match = "✅" if predicted == actual else "❌"
    print(f"{match} {tag[:50]:50s} | Predicted: {predicted:12s} | Actual: {actual}")

# 6. Evaluate on full dataset
print(f"\n📊 Evaluating on all {len(df)} tags (this may take 30-60 minutes depending on GPU)...")

correct = 0
predictions = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
    tag = row['XBRL_Tag']
    actual_category = row['Category']
    notes = row['Notes'] if 'Notes' in row else ""
    
    predicted_category, confidence = classify_with_qwen3(tag, notes)
    
    predictions.append({
        'XBRL_Tag': tag,
        'actual': actual_category,
        'predicted': predicted_category,
        'confidence': confidence,
        'correct': predicted_category == actual_category,
        'notes': notes
    })
    
    if predicted_category == actual_category:
        correct += 1
    
    # Print progress every 50 tags
    if (idx + 1) % 50 == 0:
        print(f"Progress: {idx + 1}/{len(df)} - Accuracy so far: {correct/(idx+1):.2%}")

# 7. Results
results_df = pd.DataFrame(predictions)
accuracy = correct / len(df)

print(f"\n{'='*60}")
print(f"📈 OVERALL ACCURACY: {accuracy:.2%} ({correct}/{len(df)})")
print(f"{'='*60}")

# Show confusion by category
print("\n📊 Accuracy by actual category:")
for category in sorted(df['Category'].unique()):
    cat_df = results_df[results_df['actual'] == category]
    cat_accuracy = cat_df['correct'].mean()
    cat_correct = cat_df['correct'].sum()
    cat_total = len(cat_df)
    print(f"  {category:15s}: {cat_accuracy:.2%} ({cat_correct}/{cat_total})")

# Show common misclassifications
print("\n❌ Top 10 misclassifications:")
misclassified = results_df[~results_df['correct']]
if len(misclassified) > 0:
    misclass_counts = Counter(zip(misclassified['actual'], misclassified['predicted']))
    for (actual, predicted), count in misclass_counts.most_common(10):
        print(f"  {actual:15s} -> {predicted:15s}: {count} times")
    
    # Show some examples of misclassifications
    print("\n📋 Sample misclassifications:")
    for i, row in misclassified.head(10).iterrows():
        print(f"  {row['XBRL_Tag'][:60]:60s} | Actual: {row['actual']:12s} | Predicted: {row['predicted']}")

# Save results
import os
os.makedirs('./Result', exist_ok=True)
results_df.to_csv('./Result/qwen3_llm_results.csv', index=False)
print("\n✅ Results saved to './Result/qwen3_llm_results.csv'")

# Clean up GPU memory
del model
del tokenizer
torch.cuda.empty_cache()

print("\n🎉 Classification complete!")