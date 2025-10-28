import pandas as pd
import torch
from transformers import pipeline
from collections import Counter

# 1. Load your CSV
df = pd.read_csv('./Data/labeled_financial_tags.csv')
print(f"Loaded {len(df)} tags")

# 2. Use DeBERTa-NLI (better for zero-shot classification)
print("\n🔄 Loading DeBERTa-v3 NLI model (better for financial understanding)...")
classifier = pipeline(
    "zero-shot-classification",
    model="microsoft/deberta-v3-base",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# 3. Define your categories with more descriptive labels
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

# 4. Test on a sample first
sample_tags = df['tag'].head(20).tolist()

print("\n🧪 Testing on 20 sample tags...\n")
for tag in sample_tags:
    result = classifier(tag, candidate_labels)
    predicted = result['labels'][0]
    confidence = result['scores'][0]
    actual = df[df['tag'] == tag]['category'].values[0]
    
    match = "✅" if predicted == actual else "❌"
    print(f"{match} {tag[:50]:50s} | Predicted: {predicted:12s} ({confidence:.2%}) | Actual: {actual}")

# 5. Evaluate on full dataset
print(f"\n📊 Evaluating on all {len(df)} tags...")

correct = 0
predictions = []

for idx, row in df.iterrows():
    tag = row['tag']
    actual_category = row['category']
    
    result = classifier(tag, candidate_labels)
    predicted_category = result['labels'][0]
    confidence = result['scores'][0]
    
    predictions.append({
        'tag': tag,
        'actual': actual_category,
        'predicted': predicted_category,
        'confidence': confidence,
        'correct': predicted_category == actual_category
    })
    
    if predicted_category == actual_category:
        correct += 1
    
    if (idx + 1) % 50 == 0:
        print(f"Progress: {idx + 1}/{len(df)} - Accuracy so far: {correct/(idx+1):.2%}")

# 6. Results
results_df = pd.DataFrame(predictions)
accuracy = correct / len(df)

print(f"\n{'='*60}")
print(f"📈 OVERALL ACCURACY: {accuracy:.2%}")
print(f"{'='*60}")

print("\n📊 Accuracy by actual category:")
for category in df['category'].unique():
    cat_df = results_df[results_df['actual'] == category]
    cat_accuracy = cat_df['correct'].mean()
    print(f"  {category:15s}: {cat_accuracy:.2%} ({cat_df['correct'].sum()}/{len(cat_df)})")

print("\n❌ Top 10 misclassifications:")
misclassified = results_df[~results_df['correct']]
misclass_counts = Counter(zip(misclassified['actual'], misclassified['predicted']))
for (actual, predicted), count in misclass_counts.most_common(10):
    print(f"  {actual:15s} -> {predicted:15s}: {count} times")

results_df.to_csv('./Result/deberta_zero_shot_results.csv', index=False)
print("\n✅ Results saved to './Result/deberta_zero_shot_results.csv'")