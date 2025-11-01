import pandas as pd
import torch
from transformers import pipeline
from collections import Counter

# 1. Load your CSV
df = pd.read_csv('./Data/hardest_tags_simplified.csv')

# 2. Load zero-shot classification pipeline
# This uses a pre-trained model without any fine-tuning
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# 3. Define your categories
candidate_labels = [
    'Revenue',
    'Expenses',
    'Assets',
    'Liabilities',
    'Equity',
    'Cash Flow',
    'Income',
    'Other'
]

# 4. Test on a sample first (to see how it works)
sample_tags = df['XBRL_Tag'].head(20).tolist()

print("\n🧪 Testing on 20 sample tags...\n")
for tag in sample_tags:
    result = classifier(tag, candidate_labels)
    predicted = result['labels'][0]
    confidence = result['scores'][0]
    actual = df[df['XBRL_Tag'] == tag]['Category'].values[0]
    
    match = "✅" if predicted == actual else "❌"
    print(f"{match} {tag[:50]:50s} | Predicted: {predicted:12s} ({confidence:.2%}) | Actual: {actual}")

'''
# 5. Evaluate on full dataset (this will take a while!)
print(f"\n📊 Evaluating on all {len(df)} tags (this may take 10-20 minutes)...")

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
    
    # Print progress every 50 tags
    if (idx + 1) % 50 == 0:
        print(f"Progress: {idx + 1}/{len(df)} - Accuracy so far: {correct/(idx+1):.2%}")

# 6. Results
results_df = pd.DataFrame(predictions)
accuracy = correct / len(df)

print(f"\n{'='*60}")
print(f"📈 OVERALL ACCURACY: {accuracy:.2%}")
print(f"{'='*60}")

# Show confusion by category
print("\n📊 Accuracy by actual category:")
for category in df['category'].unique():
    cat_df = results_df[results_df['actual'] == category]
    cat_accuracy = cat_df['correct'].mean()
    print(f"  {category:15s}: {cat_accuracy:.2%} ({cat_df['correct'].sum()}/{len(cat_df)})")

# Show common misclassifications
print("\n❌ Top 10 misclassifications:")
misclassified = results_df[~results_df['correct']]
misclass_counts = Counter(zip(misclassified['actual'], misclassified['predicted']))
for (actual, predicted), count in misclass_counts.most_common(10):
    print(f"  {actual:15s} -> {predicted:15s}: {count} times")

# Save results
results_df.to_csv('./Result/zero_shot_results.csv', index=False)
print("\n✅ Results saved to 'zero_shot_results.csv'")

'''