import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from data_processor import EmojiDataProcessor

def analyze_emoji_data(sample_size=1000):
    """Perform initial analysis on the emoji dataset."""
    # Initialize processor
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    processor = EmojiDataProcessor(data_dir)
    
    # Load and process data with sampling for faster analysis
    df = processor.process_tweets(sample_size)
    
    # Get basic statistics
    stats = processor.get_emoji_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Plot emoji distribution
    plt.figure(figsize=(15, 6))
    emoji_counts = df['primary_emoji'].value_counts()
    sns.barplot(x=emoji_counts.index, y=emoji_counts.values)
    plt.title('Distribution of Primary Emojis in Dataset')
    plt.xlabel('Emoji Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'data' / 'emoji_distribution.png')
    plt.close()
    
    # Analyze emoji co-occurrence
    print("\nAnalyzing emoji co-occurrence...")
    co_occurrence = {}
    for emoji_list in df['emojis']:
        if len(emoji_list) > 1:
            for i, emoji1 in enumerate(emoji_list):
                for emoji2 in emoji_list[i+1:]:
                    pair = tuple(sorted([emoji1, emoji2]))
                    co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
    
    # Get top co-occurring pairs
    top_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Save co-occurrence results to file
    results = {
        'co_occurrence': [{
            'emoji1': pair[0],
            'emoji2': pair[1],
            'count': count
        } for pair, count in top_pairs]
    }
    
    results_path = Path(__file__).parent.parent / 'data' / 'processed' / 'emoji_analysis.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nAnalysis results saved to: {results_path}")
    
    # Analyze text patterns
    print("\nAnalyzing text patterns...")
    avg_text_length = df['clean_text'].str.len().mean()
    print(f"Average tweet length: {avg_text_length:.2f} characters")
    
    # Save processed data
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'processed_tweets.csv'
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")

if __name__ == "__main__":
    analyze_emoji_data()
