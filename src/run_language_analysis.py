from pathlib import Path
from language_patterns import EmojiLanguageAnalyzer

def format_percentage(value, total):
    return f"{value} ({(value/total)*100:.1f}%)"

def main():
    # Get path to processed data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'processed_tweets.csv'
    
    if not data_path.exists():
        print(f"Error: Processed data file not found at {data_path}")
        print("Please run analysis.py first to generate the processed data.")
        return
    
    # Create analyzer and generate visualizations
    print("Starting language pattern analysis...")
    analyzer = EmojiLanguageAnalyzer(data_path)
    results = analyzer.create_language_visualizations()
    
    # Print summary
    print("\nLanguage Pattern Analysis Summary:")
    print("-" * 40)
    
    # Positioning
    total_emojis = results['positioning']['total_emojis']
    positions = results['positioning']['position_distribution']
    print("\n1. Emoji Positioning:")
    print(f"Total emojis analyzed: {total_emojis}")
    print("Position distribution:")
    for position, count in positions.items():
        print(f"- {position.title()}: {format_percentage(count, total_emojis)}")
    
    # Phrases
    print("\n2. Emoji Combinations:")
    print(f"Unique emoji pairs found: {results['phrases']['unique_pairs']}")
    print(f"Unique emoji triples found: {results['phrases']['unique_triples']}")
    
    # Save detailed results to file
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    with open(output_dir / 'language_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== Detailed Language Analysis Results ===\n\n")
        
        f.write("Top 10 Emoji Pairs:\n")
        for pair, count in results['phrases']['top_pairs']:
            f.write(f"- {pair}: {count} occurrences\n")
        
        f.write("\nTop Words by Emoji:\n")
        for emoji, words in results['words']['top_emoji_words'].items():
            f.write(f"\nEmoji {emoji}:\n")
            for word, count in words.items():
                f.write(f"- {word}: {count}\n")
    
    print("\nDetailed results have been saved to: data/processed/language_analysis_results.txt")
    
    print("\nVisualization files have been created in the data/processed/visualizations directory:")
    print("1. emoji_positioning_analysis.html - Where emojis appear in tweets")
    print("2. emoji_pairs.html - Most common emoji combinations")
    print("3. emoji_word_associations.html - Words commonly used with specific emojis")

if __name__ == "__main__":
    main()
