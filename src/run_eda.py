from pathlib import Path
from emoji_eda import EmojiEDA

def format_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"

def main():
    # Get path to processed data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'processed_tweets.csv'
    
    if not data_path.exists():
        print(f"Error: Processed data file not found at {data_path}")
        print("Please run analysis.py first to generate the processed data.")
        return
    
    print("Starting Exploratory Data Analysis...")
    analyzer = EmojiEDA(data_path)
    results = analyzer.run_complete_eda()
    
    # Save results to file
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    with open(output_dir / 'eda_results.txt', 'w', encoding='utf-8') as f:
        # Basic Statistics
        stats = results['descriptive_stats']
        f.write("Emoji Usage Analysis Summary\n")
        f.write("=" * 40 + "\n")
        
        f.write("\n1. Basic Statistics:\n")
        f.write(f"Total Tweets Analyzed: {format_number(stats['total_tweets'])}\n")
        f.write(f"Total Emojis Used: {format_number(stats['total_emojis'])}\n")
        f.write(f"Unique Emojis Found: {format_number(stats['unique_emojis'])}\n")
        f.write(f"Average Emojis per Tweet: {stats['avg_emojis_per_tweet']:.2f}\n")
        f.write(f"Median Emojis per Tweet: {stats['median_emojis_per_tweet']:.1f}\n")
        f.write(f"Maximum Emojis in a Single Tweet: {stats['max_emojis_in_tweet']}\n")
        
        # Top Emojis
        f.write("\n2. Most Popular Emojis:\n")
        for emoji, count in stats['top_emojis']:
            f.write(f"   {emoji}: {format_number(count)} uses\n")
        
        # Emoji Distribution
        f.write("\n3. Emoji Count Distribution:\n")
        for count, freq in sorted(stats['emoji_distribution'].items())[:5]:
            f.write(f"   {count} emoji{'s' if count != 1 else ''}: {format_number(freq)} tweets\n")
        if len(stats['emoji_distribution']) > 5:
            f.write("   ...\n")
        
        # Co-occurrence Patterns
        f.write("\n4. Top Emoji Pairs:\n")
        for (emoji1, emoji2), count in results['network_analysis']['top_pairs']:
            f.write(f"   {emoji1} + {emoji2}: {format_number(count)} occurrences\n")
        
        # Context Analysis
        f.write("\n5. Word Association Highlights:\n")
        word_freqs = results['context_analysis']['word_frequencies']
        for emoji, words in list(word_freqs.items())[:5]:
            f.write(f"\n   {emoji} commonly appears with:\n")
            for word, freq in words[:5]:
                f.write(f"      - {word} ({freq} times)\n")
    
    # Print basic stats to console (avoiding emoji display)
    print("\nBasic Statistics:")
    print(f"Total Tweets Analyzed: {format_number(stats['total_tweets'])}")
    print(f"Total Emojis Used: {format_number(stats['total_emojis'])}")
    print(f"Unique Emojis Found: {format_number(stats['unique_emojis'])}")
    print(f"Average Emojis per Tweet: {stats['avg_emojis_per_tweet']:.2f}")
    print(f"Median Emojis per Tweet: {stats['median_emojis_per_tweet']:.1f}")
    print(f"Maximum Emojis in a Single Tweet: {stats['max_emojis_in_tweet']}")
    
    print("\nDetailed results have been saved to: data/processed/eda_results.txt")
    print("\nVisualization files have been created in data/processed/visualizations/eda/:")
    print("1. emoji_count_distribution.html - Distribution of emoji usage")
    print("2. top_emojis.html - Bar chart of most frequent emojis")
    print("3. emoji_network.html - Network graph of emoji co-occurrences")
    print("4. emoji_heatmap.html - Heatmap of emoji co-occurrences")
    print("5. wordcloud_*.png - Word clouds for top emojis")

if __name__ == "__main__":
    main()
