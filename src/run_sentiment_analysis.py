from pathlib import Path
from sentiment_emoji_analysis import SentimentEmojiAnalyzer

def format_percentage(value, total):
    return f"{value:,} ({(value/total)*100:.1f}%)"

def main():
    # Get path to processed data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'processed_tweets.csv'
    
    if not data_path.exists():
        print(f"Error: Processed data file not found at {data_path}")
        print("Please run analysis.py first to generate the processed data.")
        return
    
    print("Starting Sentiment Analysis...")
    analyzer = SentimentEmojiAnalyzer(data_path)
    results = analyzer.run_analysis()
    
    # Save detailed results to file
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    with open(output_dir / 'sentiment_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("Sentiment Analysis Summary\n")
        f.write("=" * 40 + "\n")
        
        # Overall sentiment distribution
        vader_dist = results['sentiment_dist']['vader']
        textblob_dist = results['sentiment_dist']['textblob']
        total_tweets = sum(vader_dist.values())
        
        f.write("\n1. Overall Sentiment Distribution:\n")
        f.write("\nVADER Sentiment:\n")
        for sentiment, count in vader_dist.items():
            f.write(f"- {sentiment.title()}: {format_percentage(count, total_tweets)}\n")
        
        f.write("\nTextBlob Sentiment:\n")
        for sentiment, count in textblob_dist.items():
            f.write(f"- {sentiment.title()}: {format_percentage(count, total_tweets)}\n")
        
        # Emoji sentiment patterns
        emoji_stats = results['emoji_stats']
        top_positive = sorted(emoji_stats.items(), 
                             key=lambda x: x[1]['positive_ratio'],
                             reverse=True)[:5]
        top_negative = sorted(emoji_stats.items(),
                             key=lambda x: x[1]['negative_ratio'],
                             reverse=True)[:5]
        
        f.write("\n2. Emoji Sentiment Patterns:\n")
        f.write("\nMost Positive Emojis:\n")
        for emoji, stats in top_positive:
            pos_pct = stats['positive_ratio'] * 100
            f.write(f"- {emoji}: {pos_pct:.1f}% positive sentiment\n")
        
        f.write("\nMost Negative Emojis:\n")
        for emoji, stats in top_negative:
            neg_pct = stats['negative_ratio'] * 100
            f.write(f"- {emoji}: {neg_pct:.1f}% negative sentiment\n")
        
        # Emoji count patterns
        count_stats = results['count_stats']
        max_sentiment = max(count_stats.items(), 
                           key=lambda x: x[1]['avg_vader'])
        
        f.write("\n3. Emoji Count Patterns:\n")
        f.write(f"- Tweets with {max_sentiment[0]} emojis show the highest average sentiment\n")
        f.write(f"  (VADER score: {max_sentiment[1]['avg_vader']:.3f}, ")
        f.write(f"   TextBlob score: {max_sentiment[1]['avg_textblob']:.3f})\n")
    
    # Print basic stats to console (avoiding emoji display)
    print("\nBasic Sentiment Statistics:")
    print("\nVADER Sentiment Distribution:")
    for sentiment, count in vader_dist.items():
        print(f"- {sentiment.title()}: {format_percentage(count, total_tweets)}")
    
    print("\nTextBlob Sentiment Distribution:")
    for sentiment, count in textblob_dist.items():
        print(f"- {sentiment.title()}: {format_percentage(count, total_tweets)}")
    
    print(f"\nTweets with {max_sentiment[0]} emojis show the highest average sentiment")
    print(f"(VADER score: {max_sentiment[1]['avg_vader']:.3f}, ")
    print(f" TextBlob score: {max_sentiment[1]['avg_textblob']:.3f})")
    
    print("\nDetailed results have been saved to: data/processed/sentiment_analysis_results.txt")
    print("\nVisualization files have been created in data/processed/visualizations/sentiment/:")
    print("1. sentiment_distribution.html - Overall distribution of sentiment")
    print("2. sentiment_by_count.html - How sentiment varies with emoji count")
    print("3. emoji_sentiment_distribution.html - Sentiment patterns for top emojis")

if __name__ == "__main__":
    main()
