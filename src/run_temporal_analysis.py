from pathlib import Path
from temporal_analysis import EmojiTemporalAnalyzer

def main():
    # Get path to processed data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'processed_tweets.csv'
    
    if not data_path.exists():
        print(f"Error: Processed data file not found at {data_path}")
        print("Please run analysis.py first to generate the processed data.")
        return
    
    print("Starting temporal analysis...")
    analyzer = EmojiTemporalAnalyzer(data_path)
    results = analyzer.create_temporal_visualizations()
    
    # Print summary
    print("\nTemporal Analysis Summary:")
    print("-" * 40)
    
    print(f"\nAnalyzed emoji usage patterns over {results['total_days']} days")
    
    print("\nPeak Usage Times:")
    print(f"- Busiest hour: {int(results['peak_hour']):02d}:00")
    print(f"- Busiest day: {results['peak_day']}")
    
    # Get hourly patterns
    hourly_stats = results['hourly_stats']
    quiet_hour = int(hourly_stats.loc[hourly_stats['emoji_count']['count'].idxmin(), 'hour'])
    print(f"\nQuietest hour: {quiet_hour:02d}:00")
    
    # Get daily patterns
    daily_stats = results['daily_stats']
    weekday_avg = daily_stats[daily_stats['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
    weekend_avg = daily_stats[daily_stats['day_of_week'].isin(['Saturday', 'Sunday'])]
    
    weekday_emoji_avg = weekday_avg['emoji_count']['mean'].mean()
    weekend_emoji_avg = weekend_avg['emoji_count']['mean'].mean()
    
    print("\nEmoji Usage Patterns:")
    print(f"- Weekday average: {weekday_emoji_avg:.2f} emojis per tweet")
    print(f"- Weekend average: {weekend_emoji_avg:.2f} emojis per tweet")
    
    print("\nVisualization files have been created in the data/processed/visualizations directory:")
    print("1. hourly_patterns.html - Emoji usage patterns throughout the day")
    print("2. daily_patterns.html - Emoji usage patterns across the week")
    print("3. emoji_trends.html - Trending patterns for top emojis")
    
    print("\nNote: Since timestamp data was not available in the original dataset,")
    print("temporal patterns were simulated for demonstration purposes.")
    print("For actual temporal analysis, please ensure the dataset includes timestamp information.")

if __name__ == "__main__":
    main()
