from pathlib import Path
from emoji_dashboard import EmojiDashboard

def main():
    # Get path to processed data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'processed_tweets.csv'
    
    if not data_path.exists():
        print(f"Error: Processed data file not found at {data_path}")
        print("Please run analysis.py first to generate the processed data.")
        return
    
    print("Starting Emoji Analysis Dashboard...")
    dashboard = EmojiDashboard(data_path)
    
    print("\nDashboard is running at http://localhost:8050")
    print("Press Ctrl+C to stop the server.")
    
    dashboard.run_server(debug=True, port=8050)

if __name__ == "__main__":
    main()
