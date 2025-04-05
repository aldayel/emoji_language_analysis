from pathlib import Path
from visualizations import EmojiVisualizer

def main():
    # Get path to processed data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'processed_tweets.csv'
    
    if not data_path.exists():
        print(f"Error: Processed data file not found at {data_path}")
        print("Please run analysis.py first to generate the processed data.")
        return
    
    # Create visualizer and generate all plots
    visualizer = EmojiVisualizer(data_path)
    visualizer.generate_all_visualizations()
    
    print("\nVisualization files have been created in the data/processed/visualizations directory.")
    print("Open the HTML files in a web browser to interact with the plots.")

if __name__ == "__main__":
    main()
