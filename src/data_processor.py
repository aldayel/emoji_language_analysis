import pandas as pd
import emoji
from typing import Dict, List, Tuple
import re
from pathlib import Path
from tqdm import tqdm

class EmojiDataProcessor:
    """Handles the processing and cleaning of emoji tweet data."""
    
    def __init__(self, data_dir: str):
        """Initialize the processor with path to data directory.
        
        Args:
            data_dir: Path to the directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.df = None
        self.emoji_files = None
    
    def get_emoji_files(self) -> List[Path]:
        """Get list of all CSV files in the data directory."""
        if self.emoji_files is None:
            self.emoji_files = list(self.data_dir.glob('*.csv'))
        return self.emoji_files
    
    def load_data(self, sample_size: int = None) -> pd.DataFrame:
        """Load tweet datasets from all CSV files.
        
        Args:
            sample_size: Optional number of rows to sample from each file
            
        Returns:
            Combined DataFrame with all tweets
        """
        print(f"Loading data from {len(self.get_emoji_files())} files...")
        
        dfs = []
        for file_path in tqdm(self.emoji_files):
            # Extract emoji name from filename
            emoji_name = file_path.stem.replace('_', ' ')
            
            # Read the CSV file with proper encoding and error handling
            try:
                # Try reading with different parameters
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', lineterminator='\n')
                except:
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', lineterminator='\r\n')
                    except:
                        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
                
                # Check if we got any data
                if len(df) == 0:
                    print(f"Warning: No data read from {file_path.name}")
                    continue
                    
            except Exception as e:
                print(f"Error reading {file_path.name}: {str(e)}")
                # Try reading the file manually
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    # Remove header
                    if lines and lines[0].lower() == 'text':
                        lines = lines[1:]
                    # Create DataFrame
                    df = pd.DataFrame({'Text': lines})
                except Exception as e2:
                    print(f"Failed manual reading of {file_path.name}: {str(e2)}")
                    continue
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)))
            
            # Add emoji name column
            df['primary_emoji'] = emoji_name
            dfs.append(df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.df):,} tweets in total")
        return self.df
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract all emojis from text.
        
        Args:
            text: Input text string
            
        Returns:
            List of emojis found in the text
        """
        return [c for c in text if c in emoji.EMOJI_DATA]
    
    def clean_text(self, text: str) -> str:
        """Clean tweet text by removing URLs, mentions, etc.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def process_tweets(self, sample_size: int = None) -> pd.DataFrame:
        """Process the tweet dataset.
        
        Args:
            sample_size: Optional number of rows to sample from each file
            
        Returns:
            Processed DataFrame with additional features
        """
        if self.df is None:
            self.load_data(sample_size)
        
        print("Processing tweets...")
        # Clean text
        self.df['clean_text'] = self.df['Text'].apply(self.clean_text)
        
        # Extract emojis
        self.df['emojis'] = self.df['Text'].apply(self.extract_emojis)
        self.df['emoji_count'] = self.df['emojis'].apply(len)
        
        # Add timestamp if available
        if 'created_at' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['created_at'])
        
        print("Processing complete!")
        return self.df
    
    def get_emoji_stats(self) -> Dict:
        """Calculate basic emoji statistics.
        
        Returns:
            Dictionary containing emoji usage statistics
        """
        if self.df is None:
            self.process_tweets()
            
        stats = {
            'total_tweets': len(self.df),
            'tweets_with_emoji': len(self.df[self.df['emoji_count'] > 0]),
            'unique_emojis': len(set([emoji for sublist in self.df['emojis'] for emoji in sublist])),
            'avg_emojis_per_tweet': self.df['emoji_count'].mean(),
            'total_files': len(self.get_emoji_files())
        }
        return stats
