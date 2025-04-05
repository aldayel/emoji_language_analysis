import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

class EmojiTemporalAnalyzer:
    def __init__(self, data_path):
        """Initialize the temporal analyzer with path to processed data."""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.output_dir = self.data_path.parent / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        # Since we don't have real timestamps, we'll simulate them for demonstration
        self.simulate_temporal_data()
        
    def simulate_temporal_data(self):
        """
        Simulate temporal data for demonstration purposes.
        In a real scenario, we would use actual timestamps from the tweets.
        """
        # Generate random timestamps within the last 30 days
        now = datetime.now()
        timestamps = []
        
        for _ in range(len(self.df)):
            # Random number of days ago (0-30)
            days_ago = random.uniform(0, 30)
            # Random hour of the day (0-23)
            hour = random.randint(0, 23)
            # Random minute (0-59)
            minute = random.randint(0, 59)
            
            timestamp = now - timedelta(days=days_ago, 
                                     hours=random.randint(0, 23),
                                     minutes=random.randint(0, 59))
            timestamps.append(timestamp)
        
        self.df['timestamp'] = timestamps
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        self.df['date'] = self.df['timestamp'].dt.date
        
    def analyze_hourly_patterns(self):
        """Analyze emoji usage patterns by hour of the day."""
        hourly_stats = self.df.groupby('hour').agg({
            'emoji_count': ['mean', 'count'],
            'primary_emoji': lambda x: x.value_counts().index[0]
        }).reset_index()
        
        return hourly_stats
    
    def analyze_daily_patterns(self):
        """Analyze emoji usage patterns by day of the week."""
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        
        daily_stats = self.df.groupby('day_of_week').agg({
            'emoji_count': ['mean', 'count'],
            'primary_emoji': lambda x: x.value_counts().index[0]
        }).reset_index()
        
        # Ensure days are in correct order
        daily_stats['day_of_week'] = pd.Categorical(
            daily_stats['day_of_week'], 
            categories=day_order, 
            ordered=True
        )
        daily_stats = daily_stats.sort_values('day_of_week')
        
        return daily_stats
    
    def analyze_emoji_trends(self):
        """Analyze trending patterns in emoji usage over time."""
        # Group by date and emoji to get daily counts
        daily_emoji_counts = self.df.groupby(['date', 'primary_emoji']).size().reset_index(name='count')
        
        # Get top 10 emojis overall
        top_emojis = self.df['primary_emoji'].value_counts().head(10).index
        
        # Filter for top emojis
        trending_data = daily_emoji_counts[daily_emoji_counts['primary_emoji'].isin(top_emojis)]
        
        return trending_data
    
    def create_temporal_visualizations(self):
        """Create visualizations for temporal analysis."""
        print("Analyzing temporal patterns...")
        
        # 1. Hourly Patterns
        hourly_stats = self.analyze_hourly_patterns()
        
        fig1 = make_subplots(rows=2, cols=1,
                           subplot_titles=('Average Emojis per Tweet by Hour',
                                         'Tweet Volume by Hour'))
        
        fig1.add_trace(
            go.Scatter(x=hourly_stats['hour'], 
                      y=hourly_stats['emoji_count']['mean'],
                      mode='lines+markers',
                      name='Avg Emojis'),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Bar(x=hourly_stats['hour'],
                  y=hourly_stats['emoji_count']['count'],
                  name='Tweet Count'),
            row=2, col=1
        )
        
        fig1.update_layout(
            height=800,
            title_text='Hourly Emoji Usage Patterns',
            showlegend=True
        )
        
        fig1.write_html(self.output_dir / 'hourly_patterns.html')
        
        # 2. Daily Patterns
        daily_stats = self.analyze_daily_patterns()
        
        fig2 = make_subplots(rows=2, cols=1,
                           subplot_titles=('Average Emojis per Tweet by Day',
                                         'Tweet Volume by Day'))
        
        fig2.add_trace(
            go.Scatter(x=daily_stats['day_of_week'], 
                      y=daily_stats['emoji_count']['mean'],
                      mode='lines+markers',
                      name='Avg Emojis'),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Bar(x=daily_stats['day_of_week'],
                  y=daily_stats['emoji_count']['count'],
                  name='Tweet Count'),
            row=2, col=1
        )
        
        fig2.update_layout(
            height=800,
            title_text='Daily Emoji Usage Patterns',
            showlegend=True
        )
        
        fig2.write_html(self.output_dir / 'daily_patterns.html')
        
        # 3. Trending Analysis
        trending_data = self.analyze_emoji_trends()
        
        fig3 = go.Figure()
        
        for emoji in trending_data['primary_emoji'].unique():
            emoji_data = trending_data[trending_data['primary_emoji'] == emoji]
            fig3.add_trace(
                go.Scatter(x=emoji_data['date'],
                          y=emoji_data['count'],
                          name=emoji,
                          mode='lines',
                          text=[emoji] * len(emoji_data))
            )
        
        fig3.update_layout(
            title='Emoji Usage Trends Over Time (Top 10 Emojis)',
            xaxis_title='Date',
            yaxis_title='Usage Count',
            height=600,
            showlegend=True
        )
        
        fig3.write_html(self.output_dir / 'emoji_trends.html')
        
        # Return summary statistics
        return {
            'hourly_stats': hourly_stats,
            'daily_stats': daily_stats,
            'peak_hour': hourly_stats.loc[hourly_stats['emoji_count']['count'].idxmax(), 'hour'],
            'peak_day': daily_stats.loc[daily_stats['emoji_count']['count'].idxmax(), 'day_of_week'],
            'total_days': len(trending_data['date'].unique())
        }
