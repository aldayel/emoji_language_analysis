import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

class EmojiSentimentAnalyzer:
    def __init__(self, data_path):
        """Initialize the sentiment analyzer with path to processed data."""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.output_dir = self.data_path.parent / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_text_sentiment(self):
        """Analyze sentiment of text using both TextBlob and VADER."""
        # TextBlob analysis
        self.df['textblob_polarity'] = self.df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        self.df['textblob_subjectivity'] = self.df['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        # VADER analysis
        self.df['vader_compound'] = self.df['clean_text'].apply(lambda x: self.vader.polarity_scores(x)['compound'])
        
        # Categorize sentiment
        self.df['sentiment_category'] = self.df['vader_compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
    def analyze_emoji_sentiment(self):
        """Analyze sentiment patterns for each emoji."""
        emoji_sentiment = {}
        
        for _, row in self.df.iterrows():
            emojis = eval(row['emojis'])
            for emoji in emojis:
                if emoji not in emoji_sentiment:
                    emoji_sentiment[emoji] = {
                        'positive': 0,
                        'neutral': 0,
                        'negative': 0,
                        'total': 0,
                        'avg_compound': 0.0
                    }
                
                emoji_sentiment[emoji][row['sentiment_category']] += 1
                emoji_sentiment[emoji]['total'] += 1
                emoji_sentiment[emoji]['avg_compound'] += row['vader_compound']
        
        # Calculate averages
        for emoji in emoji_sentiment:
            emoji_sentiment[emoji]['avg_compound'] /= emoji_sentiment[emoji]['total']
        
        return emoji_sentiment
    
    def create_sentiment_visualizations(self):
        """Create visualizations for sentiment analysis."""
        print("Analyzing sentiment...")
        self.analyze_text_sentiment()
        emoji_sentiment = self.analyze_emoji_sentiment()
        
        # 1. Overall Sentiment Distribution
        fig1 = go.Figure()
        sentiment_counts = self.df['sentiment_category'].value_counts()
        
        fig1.add_trace(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            title='Overall Sentiment Distribution'
        ))
        
        fig1.update_layout(height=600)
        fig1.write_html(self.output_dir / 'sentiment_distribution.html')
        
        # 2. Top Emojis by Sentiment Category
        fig2 = make_subplots(rows=1, cols=3, subplot_titles=('Positive', 'Neutral', 'Negative'))
        
        for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
            # Sort emojis by their frequency in this sentiment category
            sorted_emojis = sorted(
                emoji_sentiment.items(),
                key=lambda x: x[1][sentiment] / x[1]['total'],
                reverse=True
            )[:10]
            
            fig2.add_trace(
                go.Bar(
                    x=[emoji for emoji, _ in sorted_emojis],
                    y=[data[sentiment] / data['total'] * 100 for _, data in sorted_emojis],
                    name=sentiment.capitalize()
                ),
                row=1, col=i+1
            )
        
        fig2.update_layout(
            height=500,
            title_text='Top Emojis by Sentiment Category (% of emoji\'s occurrences)',
            showlegend=False
        )
        
        fig2.write_html(self.output_dir / 'emoji_sentiment_distribution.html')
        
        # 3. Emoji Sentiment Correlation
        fig3 = go.Figure()
        
        # Sort emojis by average compound score
        sorted_by_compound = sorted(
            emoji_sentiment.items(),
            key=lambda x: x[1]['avg_compound'],
            reverse=True
        )
        
        fig3.add_trace(go.Bar(
            x=[emoji for emoji, _ in sorted_by_compound],
            y=[data['avg_compound'] for _, data in sorted_by_compound],
            name='Average Sentiment'
        ))
        
        fig3.update_layout(
            title='Emojis Ranked by Average Sentiment Score',
            xaxis_title='Emoji',
            yaxis_title='Average Sentiment Score',
            height=600
        )
        
        fig3.write_html(self.output_dir / 'emoji_sentiment_ranking.html')
        
        # 4. Sentiment vs. Subjectivity
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(
            x=self.df['textblob_subjectivity'],
            y=self.df['textblob_polarity'],
            mode='markers',
            marker=dict(
                color=self.df['vader_compound'],
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title='VADER Sentiment')
            ),
            text=self.df['primary_emoji'],
            name='Tweets'
        ))
        
        fig4.update_layout(
            title='Sentiment vs. Subjectivity (colored by VADER sentiment)',
            xaxis_title='Subjectivity',
            yaxis_title='Polarity',
            height=600
        )
        
        fig4.write_html(self.output_dir / 'sentiment_subjectivity.html')
        
        print("Sentiment analysis visualizations created successfully!")
        
        # Save processed data with sentiment
        output_path = self.data_path.parent / 'tweets_with_sentiment.csv'
        self.df.to_csv(output_path, index=False)
        print(f"Processed data with sentiment saved to: {output_path}")
        
        return {
            'total_tweets': len(self.df),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'most_positive_emoji': sorted_by_compound[0][0],
            'most_negative_emoji': sorted_by_compound[-1][0]
        }
