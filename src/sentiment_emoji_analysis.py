import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentEmojiAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with path to processed data."""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.output_dir = self.data_path.parent / 'visualizations' / 'sentiment'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_text_sentiment(self):
        """Analyze sentiment using both VADER and TextBlob."""
        print("Analyzing text sentiment...")
        
        # VADER sentiment
        self.df['vader_scores'] = self.df['clean_text'].apply(self.vader.polarity_scores)
        self.df['vader_compound'] = self.df['vader_scores'].apply(lambda x: x['compound'])
        self.df['vader_sentiment'] = self.df['vader_compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        # TextBlob sentiment
        self.df['textblob_polarity'] = self.df['clean_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        self.df['textblob_subjectivity'] = self.df['clean_text'].apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        self.df['textblob_sentiment'] = self.df['textblob_polarity'].apply(
            lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
        )
        
        return self.df[['vader_sentiment', 'textblob_sentiment', 'vader_compound', 
                       'textblob_polarity', 'textblob_subjectivity']]
    
    def analyze_emoji_sentiment_correlation(self):
        """Analyze correlation between emoji usage and sentiment."""
        emoji_sentiment = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0})
        emoji_subjectivity = defaultdict(list)
        
        for _, row in self.df.iterrows():
            emojis = eval(row['emojis'])
            for emoji in emojis:
                emoji_sentiment[emoji][row['vader_sentiment']] += 1
                emoji_sentiment[emoji]['total'] += 1
                emoji_subjectivity[emoji].append(row['textblob_subjectivity'])
        
        # Calculate sentiment ratios and average subjectivity
        emoji_stats = {}
        for emoji, stats in emoji_sentiment.items():
            if stats['total'] >= 10:  # Filter for emojis with sufficient data
                total = stats['total']
                emoji_stats[emoji] = {
                    'positive_ratio': stats['positive'] / total,
                    'neutral_ratio': stats['neutral'] / total,
                    'negative_ratio': stats['negative'] / total,
                    'avg_subjectivity': np.mean(emoji_subjectivity[emoji]),
                    'total_occurrences': total
                }
        
        return emoji_stats
    
    def analyze_sentiment_by_emoji_count(self):
        """Analyze how sentiment varies with number of emojis used."""
        sentiment_by_count = defaultdict(list)
        
        for _, row in self.df.iterrows():
            count = len(eval(row['emojis']))
            sentiment_by_count[count].append({
                'vader': row['vader_compound'],
                'textblob': row['textblob_polarity'],
                'subjectivity': row['textblob_subjectivity']
            })
        
        # Calculate averages
        count_stats = {}
        for count, sentiments in sentiment_by_count.items():
            if len(sentiments) >= 5:  # Filter for counts with sufficient data
                count_stats[count] = {
                    'avg_vader': np.mean([s['vader'] for s in sentiments]),
                    'avg_textblob': np.mean([s['textblob'] for s in sentiments]),
                    'avg_subjectivity': np.mean([s['subjectivity'] for s in sentiments]),
                    'sample_size': len(sentiments)
                }
        
        return count_stats
    
    def create_visualizations(self):
        """Create visualizations for sentiment analysis."""
        print("Creating sentiment visualizations...")
        
        # 1. Overall Sentiment Distribution
        fig1 = make_subplots(rows=1, cols=2, subplot_titles=('VADER Sentiment', 'TextBlob Sentiment'))
        
        vader_dist = self.df['vader_sentiment'].value_counts()
        textblob_dist = self.df['textblob_sentiment'].value_counts()
        
        fig1.add_trace(
            go.Bar(x=vader_dist.index, y=vader_dist.values, name='VADER'),
            row=1, col=1
        )
        fig1.add_trace(
            go.Bar(x=textblob_dist.index, y=textblob_dist.values, name='TextBlob'),
            row=1, col=2
        )
        
        fig1.update_layout(title='Overall Sentiment Distribution', showlegend=False)
        fig1.write_html(self.output_dir / 'sentiment_distribution.html')
        
        # 2. Emoji Count vs Sentiment
        count_stats = self.analyze_sentiment_by_emoji_count()
        counts = sorted(count_stats.keys())
        
        fig2 = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Sentiment Score by Emoji Count',
                                         'Subjectivity by Emoji Count'))
        
        fig2.add_trace(
            go.Scatter(x=counts, 
                      y=[count_stats[c]['avg_vader'] for c in counts],
                      name='VADER',
                      mode='lines+markers'),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=counts,
                      y=[count_stats[c]['avg_textblob'] for c in counts],
                      name='TextBlob',
                      mode='lines+markers'),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=counts,
                      y=[count_stats[c]['avg_subjectivity'] for c in counts],
                      name='Subjectivity',
                      mode='lines+markers'),
            row=2, col=1
        )
        
        fig2.update_layout(title='Sentiment Analysis by Emoji Count',
                          height=800)
        fig2.write_html(self.output_dir / 'sentiment_by_count.html')
        
        # 3. Top Emojis by Sentiment
        emoji_stats = self.analyze_emoji_sentiment_correlation()
        top_emojis = sorted(emoji_stats.items(), 
                          key=lambda x: x[1]['total_occurrences'],
                          reverse=True)[:15]
        
        fig3 = go.Figure()
        
        x = [e[0] for e in top_emojis]
        pos_y = [e[1]['positive_ratio'] for e in top_emojis]
        neu_y = [e[1]['neutral_ratio'] for e in top_emojis]
        neg_y = [e[1]['negative_ratio'] for e in top_emojis]
        
        fig3.add_trace(go.Bar(name='Positive', x=x, y=pos_y))
        fig3.add_trace(go.Bar(name='Neutral', x=x, y=neu_y))
        fig3.add_trace(go.Bar(name='Negative', x=x, y=neg_y))
        
        fig3.update_layout(
            title='Sentiment Distribution for Top 15 Emojis',
            barmode='stack',
            height=600
        )
        
        fig3.write_html(self.output_dir / 'emoji_sentiment_distribution.html')
        
        return {
            'sentiment_dist': {
                'vader': vader_dist.to_dict(),
                'textblob': textblob_dist.to_dict()
            },
            'emoji_stats': emoji_stats,
            'count_stats': count_stats
        }
    
    def run_analysis(self):
        """Run complete sentiment analysis."""
        self.analyze_text_sentiment()
        results = self.create_visualizations()
        return results
