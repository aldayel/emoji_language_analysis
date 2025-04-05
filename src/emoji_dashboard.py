import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EmojiDashboard:
    def __init__(self, data_path):
        """Initialize the dashboard with data path."""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def prepare_frequency_data(self):
        """Prepare emoji frequency distribution data."""
        emoji_counts = []
        for emojis in self.df['emojis'].apply(eval):
            for emoji in emojis:
                emoji_counts.append(emoji)
        
        freq_df = pd.DataFrame(pd.Series(emoji_counts).value_counts()).reset_index()
        freq_df.columns = ['emoji', 'count']
        return freq_df.head(20)  # Top 20 emojis
        
    def prepare_cluster_data(self):
        """Prepare data for clustering analysis."""
        # Create features for clustering
        emoji_features = {}
        
        for idx, row in self.df.iterrows():
            emojis = eval(row['emojis'])
            sentiment = row['vader_compound']
            
            for emoji in emojis:
                if emoji not in emoji_features:
                    emoji_features[emoji] = {
                        'avg_sentiment': [],
                        'usage_count': 0,
                        'avg_emoji_count': []
                    }
                
                emoji_features[emoji]['avg_sentiment'].append(sentiment)
                emoji_features[emoji]['usage_count'] += 1
                emoji_features[emoji]['avg_emoji_count'].append(len(emojis))
        
        # Calculate averages
        cluster_data = []
        for emoji, features in emoji_features.items():
            if features['usage_count'] >= 10:  # Filter for frequently used emojis
                cluster_data.append({
                    'emoji': emoji,
                    'avg_sentiment': np.mean(features['avg_sentiment']),
                    'usage_count': features['usage_count'],
                    'avg_emoji_count': np.mean(features['avg_emoji_count'])
                })
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # Perform clustering
        X = cluster_df[['avg_sentiment', 'usage_count', 'avg_emoji_count']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        cluster_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        return cluster_df
        
    def prepare_time_series_data(self):
        """Prepare time series data for emoji usage trends."""
        # Create synthetic timestamps for demonstration
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(len(self.df))]
        self.df['timestamp'] = timestamps
        
        # Aggregate daily emoji counts
        daily_counts = {}
        for idx, row in self.df.iterrows():
            date = row['timestamp'].date()
            emojis = eval(row['emojis'])
            
            if date not in daily_counts:
                daily_counts[date] = {}
            
            for emoji in emojis:
                if emoji not in daily_counts[date]:
                    daily_counts[date][emoji] = 0
                daily_counts[date][emoji] += 1
        
        # Convert to DataFrame
        dates = []
        emoji_counts = []
        emoji_types = []
        
        for date, counts in daily_counts.items():
            for emoji, count in counts.items():
                dates.append(date)
                emoji_counts.append(count)
                emoji_types.append(emoji)
        
        time_df = pd.DataFrame({
            'date': dates,
            'emoji': emoji_types,
            'count': emoji_counts
        })
        
        # Get top 5 emojis for time series
        top_emojis = time_df.groupby('emoji')['count'].sum().nlargest(5).index
        return time_df[time_df['emoji'].isin(top_emojis)]
    
    def create_frequency_plot(self):
        """Create emoji frequency distribution plot."""
        freq_df = self.prepare_frequency_data()
        
        fig = px.bar(
            freq_df,
            x='emoji',
            y='count',
            title='Top 20 Most Frequent Emojis',
            labels={'emoji': 'Emoji', 'count': 'Frequency'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_cluster_plot(self):
        """Create clustering visualization."""
        cluster_df = self.prepare_cluster_data()
        
        fig = px.scatter_3d(
            cluster_df,
            x='avg_sentiment',
            y='usage_count',
            z='avg_emoji_count',
            color='cluster',
            hover_data=['emoji'],
            title='Emoji Clusters based on Usage Patterns',
            labels={
                'avg_sentiment': 'Average Sentiment',
                'usage_count': 'Usage Count',
                'avg_emoji_count': 'Average Emoji Count per Tweet'
            }
        )
        
        fig.update_layout(height=600)
        return fig
    
    def create_time_series_plot(self):
        """Create time series plot for emoji usage trends."""
        time_df = self.prepare_time_series_data()
        
        fig = px.line(
            time_df,
            x='date',
            y='count',
            color='emoji',
            title='Daily Emoji Usage Trends (Top 5 Emojis)',
            labels={'date': 'Date', 'count': 'Usage Count', 'emoji': 'Emoji'}
        )
        
        fig.update_layout(height=400)
        return fig
    
    def prepare_sentiment_data(self):
        """Prepare sentiment analysis visualization data."""
        sentiment_data = []
        for idx, row in self.df.iterrows():
            emojis = eval(row['emojis'])
            for emoji in emojis:
                sentiment_data.append({
                    'emoji': emoji,
                    'vader_sentiment': row['vader_compound'],
                    'textblob_sentiment': row['textblob_polarity'],
                    'subjectivity': row['textblob_subjectivity']
                })
        
        return pd.DataFrame(sentiment_data)
    
    def prepare_cooccurrence_data(self):
        """Prepare emoji co-occurrence matrix."""
        from collections import defaultdict
        
        # Count co-occurrences
        cooc_matrix = defaultdict(lambda: defaultdict(int))
        for emojis in self.df['emojis'].apply(eval):
            for i, emoji1 in enumerate(emojis):
                for emoji2 in emojis[i+1:]:
                    cooc_matrix[emoji1][emoji2] += 1
                    cooc_matrix[emoji2][emoji1] += 1
        
        # Convert to DataFrame
        top_emojis = self.prepare_frequency_data()['emoji'].head(15)  # Top 15 emojis
        cooc_df = pd.DataFrame(0, index=top_emojis, columns=top_emojis)
        
        for emoji1 in top_emojis:
            for emoji2 in top_emojis:
                cooc_df.loc[emoji1, emoji2] = cooc_matrix[emoji1][emoji2]
        
        return cooc_df
    
    def create_sentiment_plot(self):
        """Create sentiment distribution plot."""
        sentiment_df = self.prepare_sentiment_data()
        
        # Get top 10 emojis
        top_emojis = sentiment_df['emoji'].value_counts().head(10).index
        filtered_df = sentiment_df[sentiment_df['emoji'].isin(top_emojis)]
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('VADER Sentiment', 'TextBlob Sentiment'))
        
        # VADER boxplot
        fig.add_trace(
            go.Box(x=filtered_df['emoji'],
                   y=filtered_df['vader_sentiment'],
                   name='VADER'),
            row=1, col=1
        )
        
        # TextBlob boxplot
        fig.add_trace(
            go.Box(x=filtered_df['emoji'],
                   y=filtered_df['textblob_sentiment'],
                   name='TextBlob'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Sentiment Distribution by Emoji',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_cooccurrence_plot(self):
        """Create co-occurrence heatmap."""
        cooc_df = self.prepare_cooccurrence_data()
        
        fig = go.Figure(data=go.Heatmap(
            z=cooc_df.values,
            x=cooc_df.columns,
            y=cooc_df.index,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Emoji Co-occurrence Heatmap',
            height=600,
            xaxis_title='Emoji',
            yaxis_title='Emoji'
        )
        
        return fig
    
    def create_subjectivity_plot(self):
        """Create subjectivity analysis plot."""
        sentiment_df = self.prepare_sentiment_data()
        
        # Get top 10 emojis
        top_emojis = sentiment_df['emoji'].value_counts().head(10).index
        filtered_df = sentiment_df[sentiment_df['emoji'].isin(top_emojis)]
        
        fig = go.Figure()
        
        for emoji in top_emojis:
            emoji_data = filtered_df[filtered_df['emoji'] == emoji]
            fig.add_trace(go.Violin(
                x=[emoji] * len(emoji_data),
                y=emoji_data['subjectivity'],
                name=emoji,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title='Subjectivity Distribution by Emoji',
            xaxis_title='Emoji',
            yaxis_title='Subjectivity Score',
            height=500
        )
        
        return fig
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1('Emoji Analysis Dashboard', style={'textAlign': 'center'}),
            
            # Frequency and Time Series Row
            html.Div([
                html.Div([
                    html.H2('Emoji Frequency Distribution'),
                    dcc.Graph(id='frequency-plot')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H2('Emoji Usage Trends'),
                    dcc.Graph(id='time-series-plot')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Clustering and Co-occurrence Row
            html.Div([
                html.Div([
                    html.H2('Emoji Clustering Analysis'),
                    dcc.Graph(id='cluster-plot')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H2('Emoji Co-occurrence Patterns'),
                    dcc.Graph(id='cooccurrence-plot')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Sentiment Analysis Row
            html.Div([
                html.H2('Sentiment Analysis'),
                dcc.Graph(id='sentiment-plot')
            ]),
            
            # Subjectivity Analysis Row
            html.Div([
                html.H2('Subjectivity Analysis'),
                dcc.Graph(id='subjectivity-plot')
            ])
        ])
    
    def setup_callbacks(self):
        """Set up interactive callbacks."""
        @self.app.callback(
            Output('frequency-plot', 'figure'),
            Input('frequency-plot', 'id')
        )
        def update_frequency_plot(_):
            return self.create_frequency_plot()
        
        @self.app.callback(
            Output('cluster-plot', 'figure'),
            Input('cluster-plot', 'id')
        )
        def update_cluster_plot(_):
            return self.create_cluster_plot()
        
        @self.app.callback(
            Output('time-series-plot', 'figure'),
            Input('time-series-plot', 'id')
        )
        def update_time_series_plot(_):
            return self.create_time_series_plot()
        
        @self.app.callback(
            Output('cooccurrence-plot', 'figure'),
            Input('cooccurrence-plot', 'id')
        )
        def update_cooccurrence_plot(_):
            return self.create_cooccurrence_plot()
        
        @self.app.callback(
            Output('sentiment-plot', 'figure'),
            Input('sentiment-plot', 'id')
        )
        def update_sentiment_plot(_):
            return self.create_sentiment_plot()
        
        @self.app.callback(
            Output('subjectivity-plot', 'figure'),
            Input('subjectivity-plot', 'id')
        )
        def update_subjectivity_plot(_):
            return self.create_subjectivity_plot()
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port)
