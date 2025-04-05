import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EmojiVisualizer:
    def __init__(self, data_path):
        """Initialize visualizer with path to processed data."""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.output_dir = self.data_path.parent / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
    def create_emoji_distribution_plot(self):
        """Create an interactive bar plot of emoji distribution."""
        emoji_counts = self.df['primary_emoji'].value_counts()
        
        # Create bar plot
        fig = px.bar(
            x=emoji_counts.index,
            y=emoji_counts.values,
            title='Distribution of Primary Emojis in Dataset',
            labels={'x': 'Emoji Type', 'y': 'Count'},
            height=600
        )
        
        # Update layout
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(b=100)
        )
        
        # Save plot
        fig.write_html(self.output_dir / 'emoji_distribution.html')
        print(f"Saved emoji distribution plot to {self.output_dir / 'emoji_distribution.html'}")
        
    def create_emoji_heatmap(self):
        """Create a heatmap of emoji co-occurrences."""
        # Get all unique emojis from the emojis column
        all_emojis = []
        for emoji_list in self.df['emojis'].apply(eval):  # Convert string representation to list
            all_emojis.extend(emoji_list)
        unique_emojis = pd.Series(all_emojis).value_counts().head(20).index  # Top 20 emojis
        
        # Create co-occurrence matrix
        cooc_matrix = pd.DataFrame(0, index=unique_emojis, columns=unique_emojis)
        
        for emoji_list in self.df['emojis'].apply(eval):
            emoji_list = [e for e in emoji_list if e in unique_emojis]
            for i, emoji1 in enumerate(emoji_list):
                for emoji2 in emoji_list[i:]:
                    cooc_matrix.loc[emoji1, emoji2] += 1
                    if emoji1 != emoji2:
                        cooc_matrix.loc[emoji2, emoji1] += 1
        
        # Create heatmap
        fig = px.imshow(
            cooc_matrix,
            title='Emoji Co-occurrence Heatmap (Top 20 Emojis)',
            aspect='auto',
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=800,
            xaxis_tickangle=-45
        )
        
        # Save plot
        fig.write_html(self.output_dir / 'emoji_heatmap.html')
        print(f"Saved emoji heatmap to {self.output_dir / 'emoji_heatmap.html'}")
        
    def create_emoji_stats_plots(self):
        """Create plots showing emoji usage statistics."""
        # Calculate emoji count distribution
        emoji_counts_per_tweet = self.df['emoji_count']
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution of Emojis per Tweet',
                'Cumulative Distribution of Emoji Counts',
                'Tweet Length vs. Emoji Count',
                'Top 10 Most Common Emoji Combinations'
            )
        )
        
        # 1. Histogram of emoji counts
        fig.add_trace(
            go.Histogram(x=emoji_counts_per_tweet, name='Emoji Count'),
            row=1, col=1
        )
        
        # 2. Cumulative distribution
        sorted_counts = sorted(emoji_counts_per_tweet)
        fig.add_trace(
            go.Scatter(
                x=sorted(emoji_counts_per_tweet.unique()),
                y=[sum(sorted_counts <= i) / len(sorted_counts) for i in sorted(emoji_counts_per_tweet.unique())],
                name='Cumulative Distribution'
            ),
            row=1, col=2
        )
        
        # 3. Tweet length vs emoji count scatter plot
        fig.add_trace(
            go.Scatter(
                x=self.df['emoji_count'],
                y=self.df['clean_text'].str.len(),
                mode='markers',
                opacity=0.5,
                name='Tweet Length'
            ),
            row=2, col=1
        )
        
        # 4. Top emoji combinations
        emoji_combinations = []
        for emoji_list in self.df['emojis'].apply(eval):
            if len(emoji_list) > 1:
                emoji_combinations.append(tuple(sorted(emoji_list)))
        
        top_combinations = pd.Series(emoji_combinations).value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=[' + '.join(combo) for combo in top_combinations.index],
                y=top_combinations.values,
                name='Combination Count'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            showlegend=False,
            title_text='Emoji Usage Statistics'
        )
        
        # Save plot
        fig.write_html(self.output_dir / 'emoji_stats.html')
        print(f"Saved emoji statistics plots to {self.output_dir / 'emoji_stats.html'}")
        
    def create_emoji_position_analysis(self):
        """Analyze and visualize where emojis typically appear in tweets."""
        # Calculate emoji positions
        positions = []
        relative_positions = []
        
        for text in self.df['clean_text']:
            emoji_positions = [i for i, char in enumerate(text) if char in self.df['emojis'].iloc[0]]  # Using first row's emojis as reference
            if emoji_positions:
                positions.extend(emoji_positions)
                text_length = len(text)
                relative_positions.extend([pos / text_length for pos in emoji_positions])
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Absolute Emoji Positions in Tweets',
                'Relative Emoji Positions (0=start, 1=end)'
            )
        )
        
        # Absolute positions histogram
        fig.add_trace(
            go.Histogram(x=positions, name='Absolute Position'),
            row=1, col=1
        )
        
        # Relative positions histogram
        fig.add_trace(
            go.Histogram(x=relative_positions, name='Relative Position'),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text='Emoji Position Analysis in Tweets'
        )
        
        fig.write_html(self.output_dir / 'emoji_positions.html')
        print(f"Saved emoji position analysis to {self.output_dir / 'emoji_positions.html'}")

    def create_emoji_network_graph(self):
        """Create a network graph showing emoji relationships."""
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add edges from co-occurrences
        for emoji_list in self.df['emojis'].apply(eval):
            if len(emoji_list) > 1:
                for i, emoji1 in enumerate(emoji_list):
                    for emoji2 in emoji_list[i+1:]:
                        if G.has_edge(emoji1, emoji2):
                            G[emoji1][emoji2]['weight'] += 1
                        else:
                            G.add_edge(emoji1, emoji2, weight=1)
        
        # Get top 30 edges by weight
        edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        edges.sort(key=lambda x: x[2], reverse=True)
        top_edges = edges[:30]
        
        # Create subgraph with only these edges
        nodes = set()
        for u, v, _ in top_edges:
            nodes.add(u)
            nodes.add(v)
        
        # Create the plot
        edge_x = []
        edge_y = []
        edge_weights = []
        
        pos = nx.spring_layout(G.subgraph(nodes))
        
        for u, v, weight in top_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(weight)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        node_x = [pos[node][0] for node in nodes]
        node_y = [pos[node][1] for node in nodes]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=30),
            text=list(nodes),
            textposition="middle center"
        ))
        
        fig.update_layout(
            title='Emoji Relationship Network (Top 30 Connections)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            height=800
        )
        
        fig.write_html(self.output_dir / 'emoji_network.html')
        print(f"Saved emoji network graph to {self.output_dir / 'emoji_network.html'}")

    def create_emoji_word_context(self):
        """Analyze and visualize words commonly appearing with emojis."""
        from collections import Counter
        import re
        
        # Get words before and after emojis
        before_words = []
        after_words = []
        
        for text in self.df['clean_text']:
            words = re.findall(r'\w+', text.lower())
            for i, word in enumerate(words):
                if i < len(words) - 1 and any(emoji in words[i+1] for emoji in self.df['emojis'].iloc[0]):
                    before_words.append(word)
                if i > 0 and any(emoji in words[i-1] for emoji in self.df['emojis'].iloc[0]):
                    after_words.append(word)
        
        # Get top words
        before_counts = Counter(before_words).most_common(20)
        after_counts = Counter(after_words).most_common(20)
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Top Words Before Emojis',
                'Top Words After Emojis'
            )
        )
        
        # Add before words
        fig.add_trace(
            go.Bar(
                x=[word for word, _ in before_counts],
                y=[count for _, count in before_counts],
                name='Words Before'
            ),
            row=1, col=1
        )
        
        # Add after words
        fig.add_trace(
            go.Bar(
                x=[word for word, _ in after_counts],
                y=[count for _, count in after_counts],
                name='Words After'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text='Word Context Analysis Around Emojis',
            showlegend=False
        )
        
        fig.write_html(self.output_dir / 'emoji_word_context.html')
        print(f"Saved emoji word context analysis to {self.output_dir / 'emoji_word_context.html'}")

    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("Generating visualizations...")
        self.create_emoji_distribution_plot()
        self.create_emoji_heatmap()
        self.create_emoji_stats_plots()
        self.create_emoji_position_analysis()
        self.create_emoji_network_graph()
        self.create_emoji_word_context()
        print("All visualizations generated!")
