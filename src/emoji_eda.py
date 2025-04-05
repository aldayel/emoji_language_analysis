import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter, defaultdict
import re
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class EmojiEDA:
    def __init__(self, data_path):
        """Initialize EDA with path to processed data."""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.output_dir = self.data_path.parent / 'visualizations' / 'eda'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def compute_descriptive_stats(self):
        """Compute basic descriptive statistics."""
        stats = {
            'total_tweets': len(self.df),
            'total_emojis': self.df['emoji_count'].sum(),
            'avg_emojis_per_tweet': self.df['emoji_count'].mean(),
            'median_emojis_per_tweet': self.df['emoji_count'].median(),
            'max_emojis_in_tweet': self.df['emoji_count'].max(),
            'unique_emojis': len(set([emoji for emojis in self.df['emojis'].apply(eval) for emoji in emojis])),
            'emoji_distribution': self.df['emoji_count'].value_counts().to_dict()
        }
        
        # Get top emojis
        all_emojis = []
        for emojis in self.df['emojis'].apply(eval):
            all_emojis.extend(emojis)
        stats['top_emojis'] = Counter(all_emojis).most_common(10)
        
        return stats
    
    def create_frequency_visualizations(self):
        """Create visualizations for emoji frequency analysis."""
        # 1. Emoji Count Distribution
        fig1 = px.histogram(
            self.df,
            x='emoji_count',
            nbins=20,
            title='Distribution of Emoji Count per Tweet',
            labels={'emoji_count': 'Number of Emojis', 'count': 'Frequency'}
        )
        fig1.write_html(self.output_dir / 'emoji_count_distribution.html')
        
        # 2. Top Emojis Bar Chart
        all_emojis = []
        for emojis in self.df['emojis'].apply(eval):
            all_emojis.extend(emojis)
        emoji_counts = Counter(all_emojis).most_common(15)
        
        fig2 = px.bar(
            x=[e[0] for e in emoji_counts],
            y=[e[1] for e in emoji_counts],
            title='Top 15 Most Frequent Emojis',
            labels={'x': 'Emoji', 'y': 'Frequency'}
        )
        fig2.write_html(self.output_dir / 'top_emojis.html')
        
        return {'emoji_counts': emoji_counts}
    
    def create_cooccurrence_network(self):
        """Create network visualization of emoji co-occurrences."""
        # Build co-occurrence matrix
        emoji_pairs = []
        for emojis in self.df['emojis'].apply(eval):
            if len(emojis) >= 2:
                for i in range(len(emojis)-1):
                    for j in range(i+1, len(emojis)):
                        emoji_pairs.append(tuple(sorted([emojis[i], emojis[j]])))
        
        pair_counts = Counter(emoji_pairs)
        
        # Create network
        G = nx.Graph()
        
        # Add edges with weights
        for (emoji1, emoji2), weight in pair_counts.most_common(50):  # Top 50 pairs
            G.add_edge(emoji1, emoji2, weight=weight)
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G)
        
        # Create network visualization
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        )
        
        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="top center",
                marker=dict(
                    size=20,
                    line_width=2
                )
            )
        )
        
        fig.update_layout(
            title='Emoji Co-occurrence Network (Top 50 Pairs)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            height=800
        )
        
        fig.write_html(self.output_dir / 'emoji_network.html')
        
        return {'top_pairs': pair_counts.most_common(10)}
    
    def analyze_context(self):
        """Analyze the textual context around emojis."""
        context_data = defaultdict(list)
        word_contexts = defaultdict(list)
        
        for _, row in self.df.iterrows():
            text = row['clean_text'].lower()
            emojis = eval(row['emojis'])
            
            # Get words around each emoji
            for emoji in emojis:
                try:
                    idx = text.index(emoji)
                    before_text = text[:idx].strip()
                    after_text = text[idx+len(emoji):].strip()
                    
                    # Get words before and after
                    before_words = re.findall(r'\w+', before_text)[-3:]
                    after_words = re.findall(r'\w+', after_text)[:3]
                    
                    context_data[emoji].append({
                        'before': ' '.join(before_words),
                        'after': ' '.join(after_words)
                    })
                    
                    # Collect words for word cloud
                    word_contexts[emoji].extend(before_words + after_words)
                except ValueError:
                    continue
        
        # Create word clouds for top emojis
        for emoji, words in list(word_contexts.items())[:10]:  # Top 10 emojis
            if words:
                word_freq = Counter(words)
                
                # Create and save word cloud
                wc = WordCloud(width=800, height=400, background_color='white')
                wc.generate_from_frequencies(word_freq)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Words Associated with {emoji}')
                plt.savefig(self.output_dir / f'wordcloud_{ord(emoji)}.png')
                plt.close()
        
        return {
            'context_samples': {k: v[:5] for k, v in context_data.items()},  # First 5 samples for each emoji
            'word_frequencies': {
                emoji: Counter(words).most_common(10) 
                for emoji, words in word_contexts.items()
            }
        }
    
    def create_heatmap(self):
        """Create heatmap of emoji co-occurrences."""
        # Get top 15 emojis
        all_emojis = []
        for emojis in self.df['emojis'].apply(eval):
            all_emojis.extend(emojis)
        top_emojis = [e for e, _ in Counter(all_emojis).most_common(15)]
        
        # Create co-occurrence matrix
        matrix = np.zeros((len(top_emojis), len(top_emojis)))
        
        for emojis in self.df['emojis'].apply(eval):
            for i, emoji1 in enumerate(top_emojis):
                if emoji1 in emojis:
                    for j, emoji2 in enumerate(top_emojis):
                        if emoji2 in emojis and emoji1 != emoji2:
                            matrix[i][j] += 1
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=top_emojis,
            y=top_emojis,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Emoji Co-occurrence Heatmap',
            xaxis_title='Emoji',
            yaxis_title='Emoji',
            height=800,
            width=800
        )
        
        fig.write_html(self.output_dir / 'emoji_heatmap.html')
        
        return {'matrix': matrix.tolist(), 'emojis': top_emojis}
    
    def run_complete_eda(self):
        """Run all EDA analyses and return comprehensive results."""
        print("Running comprehensive EDA...")
        
        results = {
            'descriptive_stats': self.compute_descriptive_stats(),
            'frequency_analysis': self.create_frequency_visualizations(),
            'network_analysis': self.create_cooccurrence_network(),
            'context_analysis': self.analyze_context(),
            'heatmap_analysis': self.create_heatmap()
        }
        
        return results
