import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class EmojiLanguageAnalyzer:
    def __init__(self, data_path):
        """Initialize the language analyzer with path to processed data."""
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.output_dir = self.data_path.parent / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_emoji_positioning(self):
        """Analyze where emojis appear in tweets."""
        positions = {
            'start': 0,
            'middle': 0,
            'end': 0
        }
        
        detailed_positions = []
        
        for _, row in self.df.iterrows():
            text = row['clean_text']
            emojis = eval(row['emojis'])
            
            # Skip if no emojis
            if not emojis:
                continue
                
            # Find all emoji positions
            emoji_indices = []
            for emoji in emojis:
                idx = text.find(emoji)
                while idx != -1:
                    emoji_indices.append(idx)
                    idx = text.find(emoji, idx + 1)
            
            if not emoji_indices:
                continue
                
            # Calculate relative positions
            text_length = len(text)
            for pos in emoji_indices:
                rel_pos = pos / text_length
                detailed_positions.append(rel_pos)
                
                # Categorize position
                if rel_pos < 0.2:
                    positions['start'] += 1
                elif rel_pos > 0.8:
                    positions['end'] += 1
                else:
                    positions['middle'] += 1
        
        return positions, detailed_positions
    
    def analyze_emoji_phrases(self):
        """Analyze common emoji combinations and their contexts."""
        emoji_pairs = []
        emoji_triples = []
        emoji_contexts = defaultdict(list)
        
        for _, row in self.df.iterrows():
            emojis = eval(row['emojis'])
            text = row['clean_text'].lower()
            
            # Analyze emoji sequences
            if len(emojis) >= 2:
                for i in range(len(emojis) - 1):
                    emoji_pairs.append((emojis[i], emojis[i+1]))
                    
            if len(emojis) >= 3:
                for i in range(len(emojis) - 2):
                    emoji_triples.append((emojis[i], emojis[i+1], emojis[i+2]))
            
            # Analyze context
            # Simple word tokenization
            words = [w.strip() for w in re.findall(r'\w+', text)]
            words = [w for w in words if w and w not in self.stop_words]
            
            for emoji in emojis:
                try:
                    idx = text.index(emoji)
                    # Get words before and after emoji
                    text_before = text[:idx].strip()
                    text_after = text[idx+len(emoji):].strip()
                    
                    before_words = re.findall(r'\w+', text_before)[-3:]
                    after_words = re.findall(r'\w+', text_after)[:3]
                    
                    emoji_contexts[emoji].append((before_words, after_words))
                except ValueError:
                    continue
        
        return Counter(emoji_pairs), Counter(emoji_triples), emoji_contexts
    
    def analyze_word_patterns(self):
        """Analyze words and phrases commonly associated with specific emojis."""
        emoji_collocations = defaultdict(Counter)
        emoji_phrases = defaultdict(Counter)
        
        for _, row in self.df.iterrows():
            text = row['clean_text'].lower()
            words = [w.strip() for w in re.findall(r'\w+', text)]
            words = [w for w in words if w and w not in self.stop_words]
            
            # Generate word bigrams and trigrams
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
            
            for emoji in eval(row['emojis']):
                # Add individual words
                emoji_collocations[emoji].update(words)
                
                # Add phrases
                emoji_phrases[emoji].update([' '.join(bg) for bg in bigrams])
                emoji_phrases[emoji].update([' '.join(tg) for tg in trigrams])
        
        return emoji_collocations, emoji_phrases
    
    def create_language_visualizations(self):
        """Create visualizations for language pattern analysis."""
        print("Analyzing language patterns...")
        
        # 1. Emoji Positioning Analysis
        positions, detailed_positions = self.analyze_emoji_positioning()
        
        fig1 = make_subplots(rows=1, cols=2,
                           subplot_titles=('Position Categories', 'Detailed Position Distribution'))
        
        # Position categories
        fig1.add_trace(
            go.Bar(
                x=list(positions.keys()),
                y=list(positions.values()),
                name='Position Count'
            ),
            row=1, col=1
        )
        
        # Detailed position distribution
        fig1.add_trace(
            go.Histogram(
                x=detailed_positions,
                nbinsx=50,
                name='Position Distribution'
            ),
            row=1, col=2
        )
        
        fig1.update_layout(
            title_text='Emoji Positioning Analysis',
            height=500,
            showlegend=False
        )
        
        fig1.write_html(self.output_dir / 'emoji_positioning_analysis.html')
        
        # 2. Emoji Phrases Analysis
        emoji_pairs, emoji_triples, contexts = self.analyze_emoji_phrases()
        
        # Top emoji pairs
        fig2 = go.Figure()
        
        top_pairs = emoji_pairs.most_common(20)
        fig2.add_trace(go.Bar(
            x=[f"{pair[0]} + {pair[1]}" for pair, _ in top_pairs],
            y=[count for _, count in top_pairs],
            name='Pair Frequency'
        ))
        
        fig2.update_layout(
            title='Top 20 Emoji Pairs',
            xaxis_title='Emoji Pair',
            yaxis_title='Frequency',
            height=600
        )
        
        fig2.write_html(self.output_dir / 'emoji_pairs.html')
        
        # 3. Word Pattern Analysis
        collocations, phrases = self.analyze_word_patterns()
        
        # Create visualization for top emojis and their common words
        fig3 = go.Figure()
        
        # Get top 10 emojis by word frequency
        top_emojis = sorted(collocations.items(), 
                          key=lambda x: sum(x[1].values()), 
                          reverse=True)[:10]
        
        for emoji, word_counts in top_emojis:
            top_words = word_counts.most_common(5)
            fig3.add_trace(go.Bar(
                name=emoji,
                x=[word for word, _ in top_words],
                y=[count for _, count in top_words],
                text=[word for word, _ in top_words],
            ))
        
        fig3.update_layout(
            title='Top Words Associated with Most Common Emojis',
            xaxis_title='Words',
            yaxis_title='Frequency',
            barmode='group',
            height=600
        )
        
        fig3.write_html(self.output_dir / 'emoji_word_associations.html')
        
        # Save analysis results
        results = {
            'positioning': {
                'total_emojis': sum(positions.values()),
                'position_distribution': positions
            },
            'phrases': {
                'unique_pairs': len(emoji_pairs),
                'unique_triples': len(emoji_triples),
                'top_pairs': [(f"{pair[0]} + {pair[1]}", count) 
                             for pair, count in emoji_pairs.most_common(5)]
            },
            'words': {
                'unique_collocations': sum(len(c) for c in collocations.values()),
                'top_emoji_words': {emoji: dict(counts.most_common(5)) 
                                  for emoji, counts in collocations.items()}
            }
        }
        
        print("Language pattern analysis completed!")
        return results
