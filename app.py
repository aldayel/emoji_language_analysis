from flask import Flask
from emoji_dashboard import EmojiDashboard
from pathlib import Path

# Create Flask app
server = Flask(__name__)

# Initialize dashboard
data_path = Path(__file__).parent / 'data' / 'processed' / 'processed_tweets.csv'
dashboard = EmojiDashboard(data_path)
app = dashboard.app
app.server = server

if __name__ == '__main__':
    app.run_server(debug=True)
