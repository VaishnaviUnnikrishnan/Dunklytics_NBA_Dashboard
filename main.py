from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load data and models (replace with your actual paths)
try:
    player_model = joblib.load('player_stats_model.pkl')
    team_model = joblib.load('team_rank_model.pkl')
    player_data = pd.read_csv('2023-2024 NBA Player Stats - Playoffs.csv', delimiter=';')
    team_data = player_data.groupby('Tm').agg({
        'PTS': 'sum',
        'TRB': 'sum',
        'AST': 'sum',
        'FG%': 'mean',
        '3P%': 'mean',
        'FT%': 'mean'
    }).reset_index()

    # Fill missing values with 0
    team_data = team_data.fillna(0)

    # Log team_data structure
    logging.debug("Team Data Columns: %s", team_data.columns)
    logging.debug("Team Data Sample:\n%s", team_data.head())
except Exception as e:
    logging.error(f"Error loading data or models: {e}")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/player-insights")
def player_insights():
    players = sorted(player_data['Player'].unique())
    return render_template("player_insights.html", players=players)

@app.route("/team-insights")
def team_insights():
    teams = sorted(team_data['Tm'].unique())
    return render_template("team_insights.html", teams=teams)

@app.route("/player-charts")
def player_charts():
    player = request.args.get('player')
    year = int(request.args.get('year'))

    # Filter player data
    player_stats = player_data[player_data['Player'] == player].iloc[0]

    # Create charts
    charts = {
        "monthly_trend": create_monthly_trend_chart(player_stats),
        "shooting_percentages": create_shooting_chart(player_stats),
        "radar_chart": create_radar_chart(player_stats),
    }
    return jsonify(charts)

@app.route("/team-charts")
def team_charts():
    team = request.args.get('team')
    year = int(request.args.get('year'))

    # Filter team data
    team_stats = team_data[team_data['Tm'] == team].iloc[0]

    # Create charts
    charts = {
        "team_stats": create_team_stats_chart(team_stats),
        "ranking_history": create_ranking_history_chart(team_stats, year),
        "comparison_chart": create_comparison_chart(team_stats),
    }
    return jsonify(charts)

def create_monthly_trend_chart(player_stats):
    # Create sample monthly data
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb']
    pts = np.random.normal(player_stats['PTS'], 2, 5)
    ast = np.random.normal(player_stats['AST'], 1, 5)
    trb = np.random.normal(player_stats['TRB'], 1, 5)

    fig, ax = plt.subplots()
    ax.plot(months, pts, 'o-', label='Points')
    ax.plot(months, ast, 's-', label='Assists')
    ax.plot(months, trb, '^-', label='Rebounds')
    ax.set_title('Monthly Performance Trends')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    return fig_to_base64(fig)

def create_shooting_chart(player_stats):
    shot_types = ['FG%', '3P%', 'FT%']
    percentages = [player_stats[stat] for stat in shot_types]

    fig, ax = plt.subplots()
    ax.bar(shot_types, percentages)
    ax.set_title('Shooting Percentages')
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 100)

    for i, v in enumerate(percentages):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')

    return fig_to_base64(fig)

def create_radar_chart(player_stats):
    stats = ['PTS', 'AST', 'TRB', 'STL', 'BLK']
    current_stats = [player_stats[stat] for stat in stats]

    angles = np.linspace(0, 2 * np.pi, len(stats), endpoint=False)
    current_stats = np.concatenate((current_stats, [current_stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles, current_stats, 'o-', label='Current')
    ax.fill(angles, current_stats, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(stats)
    ax.set_title('Current Stats')
    ax.legend(loc='upper right')

    return fig_to_base64(fig)

def create_team_stats_chart(team_stats):
    stats = ['PTS', 'TRB', 'AST']
    values = [team_stats[stat] for stat in stats]

    fig, ax = plt.subplots()
    ax.bar(stats, values, color='royalblue', alpha=0.7)
    ax.set_title('Team Statistics')
    ax.set_ylabel('Value')

    for i, v in enumerate(values):
        ax.text(i, v + 1, f'{v:.0f}', ha='center')

    return fig_to_base64(fig)

def create_ranking_history_chart(team_stats, year):
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Predicted']
    ranks = list(np.random.randint(1, 16, 5)) + [year % 10]  # Example prediction

    fig, ax = plt.subplots()
    ax.plot(months, ranks, 'o-', color='royalblue')
    ax.set_title('Ranking History')
    ax.set_ylabel('Rank')
    ax.set_ylim(15, 1)  # Reverse y-axis for ranks
    ax.grid(True)

    return fig_to_base64(fig)

def create_comparison_chart(team_stats):
    # Exclude non-numeric columns (e.g., 'Tm')
    numeric_columns = team_data.select_dtypes(include=[np.number]).columns
    league_avg = team_data[numeric_columns].mean()

    stats = ['PTS', 'TRB', 'AST', 'FG%', '3P%', 'FT%']
    team_values = [team_stats[stat] for stat in stats]
    league_values = [league_avg[stat] for stat in stats]

    x = np.arange(len(stats))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, team_values, width, label='Team', color='royalblue', alpha=0.7)
    ax.bar(x + width / 2, league_values, width, label='League Average', color='lightcoral', alpha=0.7)
    ax.set_title('Team vs League Average')
    ax.set_xticks(x)
    ax.set_xticklabels(stats, rotation=45)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    return fig_to_base64(fig)

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)