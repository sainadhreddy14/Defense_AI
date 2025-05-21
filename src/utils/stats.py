"""
Statistics and analysis utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..data.collector import BattleDataCollector


def display_battle_statistics(data_collector):
    """
    Display battle statistics from a data collector.
    
    Args:
        data_collector: BattleDataCollector instance
    """
    # Get basic counts
    battle_count = data_collector.get_battle_count()
    
    print(f"Total battles recorded: {battle_count}")
    
    if battle_count == 0:
        print("No battles recorded yet.")
        return
    
    # Get recent battles
    recent_battles = data_collector.get_training_data(limit=100)
    
    # Calculate win rates
    home_wins = sum(1 for b in recent_battles if b["winner"] == "HOME")
    enemy_wins = sum(1 for b in recent_battles if b["winner"] == "ENEMY")
    draws = sum(1 for b in recent_battles if b["winner"] == "DRAW")
    
    home_win_rate = home_wins / len(recent_battles) if recent_battles else 0
    enemy_win_rate = enemy_wins / len(recent_battles) if recent_battles else 0
    draw_rate = draws / len(recent_battles) if recent_battles else 0
    
    print(f"\nRecent battles (last {len(recent_battles)}):")
    print(f"  Home win rate: {home_win_rate:.2%}")
    print(f"  Enemy win rate: {enemy_win_rate:.2%}")
    print(f"  Draw rate: {draw_rate:.2%}")
    
    # Get formation stats
    formation_stats = data_collector.get_formation_stats(side="HOME", limit=5)
    
    print("\nTop performing HOME formations:")
    for i, stat in enumerate(formation_stats):
        print(f"  {i+1}. Type: {stat['formation_type']}, Win rate: {stat['win_rate']:.2%}, Battles: {stat['total_battles']}")
    
    # Analyze health outcomes
    health_diffs = [b["home_health"] - b["enemy_health"] for b in recent_battles]
    avg_health_diff = sum(health_diffs) / len(health_diffs) if health_diffs else 0
    
    print(f"\nAverage health difference (positive means HOME advantage): {avg_health_diff:.2f}")
    
    # Offer to show plots
    print("\nUse plot_win_rates() to visualize win rate trends")
    print("Use plot_formation_effectiveness() to visualize formation performance")


def plot_win_rates(data_collector, window_size=20):
    """
    Plot win rates over time.
    
    Args:
        data_collector: BattleDataCollector instance
        window_size: Size of the moving average window
    """
    # Get all battles
    battles = data_collector.get_training_data(limit=1000)
    
    if not battles:
        print("No battles recorded yet.")
        return
    
    # Create dataframe
    df = pd.DataFrame([
        {
            "winner": b["winner"],
            "home_health": b["home_health"],
            "enemy_health": b["enemy_health"],
            "health_diff": b["home_health"] - b["enemy_health"]
        }
        for b in battles
    ])
    
    # Add win indicators
    df["home_win"] = (df["winner"] == "HOME").astype(int)
    df["enemy_win"] = (df["winner"] == "ENEMY").astype(int)
    df["draw"] = (df["winner"] == "DRAW").astype(int)
    
    # Calculate moving averages
    df["home_win_rate_ma"] = df["home_win"].rolling(window=window_size).mean()
    df["enemy_win_rate_ma"] = df["enemy_win"].rolling(window=window_size).mean()
    df["draw_rate_ma"] = df["draw"].rolling(window=window_size).mean()
    df["health_diff_ma"] = df["health_diff"].rolling(window=window_size).mean()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot win rates
    ax1.plot(df["home_win_rate_ma"], label="Home Win Rate", color="blue")
    ax1.plot(df["enemy_win_rate_ma"], label="Enemy Win Rate", color="red")
    ax1.plot(df["draw_rate_ma"], label="Draw Rate", color="gray")
    ax1.set_ylabel("Win Rate")
    ax1.set_title(f"Win Rates (Moving Average, Window Size = {window_size})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot health difference
    ax2.plot(df["health_diff_ma"], label="Health Difference", color="green")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Health Difference")
    ax2.set_xlabel("Battle Number")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_formation_effectiveness(data_collector):
    """
    Plot the effectiveness of different formation types.
    
    Args:
        data_collector: BattleDataCollector instance
    """
    # Get formation stats for all types
    home_stats = data_collector.get_formation_stats(side="HOME", limit=100)
    enemy_stats = data_collector.get_formation_stats(side="ENEMY", limit=100)
    
    if not home_stats and not enemy_stats:
        print("No formation statistics available yet.")
        return
    
    # Group by formation type
    home_by_type = {}
    for stat in home_stats:
        formation_type = stat["formation_type"]
        if formation_type not in home_by_type:
            home_by_type[formation_type] = []
        home_by_type[formation_type].append(stat)
    
    enemy_by_type = {}
    for stat in enemy_stats:
        formation_type = stat["formation_type"]
        if formation_type not in enemy_by_type:
            enemy_by_type[formation_type] = []
        enemy_by_type[formation_type].append(stat)
    
    # Calculate average win rate per formation type
    home_avg_win_rates = {}
    for formation_type, stats in home_by_type.items():
        total_wins = sum(s["win_count"] for s in stats)
        total_battles = sum(s["total_battles"] for s in stats)
        if total_battles > 0:
            home_avg_win_rates[formation_type] = total_wins / total_battles
    
    enemy_avg_win_rates = {}
    for formation_type, stats in enemy_by_type.items():
        total_wins = sum(s["win_count"] for s in stats)
        total_battles = sum(s["total_battles"] for s in stats)
        if total_battles > 0:
            enemy_avg_win_rates[formation_type] = total_wins / total_battles
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Home formations
    if home_avg_win_rates:
        types = list(home_avg_win_rates.keys())
        win_rates = list(home_avg_win_rates.values())
        ax1.bar(types, win_rates, color="blue", alpha=0.7)
        ax1.set_title("HOME Formation Effectiveness")
        ax1.set_ylabel("Win Rate")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    else:
        ax1.text(0.5, 0.5, "No home formation data", 
                 horizontalalignment="center", verticalalignment="center")
    
    # Enemy formations
    if enemy_avg_win_rates:
        types = list(enemy_avg_win_rates.keys())
        win_rates = list(enemy_avg_win_rates.values())
        ax2.bar(types, win_rates, color="red", alpha=0.7)
        ax2.set_title("ENEMY Formation Effectiveness")
        ax2.set_ylabel("Win Rate")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    else:
        ax2.text(0.5, 0.5, "No enemy formation data", 
                 horizontalalignment="center", verticalalignment="center")
    
    plt.tight_layout()
    plt.show()


def analyze_formation_patterns(data_collector, side="HOME"):
    """
    Analyze patterns in formations to identify key strategies.
    
    Args:
        data_collector: BattleDataCollector instance
        side: Which side to analyze ('HOME' or 'ENEMY')
        
    Returns:
        Dictionary with pattern analysis
    """
    # Get winning formations
    battles = data_collector.get_training_data(limit=1000)
    
    if not battles:
        print("No battles recorded yet.")
        return {}
    
    # Filter for the specified side's wins
    if side == "HOME":
        winning_battles = [b for b in battles if b["winner"] == "HOME"]
        formations = [b["home_formation"] for b in winning_battles]
    else:
        winning_battles = [b for b in battles if b["winner"] == "ENEMY"]
        formations = [b["enemy_formation"] for b in winning_battles]
    
    if not formations:
        print(f"No winning formations found for {side}.")
        return {}
    
    # Analyze unit distributions
    unit_heatmaps = {}
    for unit_idx, unit_type in enumerate(UNIT_TYPES):
        unit_heatmap = np.zeros((25, 10))
        for formation in formations:
            unit_mask = formation[:, :, unit_idx] > 0
            unit_heatmap += unit_mask
        
        # Normalize
        if np.sum(unit_heatmap) > 0:
            unit_heatmap /= len(formations)
        
        unit_heatmaps[unit_type] = unit_heatmap
    
    # Analyze unit co-occurrence
    unit_counts = {unit_type: 0 for unit_type in UNIT_TYPES}
    for formation in formations:
        for unit_idx, unit_type in enumerate(UNIT_TYPES):
            unit_count = np.sum(formation[:, :, unit_idx] > 0)
            unit_counts[unit_type] += unit_count
    
    # Normalize
    total_units = sum(unit_counts.values())
    unit_frequencies = {unit_type: count / total_units if total_units > 0 else 0 
                       for unit_type, count in unit_counts.items()}
    
    return {
        "heatmaps": unit_heatmaps,
        "unit_frequencies": unit_frequencies,
        "total_winning_formations": len(formations)
    }


def plot_unit_heatmaps(analysis_results):
    """
    Plot heatmaps showing unit placement patterns.
    
    Args:
        analysis_results: Results from analyze_formation_patterns
    """
    if not analysis_results or "heatmaps" not in analysis_results:
        print("No heatmap data available.")
        return
    
    heatmaps = analysis_results["heatmaps"]
    
    # Create a grid of subplots
    num_units = len(heatmaps)
    cols = 3
    rows = (num_units + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    
    # Flatten axes for easier indexing
    if rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each unit's heatmap
    for i, (unit_type, heatmap) in enumerate(heatmaps.items()):
        if i < len(axes):
            ax = axes[i]
            im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
            ax.set_title(f"{unit_type} Placement")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            fig.colorbar(im, ax=ax, label="Frequency")
    
    # Hide unused subplots
    for i in range(len(heatmaps), len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Also plot unit frequencies
    if "unit_frequencies" in analysis_results:
        unit_freqs = analysis_results["unit_frequencies"]
        
        plt.figure(figsize=(10, 6))
        plt.bar(unit_freqs.keys(), unit_freqs.values(), color="green", alpha=0.7)
        plt.title("Unit Type Frequencies in Winning Formations")
        plt.ylabel("Frequency")
        plt.xlabel("Unit Type")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show() 