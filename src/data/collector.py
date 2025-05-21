"""
Data collection module for gathering and storing battle outcomes.
"""

import sqlite3
import pickle
import os
import datetime
import numpy as np
from ..utils.constants import DB_PATH


class BattleDataCollector:
    """Collects and stores battle data for analysis and training."""
    
    def __init__(self, db_path=DB_PATH):
        """
        Initialize the data collection system.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = self._connect_db()
        self.setup_database()
    
    def _connect_db(self):
        """Connect to the SQLite database."""
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        return sqlite3.connect(self.db_path)
    
    def setup_database(self):
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create battles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS battles (
                id INTEGER PRIMARY KEY,
                enemy_formation BLOB,
                home_formation BLOB,
                winner TEXT,
                enemy_health REAL,
                home_health REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create formations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS formations (
                id INTEGER PRIMARY KEY,
                formation BLOB,
                side TEXT,
                formation_type TEXT,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                draw_count INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create unit stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unit_stats (
                id INTEGER PRIMARY KEY,
                unit_type TEXT,
                battles_participated INTEGER DEFAULT 0,
                battles_survived INTEGER DEFAULT 0,
                total_damage_dealt REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def record_battle(self, enemy_formation, home_formation, winner, enemy_health, home_health):
        """
        Record the outcome of a battle.
        
        Args:
            enemy_formation: 3D numpy array representing enemy formation
            home_formation: 3D numpy array representing home formation
            winner: 'ENEMY', 'HOME', or 'DRAW'
            enemy_health: Remaining health of enemy units
            home_health: Remaining health of home units
        """
        cursor = self.conn.cursor()
        
        # Insert into battles table
        cursor.execute(
            "INSERT INTO battles (enemy_formation, home_formation, winner, enemy_health, home_health) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                pickle.dumps(enemy_formation),
                pickle.dumps(home_formation),
                winner,
                enemy_health,
                home_health
            )
        )
        
        # Get the battle ID
        battle_id = cursor.lastrowid
        
        # Update formation stats
        self._update_formation_stats(enemy_formation, "ENEMY", winner)
        self._update_formation_stats(home_formation, "HOME", winner)
        
        self.conn.commit()
        return battle_id
    
    def _update_formation_stats(self, formation, side, battle_winner):
        """
        Update statistics for a formation.
        
        Args:
            formation: 3D numpy array representing the formation
            side: 'ENEMY' or 'HOME'
            battle_winner: 'ENEMY', 'HOME', or 'DRAW'
        """
        cursor = self.conn.cursor()
        
        # Determine formation type (could be enhanced with more sophisticated analysis)
        formation_type = self._classify_formation(formation)
        
        # Check if this formation already exists
        cursor.execute(
            "SELECT id, win_count, loss_count, draw_count FROM formations "
            "WHERE side = ? AND formation_type = ?",
            (side, formation_type)
        )
        result = cursor.fetchone()
        
        # Update win/loss/draw counts
        if battle_winner == side:
            win_change = 1
            loss_change = 0
            draw_change = 0
        elif battle_winner == "DRAW":
            win_change = 0
            loss_change = 0
            draw_change = 1
        else:
            win_change = 0
            loss_change = 1
            draw_change = 0
        
        if result:
            formation_id, win_count, loss_count, draw_count = result
            cursor.execute(
                "UPDATE formations SET win_count = ?, loss_count = ?, draw_count = ? "
                "WHERE id = ?",
                (win_count + win_change, loss_count + loss_change, draw_count + draw_change, formation_id)
            )
        else:
            cursor.execute(
                "INSERT INTO formations (formation, side, formation_type, win_count, loss_count, draw_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    pickle.dumps(formation),
                    side,
                    formation_type,
                    win_change,
                    loss_change,
                    draw_change
                )
            )
    
    def _classify_formation(self, formation):
        """
        Classify a formation into a type based on its characteristics.
        This is a simple implementation that could be enhanced with more
        sophisticated pattern recognition.
        
        Args:
            formation: 3D numpy array representing the formation
            
        Returns:
            String representing formation type
        """
        # Simple approach: count units in different regions of the formation
        height, width, _ = formation.shape
        
        # Count units in different regions
        top_count = np.sum(formation[:height//3, :, :] > 0)
        middle_count = np.sum(formation[height//3:2*height//3, :, :] > 0)
        bottom_count = np.sum(formation[2*height//3:, :, :] > 0)
        
        left_count = np.sum(formation[:, :width//2, :] > 0)
        right_count = np.sum(formation[:, width//2:, :] > 0)
        
        # Determine formation type based on distribution
        if top_count > middle_count + bottom_count:
            return "TOP_HEAVY"
        elif bottom_count > middle_count + top_count:
            return "BOTTOM_HEAVY"
        elif abs(top_count - bottom_count) < height//4 and (top_count + bottom_count) > 1.5 * middle_count:
            return "FLANKING"
        elif left_count > 2 * right_count:
            return "LEFT_CONCENTRATED"
        elif right_count > 2 * left_count:
            return "RIGHT_CONCENTRATED"
        elif abs(top_count - middle_count - bottom_count) < height//5:
            return "BALANCED"
        else:
            return "MIXED"
    
    def get_training_data(self, limit=1000):
        """
        Retrieve battle data for training.
        
        Args:
            limit: Maximum number of battles to retrieve
            
        Returns:
            List of dictionaries with battle data
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT enemy_formation, home_formation, winner, enemy_health, home_health "
            "FROM battles ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        
        data = []
        for row in cursor.fetchall():
            data.append({
                "enemy_formation": pickle.loads(row[0]),
                "home_formation": pickle.loads(row[1]),
                "winner": row[2],
                "enemy_health": row[3],
                "home_health": row[4]
            })
        
        return data
    
    def get_battle_count(self):
        """Get the total number of recorded battles."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM battles")
        return cursor.fetchone()[0]
    
    def get_formation_stats(self, side=None, formation_type=None, limit=10):
        """
        Get statistics about formations.
        
        Args:
            side: Optional filter by side ('ENEMY' or 'HOME')
            formation_type: Optional filter by formation type
            limit: Maximum number of formations to retrieve
            
        Returns:
            List of formation statistics
        """
        cursor = self.conn.cursor()
        
        query = "SELECT formation, side, formation_type, win_count, loss_count, draw_count FROM formations"
        params = []
        
        # Add filters
        conditions = []
        if side:
            conditions.append("side = ?")
            params.append(side)
        
        if formation_type:
            conditions.append("formation_type = ?")
            params.append(formation_type)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY (win_count / (win_count + loss_count + draw_count + 0.1)) DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        stats = []
        for row in cursor.fetchall():
            formation, side, formation_type, win_count, loss_count, draw_count = row
            
            total_battles = win_count + loss_count + draw_count
            win_rate = win_count / total_battles if total_battles > 0 else 0
            
            stats.append({
                "formation": pickle.loads(formation),
                "side": side,
                "formation_type": formation_type,
                "win_count": win_count,
                "loss_count": loss_count,
                "draw_count": draw_count,
                "total_battles": total_battles,
                "win_rate": win_rate
            })
        
        return stats
    
    def clear_database(self):
        """Clear all data from the database. Use with caution!"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM battles")
        cursor.execute("DELETE FROM formations")
        cursor.execute("DELETE FROM unit_stats")
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close() 