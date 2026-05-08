"""
V7P3R AI v3.0 - Visual Training Monitor
=======================================

Real-time visual monitoring system that displays the AI's thought process
as a heatmap on the chess board. Shows move candidate consideration,
thought intensity, and decision patterns during training.

Features:
- Real-time chess board display
- Gradient heatmap showing AI attention
- Move arrows with intensity-based opacity
- Time-based fading of previous considerations
- Smooth transitions showing thought flow
"""

import pygame
import numpy as np
import chess
import chess.svg
import threading
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SquareAttention:
    """Attention data for a single square"""
    heat_level: float = 0.0          # Current heat intensity (0-1)
    last_updated: float = 0.0        # Timestamp of last update
    total_visits: int = 0            # Total times this square was considered
    decay_rate: float = 0.85         # How fast attention fades (faster decay)
    moves_since_last_consideration: int = 0  # Track moves since last hit


@dataclass
class MoveVisualization:
    """Visualization data for a candidate move"""
    from_square: int
    to_square: int
    intensity: float = 0.0           # Current arrow intensity (0-1)
    last_updated: float = 0.0        # Timestamp of last update
    consideration_count: int = 0     # How many times this move was considered
    move_str: str = ""               # String representation of move
    reset_on_next_move: bool = True  # Reset when any move is made


class ChessBoardVisualizer:
    """
    Real-time chess board visualizer with AI thought heatmap
    
    Displays the current position with colored squares and arrows
    showing where the AI is focusing its attention.
    """
    
    # Display constants
    BOARD_SIZE = 640                 # Pixel size of the board
    SQUARE_SIZE = BOARD_SIZE // 8    # Size of each square
    
    # Colors (RGB) - Darker, more visible colors
    LIGHT_SQUARE = (240, 217, 181)
    DARK_SQUARE = (181, 136, 99)
    HEAT_COLOR_LOW = (255, 200, 50)         # Darker yellow for low attention
    HEAT_COLOR_HIGH = (200, 25, 25)         # Darker red for high attention
    HEAT_COLOR_RECENT = (25, 100, 200)      # Blue for recently moved pieces
    ARROW_COLOR = (30, 100, 180)            # Darker blue for move arrows
    TEXT_COLOR = (50, 50, 50)
    LAST_MOVE_COLOR = (100, 200, 100)       # Green for last move highlight
    
    def __init__(self):
        """Initialize the chess board visualizer"""
        pygame.init()
        
        # Image paths - set before loading pieces
        self.image_dir = Path(__file__).parent.parent.parent.parent / "images"
        
        # Create display
        self.screen = pygame.display.set_mode((self.BOARD_SIZE + 200, self.BOARD_SIZE + 100))
        pygame.display.set_caption("V7P3R AI v3.0 - Training Monitor")
        
        # Load chess piece images
        self.piece_images = self._load_piece_images()
        
        # Font for text display
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Attention tracking
        self.square_attention = {i: SquareAttention() for i in range(64)}
        self.move_visualizations = {}  # Dict[str, MoveVisualization]
        
        # Current position and move tracking
        self.current_board = chess.Board()
        self.last_move = None           # Track the last move made
        self.move_count = 0             # Track total moves made
        
        # Monitoring state
        self.is_running = False
        self.update_lock = threading.Lock()
        
        # Performance tracking
        self.total_updates = 0
        self.start_time = time.time()
        
        logger.info("Chess Board Visualizer initialized")
    
    def _load_piece_images(self) -> Dict[str, pygame.Surface]:
        """Load chess piece images from the images directory"""
        pieces = {}
        
        # Mapping from piece symbols to image filenames
        piece_files = {
            'P': 'wp.png',   # White pawn
            'N': 'wN.png',   # White knight
            'B': 'wB.png',   # White bishop
            'R': 'wR.png',   # White rook
            'Q': 'wQ.png',   # White queen
            'K': 'wK.png',   # White king
            'p': 'bp.png',   # Black pawn
            'n': 'bN.png',   # Black knight
            'b': 'bB.png',   # Black bishop
            'r': 'bR.png',   # Black rook
            'q': 'bQ.png',   # Black queen
            'k': 'bK.png'    # Black king
        }
        
        target_size = (self.SQUARE_SIZE - 10, self.SQUARE_SIZE - 10)
        
        for piece_symbol, filename in piece_files.items():
            image_path = self.image_dir / filename
            
            try:
                if image_path.exists():
                    # Load and scale the piece image
                    image = pygame.image.load(str(image_path))
                    scaled_image = pygame.transform.scale(image, target_size)
                    pieces[piece_symbol] = scaled_image
                    logger.info(f"Loaded piece image: {filename}")
                else:
                    logger.warning(f"Piece image not found: {image_path}")
                    # Fallback to simple colored circles
                    pieces[piece_symbol] = self._create_fallback_piece(piece_symbol, target_size)
            except Exception as e:
                logger.error(f"Error loading piece image {filename}: {e}")
                pieces[piece_symbol] = self._create_fallback_piece(piece_symbol, target_size)
        
        return pieces
    
    def _create_fallback_piece(self, piece_symbol: str, size: Tuple[int, int]) -> pygame.Surface:
        """Create a fallback piece representation if image loading fails"""
        surface = pygame.Surface(size, pygame.SRCALPHA)
        
        # Choose color based on piece
        is_white = piece_symbol.isupper()
        color = (255, 255, 255) if is_white else (50, 50, 50)
        
        # Draw a circle with the piece letter
        center = (size[0] // 2, size[1] // 2)
        radius = size[0] // 2 - 5
        
        pygame.draw.circle(surface, color, center, radius)
        pygame.draw.circle(surface, (0, 0, 0), center, radius, 2)
        
        # Add piece letter
        font = pygame.font.Font(None, int(size[0] * 0.6))
        text = font.render(piece_symbol.upper(), True, (0, 0, 0) if is_white else (255, 255, 255))
        text_rect = text.get_rect(center=center)
        surface.blit(text, text_rect)
        
        return surface
    
    def update_position(self, board: chess.Board):
        """Update the current chess position and handle move tracking"""
        with self.update_lock:
            # Check if a move was made
            if board.fen() != self.current_board.fen():
                # A move was made - reset arrows and update move tracking
                if len(board.move_stack) > len(self.current_board.move_stack):
                    # New move made
                    if board.move_stack:
                        self.last_move = board.move_stack[-1]
                        self.move_count += 1
                        
                        # Reset all arrows
                        self.move_visualizations.clear()
                        
                        # Update moves since last consideration for all squares
                        for square_data in self.square_attention.values():
                            square_data.moves_since_last_consideration += 1
                            
                            # Fade squares that haven't been considered in many moves
                            if square_data.moves_since_last_consideration > 8:
                                square_data.heat_level *= 0.7  # Faster fade for old squares
            
            self.current_board = board.copy()
    
    def update_attention(
        self, 
        move_candidates: List[chess.Move], 
        probabilities: List[float],
        thinking_time: float = 0.0
    ):
        """
        Update attention heatmap based on current move candidates
        
        Args:
            move_candidates: List of moves being considered
            probabilities: Probability/attention for each move
            thinking_time: Time spent thinking (affects intensity)
        """
        with self.update_lock:
            current_time = time.time()
            
            # Debug logging
            if len(move_candidates) > 0:
                logger.debug(f"Updating attention: {len(move_candidates)} candidates, max prob: {max(probabilities) if probabilities else 0}")
            
            # Decay existing attention
            self._decay_attention(current_time)
            
            # Update attention for current candidates
            for i, move in enumerate(move_candidates):
                if i < len(probabilities):
                    intensity = probabilities[i]
                    
                    # Boost intensity based on thinking time
                    time_boost = min(thinking_time / 2.0, 1.0)  # Cap at 2 seconds
                    final_intensity = min(intensity + time_boost * 0.2, 1.0)
                    
                    # Debug for significant moves
                    if final_intensity > 0.3:
                        logger.debug(f"Hot move: {move} with intensity {final_intensity:.2f}")
                    
                    # Update square attention
                    self._update_square_attention(move.from_square, final_intensity, current_time)
                    self._update_square_attention(move.to_square, final_intensity, current_time)
                    
                    # Update move visualization
                    self._update_move_visualization(move, final_intensity, current_time)
            
            self.total_updates += 1
    
    def _decay_attention(self, current_time: float):
        """Decay attention over time for smooth fading effect"""
        for square_data in self.square_attention.values():
            if current_time - square_data.last_updated > 0.1:  # Decay if not updated recently
                square_data.heat_level *= square_data.decay_rate
                square_data.heat_level = max(0.0, square_data.heat_level)
        
        # Decay move visualizations much faster (quick arrow fade)
        moves_to_remove = []
        for move_str, move_viz in self.move_visualizations.items():
            if current_time - move_viz.last_updated > 0.05:  # Faster decay trigger
                move_viz.intensity *= 0.75  # Much faster decay rate
                if move_viz.intensity < 0.1:  # Higher threshold for removal
                    moves_to_remove.append(move_str)
        
        # Remove very faded moves
        for move_str in moves_to_remove:
            del self.move_visualizations[move_str]
    
    def _update_square_attention(self, square: int, intensity: float, current_time: float):
        """Update attention for a specific square"""
        square_data = self.square_attention[square]
        
        # Accumulate attention with higher intensity gain for better visibility
        square_data.heat_level = min(
            square_data.heat_level * 0.6 + intensity * 0.8, 1.0  # More emphasis on new attention
        )
        square_data.last_updated = current_time
        square_data.total_visits += 1
        square_data.moves_since_last_consideration = 0  # Reset move counter
    
    def _update_move_visualization(self, move: chess.Move, intensity: float, current_time: float):
        """Update visualization for a specific move"""
        move_str = str(move)
        
        if move_str not in self.move_visualizations:
            self.move_visualizations[move_str] = MoveVisualization(
                from_square=move.from_square,
                to_square=move.to_square,
                move_str=move_str
            )
        
        move_viz = self.move_visualizations[move_str]
        # Stronger accumulation for better visibility
        move_viz.intensity = min(move_viz.intensity * 0.5 + intensity * 0.8, 1.0)
        move_viz.last_updated = current_time
        move_viz.consideration_count += 1
    
    def render(self):
        """Render the current state of the board with heatmap"""
        with self.update_lock:
            # Clear screen
            self.screen.fill((250, 250, 250))
            
            # Draw chess board with heatmap
            self._draw_board_with_heatmap()
            
            # Draw pieces
            self._draw_pieces()
            
            # Draw move arrows
            self._draw_move_arrows()
            
            # Draw statistics panel
            self._draw_statistics_panel()
            
            # Update display
            pygame.display.flip()
    
    def _draw_board_with_heatmap(self):
        """Draw the chess board with attention heatmap overlay"""
        for rank in range(8):
            for file in range(8):
                square = rank * 8 + file
                
                # Calculate square position
                x = file * self.SQUARE_SIZE
                y = (7 - rank) * self.SQUARE_SIZE  # Flip vertically
                
                # Base square color
                is_light = (rank + file) % 2 == 0
                base_color = self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
                
                # Check if this is the last move
                is_last_move_square = False
                if self.last_move:
                    if square == self.last_move.from_square or square == self.last_move.to_square:
                        is_last_move_square = True
                
                # Get attention level for this square
                attention = self.square_attention[square].heat_level
                
                # Apply coloring based on attention and last move
                if is_last_move_square:
                    # Highlight last move squares in green
                    move_color = self.LAST_MOVE_COLOR
                    final_color = self._blend_colors(base_color, move_color, 0.5)
                elif attention > 0.01:  # Much lower threshold for training visibility
                    # Apply heat color with darker, more visible tones
                    heat_color = self._interpolate_color(
                        self.HEAT_COLOR_LOW, 
                        self.HEAT_COLOR_HIGH, 
                        attention
                    )
                    # Stronger blend for better visibility
                    blend_strength = min(attention * 1.2, 0.9)  # Darker opacity
                    final_color = self._blend_colors(base_color, heat_color, blend_strength)
                else:
                    final_color = base_color
                
                # Draw square
                pygame.draw.rect(
                    self.screen, 
                    final_color, 
                    (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)
                )
                
                # Draw square border
                pygame.draw.rect(
                    self.screen, 
                    (100, 100, 100), 
                    (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE), 
                    1
                )
    
    def _draw_pieces(self):
        """Draw chess pieces on the board"""
        for square in chess.SQUARES:
            piece = self.current_board.piece_at(square)
            if piece:
                # Calculate position
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                x = file * self.SQUARE_SIZE + 5
                y = (7 - rank) * self.SQUARE_SIZE + 5
                
                # Get piece symbol
                piece_symbol = piece.symbol()
                
                if piece_symbol in self.piece_images:
                    self.screen.blit(self.piece_images[piece_symbol], (x, y))
                else:
                    # Fallback: draw text
                    text = self.font.render(piece_symbol, True, self.TEXT_COLOR)
                    self.screen.blit(text, (x + 20, y + 20))
    
    def _draw_move_arrows(self):
        """Draw arrows showing move candidates with intensity-based opacity"""
        for move_viz in self.move_visualizations.values():
            if move_viz.intensity > 0.02:  # Much lower threshold for training
                # Calculate arrow positions
                from_file = chess.square_file(move_viz.from_square)
                from_rank = chess.square_rank(move_viz.from_square)
                to_file = chess.square_file(move_viz.to_square)
                to_rank = chess.square_rank(move_viz.to_square)
                
                start_x = from_file * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                start_y = (7 - from_rank) * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                end_x = to_file * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                end_y = (7 - to_rank) * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                
                # Thicker arrows but fewer of them
                line_width = max(2, int(move_viz.intensity * 8))  # Thinner minimum
                
                # Draw arrows with lower threshold
                if line_width >= 2:
                    pygame.draw.line(
                        self.screen, 
                        self.ARROW_COLOR,
                        (start_x, start_y), 
                        (end_x, end_y), 
                        line_width
                    )
                    
                    # Draw arrowhead for visible arrows
                    if move_viz.intensity > 0.05:  # Lower arrowhead threshold for training
                        self._draw_arrowhead(start_x, start_y, end_x, end_y, move_viz.intensity)
    
    def _draw_arrowhead(self, start_x: int, start_y: int, end_x: int, end_y: int, intensity: float):
        """Draw arrowhead at the end of a move arrow"""
        # Calculate arrow direction
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize direction
            dx /= length
            dy /= length
            
            # Arrowhead size based on intensity
            head_size = 10 + intensity * 10
            
            # Calculate arrowhead points
            head_x1 = end_x - head_size * dx + head_size * 0.5 * dy
            head_y1 = end_y - head_size * dy - head_size * 0.5 * dx
            head_x2 = end_x - head_size * dx - head_size * 0.5 * dy
            head_y2 = end_y - head_size * dy + head_size * 0.5 * dx
            
            # Draw arrowhead
            pygame.draw.polygon(
                self.screen,
                self.ARROW_COLOR,
                [(end_x, end_y), (head_x1, head_y1), (head_x2, head_y2)]
            )
    
    def _draw_statistics_panel(self):
        """Draw statistics and information panel"""
        panel_x = self.BOARD_SIZE + 10
        panel_y = 10
        
        # Background
        pygame.draw.rect(
            self.screen, 
            (240, 240, 240), 
            (panel_x, panel_y, 180, self.BOARD_SIZE - 20)
        )
        
        # Title
        title = self.font.render("AI Monitor", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Statistics
        stats_y = panel_y + 50
        
        # Runtime
        runtime = time.time() - self.start_time
        runtime_text = self.small_font.render(f"Runtime: {runtime:.1f}s", True, self.TEXT_COLOR)
        self.screen.blit(runtime_text, (panel_x + 10, stats_y))
        stats_y += 25
        
        # Total updates
        updates_text = self.small_font.render(f"Updates: {self.total_updates}", True, self.TEXT_COLOR)
        self.screen.blit(updates_text, (panel_x + 10, stats_y))
        stats_y += 25
        
        # Active moves
        active_moves = len([m for m in self.move_visualizations.values() if m.intensity > 0.1])
        moves_text = self.small_font.render(f"Active moves: {active_moves}", True, self.TEXT_COLOR)
        self.screen.blit(moves_text, (panel_x + 10, stats_y))
        stats_y += 25
        
        # Hottest squares
        hot_squares = sorted(
            [(sq, data.heat_level) for sq, data in self.square_attention.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        stats_y += 20
        hot_title = self.small_font.render("Hot Squares:", True, self.TEXT_COLOR)
        self.screen.blit(hot_title, (panel_x + 10, stats_y))
        stats_y += 20
        
        for i, (square, heat) in enumerate(hot_squares):
            if heat > 0.01:  # Much lower threshold for training heatmaps
                file_char = chr(ord('a') + chess.square_file(square))
                rank_char = str(chess.square_rank(square) + 1)
                square_name = f"{file_char}{rank_char}"
                square_text = self.small_font.render(
                    f"{square_name}: {heat:.2f}", True, self.TEXT_COLOR
                )
                self.screen.blit(square_text, (panel_x + 20, stats_y))
                stats_y += 18
        
        # Current position info
        stats_y += 20
        pos_title = self.small_font.render("Position:", True, self.TEXT_COLOR)
        self.screen.blit(pos_title, (panel_x + 10, stats_y))
        stats_y += 20
        
        # Move number
        move_num = self.current_board.fullmove_number
        move_text = self.small_font.render(f"Move: {move_num}", True, self.TEXT_COLOR)
        self.screen.blit(move_text, (panel_x + 20, stats_y))
        stats_y += 18
        
        # Side to move
        side = "White" if self.current_board.turn else "Black"
        side_text = self.small_font.render(f"Turn: {side}", True, self.TEXT_COLOR)
        self.screen.blit(side_text, (panel_x + 20, stats_y))
        stats_y += 18
        
        # Last move
        if self.last_move:
            last_move_text = self.small_font.render(f"Last: {self.last_move}", True, self.TEXT_COLOR)
            self.screen.blit(last_move_text, (panel_x + 20, stats_y))
            stats_y += 18
    
    def _interpolate_color(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        """Interpolate between two colors"""
        t = max(0.0, min(1.0, t))
        return (
            int(color1[0] * (1 - t) + color2[0] * t),
            int(color1[1] * (1 - t) + color2[1] * t),
            int(color1[2] * (1 - t) + color2[2] * t)
        )
    
    def _blend_colors(self, base: Tuple[int, int, int], overlay: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
        """Blend two colors with alpha"""
        alpha = max(0.0, min(1.0, alpha))
        return (
            int(base[0] * (1 - alpha) + overlay[0] * alpha),
            int(base[1] * (1 - alpha) + overlay[1] * alpha),
            int(base[2] * (1 - alpha) + overlay[2] * alpha)
        )
    
    def start_monitoring(self):
        """Start the monitoring display loop"""
        self.is_running = True
        clock = pygame.time.Clock()
        
        logger.info("Visual monitoring started - Controls: ESC=exit, R=reset heatmap")
        
        try:
            while self.is_running:
                # Handle events with error protection
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.is_running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.is_running = False
                            elif event.key == pygame.K_r:
                                # Reset heatmap
                                self._reset_heatmap()
                except pygame.error as e:
                    logger.warning(f"Pygame event error: {e}")
                    continue
                
                # Render frame with error protection
                try:
                    self.render()
                except Exception as e:
                    logger.warning(f"Render error: {e}")
                    continue
                
                # Limit to 30 FPS
                try:
                    clock.tick(30)
                except Exception as e:
                    logger.warning(f"Clock tick error: {e}")
                    time.sleep(0.033)  # Fallback timing
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            try:
                pygame.quit()
            except:
                pass
            logger.info("Visual monitoring stopped")
    
    def _reset_heatmap(self):
        """Reset the attention heatmap"""
        with self.update_lock:
            for square_data in self.square_attention.values():
                square_data.heat_level = 0.0
                square_data.total_visits = 0
                square_data.moves_since_last_consideration = 0
            self.move_visualizations.clear()
        logger.info("Heatmap reset")
    
    def stop_monitoring(self):
        """Stop the monitoring display"""
        self.is_running = False


class TrainingMonitor:
    """
    Integration interface for the visual monitoring system
    
    Connects the AI training process with the visual display
    """
    
    def __init__(self, enable_visual: bool = True):
        self.enable_visual = enable_visual
        self.visualizer = None
        self.monitor_thread = None
        
        if enable_visual:
            self.visualizer = ChessBoardVisualizer()
    
    def start_visual_monitoring(self):
        """Start visual monitoring in a separate thread"""
        if not self.enable_visual or not self.visualizer:
            return
        
        def monitor_wrapper():
            """Wrapper to handle exceptions in monitoring thread"""
            try:
                self.visualizer.start_monitoring()
            except Exception as e:
                logger.error(f"Visual monitoring error: {e}")
            finally:
                logger.info("Visual monitoring thread exiting")
        
        self.monitor_thread = threading.Thread(
            target=monitor_wrapper,
            daemon=True,
            name="VisualMonitor"
        )
        self.monitor_thread.start()
        logger.info("Visual monitoring thread started")
    
    def update_position(self, board: chess.Board):
        """Update the current chess position"""
        if self.visualizer:
            try:
                self.visualizer.update_position(board)
            except Exception as e:
                logger.warning(f"Position update error: {e}")
    
    def update_ai_attention(
        self, 
        move_candidates: List[chess.Move], 
        probabilities: List[float],
        thinking_time: float = 0.0
    ):
        """Update AI attention heatmap"""
        if self.visualizer:
            try:
                # Debug logging for training
                if move_candidates and probabilities:
                    max_prob = max(probabilities)
                    logger.debug(f"TrainingMonitor: {len(move_candidates)} candidates, max_prob: {max_prob:.4f}")
                    
                self.visualizer.update_attention(move_candidates, probabilities, thinking_time)
            except Exception as e:
                logger.warning(f"Attention update error: {e}")
    
    def stop_monitoring(self):
        """Stop visual monitoring with proper cleanup"""
        if self.visualizer:
            try:
                self.visualizer.stop_monitoring()
            except:
                pass
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            try:
                self.monitor_thread.join(timeout=3.0)
                if self.monitor_thread.is_alive():
                    logger.warning("Monitor thread did not stop gracefully")
            except Exception as e:
                logger.warning(f"Error stopping monitor thread: {e}")
        
        logger.info("Training monitor stopped")


def test_visual_monitor():
    """Test the visual monitoring system"""
    logger.info("Testing Visual Training Monitor...")
    
    # Create monitor
    monitor = TrainingMonitor(enable_visual=True)
    
    # Start visual monitoring
    monitor.start_visual_monitoring()
    
    # Simulate AI thinking on different positions
    board = chess.Board()
    monitor.update_position(board)
    
    # Simulate move considerations
    import random
    legal_moves = list(board.legal_moves)
    
    for i in range(100):
        # Simulate AI considering different moves
        candidates = random.sample(legal_moves, min(5, len(legal_moves)))
        probabilities = [random.random() for _ in candidates]
        thinking_time = random.uniform(0.1, 2.0)
        
        monitor.update_ai_attention(candidates, probabilities, thinking_time)
        
        time.sleep(0.1)  # Small delay to see the visualization
        
        # Occasionally make a move to change position
        if i % 20 == 0 and legal_moves:
            move = random.choice(legal_moves)
            board.push(move)
            monitor.update_position(board)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                board = chess.Board()  # Reset if game over
                legal_moves = list(board.legal_moves)
                monitor.update_position(board)
    
    logger.info("Test completed! Visual monitor should be running.")
    logger.info("Press ESC or close window to stop monitoring.")


if __name__ == "__main__":
    test_visual_monitor()
