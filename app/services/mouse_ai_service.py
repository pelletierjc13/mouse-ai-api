"""
Mouse AI service compatible with frontend format.
"""
from typing import List
import logging
import random

from app.core.utils import is_valid_position, get_adjacent_positions
from app.services.log_service import log_service

logger = logging.getLogger(__name__)


class MouseAIService:
    """Service for handling mouse AI logic compatible with frontend."""
    
    def __init__(self, mouse_id: str = "default"):
        """Initialize the AI service with position history tracking for a specific mouse."""
        self.mouse_id = mouse_id
        self.position_history = []  # [previous_positions] for this specific mouse
        self.last_direction = None
        logger.info(f"- Thread {mouse_id} - Initialized MouseAIService for mouse: {mouse_id}")
    
    def calculate_next_position(
        self, 
        labyrinth: List[List[int]], 
        current_position: List[int], 
        goal_position: List[int],
        mouse_id: str = "default",
        available_cheeses: List[List[int]] = None,
        algorithm: str = "greedy",
    ) -> List[int]:
        """
        Calculate the next position for the mouse using intelligent algorithm.
        
        Args:
            labyrinth: 2D maze representation (0=free, 1=wall)
            current_position: Current mouse position [x, y]
            goal_position: Target goal position [x, y]
            mouse_id: Unique identifier for the mouse
            available_cheeses: List of available cheese positions [[x, y], ...]
            
        Returns:
            List[int]: Next position [x, y]
        """
        logger.info(f"- Thread {mouse_id} - Starting calculation for position {current_position}, goal {goal_position}")
        
        # Log du début du calcul d'IA
        log_service.add_custom_log(
            message=f" Thread {mouse_id} - Starting AI calculation for position {current_position}, goal {goal_position}",
            level="DEBUG",
            mouse_id=mouse_id,
            current_position=current_position,
            goal_position=goal_position,
            available_cheeses=available_cheeses,
            action="ai_calculation_start"
        )
        
        # Validate current position
        if not is_valid_position(current_position, labyrinth):
            logger.warning(f"Current position {current_position} is invalid, trying to find valid position")
            # Try to find a valid position near the current one
            valid_position = self._find_nearest_valid_position(current_position, labyrinth)
            if valid_position:
                logger.info(f"Found valid position {valid_position} near {current_position}")
                current_position = valid_position
            else:
                logger.error(f"No valid position found near {current_position}")
                return current_position
        
        # If multiple cheeses available, choose the nearest one
        if available_cheeses and len(available_cheeses) > 1:
            optimal_cheese = self._find_nearest_cheese(current_position, available_cheeses, labyrinth)
            if optimal_cheese:
                goal_position = optimal_cheese
                logger.info(f"- Thread {mouse_id} - Mouse {mouse_id} targeting nearest cheese at {goal_position}")
        
        # If already at goal, stay in place
        if current_position == goal_position:
            logger.info(f"- Thread {mouse_id} - Mouse {mouse_id} is already at goal position {goal_position}")
            return current_position
        
        # NEW: choose algorithm
        if algorithm == "random":
            next_position = self._random_move(labyrinth, current_position)
        elif algorithm == "straight":
            next_position = self._straight_move(labyrinth, current_position)
        elif algorithm == "intelligent":
            # old A* + smart logic
            next_position = self._intelligent_move(labyrinth, current_position, goal_position, mouse_id)
        else:
            # default: greedy
            next_position = self._greedy_move(labyrinth, current_position, goal_position, mouse_id)
        
        # Update position history
        self._update_position_history(current_position, next_position)
        
        logger.info(f"- Thread {mouse_id} - Calculated next position: {next_position}")
        
        # Log du résultat du calcul d'IA
        log_service.add_custom_log(
            message=f" Thread {mouse_id} - AI calculation completed: {current_position} -> {next_position}",
            level="DEBUG",
            mouse_id=mouse_id,
            current_position=current_position,
            next_position=next_position,
            goal_position=goal_position,
            action="ai_calculation_complete"
        )
        
        return next_position
    
    def _intelligent_move(
        self, 
        labyrinth: List[List[int]], 
        current_position: List[int], 
        goal_position: List[int],
        mouse_id: str = "default"
    ) -> List[int]:
        """
        Intelligent movement algorithm using A* pathfinding.
        
        Args:
            labyrinth: 2D maze representation
            current_position: Current position [x, y]
            goal_position: Goal position [x, y]
            
        Returns:
            List[int]: Next position using intelligent approach
        """
        # Check if mouse cannot move towards goal and needs exploration
        if self._detect_no_movement_possible(labyrinth, current_position, goal_position):
            logger.info(f"Mouse {mouse_id} cannot move towards goal, forcing exploration")
            forced_move = self._force_direction_change(labyrinth, current_position, goal_position, mouse_id)
            if forced_move:
                return forced_move
        
        # Try to find a path using A* algorithm
        path = self._find_path_astar(labyrinth, current_position, goal_position)
        
        if path and len(path) > 1:
            next_pos = path[1]
            
            # Check if mouse is stuck
            if self._is_stuck(mouse_id, current_position):
                logger.info(f"Mouse {mouse_id} is stuck, forcing direction change")
                forced_move = self._force_direction_change(labyrinth, current_position, goal_position, mouse_id)
                if forced_move:
                    return forced_move
            
            # Check if this would be a back-and-forth movement
            if self._is_back_and_forth_move(current_position, next_pos):
                logger.info(f"Avoiding back-and-forth move from {current_position} to {next_pos}")
                # Try alternative moves
                alternative_move = self._find_alternative_move(labyrinth, current_position, goal_position, mouse_id)
                if alternative_move:
                    return alternative_move
            return next_pos
        
        # Fallback to greedy approach if A* fails
        logger.warning(f"A* pathfinding failed from {current_position} to {goal_position}, falling back to greedy.")
        return self._greedy_move(labyrinth, current_position, goal_position, mouse_id)
    
    def _find_path_astar(
        self, 
        labyrinth: List[List[int]], 
        start: List[int], 
        goal: List[int]
    ) -> List[List[int]]:
        """
        A* pathfinding algorithm implementation.
        """
        from collections import deque
        import heapq
        
        # Priority queue: (f_score, position)
        open_set = [(0, tuple(start))]
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): self._calculate_heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if list(current) == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(list(current))
                    current = came_from[current]
                path.append(list(current))
                return path[::-1]
            
            # Check all adjacent positions
            for neighbor in get_adjacent_positions(list(current)):
                if not is_valid_position(neighbor, labyrinth):
                    continue
                
                neighbor_tuple = tuple(neighbor)
                tentative_g_score = g_score[current] + 1
                
                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score[neighbor_tuple] = tentative_g_score + self._calculate_heuristic(neighbor, goal)
                    
                    if neighbor_tuple not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor_tuple], neighbor_tuple))
        
        return []  # No path found
    
    def _random_move(
        self,
        labyrinth: List[List[int]],
        current_position: List[int],
    ) -> List[int]:
        """
        Algorithm 2: Random movement.
        Pick any valid adjacent cell at random.
        """
        neighbors = [
            pos for pos in get_adjacent_positions(current_position)
            if is_valid_position(pos, labyrinth)
        ]
        
        if not neighbors:
            return current_position
        
        return random.choice(neighbors)

    def _straight_move(
        self,
        labyrinth: List[List[int]],
        current_position: List[int],
    ) -> List[int]:
        """
        Algorithm 3: Straight until wall.
        
        Behavior:
        - If we have a previous direction and it is still valid, keep using it.
        - If blocked or no previous direction, pick a new valid direction
          (random among up/right/down/left) and save it as last_direction.
        """
        x, y = current_position
        
        # If we have a direction, try to keep going
        if self.last_direction is not None:
            dx, dy = self.last_direction
            next_pos = [x + dx, y + dy]
            if is_valid_position(next_pos, labyrinth):
                return next_pos
        
        # Need to choose a new direction
        directions = [
            (0, -1),  # up
            (1, 0),   # right
            (0, 1),   # down
            (-1, 0),  # left
        ]
        
        valid_dirs = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_position([nx, ny], labyrinth):
                valid_dirs.append((dx, dy))
        
        if not valid_dirs:
            # Nowhere to go, stay in place
            return current_position
        
        # Pick a new direction and remember it
        self.last_direction = random.choice(valid_dirs)
        dx, dy = self.last_direction
        return [x + dx, y + dy]
    
    def _greedy_move(
        self, 
        labyrinth: List[List[int]], 
        current_position: List[int], 
        goal_position: List[int],
        mouse_id: str = "default"
    ) -> List[int]:
        """
        Greedy movement algorithm as fallback.
        """
        x, y = current_position
        gx, gy = goal_position
        
        # Calculate direction to goal
        dx = gx - x
        dy = gy - y
        
        # Priority moves towards goal
        candidate_moves = []
        
        # X axis movement (horizontal priority)
        if dx > 0:  # Move right
            candidate_moves.append([x + 1, y])
        elif dx < 0:  # Move left
            candidate_moves.append([x - 1, y])
        
        # Y axis movement (vertical secondary)
        if dy > 0:  # Move down
            candidate_moves.append([x, y + 1])
        elif dy < 0:  # Move up
            candidate_moves.append([x, y - 1])
        
        # Check if mouse is stuck first
        if self._is_stuck(mouse_id, current_position):
            logger.info(f"Mouse {mouse_id} is stuck in greedy mode, forcing exploration")
            forced_move = self._force_direction_change(labyrinth, current_position, goal_position, mouse_id)
            if forced_move:
                return forced_move
        
        # Try moves in priority order, avoiding back-and-forth
        for move in candidate_moves:
            if is_valid_position(move, labyrinth) and not self._is_back_and_forth_move(current_position, move):
                return move
        
        # If all preferred moves are back-and-forth, try any valid move
        for move in candidate_moves:
            if is_valid_position(move, labyrinth):
                return move
        
        # If no direct move towards goal is possible, try any valid adjacent position
        adjacent_positions = get_adjacent_positions(current_position)
        
        # Shuffle to add some randomness
        import random
        random.shuffle(adjacent_positions)
        
        for adjacent_pos in adjacent_positions:
            if is_valid_position(adjacent_pos, labyrinth):
                return adjacent_pos
        
        # If no move is possible, stay in place
        logger.warning(f"No valid moves from position {current_position}")
        return current_position
    
    def _calculate_heuristic(self, pos1: List[int], pos2: List[int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_dead_end(self, position: List[int], labyrinth: List[List[int]]) -> bool:
        """Check if position is a dead end (only one valid adjacent position)."""
        adjacent_positions = get_adjacent_positions(position)
        valid_adjacent = [pos for pos in adjacent_positions if is_valid_position(pos, labyrinth)]
        return len(valid_adjacent) <= 1
    
    def _find_nearest_valid_position(self, position: List[int], labyrinth: List[List[int]]) -> List[int]:
        """Find the nearest valid position to the given position."""
        # First, try adjacent positions
        for adjacent_pos in get_adjacent_positions(position):
            if is_valid_position(adjacent_pos, labyrinth):
                return adjacent_pos
        
        # If no adjacent valid position, search in a larger radius
        for radius in range(2, 5):  # Search up to radius 4
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        new_pos = [position[0] + dx, position[1] + dy]
                        if is_valid_position(new_pos, labyrinth):
                            return new_pos
        
        return None  # No valid position found
    
    def _update_position_history(self, current_pos: List[int], next_pos: List[int]):
        """Update position history for this mouse."""
        # Add current position to history
        self.position_history.append(current_pos)
        
        # Keep only last 3 positions to avoid memory issues
        if len(self.position_history) > 3:
            self.position_history = self.position_history[-3:]
    
    def _is_back_and_forth_move(self, current_pos: List[int], next_pos: List[int]) -> bool:
        """Check if the next move would be a back-and-forth movement."""
        if len(self.position_history) < 1:
            return False
        
        # Get the previous position (where the mouse came from)
        previous_pos = self.position_history[-1]
        
        # Check if next_pos is the same as previous_pos (going back)
        return next_pos == previous_pos
    
    def _find_alternative_move(self, labyrinth: List[List[int]], current_pos: List[int], goal_pos: List[int], mouse_id: str) -> List[int]:
        """Find an alternative move that doesn't go back to previous position."""
        adjacent_positions = get_adjacent_positions(current_pos)
        
        # Filter out back-and-forth moves
        valid_moves = []
        for pos in adjacent_positions:
            if (is_valid_position(pos, labyrinth) and 
                not self._is_back_and_forth_move(current_pos, pos)):
                valid_moves.append(pos)
        
        if not valid_moves:
            # If no alternative moves, return None to use original move
            return None
        
        # Choose the move that gets closest to the goal
        best_move = None
        min_distance = float('inf')
        
        for move in valid_moves:
            distance = abs(move[0] - goal_pos[0]) + abs(move[1] - goal_pos[1])
            if distance < min_distance:
                min_distance = distance
                best_move = move
        
        return best_move
    
    def _is_stuck(self, mouse_id: str, current_pos: List[int]) -> bool:
        """Check if the mouse is stuck (same position for multiple turns)."""
        if mouse_id not in self.position_history or len(self.position_history[mouse_id]) < 3:
            return False
        
        # Check if the mouse has been in the same position for the last 3 turns
        recent_positions = self.position_history[mouse_id][-3:]
        return all(pos == current_pos for pos in recent_positions)
    
    def _force_direction_change(self, labyrinth: List[List[int]], current_pos: List[int], goal_pos: List[int], mouse_id: str) -> List[int]:
        """Force a direction change when the mouse is stuck."""
        adjacent_positions = get_adjacent_positions(current_pos)
        
        # Get all valid moves (including back-and-forth if necessary)
        valid_moves = []
        for pos in adjacent_positions:
            if is_valid_position(pos, labyrinth):
                valid_moves.append(pos)
        
        if not valid_moves:
            return current_pos  # Stay in place if no moves available
        
        # If stuck, try to move away from the goal temporarily to explore
        if self._is_stuck(mouse_id, current_pos):
            logger.info(f"Mouse {mouse_id} is stuck at {current_pos}, forcing exploration")
            
            # Try moves that are NOT towards the goal (exploration)
            exploration_moves = []
            for move in valid_moves:
                current_distance = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
                new_distance = abs(move[0] - goal_pos[0]) + abs(move[1] - goal_pos[1])
                
                # If this move increases distance from goal, it's exploration
                if new_distance > current_distance:
                    exploration_moves.append(move)
            
            if exploration_moves:
                # Choose the exploration move that goes furthest from goal
                best_exploration = None
                max_distance = 0
                for move in exploration_moves:
                    distance = abs(move[0] - goal_pos[0]) + abs(move[1] - goal_pos[1])
                    if distance > max_distance:
                        max_distance = distance
                        best_exploration = move
                return best_exploration
        
        # If not stuck or no exploration moves, use normal logic
        return self._find_alternative_move(labyrinth, current_pos, goal_pos, mouse_id)
    
    def _detect_no_movement_possible(self, labyrinth: List[List[int]], current_pos: List[int], goal_pos: List[int]) -> bool:
        """Detect if the mouse cannot move towards the goal and needs to explore."""
        adjacent_positions = get_adjacent_positions(current_pos)
        valid_moves = [pos for pos in adjacent_positions if is_valid_position(pos, labyrinth)]
        
        if not valid_moves:
            return True  # No moves possible at all
        
        # Check if all valid moves would increase distance from goal
        current_distance = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
        moves_towards_goal = 0
        
        for move in valid_moves:
            new_distance = abs(move[0] - goal_pos[0]) + abs(move[1] - goal_pos[1])
            if new_distance <= current_distance:
                moves_towards_goal += 1
        
        # If no moves towards goal are possible, we need to explore
        return moves_towards_goal == 0
    
    def _find_nearest_cheese(self, current_position: List[int], available_cheeses: List[List[int]], labyrinth: List[List[int]]) -> List[int]:
        """
        Find the nearest cheese using pathfinding distance, not just Manhattan distance.
        
        Args:
            current_position: Current mouse position [x, y]
            available_cheeses: List of available cheese positions [[x, y], ...]
            labyrinth: 2D maze representation
            
        Returns:
            List[int]: Position of the nearest cheese [x, y]
        """
        if not available_cheeses:
            return None
        
        best_cheese = None
        shortest_path_length = float('inf')
        
        for cheese_pos in available_cheeses:
            # Calculate actual path length using A* algorithm
            path = self._find_path_astar(labyrinth, current_position, cheese_pos)
            
            if path:
                path_length = len(path) - 1  # -1 because path includes start position
                
                # If this cheese is closer (or same distance but better path), choose it
                if path_length < shortest_path_length:
                    shortest_path_length = path_length
                    best_cheese = cheese_pos
                elif path_length == shortest_path_length:
                    # If same path length, prefer cheese that's closer in Manhattan distance
                    current_manhattan = abs(current_position[0] - cheese_pos[0]) + abs(current_position[1] - cheese_pos[1])
                    best_manhattan = abs(current_position[0] - best_cheese[0]) + abs(current_position[1] - best_cheese[1])
                    
                    if current_manhattan < best_manhattan:
                        best_cheese = cheese_pos
            else:
                # If no path found to this cheese, use Manhattan distance as fallback
                manhattan_distance = abs(current_position[0] - cheese_pos[0]) + abs(current_position[1] - cheese_pos[1])
                if manhattan_distance < shortest_path_length:
                    shortest_path_length = manhattan_distance
                    best_cheese = cheese_pos
        
        return best_cheese
