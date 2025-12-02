"""
Mouse movement endpoints compatible with the frontend.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
import logging

from app.services.mouse_ai_service import MouseAIService
from app.services.log_service import log_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["mouse"])

# Dictionary to store separate AI service instances for each mouse
mouse_ai_services: Dict[str, MouseAIService] = {}


@router.post("/move")
async def get_mouse_move(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get next move for a mouse based on the frontend request format.
    
    Expected request format:
    {
        "mouseId": "string",
        "position": {"x": int, "y": int},
        "environment": {
            "grid": [["wall", "path", ...], ...],
            "width": int,
            "height": int,
            "cheesePositions": [{"x": int, "y": int}, ...],
            "otherMice": [...],
            "walls": [...],
            "paths": [...]
        },
        "mouseState": {
            "health": int,
            "happiness": int,
            "energy": int,
            "cheeseFound": int
        },
        "availableMoves": ["north", "south", "east", "west"]
    }
    
    Returns:
    {
        "mouseId": "string",
        "move": "north|south|east|west",
        "reasoning": "string"
    }
    """
    try:
        # Extract data from request
        mouse_id = request.get("mouseId", "unknown")
        position = request.get("position", {"x": 0, "y": 0})
        environment = request.get("environment", {})
        mouse_state = request.get("mouseState", {})
        available_moves = request.get("availableMoves", ["north", "south", "east", "west"])

        # NEW: read algorithm from mouseState (default = "greedy")
        algorithm = mouse_state.get("algorithm", "greedy")
        
        # Extract mouse tag from mouse_state or use mouse_id as fallback
        mouse_tag = mouse_state.get("tag", mouse_id)
        if isinstance(mouse_tag, str) and mouse_tag.isdigit():
            mouse_tag = int(mouse_tag)
        elif isinstance(mouse_tag, str):
            # Try to extract number from mouse_id (e.g., "souris1" -> 1)
            import re
            match = re.search(r'(\d+)', mouse_id)
            mouse_tag = int(match.group(1)) if match else 1
        else:
            mouse_tag = mouse_tag if isinstance(mouse_tag, int) else 1
            
        logger.info(f"- Thread {mouse_tag} - Received move request for mouse: {mouse_id}")
        
        # Log du mouvement de souris pour le frontend
        log_service.add_custom_log(
            message=f"Thread {mouse_tag} - Received move request for mouse: {mouse_id}",
            level="INFO",
            mouse_id=mouse_id,
            mouse_tag=mouse_tag,
            position=position,
            mouse_state=mouse_state
        )
        
        # Create or get AI service instance for this specific mouse
        if mouse_id not in mouse_ai_services:
            mouse_ai_services[mouse_id] = MouseAIService(f"Thread {mouse_tag}")
            logger.info(f"- Thread {mouse_tag} - Created new AI service instance for mouse: {mouse_id}")
            
            # Log de création du service
            log_service.add_custom_log(
                message=f" Thread {mouse_tag} - Created new AI service instance for mouse: {mouse_id}",
                level="INFO",
                mouse_id=mouse_id,
                mouse_tag=mouse_tag,
                action="service_created"
            )
        
        mouse_ai_service = mouse_ai_services[mouse_id]
        
        # Convert frontend grid format to Python format
        grid = environment.get("grid", [])
        python_grid = []
        for row in grid:
            python_row = []
            for cell in row:
                if cell == "wall":
                    python_row.append(1)  # 1 = wall (impassable)
                else:  # path, cheese, start = 0 (passable)
                    python_row.append(0)
            python_grid.append(python_row)
        
        # Debug: Log the grid conversion
        print(f"Frontend grid: {grid}")
        print(f"Python grid: {python_grid}")
        print(f"Mouse position: {position}")
        
        # Find the nearest cheese as goal
        cheese_positions = environment.get("cheesePositions", [])
        if not cheese_positions:
            # No cheese, use random movement
            import random
            move = random.choice(available_moves)
            return {
                "mouseId": mouse_id,
                "move": move,
                "reasoning": "No cheese found, random movement"
            }
        
        # Find closest cheese
        current_pos = [position["x"], position["y"]]
        
        # Check if mouse is already on a cheese
        for cheese in cheese_positions:
            if current_pos[0] == cheese["x"] and current_pos[1] == cheese["y"]:
                return {
                    "mouseId": mouse_id,
                    "move": "north",  # Use a valid direction but the frontend should handle this
                    "reasoning": f"Mouse is already on cheese at ({cheese['x']}, {cheese['y']}) - staying in place"
                }
        
        closest_cheese = cheese_positions[0]
        min_distance = abs(current_pos[0] - closest_cheese["x"]) + abs(current_pos[1] - closest_cheese["y"])
        
        for cheese in cheese_positions[1:]:
            distance = abs(current_pos[0] - cheese["x"]) + abs(current_pos[1] - cheese["y"])
            if distance < min_distance:
                min_distance = distance
                closest_cheese = cheese
        
        goal_position = [closest_cheese["x"], closest_cheese["y"]]
        
        # Convert cheese positions to list format for AI optimization
        available_cheeses_list = []
        for cheese in cheese_positions:
            available_cheeses_list.append([cheese["x"], cheese["y"]])
        
        # Get next move using the AI service with available cheeses
        next_position = mouse_ai_service.calculate_next_position(
            labyrinth=python_grid,
            current_position=current_pos,
            goal_position=goal_position,
            mouse_id=mouse_id,
            available_cheeses=available_cheeses_list,
            algorithm=algorithm,
        )
        
        # Convert position change to direction
        move = _position_to_direction(current_pos, next_position)
        
        # Generate intelligent reasoning
        distance_to_cheese = abs(current_pos[0] - closest_cheese['x']) + abs(current_pos[1] - closest_cheese['y'])
        
        if next_position == current_pos:
            reasoning = f"Staying in place - no valid moves available"
        elif move == "north":
            reasoning = f"Moving north towards cheese at ({closest_cheese['x']}, {closest_cheese['y']}) - distance: {distance_to_cheese}"
        elif move == "south":
            reasoning = f"Moving south towards cheese at ({closest_cheese['x']}, {closest_cheese['y']}) - distance: {distance_to_cheese}"
        elif move == "east":
            reasoning = f"Moving east towards cheese at ({closest_cheese['x']}, {closest_cheese['y']}) - distance: {distance_to_cheese}"
        elif move == "west":
            reasoning = f"Moving west towards cheese at ({closest_cheese['x']}, {closest_cheese['y']}) - distance: {distance_to_cheese}"
        else:
            reasoning = f"Moving {move} towards cheese at ({closest_cheese['x']}, {closest_cheese['y']})"
        
        logger.info(f"- Thread {mouse_tag} - Returning move: {move} - {reasoning}")
        
        # Log du mouvement calculé
        log_service.add_custom_log(
            message=f"Thread {mouse_tag} - Calculated move: {move} for mouse {mouse_id}",
            level="INFO",
            mouse_id=mouse_id,
            mouse_tag=mouse_tag,
            move=move,
            current_position=current_pos,
            next_position=next_position,
            reasoning=reasoning,
            cheese_target=closest_cheese,
            distance_to_cheese=distance_to_cheese
        )
        
        return {
            "mouseId": mouse_id,
            "move": move,
            "reasoning": reasoning
        }
        
    except Exception as e:
        logger.error(f"Error processing mouse move request: {str(e)}")
        
        # Log de l'erreur
        log_service.add_custom_log(
            message=f" Thread {mouse_tag} - Error processing move request: {str(e)}",
            level="ERROR",
            mouse_id=mouse_id,
            mouse_tag=mouse_tag,
            error=str(e),
            action="error_fallback"
        )
        
        # Fallback to random movement
        import random
        available_moves = request.get("availableMoves", ["north", "south", "east", "west"])
        move = random.choice(available_moves)
        
        # Log du mouvement de fallback
        log_service.add_custom_log(
            message=f" Thread {mouse_tag} - Using random fallback move: {move}",
            level="WARNING",
            mouse_id=mouse_id,
            mouse_tag=mouse_tag,
            move=move,
            action="random_fallback"
        )
        
        return {
            "mouseId": request.get("mouseId", "unknown"),
            "move": move,
            "reasoning": f"Error occurred, using random movement: {str(e)}"
        }


def _position_to_direction(current_pos: List[int], next_pos: List[int]) -> str:
    """Convert position change to direction string."""
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    if dx > 0:
        return "east"
    elif dx < 0:
        return "west"
    elif dy > 0:
        return "south"
    elif dy < 0:
        return "north"
    else:
        return "north"  # Default fallback


@router.post("/cleanup")
async def cleanup_mouse_services() -> Dict[str, str]:
    """Clean up all mouse AI service instances."""
    global mouse_ai_services
    count = len(mouse_ai_services)
    mouse_ai_services.clear()
    logger.info(f"Cleaned up {count} mouse AI service instances")
    return {"status": "cleaned", "instances_removed": str(count)}


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Mouse AI service is running",
        "active_instances": str(len(mouse_ai_services))
    }
