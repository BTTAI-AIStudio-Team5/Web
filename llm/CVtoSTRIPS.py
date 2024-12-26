#template by ivi fung

import ast  # To safely evaluate the string representation of a tuple/list

def get_initial_state(info):
    # List to store all objects with category, bounding box, and box identification
    objects = []

    # Process the info list directly
    for item in info:
        # Extract category and bounding box coordinates
        category = item['category']
        bbox = item['bbox']
        
        # Determine if the category is a box by checking if it starts with 'box'
        is_box = category.lower().startswith('box')
        
        # Add the category, bounding box, and box identification to objects list
        objects.append({
            'category': category,
            'bounding_box': bbox,
            'is_box': is_box
        })

    # Sort the objects list based on the x-coordinate (our right to left, monkey's left to right)
    sorted_objects = sorted(objects, key=lambda obj: obj["bounding_box"][0], reverse=True)

    # Build the initial state string
    initial_state = "Initial state: "
    state_parts = []
    # Process sorted objects to build STRIPS format
    for obj in sorted_objects:
        if obj['category'] == 'monkey':
            # Get position relative to nearest box
            monkey_x = obj['bounding_box'][0]
            state_parts.append(f"At({int(monkey_x)})")
            # Add default level
            monkey_y = obj['bounding_box'][1]
            state_parts.append(f"Level({'low' if monkey_y > 275 else 'high'})")
            
        elif obj['category'] == 'banana':
            # Similar logic for banana position
            banana_x = obj['bounding_box'][0]
            banana_y = obj['bounding_box'][1]
            state_parts.append(f"BananasAtHigh({int(banana_x)})" if banana_y < 430 else f"BananasAtLow({int(banana_x)})")
        elif obj['is_box']:
            box_letter = obj['category'][-1]
            box_x = obj['bounding_box'][0]
            state_parts.append(f"BoxAt({int(box_x)})")
    
    initial_state += ", ".join(state_parts)
    return initial_state

def generate_strips_plan(initial_state, filename):
    strips_planner_file_text = f"""{initial_state}
Goal state: Have(Bananas)

Actions:
            // move from X to Y
            Move(X, Y)
            Preconditions:  At(X), Level(low)
            Postconditions: !At(X), At(Y)
            
            // climb up on the box
            ClimbUp(Location)
            Preconditions:  At(Location), BananasAtHigh(Location), Level(low)
            Postconditions: Level(high), !Level(low)
            
            // climb down from the box
            ClimbDown(Location)
            Preconditions:  At(Location), Level(high)
            Postconditions: Level(low), !Level(high)
            
            // move monkey and box from X to Y
            MoveBox(X, Y)
            Preconditions:  At(X), BoxAt(X), Level(low)
            Postconditions: BoxAt(Y), !BoxAt(X), At(Y), !At(X)
            
            // take the bananas when at a high level 
            TakeBananasHigh(Location)
            Preconditions:  BananasAtHigh(Location), At(Location), Level(high)
            Postconditions: Have(Bananas)

            // take the bananas when at a low level
            TakeBananasLow(Location)
            Preconditions:  BananasAtLow(Location), At(Location), Level(low)
            Postconditions: Have(Bananas)
"""
    # Save the STRIPS problem to a file
    with open(f'{filename}', 'w') as f:
        f.write(strips_planner_file_text)


if __name__ == "__main__":
    info = sorted([{'category': 'banana', 'bbox': [974.0, 476.5]}, {'category': 'boxC', 'bbox': [1023.5, 476.0]}, {'category': 'boxB', 'bbox': [475.0, 476.0]}, {'category': 'boxA', 'bbox': [125.0, 476.0]}, {'category': 'monkey', 'bbox': [695.0, 400.0]}], key=lambda x: x['bbox'][0], reverse=True)
    print(get_initial_state(info))


    info =  [{'category': 'boxA', 'bbox': [725.0, 476.0]}, {'category': 'boxC', 'bbox': [625.0, 476.0]}, 
             {'category': 'boxB', 'bbox': [525.0, 476.0]}, {'category': 'banana', 'bbox': [725.0, 426.5]}, 
             {'category': 'monkey', 'bbox': [295.0, 400.0]}]
    


