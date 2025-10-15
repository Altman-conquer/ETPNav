import json
import os
import time
from typing import List

import requests


def _query_text_llm(prompt):
    base_url = 'http://127.0.0.1:5001'

    payload = {
        "instruction": json.dumps(prompt)
    }
    kimi_response = requests.post(f"{base_url}/query_text_llm", json=payload).json()
    print("query_kimi response:", kimi_response)
    return kimi_response['result']


def parse_object_goal_instruction(language_instr):
    """
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    text = _query_text_llm([
        {
            "role": "user",
            "content": "go to the kitchen and then go to the toilet",
        },
        {
            "role": "assistant",
            "content": "kitchen, toilet"
        },
        {
            "role": "user",
            "content": "go to the chair and then go to another chair"
        },
        {
            "role": "assistant",
            "content": "chair, chair"
        },
        {
            "role": "user",
            "content": "navigate to the green sofa and turn right and find several chairs, finally go to the painting"
        },
        {
            "role": "assistant",
            "content": "green sofa, chairs, painting"
        },
        {
            "role": "user",
            "content": "approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen"
        },
        {
            "role": "assistant",
            "content": "window, television, oven, kitchen"
        },
        {
            "role": "user",
            "content": "walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed"
        },
        {
            "role": "assistant",
            "content": "plant, table, bedroom, bed"
        },
        {
            "role": "user",
            "content": "go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room"
        },
        {
            "role": "assistant",
            "content": "stairs, room, book shelf, table, next room"
        },
        {
            "role": "user",
            "content": "Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances"
        },
        {
            "role": "assistant",
            "content": "table, cushion, column, appliances"
        },
        {
            "role": "user",
            "content": "Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table"
        },
        {
            "role": "assistant",
            "content": "chair, table"
        },
        {
            "role": "user",
            "content": "move in between the couch and bookshelf"
        },
        {
            "role": "assistant",
            "content": "couch, bookshelf"
        },
        {
            "role": "user",
            "content": "Walk forward down the hall past the table on the left. Continue going forward to you reach the open doorway to the left."
        },
        {
            "role": "assistant",
            "content": "hall, table, open doorway"
        },
        {
            "role": "user",
            "content": language_instr
        }
    ])
    return [x.strip() for x in text.split(",")]


def parse_spatial_instruction(language_instr):
    text = _query_text_llm([
        {"role": "user", "content": "move a bit to the right of the refrigerator"},
        {"role": "assistant", "content": "robot.move_to_right('refrigerator')"},
        {"role": "user", "content": "move in between the couch and bookshelf"},
        {"role": "assistant", "content": "robot.move_in_between('couch', 'bookshelf')"},
        {"role": "user", "content": "move in the middle of the stairs and oven"},
        {"role": "assistant", "content": "robot.move_in_between('stairs', 'oven')"},
        {"role": "user", "content": "face the toilet"},
        {"role": "assistant", "content": "robot.face('toilet')"},
        {"role": "user", "content": "move to the south side of the table"},
        {"role": "assistant", "content": "robot.move_south('table')"},
        {"role": "user", "content": "move to the west of the chair"},
        {"role": "assistant", "content": "robot.move_west('chair')"},
        {"role": "user", "content": "move to the east of the chair"},
        {"role": "assistant", "content": "robot.move_east('chair')"},
        {"role": "user", "content": "turn left 60 degrees"},
        {"role": "assistant", "content": "robot.turn(-60)"},
        {"role": "user", "content": "turn right 20 degrees"},
        {"role": "assistant", "content": "robot.turn(20)"},
        {"role": "user", "content": "find any chairs in the environment"},
        {"role": "assistant", "content": "robot.move_to_object('chair')"},
        {"role": "user", "content": "come to the table"},
        {"role": "assistant", "content": "robot.move_to_object('table')"},
        {"role": "user", "content": "with the television on your left"},
        {"role": "assistant", "content": "robot.with_object_on_left('television')"},
        {"role": "user", "content": "with the toilet to your right, turn left 30 degrees"},
        {"role": "assistant", "content": "robot.with_object_on_right('toilet')\nrobot.turn(-30)"},
        {"role": "user", "content": "with the toilet to your left, turn right 30 degrees"},
        {"role": "assistant", "content": "robot.with_object_on_left('toilet')\nrobot.turn(30)"},
        {"role": "user", "content": "with the television behind you"},
        {"role": "assistant", "content": "robot.face('television')\nrobot.turn(180)"},
        {"role": "user",
         "content": "go to the south of the table in front of you, with the table on your left, turn right 45 degrees"},
        {"role": "assistant",
         "content": "robot.move_south('table')\nrobot.with_object_on_left('table')\nrobot.turn(45)"},
        {"role": "user", "content": "move forward for 3 meters"},
        {"role": "assistant", "content": "robot.move_forward(3)"},
        {"role": "user", "content": "move right 2 meters"},
        {"role": "assistant", "content": "robot.turn(90)\nrobot.move_forward(2)"},
        {"role": "user", "content": "move left 3 meters"},
        {"role": "assistant", "content": "robot.turn(-90)\nrobot.move_forward(3)"},
        {"role": "user", "content": "move back and forth between the chair and the table 3 times"},
        {"role": "assistant",
         "content": "pos1 = robot.get_pos('chair')\npos2 = robot.get_pos('table')\nfor i in range(3):\n    robot.move_to(pos1)\n    robot.move_to(pos2)"},
        {"role": "user", "content": "go to the television"},
        {"role": "assistant", "content": "robot.move_to_object('television')"},
        {"role": "user", "content": "monitor the chair from 4 view points"},
        {"role": "assistant",
         "content": "contour = robot.get_contour('chair')\nnp.random.shuffle(contour)\nfor pos in contour[:4]:\n    robot.move_to(pos)\n    robot.face('chair')"},
        {"role": "user", "content": "navigate to 3 meters right of the table"},
        {"role": "assistant",
         "content": "robot.move_to_right('table')\nrobot.face('table')\nrobot.turn(180)\nrobot.move_forward(3)"},
        {"role": "user", "content": "turn west"},
        {"role": "assistant", "content": "robot.turn_absolute(-90)"},
        {"role": "user", "content": "turn east"},
        {"role": "assistant", "content": "robot.turn_absolute(90)"},
        {"role": "user", "content": "turn south"},
        {"role": "assistant", "content": "robot.turn_absolute(180)"},
        {"role": "user", "content": "turn north"},
        {"role": "assistant", "content": "robot.turn_absolute(0)"},
        {"role": "user", "content": "turn east and then turn left 90 degrees"},
        {"role": "assistant", "content": "robot.turn_absolute(90)\nrobot.turn(-90)"},
        {"role": "user", "content": "move 3 meters north of the chair"},
        {"role": "assistant", "content": "robot.move_north('chair')\nrobot.move_forward(3)"},
        {"role": "user", "content": "move 1 meters south of the chair"},
        {"role": "assistant", "content": "robot.move_south('chair')\nrobot.move_forward(3)"},
        {"role": "user", "content": "move 3 meters west"},
        {"role": "assistant", "content": "robot.turn_absolute(-90)\nrobot.move_forward(3)"},
        {"role": "user", "content": language_instr},
    ])
    return text

def get_navigation_prompt(instruction: str, map_img_path: str, waypoint_count: int, waypoint_dirs: List[str] = []):
    directions_str = "; ".join(f"{i}={d}" for i, d in enumerate(waypoint_dirs))

    promptv1 = f"""
            Indoor map navigation — Next Waypoint Selection and Relative Directions
            Legend:
            - Red points: objects with label at top-right (e.g., "table", "sink").
            - Green circles with white text: navigation waypoints (e.g., "0", "1").
            - White circle with blue arrow: agent position and rotation.
            - Blue circles with connected line: agent path (or trajectory).

            Task:
            - Choose exactly one next waypoint based on the given command.
            - Additionally, report the relative direction of each visible green waypoint w.r.t. the agent's facing.
              Use only one of [front, front-left, left, back-left, back, back-right, right, front-right].

            Selection rules:
            1) Choose exactly one waypoint from the green waypoints.
            2) If all instructions in the command have already been completed at the agent's current position (no further movement needed), output waypoint "-1" to terminate.
            3) If the command is unclear or ambiguous, choose the waypoint closest to the agent.
            4) Prefer waypoints that move toward the referenced objects and respect left/right/front/back relations.
            5) Choose only the immediate next waypoint; do not output a multi-step plan.
            6) For Waypoint Directions, include every visible green waypoint exactly once as "id=direction", separated by "; ".
            7) In Reason, analyze each waypoint in the format "id: short reason", then add a final short sentence explaining why the chosen id is best.

            Output format (strict, only these three lines):
            Waypoint Chosen: <waypoint_id>/-1
            Waypoint Directions: <id=direction; id=direction; ...>
            Reason: <id: reason; id: reason; ...; final choice rationale>

            The image is the map and the command: "{instruction}"
            Besides, there are two waypoint in the map image, their id are 0, 1, respectively. And the agent is facing to the top of the image.

            Template Response (ongoing navigation):
            Waypoint Chosen: 3
            Waypoint Directions: 0=front-left; 1=left; 2=back; 3=front
            Reason: 0: farther and not aligned with target; 1: left but detours; 2: behind and away from target corridor; 3: forward and closest toward the doorway. Choose 3 because it best follows the command and minimizes travel.

            Template Response (terminate):
            Waypoint Chosen: -1
            Waypoint Directions: 0=left; 1=front
            Reason: 0: not needed; 1: not needed; all instructions already satisfied at current position, so terminate.
            """

    promptv2 = f"""
            Indoor Map Navigation — Next Waypoint Selection with Agent-Centric Relative Directions

            Legend:
            - Red points: labeled objects (e.g., "table", "sink").
            - Green circles with white text: navigation waypoints (IDs: e.g., "0", "1").
            - White circle with blue arrow: agent's current position and **facing direction**.
            - Blue line: agent's past trajectory.

            Important: All directional terms (front, left, etc.) must be interpreted **relative to the agent’s current facing direction**, NOT the image’s top.  
            Assume the agent’s forward direction is where the blue arrow points. Define the 8 cardinal-relative directions as:
            - front: directly ahead (±22.5°)
            - front-right: 22.5°–67.5° right of front
            - right: 67.5°–112.5° right of front
            - back-right: 112.5°–157.5° right of front
            - back: directly behind (±22.5° from rear)
            - back-left: 112.5°–157.5° left of front
            - left: 67.5°–112.5° left of front
            - front-left: 22.5°–67.5° left of front

            Task:
            1. Select exactly one next waypoint (from visible green waypoints) based on the command: "{instruction}".
            2. For **every visible green waypoint**, report its direction relative to the agent using only one of:  
               [front, front-left, left, back-left, back, back-right, right, front-right].
            3. If the task is already complete at the agent’s current location, output waypoint "-1" to terminate.

            Selection Rules:
            - Choose only one waypoint ID (or -1). Do not plan multiple steps.
            - If the command is ambiguous or underspecified, pick the **closest** visible waypoint.
            - Prefer waypoints that move the agent toward referenced objects while respecting spatial relations (e.g., “go to the sink on your right” → favor right-side waypoints near sink).
            - All visible green waypoints must appear exactly once in the "Waypoint Directions" line.

            Output Format (STRICT — exactly three lines, no extra text):
            Waypoint Chosen: <ID or -1>
            Waypoint Directions: <id=direction; id=direction>
            Reason: <id: short justification; id: short justification; final rationale for chosen ID>

            Note: In this map, only two green waypoints exist: IDs 0 and 1. The agent is currently facing the top of the image (i.e., image-up = agent-front).

            Example (ongoing navigation):
            Waypoint Chosen: 1
            Waypoint Directions: 0=left; 1=front
            Reason: 0: left but leads away from target kitchen; 1: directly ahead and closer to the sink. Choose 1 because it aligns with the command to approach the sink.

            Example (termination):
            Waypoint Chosen: -1
            Waypoint Directions: 0=back-left; 1=right
            Reason: 0: not required; 1: not required; agent is already at the dining table as instructed. Terminate since task is complete.
            """

    promptv3 = f"""
            Indoor Map Navigation — Next Waypoint Selection with Agent-Relative Directions

            Map Legend (for reference only):

            Red dots: labeled objects (e.g., “table”, “sink”).
            Green circles with white ID: navigation waypoints (e.g., “0”, “1”).
            White circle + blue arrow: agent’s current position and heading (the agent is located in the center of the map and facing upward).
            Blue path line: agent’s past trajectory.
            Task:
            Given the indoor map and command, select exactly one next waypoint to move toward or output -1 if no movement is needed.

            Additionally, for every visible green waypoint, report its direction relative to the agent’s current facing direction using exactly one of:

            front, front-left, left, back-left, back, back-right, right, front-right 

            Selection Rules:

            Output only one waypoint ID (or -1).
            If all command goals are already satisfied at current position → output -1.
            If command is ambiguous → choose the closest waypoint to agent.
            Prefer waypoints that move toward referenced objects, respecting spatial relations (left/right/front/back relative to agent).
            Only plan the immediate next step — no multi-hop paths.
            In “Waypoint Directions”, list every visible green waypoint once, in format: id=direction, separated by ; .
            In “Reason”, analyze each waypoint briefly: id: reason, then conclude with a single sentence justifying the final choice.
            Output Format (STRICT — ONLY THESE THREE LINES):
            Waypoint Chosen: <waypoint_id>/-1
            Waypoint Directions: <id=direction; id=direction; ...>
            Reason: <id: reason; id: reason; ...; final rationale>

            Input Context:

            The image is the indoor map.
            Command: "{instruction}"
            Visible waypoints: {', '.join([str(i) for i in range(waypoint_count)])}.
            {'Visible waypoint directions: ' + directions_str + '.' if len(waypoint_dirs) > 0 else ''}
            Agent is facing top of image, which defines “front” direction.
            The agent is currently located in the center of the map and facing upward.
            The top side of the image corresponds to the agent's front side.
            The right side of the image corresponds to the agent's right side.
            The left side of the image corresponds to the agent's left side.
            The bottom side of the image corresponds to the agent's back side.

            Template Response (ongoing navigation):
            Waypoint Chosen: 3
            Waypoint Directions: 0=front-left; 1=left; 2=back; 3=front
            Reason: 0: farther and not aligned with target; 1: left but detours; 2: behind and away from target corridor; 3: forward and closest toward the doorway. Choose 3 because it best follows the command and minimizes travel.

            Template Response (terminate):
            Waypoint Chosen: -1
            Waypoint Directions: 0=left; 1=front
            Reason: 0: not needed; 1: not needed; all instructions already satisfied at current position, so terminate.
            """

    prompt = f"""
            Indoor Map Navigation — Next Waypoint Selection with Agent-Relative Directions

            Map Legend (for reference only):

            Red dots: labeled objects (e.g., “table”, “sink”).
            Green circles with white ID: navigation waypoints (e.g., “0”, “1”).
            White circle + blue arrow: agent’s current position and heading (the agent is located in the center of the map and facing upward).
            Blue path line: agent’s past trajectory.
            Task:
            Given the indoor map and command, select exactly one next waypoint to move toward or output -1 if no movement is needed.

            Additionally, for every visible green waypoint, report its direction relative to the agent’s current facing direction using exactly one of:

            front, front-left, left, back-left, back, back-right, right, front-right 

            Selection Rules:

            Output only one waypoint ID (or -1).
            If all command goals are already satisfied at current position → output -1.
            If command is ambiguous → choose the closest waypoint to agent.
            Prefer waypoints that move toward referenced objects, respecting spatial relations (left/right/front/back relative to agent).
            Only plan the immediate next step — no multi-hop paths.
            In “Waypoint Directions”, list every visible green waypoint once, in format: id=direction, separated by ; .
            In “Reason”, analyze each waypoint briefly: id: reason, then conclude with a single sentence justifying the final choice.
            Output Format (STRICT — ONLY THESE THREE LINES):
            Waypoint Chosen: <waypoint_id>/-1
            Waypoint Directions: <id=direction; id=direction; ...>
            Reason: <id: reason; id: reason; ...; final rationale>

            Input Context:

            The image is the indoor map.
            Command: "{instruction}"
            Visible waypoints: {', '.join([str(i) for i in range(waypoint_count)])}.
            {'Visible waypoint directions: ' + directions_str + '.' if len(waypoint_dirs) > 0 else ''}
            Agent is facing top of image, which defines “front” direction.
            The agent is currently located in the center of the map and facing upward.
            The top side of the image corresponds to the agent's front side.
            The right side of the image corresponds to the agent's right side.
            The left side of the image corresponds to the agent's left side.
            The bottom side of the image corresponds to the agent's back side.

            Template Response (ongoing navigation):
            Waypoint Chosen: 3
            Waypoint Directions: 0=front-left; 1=left; 2=back; 3=front

            Template Response (terminate):
            Waypoint Chosen: -1
            Waypoint Directions: 0=left; 1=front
            """

    return prompt

def navigation_using_llm(instruction: str, map_img_path: str, waypoint_count: int, waypoint_dirs: List[str] = []):
    if not os.path.isabs(map_img_path):
        map_img_path = os.path.abspath(map_img_path)
    base_url = 'http://127.0.0.1:5001'

    prompt = get_navigation_prompt(instruction, map_img_path, waypoint_count, waypoint_dirs)

    qwen_payload = {
        "image_paths": [map_img_path],
        "prompt": prompt
    }

    qwen_response = requests.post(f"{base_url}/query_qwen", json=qwen_payload).json()

    if qwen_response['status'] != 'success':
        print(f"Error: {qwen_response}, retrying...")
        time.sleep(0.5)
        qwen_response = requests.post(f"{base_url}/query_qwen", json=qwen_payload).json()

    print("query_qwen response:", qwen_response)
    return qwen_response['result']


if __name__ == '__main__':
    """
    Walk forward down the hall past the table on the left. Continue going forward to you reach the open doorway to the left. Turn left and walk forward, stop in front of the doorway to the bathroom. Turn right and enter that hallway stop and wait in front of the sink on your right.
    """

    instruction = ('Walk forward down the hall past the table on the left.'
                   ' Continue going forward to you reach the open doorway to the left. '
                   'Turn left and walk forward, stop in front of the doorway to the bathroom. '
                   'Turn right and enter that hallway stop and wait in front of the sink on your right.')

    # text = parse_spatial_instruction(instruction)
    # print(text)

    """
    all commands:
    robot.turn_left() # turn left 90 degrees
    robot.turn_right() # turn right 90 degrees
    robot.goto('object_name', side='left') # goto the object with the object located at the {side} of the object, side = left/right/front/back, default is front
    robot.goto_between('object_a', 'object_b') # goto the position in between object_a and object_b

    robot.goto('table', side='left') # side = left/right/front/back, default is front
    robot.goto('doorway', side='left')
    robot.turn_right()
    robot.goto('sink', side='right')
    """

    # text = parse_object_goal_instruction(instruction)
    # print(text)

    navigation_using_llm(
        instruction, waypoint_count=3,
        map_img_path='/home/zhandijia/DockerData/zhandijia-root/ETPNav/tmp/rgb_map/final_cropped_rotated_rgb_map.jpg')
