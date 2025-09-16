import json

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


if __name__ == '__main__':
    instruction = 'Walk forward down the hall past the table on the left. Continue going forward to you reach the open doorway to the left. Turn left and walk forward, stop in front of the doorway to the bathroom. Turn right and enter that hallway stop and wait in front of the sink on your right.'

    text = parse_spatial_instruction(instruction)
    print(text)

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

    text = parse_object_goal_instruction(instruction)
    print(text)
