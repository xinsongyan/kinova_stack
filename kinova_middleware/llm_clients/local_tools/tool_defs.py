from __future__ import annotations


CHECK_SORTING_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_sorting_status",
        "description": "Analyze the scene to see which cubes are sorted, which are at start, and which are out of workspace.",
        "parameters": {"type": "object", "properties": {}},
    },
}

CHECK_STACKING_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_stacking_status",
        "description": "Analyze the cubes and report which cube is stacked on top of which other cube based on z height and xy alignment.",
        "parameters": {"type": "object", "properties": {}},
    },
}

GRAB_SHAPES_TARGETS = ("box", "sphere", "cylinder")

VERIFY_OBJECT_LIFT_TOOL = {
    "type": "function",
    "function": {
        "name": "verify_object_lift",
        "description": "Check whether a named object has been lifted high enough after a grasp.",
        "parameters": {
            "type": "object",
            "properties": {
                "body_name": {
                    "type": "string",
                    "description": "The object name to verify, for example box, sphere, or cylinder.",
                },
                "min_height": {
                    "type": "number",
                    "description": "Minimum Z height that counts as a successful lift. Default is 0.12.",
                },
            },
            "required": ["body_name"],
        },
    },
}

FINISH_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_task",
        "description": "Call this when the task is truly complete or cannot continue safely. This is the only normal way to end the agent loop.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short final summary of what was completed or why the task stopped.",
                }
            },
            "required": ["summary"],
        },
    },
}
