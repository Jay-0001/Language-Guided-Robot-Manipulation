import re

# --------------------------------------------
# Vocabulary (expand anytime)
# --------------------------------------------

COLORS = [
    "red", "green", "blue", "yellow", "black", "white",
    "pink", "purple", "orange", "gray"
]

OBJECTS = [
    "sphere", "ball", "cube", "block", "object", "item",
    "robot arm", "arm", "franka", "panda"
]

ACTIONS = [
    "pick", "grab", "lift", "push", "place", "move"
]

RELATIONS = {
    "left": ["left", "to the left", "on the left"],
    "right": ["right", "to the right", "on the right"],
    "near": ["near", "close to", "next to", "beside"],
    "front": ["in front", "front of"],
    "behind": ["behind"],
}

# --------------------------------------------
# Helper functions
# --------------------------------------------

def extract_from_list(sentence, vocab_list):
    for word in vocab_list:
        if word in sentence:
            return word
    return None

def extract_relation(sentence):
    for key, phrases in RELATIONS.items():
        for p in phrases:
            if p in sentence:
                return key
    return None

# --------------------------------------------
# Main parser function
# --------------------------------------------

def parse_query(query: str):
    q = query.lower()

    color = extract_from_list(q, COLORS)
    obj = extract_from_list(q, OBJECTS)
    action = extract_from_list(q, ACTIONS)
    relation = extract_relation(q)

    # Build grounding phrase
    if color and obj:
        target_phrase = f"{color} {obj}"
    elif obj:
        target_phrase = obj
    else:
        target_phrase = None

    return {
        "object": obj,
        "color": color,
        "action": action,
        "relation": relation,
        "target_phrase": target_phrase
    }

# --------------------------------------------
# Interactive loop
# --------------------------------------------

def language_module():
    print("Which object should the robot pick ?")

    user_query = input("User: ")
    parsed = parse_query(user_query)
    print("Parsed Intent:", parsed, "\n")
    return parsed["target_phrase"]+ "."
