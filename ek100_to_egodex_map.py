# ek100_to_egodex_map.py

# Maps EPIC-KITCHENS class indices (0–96) to 24 EgoDex class labels
# Unmapped indices will return "Unknown" in your evaluation
ek100_to_egodex_map = {
    0: "wash_kitchen_dishes",
    1: "wash_put_away_dishes",
    2: "clean_tableware",
    3: "clean_cups",
    4: "clean_surface",
    5: "wipe_kitchen_surfaces",
    6: "wipe_screen",
    7: "stack_unstack_plates",
    8: "stack_unstack_bowls",
    9: "stack_unstack_cups",
    10: "stack_unstack_tupperware",
    11: "stack_remove_jenga",
    12: "pour",
    13: "scoop_dump_ice",
    14: "load_dispense_ice",
    15: "boil_serve_egg",
    16: "put_toothpaste_on_toothbrush",
    17: "pick_place_food",
    18: "make_sandwich",
    19: "wrap_unwrap_food",
    20: "add_remove_lid",
    21: "open_close_insert_remove_tupperware",
    22: "peel_place_sticker",
    23: "fry_bread"
}
