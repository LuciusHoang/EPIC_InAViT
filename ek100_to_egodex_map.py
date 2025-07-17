# scripts/ek100_to_egodex_map.py

# Maps EPIC-KITCHENS class indices → your 24 EgoDex kitchen actions

ek100_to_egodex_map = {
    142: "wash_kitchen_dishes",                  # wash + dish
    3145: "wash_put_away_dishes",                # wash + dish, put + dish
    1023: "clean_tableware",                     # clean + plate
    972: "clean_cups",                           # clean + cup
    3812: "clean_surface",                       # clean + surface
    3581: "wipe_kitchen_surfaces",               # wipe + counter
    3578: "wipe_screen",                         # wipe + screen
    832: "stack_unstack_plates",                 # put + plate
    914: "stack_unstack_bowls",                  # put + bowl
    820: "stack_unstack_cups",                   # put + cup
    1333: "stack_unstack_tupperware",            # put + tupperware
    1598: "stack_remove_jenga",                  # take + block
    233: "pour",                                 # pour + water
    2852: "scoop_dump_ice",                      # scoop + ice
    1403: "load_dispense_ice",                   # take + ice
    1154: "boil_serve_egg",                      # cook + egg
    1674: "put_toothpaste_on_toothbrush",        # put + toothpaste
    723: "pick_place_food",                      # take + food
    2132: "make_sandwich",                       # prepare + sandwich
    1233: "wrap_unwrap_food",                    # wrap + food
    1112: "add_remove_lid",                      # put + lid
    1312: "open_close_insert_remove_tupperware", # open + tupperware
    1410: "peel_place_sticker",                  # peel + label
    1943: "fry_bread"                            # cook + bread
}
