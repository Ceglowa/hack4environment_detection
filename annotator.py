from collections import defaultdict
import pandas as pd

categories = defaultdict(set,
                         {'brands': {'adidas',
                                     'aldi',
                                     'amazon',
                                     'apple',
                                     'asahi',
                                     'bewleys',
                                     'budweiser',
                                     'bulmers',
                                     'burgerking',
                                     'cadburys',
                                     'cafe_nero',
                                     'camel',
                                     'carlsberg',
                                     'centra',
                                     'circlek',
                                     'coke',
                                     'coles',
                                     'colgate',
                                     'corona',
                                     'costa',
                                     'doritos',
                                     'drpepper',
                                     'dunnes',
                                     'duracell',
                                     'durex',
                                     'esquires',
                                     'frank_and_honest',
                                     'fritolay',
                                     'gatorade',
                                     'gillette',
                                     'guinness',
                                     'haribo',
                                     'heineken',
                                     'insomnia',
                                     'kellogs',
                                     'kfc',
                                     'lego',
                                     'lidl',
                                     'lindenvillage',
                                     'lolly_and_cookes',
                                     'loreal',
                                     'lucozade',
                                     'marlboro',
                                     'mars',
                                     'mcdonalds',
                                     'nescafe',
                                     'nestle',
                                     'nike',
                                     'obriens',
                                     'pepsi',
                                     'powerade',
                                     'redbull',
                                     'ribena',
                                     'sainsburys',
                                     'samsung',
                                     'spar',
                                     'starbucks',
                                     'stella',
                                     'subway',
                                     'supermacs',
                                     'supervalu',
                                     'tayto',
                                     'tesco',
                                     'thins',
                                     'volvic',
                                     'waitrose',
                                     'walkers',
                                     'wilde_and_greene',
                                     'woolworths',
                                     'wrigleys'},
                          'food': {'aluminium_foil',
                                   'crisp_large',
                                   'crisp_small',
                                   'foodOther',
                                   'glass_jar',
                                   'glass_jar_lid',
                                   'napkins',
                                   'paperFoodPackaging',
                                   'pizza_box',
                                   'plasticCutlery',
                                   'plasticFoodPackaging',
                                   'styrofoam_plate',
                                   'sweetWrappers'},
                          'dumping': {'large', 'medium', 'small'},
                          'industrial': {'bricks', 'chemical', 'oil', 'tape'},
                          'coastal': {'balloons',
                                      'buoys',
                                      'coastal_other',
                                      'degraded_lighters',
                                      'degraded_plasticbag',
                                      'degraded_plasticbottle',
                                      'degraded_straws',
                                      'fishing_gear_nets',
                                      'lego',
                                      'macroplastics',
                                      'mediumplastics',
                                      'microplastics',
                                      'rope_large',
                                      'rope_medium',
                                      'rope_small',
                                      'shotgun_cartridges',
                                      'styro_large',
                                      'styro_medium',
                                      'styro_small'},
                          'sanitary': {'condoms',
                                       'deodorant',
                                       'facemask',
                                       'gloves',
                                       'hand_sanitiser',
                                       'menstral',
                                       'nappies',
                                       'sanitaryOther',
                                       'tooth_brush',
                                       'tooth_pick',
                                       'wetwipes'},
                          'alcohol': {'alcoholOther',
                                      'alcohol_plastic_cups',
                                      'beerBottle',
                                      'beerCan',
                                      'bottleTops',
                                      'brokenGlass',
                                      'paperCardAlcoholPackaging',
                                      'pint',
                                      'plasticAlcoholPackaging',
                                      'six_pack_rings',
                                      'spiritBottle',
                                      'wineBottle'},
                          'smoking': {'butts',
                                      'cigaretteBox',
                                      'filterbox',
                                      'filters',
                                      'lighters',
                                      'skins',
                                      'smokingOther',
                                      'smoking_plastic',
                                      'tobaccoPouch',
                                      'vape_oil',
                                      'vape_pen'},
                          'other': {'bags_litter',
                                    'balloons',
                                    'batteries',
                                    'books',
                                    'cable_tie',
                                    'dogshit',
                                    'dump',
                                    'ear_plugs',
                                    'elec_large',
                                    'elec_small',
                                    'election_posters',
                                    'forsale_posters',
                                    'hair_tie',
                                    'metal',
                                    'other',
                                    'overflowing_bins',
                                    'paper',
                                    'plastic',
                                    'random_litter',
                                    'stationary',
                                    'tyre',
                                    'washing_up'},
                          'softdrinks': {'bottleLabel',
                                         'bottleLid',
                                         'energy_can',
                                         'fizzyDrinkBottle',
                                         'ice_tea_bottles',
                                         'ice_tea_can',
                                         'juice_bottles',
                                         'juice_cartons',
                                         'juice_packet',
                                         'milk_bottle',
                                         'milk_carton',
                                         'paper_cups',
                                         'plastic_cup_tops',
                                         'plastic_cups',
                                         'sportsDrink',
                                         'straws',
                                         'tinCan',
                                         'waterBottle'},
                          'coffee': {'coffeeCups', 'coffeeLids',
                                     'coffeeOther'}})

better_cats = [f'{cat}.{subcat}' for cat in categories for subcat in categories[cat]]

print('\n'.join(better_cats))




