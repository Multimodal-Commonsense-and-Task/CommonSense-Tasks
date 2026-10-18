import re
import json
import nltk
import random
from classes import *
from nltk.corpus import wordnet as wn
from openai import OpenAI
from tqdm import tqdm

random.seed(42)
nltk.download('wordnet')

prompt_json_path = './prompts/fake_f_ImageNet_200.json'
dataset_count = ImageNet_count; imagenet=True

places365_list = ['valley', 'butte', 'gymnasium', 'kasbah', 'moat', 'archive', 'rice_paddy', 'landfill', 'lighthouse', 'dressing_room', 'sky', 'mausoleum', 'roof_garden', 'yard', 'airfield', 'clean_room', 'campsite', 'mosque', 'train_interior', 'entrance_hall', 'lecture_room', 'childs_room', 'boardwalk', 'closet', 'hotel', 'art_school', 'basement', 'diner', 'mansion', 'oilrig', 'pub', 'lock_chamber', 'hayfield', 'wet_bar', 'residential_neighborhood', 'catacomb', 'conference_center', 'bridge', 'art_studio', 'water_tower', 'wave', 'basketball_court', 'waiting_room', 'desert_road', 'elevator_shaft', 'mountain_path', 'bus_station', 'home_office', 'baseball_field', 'butchers_shop', 'beach', 'village', 'army_base', 'underwater', 'staircase', 'hospital_room', 'canal', 'aquarium', 'throne_room', 'laundromat', 'alley', 'utility_room', 'bar', 'corral', 'glacier', 'islet', 'physics_laboratory', 'soccer_field', 'marsh', 'corridor', 'train_station', 'youth_hostel', 'temple', 'construction_site', 'auto_showroom', 'hotel_room', 'bow_window', 'manufactured_home', 'office_cubicles', 'recreation_room', 'pond', 'slum', 'motel', 'tree_house', 'water_park', 'crevasse', 'field', 'swimming_pool', 'beach_house', 'ice_shelf', 'playground', 'landing_deck', 'sauna', 'chalet', 'tundra', 'biology_laboratory', 'arcade', 'forest_path', 'bowling_alley', 'park', 'martial_arts_gym', 'botanical_garden', 'courtyard', 'booth', 'trench', 'storage_room', 'schoolhouse', 'shower', 'greenhouse', 'ball_pit', 'racecourse', 'forest_road', 'lobby', 'airport_terminal', 'bakery', 'lagoon', 'hot_spring', 'science_museum', 'dining_room', 'downtown', 'topiary_garden', 'living_room', 'athletic_field', 'phone_booth', 'bullring', 'doorway', 'bathroom', 'office', 'plaza', 'pharmacy', 'crosswalk', 'attic', 'sushi_bar', 'hunting_lodge', 'rainforest', 'playroom', 'fastfood_restaurant', 'aqueduct', 'vegetable_garden', 'pet_shop', 'escalator', 'shopfront', 'restaurant', 'junkyard', 'general_store', 'coffee_shop', 'chemistry_lab', 'castle', 'carrousel', 'operating_room', 'pizzeria', 'amusement_arcade', 'restaurant_patio', 'igloo', 'corn_field', 'embassy', 'bazaar', 'cemetery', 'delicatessen', 'arch', 'industrial_area', 'elevator_lobby', 'boat_deck', 'hardware_store', 'tower', 'engine_room', 'kitchen', 'palace', 'heliport', 'waterfall', 'pantry', 'harbor', 'watering_hole', 'picnic_area', 'snowfield', 'ice_cream_parlor', 'alcove', 'promenade', 'art_gallery', 'discotheque', 'ice_skating_rink', 'barn', 'museum', 'amusement_park', 'windmill', 'berth', 'florist_shop', 'shed', 'kennel', 'loading_dock', 'parking_lot', 'library', 'bookstore', 'bedchamber', 'viaduct', 'gas_station', 'kindergarden_classroom', 'volleyball_court', 'formal_garden', 'raceway', 'nursing_home', 'restaurant_kitchen', 'wheat_field', 'excavation', 'locker_room', 'swimming_hole', 'elevator', 'artists_loft', 'jacuzzi', 'fabric_store', 'hospital', 'swamp', 'rock_arch', 'orchestra_pit', 'ballroom', 'wind_farm', 'building_facade', 'dam', 'hangar', 'inn', 'oast_house', 'server_room', 'mountain', 'jail_cell', 'street', 'football_field', 'pasture', 'banquet_hall', 'parking_garage', 'desert', 'home_theater', 'medina', 'ski_resort', 'apartment_building', 'ocean', 'bank_vault', 'television_studio', 'synagogue', 'fire_escape', 'veterinarians_office', 'candy_store', 'cabin', 'beauty_salon', 'runway', 'farm', 'creek', 'beer_garden', 'ruin', 'repair_shop', 'auditorium', 'fountain', 'patio', 'supermarket', 'office_building', 'pier', 'lake', 'shopping_mall', 'stage', 'dining_hall', 'department_store', 'railroad_track', 'movie_theater', 'flea_market', 'courthouse', 'church', 'sandbox', 'garage', 'skyscraper', 'conference_room', 'rope_bridge', 'legislative_chamber', 'river', 'bedroom', 'campus', 'computer_room', 'classroom', 'amphitheater', 'grotto', 'burial_chamber', 'subway_station', 'cockpit', 'tree_farm', 'music_studio', 'badlands', 'balcony', 'fire_station', 'clothing_store', 'assembly_line', 'cottage', 'fishpond', 'television_room', 'cafeteria', 'ice_floe', 'forest', 'cliff', 'stable', 'vineyard', 'porch', 'atrium', 'dorm_room', 'mountain_snowy', 'ski_slope', 'reception', 'bamboo_forest', 'boathouse', 'golf_course', 'airplane_cabin', 'boxing_ring', 'car_interior', 'market', 'zen_garden', 'arena', 'field_road', 'pagoda', 'highway', 'nursery', 'barndoor', 'japanese_garden', 'toyshop', 'coast', 'auto_factory', 'natural_history_museum', 'stadium', 'shoe_shop', 'lawn', 'raft', 'iceberg', 'driveway', 'mezzanine', 'bus_interior', 'orchard', 'archaelogical_excavation', 'house', 'drugstore', 'galley', 'gift_shop', 'beer_hall', 'volcano', 'pavilion', 'gazebo', 'food_court', 'ticket_booth', 'canyon', 'jewelry_shop']
places365_list_subsampled = random.sample(places365_list, 200)

def generate_hypernym(client, cls, imagenet=False):
    if imagenet:
        cls_name = ImageNet_description[cls]
    else:
        cls_name = cls
    base_message = f"Find the hypernym of the following word. For example, if the class is pirate ship, the hypernym can be ship. Or if the class is papillon, the hypernym can be toy spaniel.\nclass: {cls_name}\nPlease make sure that the response format is in the form of hypernym: answer.\n"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message}
        ]
    )
    response_content = response.choices[0].message.content
    match = re.search(r'hypernym:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

client = OpenAI()

totalprompt_dict = {}

for cls in tqdm(dataset_count):
    totalprompt_dict[cls] = {'metaprompts': []}
    if imagenet:
        cls_name = ImageNet_description[cls]
    else:
        cls_name = cls
    synsets = wn.synsets(cls_name)
    if len(synsets) == 0:
        hypernym = generate_hypernym(client, cls_name)
    else:
        try:
            hypernym = synsets[0].hypernyms()[0].name().split('.')[0]
        except:
            hypernym = generate_hypernym(client, cls_name)
    
    print(f"idx: {cls}, cls: {cls_name}, hypernym: {hypernym}")
    # Diversify 50 prompts using places365_list_subsampled
    class_prompt_list = []
    for i in range(200):
        background = places365_list_subsampled[i].replace('_', ' ')
        diversified_prompt = f"{cls_name.replace('_', ' ')}, {hypernym.replace('_', ' ')} inside {background}"
        totalprompt_dict[cls]['metaprompts'].append(
            {
                'index': i,
                'metaprompt': 'dummy',
                'prompts': [
                    {
                        'index': 0,
                        'content': diversified_prompt
                    }
                ]
            }
        )

with open(prompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
