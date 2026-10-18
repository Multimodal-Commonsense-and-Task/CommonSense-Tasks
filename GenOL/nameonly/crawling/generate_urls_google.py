import os
import json
import logging
from url_generator.google_generator import GoogleURLGenerator
from classes import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

save_dir = './urls/bongard_hoi'
error_file_path = './hoi_error.txt'

# generator = GoogleURLGenerator(max_scroll=1, sleep_time=2, mode='headless')

# # Check the save directory to remove already generated URLs
# class_txt_files = os.listdir(save_dir)
# class_txt_files = [f.split('.')[0] for f in class_txt_files]
# domainnet = [cls for cls in domainnet if cls not in class_txt_files]

# # For classification
# for cls in tqdm(food101_count):
#     logger.info(f"Generating URL for {cls}")
#     url_list = generator.generate_url(query=cls, total_images=10, image_type='None')
#     with open(os.path.join(save_dir, f'{cls}.txt'), 'w') as f:
#         for url in url_list:
#             f.write(url + '\n')


# For Bongard-HoI
dataset_list = [('orange', 'eat', 'A person eating an orange', 28), ('orange', 'peel', 'Peeling an orange', 22), ('orange', 'wash', 'Washing an orange', 2), ('orange', 'hold', 'Holding an orange', 70), ('orange', 'no_interaction', 'An orange', 3), ('orange', 'squeeze', 'Squeezing an orange', 35), ('orange', 'cut', 'Cutting an orange', 1), ('toilet', 'clean', 'Cleaning a toilet', 31), ('toilet', 'sit_on', 'Sitting on a toilet', 26), ('toilet', 'wash', 'Washing a toilet', 1), ('toilet', 'flush', 'Flushing a toilet', 1), ('toilet', 'stand_on', 'Standing on a toilet', 2), ('toilet', 'open', 'Opening a toilet', 2), ('teddy_bear', 'hold', 'Holding a teddy bear', 7), ('teddy_bear', 'no_interaction', 'A teddy bear', 4), ('teddy_bear', 'kiss', 'Kissing a teddy bear', 1), ('teddy_bear', 'hug', 'Hugging a teddy bear', 2), ('broccoli', 'hold', 'Holding broccoli', 7), ('broccoli', 'eat', 'Eating broccoli', 2), ('broccoli', 'no_interaction', 'Broccoli', 2), ('broccoli', 'wash', 'Washing broccoli', 1), ('broccoli', 'stir', 'Stirring broccoli', 1), ('broccoli', 'cut', 'Cutting broccoli', 1), ('carrot', 'hold', 'Holding a carrot', 16), ('carrot', 'peel_or_cut', 'Peeling or cutting a carrot', 1), ('carrot', 'wash', 'Washing a carrot', 4), ('carrot', 'cook', 'Cooking a carrot', 2), ('carrot', 'no_interaction', 'A carrot', 6), ('carrot', 'eat', 'Eating a carrot', 2), ('wine_glass', 'sip', 'Sipping from a wine glass', 7), ('wine_glass', 'wash', 'Washing a wine glass', 3), ('wine_glass', 'fill', 'Filling a wine glass', 4), ('car', 'drive', 'Driving a car', 32), ('car', 'wash', 'Washing a car', 27), ('car', 'no_interaction', 'A car', 36), ('car', 'inspect', 'Inspecting a car', 7), ('cat', 'pet', 'Petting a cat', 22), ('cat', 'kiss', 'Kissing a cat', 7), ('cat', 'chase', 'Chasing a cat', 3), ('cat', 'hold', 'Holding a cat', 6), ('cat', 'feed', 'Feeding a cat', 7), ('cat', 'wash', 'Washing a cat', 1), ('train', 'exit', 'Exiting a train', 7), ('train', 'ride', 'Riding a train', 7), ('train', 'sit_on', 'Sitting on a train', 19), ('boat', 'sit_on', 'Sitting on a boat', 7), ('boat', 'drive', 'Driving a boat', 8), ('boat', 'sail', 'Sailing a boat', 7), ('boat', 'repair', 'Repairing a boat', 7), ('boat', 'tie', 'Tying a boat', 2), ('boat', 'board', 'Boarding a boat', 1), ('airplane', 'exit', 'Exiting an airplane', 7), ('airplane', 'no_interaction', 'An airplane', 3), ('airplane', 'inspect', 'Inspecting an airplane', 3), ('airplane', 'wash', 'Washing an airplane', 2), ('airplane', 'ride', 'Riding an airplane', 1), ('airplane', 'load', 'Loading an airplane', 7), ('tv', 'control', 'Controlling a TV', 36), ('tv', 'watch', 'Watching TV', 104), ('tv', 'repair', 'Repairing a TV', 8), ('tv', 'no_interaction', 'A TV', 7), ('person', 'teach', 'A person teaching', 7), ('person', 'hug', 'A person hugging', 244), ('person', 'hold', 'A person holding', 26), ('person', 'stab', 'A person stabbing', 7), ('person', 'carry', 'A person carrying', 3), ('person', 'lick', 'A person licking', 3), ('umbrella', 'stand_under', 'Standing under an umbrella', 134), ('umbrella', 'sit_under', 'Sitting under an umbrella', 5), ('umbrella', 'no_interaction', 'An umbrella', 3), ('umbrella', 'hold', 'Holding an umbrella', 31), ('knife', 'cut_with', 'Cutting with a knife', 7), ('knife', 'no_interaction', 'A knife', 7), ('refrigerator', 'clean', 'Cleaning a refrigerator', 30), ('refrigerator', 'move', 'Moving a refrigerator', 1), ('refrigerator', 'no_interaction', 'A refrigerator', 7), ('refrigerator', 'open', 'Opening a refrigerator', 19), ('refrigerator', 'hold', 'Holding a refrigerator', 8), ('kite', 'fly', 'Flying a kite', 230), ('kite', 'hold', 'Holding a kite', 63), ('laptop', 'type_on', 'Typing on a laptop', 112), ('laptop', 'repair', 'Repairing a laptop', 20), ('laptop', 'no_interaction', 'A laptop', 7), ('laptop', 'open', 'Opening a laptop', 1), ('cow', 'pet', 'Petting a cow', 7), ('cow', 'lasso', 'Lassoing a cow', 3), ('cow', 'ride', 'Riding a cow', 2), ('cow', 'hug', 'Hugging a cow', 2), ('cake', 'make', 'Making a cake', 8), ('cake', 'no_interaction', 'A cake', 4), ('cake', 'eat', 'Eating a cake', 6), ('cake', 'cut', 'Cutting a cake', 34), ('cake', 'blow', 'Blowing on a cake', 13), ('cake', 'hold', 'Holding a cake', 98), ('donut', 'eat', 'Eating a donut', 62), ('donut', 'hold', 'Holding a donut', 46), ('donut', 'no_interaction', 'A donut', 7), ('donut', 'make', 'Making a donut', 7), ('donut', 'cut', 'Cutting a donut', 2), ('sandwich', 'eat', 'Eating a sandwich', 23), ('sandwich', 'make', 'Making a sandwich', 5), ('sandwich', 'cut', 'Cutting a sandwich', 4), ('sandwich', 'no_interaction', 'A sandwich', 8), ('snowboard', 'adjust', 'Adjusting a snowboard', 8), ('snowboard', 'wear', 'Wearing a snowboard', 5), ('snowboard', 'ride', 'Riding a snowboard', 18), ('snowboard', 'no_interaction', 'A snowboard', 2), ('snowboard', 'hold', 'Holding a snowboard', 2), ('bird', 'hold', 'Holding a bird', 50), ('bird', 'watch', 'Watching a bird', 6), ('bird', 'feed', 'Feeding a bird', 19), ('bird', 'no_interaction', 'A bird', 8), ('bird', 'chase', 'Chasing a bird', 4), ('bear', 'feed', 'Feeding a bear', 7), ('bear', 'no_interaction', 'A bear', 3), ('bear', 'hunt', 'Hunting a bear', 7), ('bear', 'watch', 'Watching a bear', 2), ('hot_dog', 'make', 'Making a hot dog', 7), ('hot_dog', 'eat', 'Eating a hot dog', 14), ('hot_dog', 'no_interaction', 'A hot dog', 5), ('clock', 'hold', 'Holding a clock', 7), ('clock', 'no_interaction', 'A clock', 3), ('clock', 'set', 'Setting a clock', 7), ('clock', 'check', 'Checking a clock', 4), ('bicycle', 'ride', 'Riding a bicycle', 849), ('bicycle', 'hold', 'Holding a bicycle', 20), ('bicycle', 'straddle', 'Straddling a bicycle', 33), ('bicycle', 'wash', 'Washing a bicycle', 18), ('bicycle', 'walk', 'Walking a bicycle', 22), ('bicycle', 'hop_on', 'Hopping on a bicycle', 2), ('bicycle', 'no_interaction', 'A bicycle', 7), ('bicycle', 'carry', 'Carrying a bicycle', 5), ('skateboard', 'carry', 'Carrying a skateboard', 10), ('skateboard', 'ride', 'Riding a skateboard', 39), ('skateboard', 'no_interaction', 'A skateboard', 8), ('banana', 'eat', 'Eating a banana', 152), ('banana', 'peel', 'Peeling a banana', 37), ('banana', 'inspect', 'Inspecting a banana', 5), ('banana', 'no_interaction', 'A banana', 7), ('banana', 'buy', 'Buying a banana', 1), ('banana', 'cut', 'Cutting a banana', 5), ('elephant', 'ride', 'Riding an elephant', 95), ('elephant', 'no_interaction', 'An elephant', 2), ('elephant', 'pet', 'Petting an elephant', 13), ('elephant', 'hop_on', 'Hopping on an elephant', 7), ('elephant', 'wash', 'Washing an elephant', 7), ('elephant', 'hold', 'Holding an elephant', 4), ('elephant', 'watch', 'Watching an elephant', 4), ('elephant', 'hug', 'Hugging an elephant', 3), ('elephant', 'walk', 'Walking an elephant', 7), ('elephant', 'kiss', 'Kissing an elephant', 1), ('cup', 'hold', 'Holding a cup', 332), ('cup', 'drink_with', 'Drinking with a cup', 45), ('cup', 'fill', 'Filling a cup', 5), ('dog', 'hold', 'Holding a dog', 142), ('dog', 'walk', 'Walking a dog', 76), ('dog', 'hug', 'Hugging a dog', 50), ('dog', 'wash', 'Washing a dog', 47), ('dog', 'feed', 'Feeding a dog', 8), ('dog', 'dry', 'Drying a dog', 1), ('dog', 'straddle', 'Straddling a dog', 4), ('dog', 'run', 'Running with a dog', 3), ('dog', 'no_interaction', 'A dog', 2), ('dog', 'chase', 'Chasing a dog', 3), ('dog', 'straddle,pet', 'Straddling and petting a dog', 1), ('dining_table', 'sit_at', 'Sitting at a dining table', 1052), ('dining_table', 'eat_at', 'Eating at a dining table', 215), ('dining_table', 'clean', 'Cleaning a dining table', 1), ('keyboard', 'type_on', 'Typing on a keyboard', 90), ('keyboard', 'clean', 'Cleaning a keyboard', 87), ('bus', 'exit', 'Exiting a bus', 7), ('bus', 'no_interaction', 'A bus', 2), ('bus', 'wave', 'Waving at a bus', 1), ('bus', 'inspect', 'Inspecting a bus', 1), ('bus', 'board', 'Boarding a bus', 2), ('bus', 'wash', 'Washing a bus', 1), ('skis', 'ride', 'Riding skis', 65), ('skis', 'carry', 'Carrying skis', 4), ('skis', 'no_interaction', 'Skis', 2), ('skis', 'jump', 'Jumping with skis', 26), ('skis', 'inspect', 'Inspecting skis', 1), ('stop_sign', 'hold', 'Holding a stop sign', 7), ('stop_sign', 'no_interaction', 'A stop sign', 7), ('giraffe', 'feed', 'Feeding a giraffe', 32), ('giraffe', 'ride', 'Riding a giraffe', 2), ('giraffe', 'watch', 'Watching a giraffe', 8), ('giraffe', 'no_interaction', 'A giraffe', 3), ('giraffe', 'pet', 'Petting a giraffe', 8), ('vase', 'make', 'Making a vase', 7), ('vase', 'hold', 'Holding a vase', 12), ('sports_ball', 'spin', 'Spinning a sports ball', 8), ('sports_ball', 'dribble', 'Dribbling a sports ball', 20), ('sports_ball', 'hit', 'Hitting a sports ball', 7), ('sports_ball', 'catch', 'Catching a sports ball', 7), ('sports_ball', 'kick', 'Kicking a sports ball', 105), ('sports_ball', 'sign', 'Signing a sports ball', 1), ('sports_ball', 'no_interaction', 'A sports ball', 3), ('bowl', 'stir', 'Stirring a bowl', 8), ('bowl', 'hold', 'Holding a bowl', 17), ('bowl', 'no_interaction', 'A bowl', 5), ('bowl', 'wash', 'Washing a bowl', 2), ('bowl', 'eat_with', 'Eating with a bowl', 3), ('bottle', 'drink_with', 'Drinking with a bottle', 25), ('bottle', 'hold', 'Holding a bottle', 161), ('bottle', 'inspect', 'Inspecting a bottle', 2), ('bottle', 'no_interaction', 'A bottle', 8), ('bottle', 'open', 'Opening a bottle', 2), ('motorcycle', 'ride', 'Riding a motorcycle', 1238), ('motorcycle', 'wash', 'Washing a motorcycle', 65), ('motorcycle', 'inspect', 'Inspecting a motorcycle', 51), ('motorcycle', 'sit_on', 'Sitting on a motorcycle', 88), ('motorcycle', 'jump', 'Jumping on a motorcycle', 209), ('motorcycle', 'no_interaction', 'A motorcycle', 5), ('motorcycle', 'park', 'Parking a motorcycle', 2), ('motorcycle', 'walk', 'Walking with a motorcycle', 2), ('motorcycle', 'hold', 'Holding a motorcycle', 12), ('spoon', 'lick_or_sip', 'Licking or sipping with a spoon', 38), ('spoon', 'hold', 'Holding a spoon', 63), ('spoon', 'wash', 'Washing a spoon', 1), ('suitcase', 'drag', 'Dragging a suitcase', 39), ('suitcase', 'pack', 'Packing a suitcase', 2), ('suitcase', 'no_interaction', 'A suitcase', 10), ('suitcase', 'hold', 'Holding a suitcase', 14), ('suitcase', 'hug', 'Hugging a suitcase', 1), ('bed', 'sit_on', 'Sitting on a bed', 106), ('bed', 'lie_on', 'Lying on a bed', 311), ('bed', 'clean', 'Cleaning a bed', 2), ('bed', 'no_interaction', 'A bed', 1)]

generator = GoogleURLGenerator(mode='headless', use_color=True, use_size=False, scroll_patience=20)
for i, (_, _, prompt, count) in enumerate(tqdm(dataset_list)):
    # if i not in [227]: # [124, 165, 167, 207, 212, 225]
    #     continue
    logger.info(f"Generating URL for {prompt}")
    url_list = generator.generate_url(query=prompt, total_images=count * 3, image_type='None')
    with open(os.path.join(save_dir, f'{i}.txt'), 'w') as f:
        for url in url_list:
            f.write(url + '\n')




# # For Bongard-Openworld - positive 7, negative 7
# # Process jsonl file
# data_list = []
# with open('../prompt_generation/prompts/openworld_base.json', 'r') as f:
#     data_dict = json.load(f)

# # Crawl positive images
# generator = GoogleURLGenerator(max_scroll=5, sleep_time=2, mode='headless')
# for uid, pos_neg_dict in tqdm(data_dict.items()):
#     pos_save_dir = os.path.join(save_dir, 'pos')
#     txt_file_path = os.path.join(pos_save_dir, f"{uid}.txt")

#     if os.path.exists(txt_file_path):
#         with open(txt_file_path, 'r') as f:
#             lines = f.readlines()
#         if len(lines) > 10:
#             logger.info(f"Skipping already processed uid - {uid}")
#             continue
#         else:
#             logger.info(f"uid - {uid} file exists, but not enough ({len(lines)}) urls!")
    
#     logger.info(f"Generating URLs for uid - {uid}")
#     pos_url_list = generator.generate_url(query=pos_neg_dict['positive_prompts'][0], total_images=100, image_type='None')
#     if len(pos_url_list) < 10:
#         with open(error_file_path, 'a') as error_file:
#             error_file.write(f"uid: {uid}, List Length: {len(pos_url_list)}\n")
#     pos_save_dir = os.path.join(save_dir, 'pos')
#     os.makedirs(pos_save_dir, exist_ok=True)
#     with open(os.path.join(pos_save_dir, f"{uid}.txt"), 'w') as f:
#         for url in pos_url_list:
#             f.write(url + '\n')

# # Crawl positive images
# generator = GoogleURLGenerator(max_scroll=1, sleep_time=2, mode='headless')
# for uid, pos_neg_dict in tqdm(data_dict.items()):
#     neg_save_dir = os.path.join(save_dir, 'neg')
#     txt_file_path = os.path.join(neg_save_dir, f"{uid}.txt")

#     if os.path.exists(txt_file_path):
#         with open(txt_file_path, 'r') as f:
#             lines = f.readlines()
#         if len(lines) > 10:
#             logger.info(f"Skipping already processed uid - {uid}")
#             continue
#         else:
#             logger.info(f"uid - {uid} file exists, but not enough ({len(lines)}) urls!")
    
#     logger.info(f"Generating URLs for uid - {uid}")
#     neg_url_list = []
#     negative_prompts = pos_neg_dict['negative_prompts']
#     for neg_prompt in negative_prompts:
#         try:
#             neg_urls = generator.generate_url(query=neg_prompt, total_images=15, image_type='None')
#         except Exception as e:
#             print(f"Error occured while processing uid - {uid} - {e}")
#         if len(neg_urls) > 15:
#             neg_urls = neg_urls[:15]
#         neg_url_list.extend(neg_urls)

#     if len(neg_url_list) < 10:
#         with open(error_file_path, 'a') as error_file:
#             error_file.write(f"uid: {uid}, List Length: {len(neg_url_list)}\n")
#     os.makedirs(neg_save_dir, exist_ok=True)
#     with open(os.path.join(neg_save_dir, f"{uid}.txt"), 'w') as f:
#         for url in neg_url_list:
#             f.write(url + '\n')
