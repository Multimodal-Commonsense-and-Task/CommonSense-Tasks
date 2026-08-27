import os
import argparse
from sc2 import maps
from datetime import datetime
from sc2.main import run_game
from bots.zerg_bot import Zerg_Bot
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from bots.swarmbrain import SwarmBrain
from bots.terran_bot import Terran_Bot
from bots.protoss_bot import Protoss_Bot
from bots.textstarcraft import TextStarCraft


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StarCraft II agent')
    parser.add_argument('--port', default=8080)
    parser.add_argument('--num_server', default=3)
    parser.add_argument('--save_path', default='tmp')
    parser.add_argument('--temperature', default=0.7)
    parser.add_argument('--LLM_api_text', default='gpt-4o-mini')
    parser.add_argument('--LLM_api_key', default="YOUR_API_KEY")
    parser.add_argument('--realtime', action='store_true', default=False)

    parser.add_argument(
        '--own_race', default='Terran',
        choices=['Protoss', 'Zerg', 'Terran']
    )
    parser.add_argument('--mode', default='bot', choices=['bot', 'agent'])

    # if mode is bot
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument(
        '--enemy_race', default='Terran',
        choices=['Protoss', 'Zerg', 'Terran']
    )
    parser.add_argument(
        '--difficulty', default='Hard', 
        choices=[
            'VeryEasy', 'Easy', 'Medium', 'MediumHard', 'Hard', 'Harder',
            'VeryHard', 'CheatVision', 'CheatMoney', 'CheatInsane'
        ]
    )
    
    # if mode is agent
    parser.add_argument(
        '--enemy_agent',
        default='HEP-TextStarCraft',
        choices=['TextStarCraft', 'SwarmBrain', 'HEP-TextStarCraft']
    )

    args = parser.parse_args()
    args.own_race = Race[args.own_race]
    args.enemy_race = Race[args.enemy_race]
    args.difficulty = Difficulty[args.difficulty]
    args.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_replay_folder = os.path.join(current_dir, args.save_path)
    os.makedirs(os.path.join(temp_replay_folder), exist_ok=True)
    temp_replay_path = f'{temp_replay_folder}/{args.current_time}_{args.difficulty}_{args.enemy_race}_temp.SC2Replay'
    if args.own_race == Race.Protoss:
        our_bot = Protoss_Bot(args)
    elif args.own_race == Race.Zerg:
        our_bot = Zerg_Bot(args)
    elif args.own_race == Race.Terran:
        our_bot = Terran_Bot(args)
    if args.mode == 'bot':
        enemy = Computer(args.enemy_race, args.difficulty)
        result = run_game(maps.get("Ancient Cistern LE"), [Bot(args.own_race, our_bot), enemy], realtime=args.realtime, save_replay_as=temp_replay_path, random_seed=args.seed)
        result = str(result).split(".")[1]
        final_replay_path = f'{temp_replay_folder}/{args.current_time}_{args.difficulty}_{args.enemy_race}_{result}.SC2Replay'
        os.rename(temp_replay_path, final_replay_path)
    elif args.mode == 'agent':
        if args.enemy_agent == 'TextStarCraft':
            enemy = Bot(Race.Protoss, TextStarCraft(args))
        elif args.enemy_agent == 'SwarmBrain':
            enemy = Bot(Race.Zerg, SwarmBrain(args))
        elif args.enemy_agent == 'HEP-TextStarCraft':
            enemy = Bot(Race.Protoss, TextStarCraft(args, hep=True))
        run_game(maps.get("Ancient Cistern LE"), [Bot(args.own_race, our_bot), enemy], realtime=args.realtime, save_replay_as=temp_replay_path, random_seed=args.seed)
