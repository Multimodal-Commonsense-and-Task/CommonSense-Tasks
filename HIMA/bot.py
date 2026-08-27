import os
import time
import json
import utils
import random
import requests
import constants
import numpy as np
from prompt import Prompt
from openai import OpenAI
from sc2.data import Race
from sc2.bot_ai import BotAI
from collections import deque
from collections import Counter
from sc2.ids.unit_typeid import UnitTypeId


class HIMA(BotAI):
    def __init__(self, args):
        self.args = args
        self.own_race = args.own_race
        self.enemy_race = args.enemy_race
        self.leader = OpenAI(api_key=args.LLM_api_key)
        self.server = (self.args.seed % self.args.num_server) + self.args.port

        self.prompt = Prompt(self.own_race)
        self.action = utils.ActionDescriptions(self.own_race)
        self.text_prompt = self.prompt.generate_prompts()
        self.action_extractor = utils.ActionExtractor(constants.ACTION_DICT[self.own_race], self.own_race)
        self.game_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_path)

        with open(os.path.join(self.game_folder, 'prompt.txt'), "a", encoding='utf-8') as file:
            file.write(f"{self.text_prompt}")

        self.advice = None
        self.under_attack = False
        self.current_time = time.time()
        self.action_queue, self.enemy_cluster = deque(), deque()
        self.scout_period, self.refresh_period, self.inference_period, self.combat_period = 4000, 2500, 400, 100

    def move_to_nexus(self, units):
        nexus = sorted(self.townhalls, key=lambda base: base.distance_to(self.enemy_start_locations[0]))[0]
        ramp = sorted(constants.MAP_RAMPS, key=lambda ramp: ramp.distance_to(nexus))[0]
        for unit in units:
            if unit.distance_to(ramp) > 7:
                unit.move(ramp)

    def scout(self):
        if self.start_position == 'C':
            POINT = [constants.MAP_POINTS[point] for point in ['H', 'J', 'K', 'L', 'O']]
        else:
            POINT = [constants.MAP_POINTS[point] for point in ['B', 'E', 'F', 'G', 'I']]
        if self.units(self.scout_unit):
            unit = random.choice(self.units(self.scout_unit))
            points = random.sample(POINT, 3)
            points.append(self.enemy_start_locations[0])
            points.append(self.start_location)
            for point in points:
                unit.patrol(point, queue=True)

    def find_enemy_clusters(self):
        if not self.enemy_cluster:
            positions = []
            for enemy in self.enemy_units.of_type(constants.UNITS[self.enemy_race]):
                dist = min(
                    (own.distance_to(enemy) for own in self.units | self.structures),
                    default=float('inf')
                )
                if dist <= 10:
                    positions.append(enemy.position)

            positions.sort(key=lambda p: min(own.distance_to(p) for own in self.units | self.structures))
            for pos in positions:
                if not any(pos.distance_to(p) < 20 for p in self.enemy_cluster):
                    self.enemy_cluster.appendleft(pos)

        return self.enemy_cluster.popleft() if self.enemy_cluster else None

    async def update_cycle(self, observation):
        if self.enemy_units:
            await self.unit_attack(self.units.of_type(constants.UNITS[self.own_race]))
        if self.iteration > self.next_scout:
            self.next_scout = self.iteration + self.scout_period
            self.scout()
        self.troop.update_army()
        if self.own_race == Race.Protoss:
            self.race_specific_tactic()
        else:
            await self.race_specific_tactic()
        await self.distribute_workers(resource_ratio=1)
        enemy_count = sum(observation['enemy'].values())
        self.under_attack = True if enemy_count else False

        if self.troop.is_attack:
            self.troop.add_army(self.units.of_type(constants.UNITS[self.own_race]) - self.units_before)
            if self.troop.check_power() <= self.troop.retreat_threshold:
                self.troop.clear_army()
                self.move_to_nexus(self.troop)
            if self.troop.check_fighting() > 0.8 * len(self.troop):
                self.troop.clear_army()
                self.move_to_nexus(self.troop)
        elif not self.under_attack:
            self.move_to_nexus(self.units.of_type(constants.UNITS[self.own_race]))
        self.units_before = self.units.of_type(constants.UNITS[self.own_race])

    def get_information(self):
        enemy_units = []
        for unit in [u for u in self.enemy_units if 'Changeling' not in u.name and u.name not in  ['Probe', 'Observer', 'Drone', 'Overlord', 'Overseer', 'SCV']]:
            if any([unit.distance_to(stru) < self.warning_range for stru in self.structures]):
                enemy_units.append(unit)
        information = {
            "resource": {
                'game_time': self.time_formatted,
                'supply_cap': self.supply_cap,
                'supply_used': self.supply_used,
            },
            "unit": {key: self.units(UnitTypeId[key.upper()]).amount for key in constants.ACTION_DICT[self.own_race]['Train Unit'].values()},
            "building": {key: sum([int(i.build_progress) for i in self.structures(UnitTypeId[key.upper()])]) for key in constants.ACTION_DICT[self.own_race]['Build Structure'].values()},
            "research": {key: self.already_pending_upgrade(constants.RESEARCHS[self.own_race][key][0]) for key in constants.ACTION_DICT[self.own_race]['Research Technique'].values()},
            "enemy": dict(Counter([unit.name for unit in enemy_units])) if enemy_units else {},
            "previous_action": [action for t, action in self.successful_actions if self.time - 60 <= t < self.time]
        }
        gas_building = len([i for i in self.gas_buildings if int(i.build_progress) and i.vespene_contents])
        if self.own_race == 'Protoss':
            information['building']['Assimilator'] = gas_building
            information['building']['Gateway'] = sum([int(i.build_progress) for i in self.structures(UnitTypeId.GATEWAY)]) + sum([int(i.build_progress) for i in self.structures(UnitTypeId.WARPGATE)])
        elif self.own_race == 'Terran':
            information['building']['Refinery'] = gas_building
        return information

    def get_action(self, observation):
        self.agent_call += 1
        self.next_inference = self.iteration + self.inference_period
        command = self.agent_inference(observation)
        action_ids = self.action_extractor.extract_actions_from_command(command)
        for id in action_ids:
            self.action_queue.append(id)
            if self.own_race == Race.Terran:
                if id == 110:
                    self.action_queue.append(id)
                if id == 201 and not self.structures(UnitTypeId.REFINERY).exists:
                    self.action_queue.append(id)

    def record_succeed(self, action):
        self.executed_action = self.action.dict[action]
        self.next_refresh = self.iteration + self.refresh_period
        self.successful_actions.append((self.time, self.action.dict[action]))

    def record_failure(self, action, reason, precondition=None):
        self.failed_action = action
        self.failure_reason = reason
        self.precondition = precondition

    def generate_input(self, information, agent='multi-LM', actions=None):
        summary = {
            'supply_used': information['resource']['supply_used'],
            'supply_capacity': information['resource']['supply_cap'],
            'unit': {k: v for k, v in list(information['unit'].items()) + list(information['building'].items()) if v > 0},
            'research': [research for research in information['research'] if research == 1],
            'previous_action': information['previous_action']
        }
        if agent == 'multi-LM':
            if information['enemy']:
                summary['observed enemy'] = information['enemy']
            if self.advice:
                summary['failed action'] = self.advice[1]
                summary['failed reason'] = f"{self.advice[2]} is {self.advice[0].replace('_', ' ')}."
                self.advice = None
            summary = f'Game Status: {json.dumps(summary)}\n{actions}'
        return summary

    def check_supply_cost(self, tgt):
        supply_building = {Race.Protoss: [UnitTypeId.PYLON, 205], Race.Zerg: [UnitTypeId.OVERLORD, 101], Race.Terran: [UnitTypeId.SUPPLYDEPOT, 213]}
        action_id = self.action.r_dict[tgt]
        if self.supply_left < self.calculate_supply_cost(UnitTypeId[tgt.upper()]):
            if self.already_pending(supply_building[self.own_race][0]):
                return self.record_failure(action_id, 'not_afford')
            else:
                return self.record_failure(action_id, 'not_exist', supply_building[self.own_race][1])
        return True

    def check_affordability(self, tgt, research=False):
        action_id = self.action.r_dict[tgt]
        if tgt == 'MULE':
            return True
        check = constants.RESEARCHS[self.own_race][tgt][0] if research else UnitTypeId[tgt.upper()]
        if not self.can_afford(check):
            return self.record_failure(action_id, 'not_afford')
        return True

    def check_research_status(self, tgt):
        action_id = self.action.r_dict[tgt]
        if self.already_pending(constants.RESEARCHS[self.own_race][tgt][0]) == 1:
            return self.record_failure(action_id, 'already_done')
        elif self.already_pending(constants.RESEARCHS[self.own_race][tgt][0]) != 0:
            return self.record_failure(action_id, 'already_on')
        return True

    def check_ability(self, tgt, abilities, train=False):
        action_id = self.action.r_dict[tgt]
        check = constants.ABILITYS[UnitTypeId[tgt.upper()]] if train else constants.RESEARCHS[self.own_race][tgt][1]
        if check not in abilities:
            return self.record_failure(action_id, 'not_ability')
        return True

    def check_building_condition(self, tgt, buildings):
        action_id = self.action.r_dict[tgt]
        for building in buildings:
            id = self.action.r_dict[building]
            if self.already_pending(UnitTypeId[building.upper()]) and len([gate for gate in self.structures(UnitTypeId[building.upper()]) if gate.is_ready and gate.is_idle]) == 0:
                if self.time < 800 or (
                    (
                        1 - max([
                            structure.build_progress
                            for structure in self.structures(UnitTypeId[building.upper()])
                        ])
                    ) if self.structures(UnitTypeId[building.upper()]) else 1
                ) * constants.BUILDINGS[self.own_race][UnitTypeId[building.upper()]] < 20:
                    return self.record_failure(action_id, 'not_afford')
                else:
                    return self.record_failure(action_id, 'not_ready', id)
            if not self.structures(UnitTypeId[building.upper()]).exists:
                return self.record_failure(action_id, 'not_exist', id)
        return True

    async def expand_now(self, building, max_distance=10, location=None):
        location = await self.get_next_expansion()
        if not location:
            return False
        if await self.build(building, near=location, max_distance=max_distance, random_alternative=False, placement_step=1):
            return True
        return False

    def check_condition(self, action_id, tgt, prerequisites):
        if not self.check_building_condition(tgt, prerequisites):
            return False
        if not self.check_supply_cost(tgt):
            return False
        if not self.check_affordability(tgt):
            return False
        if action_id == 100:
            if self.workers.amount > 75:
                return self.record_failure(action_id, 'many')
        if action_id == 200:
            if self.already_pending(UnitTypeId[tgt.upper()]):
                return self.record_failure(action_id, 'pending_nexus')
            if self.structures(UnitTypeId[tgt.upper()]).amount == 7:
                return self.record_failure(action_id, 'many')
        return True

    async def handle_build(self, tgt, prerequisites):
        # TODO: ZERG two choices buildings
        action_id = self.action.r_dict[tgt]
        if not self.check_condition(action_id, tgt, prerequisites):
            return
        if action_id == 200:
            if await self.expand_now(UnitTypeId[tgt.upper()]):
                return self.record_succeed(action_id)
            else:
                return self.record_failure(action_id, 'nexus_fail')
        if action_id == 201:
            for nexus in self.townhalls:
                for vespene in self.vespene_geyser.closer_than(10, nexus):
                    if vespene.position not in self.gas_list:
                        if await self.build(UnitTypeId[tgt.upper()], vespene):
                            self.gas_list.append(vespene.position)
                            return self.record_succeed(action_id)
            return self.record_failure(action_id, 'gas_fail')
        await self.action_build(tgt)

    async def handle_research(self, tgt, prerequisites):
        action_id = self.action.r_dict[tgt]
        if not self.check_building_condition(tgt, prerequisites):
            return
        if not self.check_research_status(tgt):
            return
        building = self.structures(UnitTypeId[prerequisites[0].upper()]).ready[0]
        abilities = await self.get_available_abilities(building)
        if not self.check_ability(tgt, abilities):
            return
        if not self.check_affordability(tgt, research=True):
            return
        id = self.action.r_dict[prerequisites[0]]
        if len([gate for gate in self.structures(UnitTypeId[prerequisites[0].upper()]) if gate.is_ready]) == 0:
            return self.record_failure(action_id, 'not_ready', id)
        if building.research(constants.RESEARCHS[self.own_race][tgt][0]):
            return self.record_succeed(action_id)
        else:
            return self.record_failure(action_id, 'fail')

    def leader_inference(self, user_input):
        utils.save_data_to_file(user_input, os.path.join(self.game_folder, "input.txt"))
        self.messages = [
            {"role": "system", "content": self.text_prompt[0]},
            {"role": "user", "content": self.text_prompt[1]},
            {"role": "assistant", "content": self.text_prompt[2]},
            {"role": "user", "content": user_input}
        ]

        while True:
            try:
                output = self.leader.chat.completions.create(
                    model=self.args.LLM_api_text,
                    temperature=self.args.temperature,
                    messages=self.messages
                )
                response = output.choices[0].message.content
                break
            except Exception as e:
                print(e)
                time.sleep(7)

        utils.save_data_to_file(f"{self.time_formatted}\n{response.strip()}", os.path.join(self.game_folder, "output.txt"))
        return response

    def agent_inference(self, observation):
        query_input = self.generate_input(observation, agent='LM')
        user_input = {
            "prompt": json.dumps(query_input),
            "temperature": self.args.temperature
        }
        r = requests.post(f"http://localhost:{self.server}/infer", json=user_input)
        r.raise_for_status()
        query_input = self.generate_input(observation, actions=r.json()["text"])
        command = self.leader_inference(query_input)
        return command

    def attack(self):
        self.action_queue.appendleft(f'attack_{self.enemy_position}')
        self.troop.is_attack = self.enemy_position
        self.executed_action = f"ATTACK TO {self.enemy_position}"
        attack_units = sorted(
            self.units.of_type(constants.UNITS[self.own_race]),
            key=lambda unit: (
                -self.calculate_supply_cost(unit.type_id),
                unit.distance_to(constants.MAP_POINTS[self.troop.is_attack])
            )
        )[:int(self.army_count * 0.85)]
        self.troop.add_army(attack_units)

    def save_metric(self):
        if self.supply_cap != 200:
            if self.supply_cap != 0:
                self.apu.append(self.supply_used / max(self.supply_used, self.supply_cap))
        else:
            if not hasattr(self, "rur"):
                self.rur = self.state.score.spent_minerals + self.state.score.spent_vespene
            if not hasattr(self, "pbr") and self.supply_used == 200:
                self.pbr = self.time
        if self.executed_action:
            utils.save_data_to_file(f"{self.time_formatted} <{self.executed_action}>", os.path.join(self.game_folder, "command.txt"))
        if self.failed_action and self.failure_reason != 'not_afford':
            utils.save_data_to_file(f"{self.time_formatted} <{self.action.dict[self.failed_action]}> {self.failure_reason}", os.path.join(self.game_folder, "command.txt"))

    async def on_start(self):
        self.client.game_step = 1
        if self.townhalls[0].position == constants.MAP_POINTS['C']:
            self.start_position = 'C'
            self.enemy_position = 'N'
            self.location = -1
        else:
            self.start_position = 'N'
            self.enemy_position = 'C'
            self.location = 1
        self.next_scout, self.next_refresh, self.next_inference, self.next_combat, self.agent_call = 4000, 9999, 0, 0, 0
        self.apu, self.successful_actions, self.gas_list, self.units_before = [], [], [], []

    async def on_step(self, iteration):
        if not self.townhalls:
            return
        self.iteration = iteration
        observation = self.get_information()
        self.executed_action, self.failed_action = None, None
        await self.update_cycle(observation)

        if self.iteration > self.next_refresh:
            self.action_queue = deque()
        if self.supply_used <= self.supply_cap and self.supply_used < 198:
            if not self.action_queue and self.iteration > self.next_inference:
                self.get_action(observation)
        if not self.troop.is_attack and not self.under_attack:
            if self.supply_used > 190 or sum([self.calculate_supply_cost(unit.type_id) for unit in self.units.of_type(constants.UNITS[self.own_race])]) > self.troop.attack_threshold:
                self.attack()

        action = self.action_queue.popleft() if self.action_queue else None
        if isinstance(action, int):
            method_name = f'handle_action_{action}'
            method = getattr(self, method_name, None)
            await method()

        if self.failed_action:
            if self.failure_reason == 'not_afford':
                self.action_queue.appendleft(self.failed_action)
            elif self.failure_reason == 'not_exist':
                if self.supply_used < 190 and self.supply_used <= self.supply_cap:
                    required = self.action.dict[self.precondition]
                    if required in [
                        'Gateway', 'RoboticsFacility', 'Stargate', 'Pylon', 'Forge', 'CyberneticsCore', 'FleetBeacon',
                        'Overlord', 'SpawningPool', 'Spire', 'GreaterSpire',
                        'Refinery', 'Barracks', 'Factory', 'SupplyDepot', 'Armory', 'FusionCore'
                        ]:
                        self.action_queue = deque()
                        self.advice = (self.failure_reason, self.action.dict[self.failed_action], required)

        self.save_metric()

    async def on_end(self, game_result):
        observation = self.get_information()
        if not hasattr(self, "rur"):
            self.rur = 0
        if not hasattr(self, "pbr"):
            self.pbr = 0
        with open(os.path.join(self.game_folder, "metric.json"), "w") as f:
            json.dump({"result": game_result.name, "time": self.time_formatted, "agent_call": self.agent_call, 
                       "elapsed_time": int(time.time() - self.current_time),
                       "apu": round(np.mean(self.apu), 2), "rur": self.rur, "pbr": round((self.pbr / self.time), 2),
                       "tr": sum([research for research in observation["research"].values() if research == 1])}, f, indent=4)
