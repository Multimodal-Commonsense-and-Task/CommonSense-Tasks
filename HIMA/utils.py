import re
from constants import ACTION_DICT, MAP_POINTS


class ActionExtractor:
    def __init__(self, action_dict, own_race):
        self.own_race = own_race
        self.full_action_dict = {}
        for category in action_dict:
            for key, value in action_dict[category].items():
                self.full_action_dict[value] = key

    def extract_actions_from_command(self, command):
        action_ids = []
        extracted_decisions = self.extract_actions_from_text(command)
        for decision in extracted_decisions:
            action_ids.append(self.full_action_dict[decision])
        return action_ids

    def extract_actions_from_text(self, text):
        if 'Final Actions Summary' in text:
            text = text.split("Final Actions Summary", 1)[1]
            match = re.search(r"((?:\s*<[^>\n]+>(?:\s*[xX]\s*\d+)?)+)", text)
            if match:
                text = match.group(1)
            else:
                return []

        actions = []
        action_pattern = r"<([^>\n]+)>(?:\s*[xX]\s*(\d+))?"
        matches = re.findall(action_pattern, text)
        for action, multiplier in matches:
            if action in self.full_action_dict:
                count = int(multiplier) if multiplier else 1
                actions.extend([action] * count)

        return actions


class ActionDescriptions:
    def __init__(self, race):
        self.race = race
        self.dict, self.r_dict = {}, {}
        for _, value in ACTION_DICT[self.race].items():
            for inner_key, inner_value in value.items():
                self.dict[inner_key] = inner_value
                self.r_dict[inner_value] = inner_key


class Troop:
    def __init__(self, bot, attack_threshold=80, retreat_threshold=10):
        self._bot = bot
        self._tags = set()
        self.is_attack = False
        self.attack_threshold = attack_threshold
        self.retreat_threshold = retreat_threshold

    @property
    def units(self):
        return [u for u in self._bot.units if u.tag in self._tags]

    def __len__(self):
        return len(self._tags)

    def __iter__(self):
        yield from self.units

    def __contains__(self, unit):
        return unit.tag in self._tags

    def add_army(self, units):
        if not isinstance(units, list):
            units = [units]
        for unit in units:
            self._tags.add(unit.tag)
            unit.attack(MAP_POINTS[self.is_attack])

    def clear_army(self):
        self.is_attack = False
        self._tags.clear()

    def update_army(self):
        if self.is_attack:
            live = {u.tag for u in self._bot.units}
            self._tags.intersection_update(live)
            for unit in self.units:
                if not unit.is_active:
                    unit.attack(MAP_POINTS[self.is_attack])

    def check_power(self):
        return sum(self._bot.calculate_supply_cost(u.type_id) for u in self.units)

    def check_fighting(self):
        return len([unit for unit in self.units if MAP_POINTS[self.is_attack].distance_to(unit) > 5 and not unit.is_attacking])


def save_data_to_file(data, path, add_seperator=False):
    with open(path, "a", encoding='utf-8') as file:
        file.write(f"{data}\n")
        if add_seperator:
            file.write("------------------------------------------------------\n")
