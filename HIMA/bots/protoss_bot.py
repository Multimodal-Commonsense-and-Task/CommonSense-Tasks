import math
import utils
import random
from bot import HIMA
from sc2.position import Point2
from sc2.ids.buff_id import BuffId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.unit_typeid import UnitTypeId
from constants import ABILITYS, CHRONO_PRIORITY


class Protoss_Bot(HIMA):
    def __init__(self, args):
        self.warning_range = 10
        self.scout_unit = UnitTypeId.PROBE
        self.troop = utils.Troop(self)
        super().__init__(args)

    async def unit_attack(self, units):
        for unit in units:
            ability = await self.get_available_abilities(unit)
            enemy, distance = self.enemy_units.closest_to(unit), self.enemy_units.closest_distance_to(unit)
            mechanic_enemy = self.enemy_units.filter(lambda x: x.is_armored or x.is_mechanical)
            if mechanic_enemy:
                v_enemy, v_distance = mechanic_enemy.closest_to(unit), mechanic_enemy.closest_distance_to(unit)
            if unit.type_id == UnitTypeId.VOIDRAY and mechanic_enemy:
                unit.attack(v_enemy.position)
            else:
                unit.attack(enemy.position)
            if unit.type_id == UnitTypeId.ZEALOT:
                if distance < 30:
                    if AbilityId.EFFECT_CHARGE in ability and not unit.is_attacking:
                        unit(AbilityId.EFFECT_CHARGE, enemy)
            elif unit.type_id == UnitTypeId.STALKER:
                if distance < 30:
                    if AbilityId.EFFECT_BLINK_STALKER in ability:
                        if unit.shield_percentage < 0.05:
                            unit(AbilityId.EFFECT_BLINK_STALKER, self.townhalls[0].position)
                        else:
                            unit(AbilityId.EFFECT_BLINK_STALKER, enemy.position)
            elif unit.type_id == UnitTypeId.SENTRY:
                if distance < 8:
                    if AbilityId.GUARDIANSHIELD_GUARDIANSHIELD in ability:
                        unit(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD)
            elif unit.type_id == UnitTypeId.HIGHTEMPLAR:
                if distance < 9:
                    if AbilityId.PSISTORM_PSISTORM in ability:
                        stormable_enemies = [enemy.position for enemy in self.enemy_units if enemy.distance_to(unit) < 9]
                        unit(AbilityId.PSISTORM_PSISTORM, Point2.center(stormable_enemies))
                if distance < 10:
                    if AbilityId.FEEDBACK_FEEDBACK in ability:
                        unit(AbilityId.FEEDBACK_FEEDBACK, enemy)
            elif unit.type_id == UnitTypeId.DARKTEMPLAR:
                if distance < 30:
                    if AbilityId.EFFECT_SHADOWSTRIDE in ability:
                        unit(AbilityId.EFFECT_SHADOWSTRIDE, enemy.position)
            elif unit.type_id == UnitTypeId.VOIDRAY:
                if mechanic_enemy and v_distance < 5:
                    if AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT in ability:
                        unit(AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT)
                else:
                    if AbilityId.CANCEL_VOIDRAYPRISMATICALIGNMENT in ability:
                        unit(AbilityId.CANCEL_VOIDRAYPRISMATICALIGNMENT)
            elif unit.type_id == UnitTypeId.ORACLE:
                if distance < 5 and unit.energy > 40:
                    if AbilityId.BEHAVIOR_PULSARBEAMON in ability:
                        unit(AbilityId.BEHAVIOR_PULSARBEAMON)
                else:
                    if AbilityId.BEHAVIOR_PULSARBEAMOFF in ability:
                        unit(AbilityId.BEHAVIOR_PULSARBEAMOFF)
            elif unit.type_id == UnitTypeId.PHOENIX:
                if distance < 10:
                    if AbilityId.GRAVITONBEAM_GRAVITONBEAM in ability:
                        unit(AbilityId.GRAVITONBEAM_GRAVITONBEAM, enemy)
            elif unit.type_id == UnitTypeId.IMMORTAL:
                if unit.health != unit.health_max:
                    if AbilityId.EFFECT_IMMORTALBARRIER in ability:
                        unit(AbilityId.EFFECT_IMMORTALBARRIER)
            elif unit.type_id == UnitTypeId.MOTHERSHIP:
                if self.enemy_units.amount >= 10:
                    if AbilityId.EFFECT_TIMEWARP in ability:
                        unit(AbilityId.EFFECT_TIMEWARP, self.enemy_units.center)

    def chronoboost_building(self):
        if any([self.structures(x).exists for x in CHRONO_PRIORITY]):
            nexuses = [nexus for nexus in self.structures(UnitTypeId.NEXUS) if nexus.energy > 50]
            for nexus in nexuses:
                for type_id in CHRONO_PRIORITY:
                    structures = [
                        structure for structure in self.structures(type_id)
                        if structure.is_ready and not structure.is_idle and not structure.has_buff(BuffId.CHRONOBOOSTENERGYCOST)
                    ]
                    if structures:
                        nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, random.choice(structures))
                        break

    def race_specific_tactic(self):
        self.chronoboost_building()

    async def action_build(self, tgt):
        action_id = self.action.r_dict[tgt]
        position = self.find_position(tgt)
        if await self.build(UnitTypeId[tgt.upper()], near=position):
            return self.record_succeed(action_id)
        else:
            return self.record_failure(action_id, 'fail')

    async def handle_train(self, tgt, prerequisites):
        action_id = self.action.r_dict[tgt]
        has_warpgate = True if self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) and self.structures(UnitTypeId.WARPGATE).exists and 1 <= action_id <= 6 else False
        if has_warpgate:
            prerequisites = [building for building in prerequisites if building != 'Gateway']
        if not self.check_condition(action_id, tgt, prerequisites):
            return
        if tgt == 'Mothership':
            abilities = await self.get_available_abilities(self.structures(UnitTypeId.NEXUS))
            if not self.check_ability(tgt, abilities, train=True):
                return
        if tgt == 'Archon':
            templar = self.units([UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR])
            if len(templar) < 2:
                return self.record_failure(action_id, 'no_templar')
            for tem in templar:
                tem(AbilityId.MORPH_ARCHON)
                return self.record_succeed(action_id)

        if has_warpgate:
            available_warpgate = [warpgate for warpgate in self.structures(UnitTypeId.WARPGATE).ready.idle
                                  if ABILITYS[UnitTypeId[tgt.upper()]] in await self.get_available_abilities(warpgate)]
            if not available_warpgate:
                return self.record_failure(action_id, 'not_afford', 202)
            pylon = self.structures(UnitTypeId.PYLON).ready.closest_to(self.enemy_start_locations[0])
            warp_place = await self.find_placement(ABILITYS[UnitTypeId[tgt.upper()]], pylon.position, placement_step=1)
            if warp_place:
                if available_warpgate[0].warp_in(UnitTypeId[tgt.upper()], warp_place):
                    return self.record_succeed(action_id)
        else:
            id = self.action.r_dict[prerequisites[0]]
            building = [gate for gate in self.structures(UnitTypeId[prerequisites[0].upper()]) if gate.is_ready]
            gate = sorted(building, key=lambda g: len(g.orders))[0]
            if len(gate.orders) == 5:
                if not [structure for structure in self.structures(UnitTypeId[prerequisites[0].upper()]) if structure.build_progress != 1]:
                    return self.record_failure(action_id, 'not_exist', id)
                else:
                    return self.record_failure(action_id, 'not_ready', id)
            if gate.train(UnitTypeId[tgt.upper()]):
                return self.record_succeed(action_id)
            else:
                return self.record_failure(action_id, 'fail_train', id)

    def find_optimal_pylon_position(self):
        def is_position_blocking_resources(position):
            for resource in self.resources:
                if position.distance_to(resource.position) < 3:
                    return True
            return False

        def is_position_valid_for_pylon(position):
            for structure in self.structures:
                if position.distance_to(structure.position) < 3:
                    return False
            for assimilator in self.structures(UnitTypeId.ASSIMILATOR):
                if position.distance_to(assimilator.position) < 4.5:
                    return False
            for resource in self.resources:
                if position.distance_to(resource.position) < 3:
                    return False
            return True

        def score_position(pos):
            total_score = 0
            for pylon_pos in pylon_positions:
                distance = pos.distance_to(pylon_pos)
                score = 1 / (1 + math.exp(-distance + 5))
                total_score += score

            map_width = self.game_info.map_size[0]
            map_height = self.game_info.map_size[1]

            edge_dist_x = min(pos.x, map_width - pos.x)
            edge_dist_y = min(pos.y, map_height - pos.y)
            edge_distance = min(edge_dist_x, edge_dist_y)
            if edge_distance < 30:
                edge_penalty = - (30 - edge_distance) * 0.2
            else:
                edge_penalty = 0

            total_score += edge_penalty
            return total_score

        pylons = self.structures(UnitTypeId.PYLON)
        base_positions = [base.position for base in self.townhalls]
        pylon_positions = [pylon.position for pylon in pylons]
        base_pylon_counts = {}

        for base_pos in base_positions:
            count = sum(1 for pylon_pos in pylon_positions if base_pos.distance_to(pylon_pos) <= 10)
            base_pylon_counts[base_pos] = count

        base_position = sorted(base_pylon_counts.keys(), key=lambda base: base_pylon_counts[base])[0]
        candidate_positions = [base_position] + [pos for i in range(1, 15) for pos in self.neighbors8(base_position, d=i)]
        valid_positions = [pos for pos in candidate_positions if not is_position_blocking_resources(pos) and is_position_valid_for_pylon(pos)]
        if not valid_positions:
            return base_position
        pylons = self.structures(UnitTypeId.PYLON)
        pylon_positions = [pylon.position for pylon in pylons]
        score = [(pos, score_position(pos), pos.distance_to(self.enemy_start_locations[0])) for pos in valid_positions]
        best_position = sorted(score, key=lambda x: (-x[1], x[2]))[0][0]
        return best_position

    def original_building_position_search(self, base_position, max_distance=15):
        def compute_score(pos):
            distance_to_own_base = pos.distance_to(base_position)
            distance_to_enemy_base = pos.distance_to(enemy_base_position)
            score = weight_own_base * distance_to_own_base + weight_enemy_base / (distance_to_enemy_base + 1)
            return score

        def is_position_valid_for_building(position):
            for structure in self.structures:
                if structure in self.structures({UnitTypeId.ASSIMILATOR, UnitTypeId.NEXUS}):
                    if position.distance_to(structure.position) < structure.radius + 7:
                        return False
                else:
                    if position.distance_to(structure.position) < structure.radius + 3:
                        return False
            for resource in self.resources:
                if position.distance_to(resource.position) < 5:
                    return False
            for base_location in self.expansion_locations_list:
                if position.distance_to(base_location) < 10:
                    return False
            return True

        enemy_base_position = self.enemy_start_locations[0]
        weight_own_base = 1.0
        weight_enemy_base = -2.0
        for distance in range(1, max_distance + 1):
            candidate_positions = list(self.neighbors8(base_position, distance))
            valid_positions = [pos for pos in candidate_positions if is_position_valid_for_building(pos)]
            if valid_positions:
                best_position = max(valid_positions, key=compute_score)
                return best_position
        return None

    def find_best_base_for_building(self, building_type):
        base_positions = [base.position for base in self.townhalls]
        building_positions = [building.position for building in self.structures(building_type)]
        base_building_counts = {}
        for base_pos in base_positions:
            count = sum(1 for building_pos in building_positions if base_pos.distance_to(building_pos) <= 12)
            base_building_counts[base_pos] = count
        sorted_bases = sorted(base_building_counts.keys(), key=lambda base: base_building_counts[base])
        return sorted_bases

    def find_optimal_base_position(self, base_position, max_distance=15):
        pylon_scores = {}
        for pylon in self.structures(UnitTypeId.PYLON):
            distance_to_base = pylon.distance_to(base_position)
            nearby_buildings = sum(1 for building in self.structures if building.distance_to(pylon) < 12)
            score = 1 * distance_to_base - 2 * nearby_buildings
            pylon_scores[pylon] = score

        best_pylon = max(pylon_scores, key=pylon_scores.get)
        best_position = self.original_building_position_search(best_pylon.position, max_distance)
        return best_position

    def find_position(self, tgt):
        if tgt == 'Pylon':
            return self.find_optimal_pylon_position()
        best_base = self.find_best_base_for_building(UnitTypeId[tgt.upper()])[0]
        best_position = self.find_optimal_base_position(best_base)
        if not best_position:
            for base in self.townhalls:
                if base.position != best_base.position:
                    best_position = self.find_optimal_base_position(base.position)
                    if best_position:
                        break

        if not best_position:
            best_position = self.find_optimal_base_position(best_base.position, max_distance=30)

        return best_position

    @staticmethod
    def neighbors4(p, d=1):
        return {
            Point2((p.x - d, p.y)),
            Point2((p.x + d, p.y)),
            Point2((p.x, p.y - d)),
            Point2((p.x, p.y + d))
        }

    def neighbors8(self, p, d=1):
        direct_neighbors = self.neighbors4(p, d)
        diagonal_neighbors = {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d))
        }
        return direct_neighbors | diagonal_neighbors

    async def handle_action_100(self):
        await self.handle_train('Probe', ['Nexus'])

    async def handle_action_101(self):
        await self.handle_train('Zealot', ['Gateway'])

    async def handle_action_102(self):
        await self.handle_train('Sentry', ['Gateway', 'CyberneticsCore'])

    async def handle_action_103(self):
        await self.handle_train('Stalker', ['Gateway', 'Assimilator', 'CyberneticsCore'])

    async def handle_action_104(self):
        await self.handle_train('Adept', ['Gateway', 'Assimilator', 'CyberneticsCore'])

    async def handle_action_105(self):
        await self.handle_train('HighTemplar', ['Gateway', 'TemplarArchive'])

    async def handle_action_106(self):
        await self.handle_train('DarkTemplar', ['Gateway', 'DarkShrine'])

    async def handle_action_107(self):
        await self.handle_train('Observer', ['RoboticsFacility'])

    async def handle_action_108(self):
        await self.handle_train('WarpPrism', ['RoboticsFacility'])

    async def handle_action_109(self):
        await self.handle_train('Immortal', ['RoboticsFacility'])

    async def handle_action_110(self):
        await self.handle_train('Colossus', ['RoboticsFacility', 'RoboticsBay'])

    async def handle_action_111(self):
        await self.handle_train('Disruptor', ['RoboticsFacility', 'RoboticsBay'])

    async def handle_action_112(self):
        await self.handle_train('Phoenix', ['Stargate'])

    async def handle_action_113(self):
        await self.handle_train('VoidRay', ['Stargate'])

    async def handle_action_114(self):
        await self.handle_train('Oracle', ['Stargate'])

    async def handle_action_115(self):
        await self.handle_train('Carrier', ['Stargate', 'FleetBeacon'])

    async def handle_action_116(self):
        await self.handle_train('Tempest', ['Stargate', 'FleetBeacon'])

    async def handle_action_117(self):
        await self.handle_train('Mothership', ['Nexus', 'FleetBeacon'])

    async def handle_action_118(self):
        await self.handle_train('Archon', ['RoboticsFacility', 'RoboticsBay'])

    async def handle_action_200(self):
        await self.handle_build('Nexus', [])

    async def handle_action_201(self):
        await self.handle_build('Assimilator', [])

    async def handle_action_202(self):
        await self.handle_build('Gateway', ['Pylon'])

    async def handle_action_203(self):
        await self.handle_build('RoboticsFacility', ['CyberneticsCore', 'Pylon'])

    async def handle_action_204(self):
        await self.handle_build('Stargate', ['CyberneticsCore', 'Pylon'])

    async def handle_action_205(self):
        await self.handle_build('Pylon', [])

    async def handle_action_206(self):
        await self.handle_build('Forge', ['Pylon'])

    async def handle_action_207(self):
        await self.handle_build('CyberneticsCore', ['Gateway', 'Pylon'])

    async def handle_action_208(self):
        await self.handle_build('PhotonCannon', ['Forge', 'Pylon'])

    async def handle_action_209(self):
        await self.handle_build('ShieldBattery', ['CyberneticsCore', 'Pylon'])

    async def handle_action_210(self):
        await self.handle_build('TwilightCouncil', ['CyberneticsCore', 'Pylon'])

    async def handle_action_211(self):
        await self.handle_build('TemplarArchive', ['TwilightCouncil', 'CyberneticsCore', 'Pylon'])

    async def handle_action_212(self):
        await self.handle_build('DarkShrine', ['TwilightCouncil', 'CyberneticsCore', 'Pylon'])

    async def handle_action_213(self):
        await self.handle_build('FleetBeacon', ['Stargate', 'CyberneticsCore', 'Pylon'])

    async def handle_action_214(self):
        await self.handle_build('RoboticsBay', ['RoboticsFacility', 'CyberneticsCore', 'Pylon'])

    async def handle_action_300(self):
        await self.handle_research('ProtossGroundWeaponsLevel1', ['Forge'])

    async def handle_action_301(self):
        await self.handle_research('ProtossGroundWeaponsLevel2', ['Forge'])

    async def handle_action_302(self):
        await self.handle_research('ProtossGroundWeaponsLevel3', ['Forge'])

    async def handle_action_303(self):
        await self.handle_research('ProtossGroundArmorsLevel1', ['Forge'])

    async def handle_action_304(self):
        await self.handle_research('ProtossGroundArmorsLevel2', ['Forge'])

    async def handle_action_305(self):
        await self.handle_research('ProtossGroundArmorsLevel3', ['Forge'])

    async def handle_action_306(self):
        await self.handle_research('ProtossShieldsLevel1', ['Forge'])

    async def handle_action_307(self):
        await self.handle_research('ProtossShieldsLevel2', ['Forge'])

    async def handle_action_308(self):
        await self.handle_research('ProtossShieldsLevel3', ['Forge'])

    async def handle_action_309(self):
        await self.handle_research('ProtossAirWeaponsLevel1', ['CyberneticsCore'])

    async def handle_action_310(self):
        await self.handle_research('ProtossAirWeaponsLevel2', ['CyberneticsCore', 'FleetBeacon'])

    async def handle_action_311(self):
        await self.handle_research('ProtossAirWeaponsLevel3', ['CyberneticsCore', 'FleetBeacon'])

    async def handle_action_312(self):
        await self.handle_research('ProtossAirArmorsLevel1', ['CyberneticsCore'])

    async def handle_action_313(self):
        await self.handle_research('ProtossAirArmorsLevel2', ['CyberneticsCore', 'FleetBeacon'])

    async def handle_action_314(self):
        await self.handle_research('ProtossAirArmorsLevel3', ['CyberneticsCore', 'FleetBeacon'])

    async def handle_action_315(self):
        await self.handle_research('WarpGateResearch', ['CyberneticsCore'])

    async def handle_action_316(self):
        await self.handle_research('Charge', ['TwilightCouncil'])

    async def handle_action_317(self):
        await self.handle_research('BlinkTech', ['TwilightCouncil'])

    async def handle_action_318(self):
        await self.handle_research('AdeptPiercingAttack', ['TwilightCouncil'])

    async def handle_action_319(self):
        await self.handle_research('PsiStormTech', ['TemplarArchive'])

    async def handle_action_320(self):
        await self.handle_research('DarkTemplarBlinkUpgrade', ['DarkShrine'])

    async def handle_action_321(self):
        await self.handle_research('ObserverGraviticBooster', ['RoboticsBay', 'RoboticsFacility'])

    async def handle_action_322(self):
        await self.handle_research('GraviticDrive', ['RoboticsBay', 'RoboticsFacility'])

    async def handle_action_323(self):
        await self.handle_research('ExtendedThermalLance', ['RoboticsBay', 'RoboticsFacility'])

    async def handle_action_324(self):
        await self.handle_research('PhoenixRangeUpgrade', ['FleetBeacon', 'Stargate'])

    async def handle_action_325(self):
        await self.handle_research('VoidRaySpeedUpgrade', ['FleetBeacon', 'Stargate'])
