import math
import utils
import random
from bot import HIMA
from constants import MAP_RAMPS
from sc2.position import Point2
from sc2.data import ActionResult
from sc2.ids.buff_id import BuffId
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId


class Zerg_Bot(HIMA):
    def __init__(self, args):
        self.warning_range = 20
        self.townhall_queens = {}
        self.scout_unit = UnitTypeId.DRONE
        self.troop = utils.Troop(self, attack_threshold=60)
        super().__init__(args)

    async def unit_attack(self, units):
        for unit in units:
            ability = await self.get_available_abilities(unit)
            enemy, distance = (self.enemy_units + self.enemy_structures).closest_to(unit), (self.enemy_units + self.enemy_structures).closest_distance_to(unit)
            if unit.type_id == UnitTypeId.BANELING:
                if not unit.is_attacking:
                    unit.attack(enemy.position)
            else:
                unit.attack(enemy)
            if unit.type_id == UnitTypeId.QUEEN:
                ability = await self.get_available_abilities(unit)
                if AbilityId.TRANSFUSION_TRANSFUSION in ability:
                    for target in self.units + self.structures:
                        if target.health_max - target.health > 70 and not target.has_buff(BuffId.TRANSFUSION) and target.distance_to(unit) < 7:
                            unit(AbilityId.TRANSFUSION_TRANSFUSION, target)
                            break
            elif unit.type_id == UnitTypeId.ROACH:
                ability = await self.get_available_abilities(unit)
                if unit.is_burrowed and unit.health_percentage > 0.7:
                    if AbilityId.BURROWUP_ROACH in ability:
                        unit(AbilityId.BURROWUP_ROACH)
                if not unit.is_burrowed and unit.health_percentage < 0.4:
                    if AbilityId.BURROWDOWN_ROACH in ability:
                        unit(AbilityId.BURROWDOWN_ROACH)
            elif unit.type_id == UnitTypeId.RAVAGER:
                ability = await self.get_available_abilities(unit)
                if distance < 9:
                    if AbilityId.EFFECT_CORROSIVEBILE in ability:
                        # TODO: can attack building
                        stormable_enemies = [enemy.position for enemy in self.enemy_units if enemy.distance_to(unit) < 9]
                        if stormable_enemies:
                            unit(AbilityId.EFFECT_CORROSIVEBILE, Point2.center(stormable_enemies))
            elif unit.type_id == UnitTypeId.INFESTOR:
                ability = await self.get_available_abilities(unit)
                if distance < 10:
                    if AbilityId.FUNGALGROWTH_FUNGALGROWTH in ability:
                        stormable_enemies = [enemy.position for enemy in self.enemy_units if enemy.distance_to(unit) < 10]
                        unit(AbilityId.FUNGALGROWTH_FUNGALGROWTH, Point2.center(stormable_enemies))
                if distance < 8:
                    if AbilityId.NEURALPARASITE_NEURALPARASITE in ability:
                        unit(AbilityId.NEURALPARASITE_NEURALPARASITE, self.enemy_units.closest_to(unit))

    async def find_placement(self, building, near, max_distance=20, random_alternative=False, placement_step=3,
                             min_distance=0, minDistanceToResources=3):
        if await self.can_place(building, near):
            return near

        for distance in range(min_distance, max_distance, placement_step):
            possible_positions = [Point2(p).offset(near).to2 for p in (
                    [(dx, -distance) for dx in range(-distance, distance + 1, placement_step)] +
                    [(dx, distance) for dx in range(-distance, distance + 1, placement_step)] +
                    [(-distance, dy) for dy in range(-distance, distance + 1, placement_step)] +
                    [(distance, dy) for dy in range(-distance, distance + 1, placement_step)]
            )]
            if (self.townhalls | self.mineral_field | self.vespene_geyser).exists and minDistanceToResources > 0:
                possible_positions = [x for x in possible_positions if (self.mineral_field | self.vespene_geyser).closest_to(x).distance_to(x) >= minDistanceToResources]

            res = await self._client.query_building_placement(self._game_data.units[building.value].creation_ability, possible_positions)
            possible = [p for r, p in zip(res, possible_positions) if r == ActionResult.Success]
            if not possible:
                continue

            if random_alternative:
                return random.choice(possible)
            else:
                return min(possible, key=lambda p: p.distance_to(near))

    async def handle_train(self, tgt, prerequisites, morph=None):
        action_id = self.action.r_dict[tgt]
        if not self.check_condition(action_id, tgt, prerequisites):
            return
        if tgt == 'Overlord':
            if self.supply_cap == 200:
                return self.record_failure(action_id, 'supply_200')
            if self.supply_left > 30:
                return self.record_failure(action_id, 'supply_many')
        elif tgt == 'Queen':
            if self.townhalls.random.train(UnitTypeId[tgt.upper()]):
                return self.record_succeed(action_id)
            else:
                return self.record_failure(action_id, 'fail_train')
        if morph:
            if not self.units(UnitTypeId[morph.upper()]).exists:
                return self.record_failure(action_id, 'fail')
            idle_units = [u for u in self.units(UnitTypeId[morph.upper()]) if u.is_idle]
            if idle_units:
                closest_unit = min(idle_units, key=lambda u: u.distance_to(self.start_location))
            else:
                closest_unit = min(self.units(UnitTypeId[morph.upper()]), key=lambda u: u.distance_to(self.start_location))
            closest_unit.build(UnitTypeId[tgt.upper()])
            return self.record_succeed(action_id)
        else:
            if not self.larva:
                return self.record_failure(action_id, 'not_afford')
            if self.larva.random.train(UnitTypeId[tgt.upper()]):
                return self.record_succeed(action_id)
            else:
                return self.record_failure(action_id, 'fail_train')

    async def race_specific_tactic(self):
        if not self.townhalls:
            return
        tmp = self.townhall_queens.copy()
        self.townhall_queens = {}
        for townhall in self.townhalls:
            if townhall not in tmp:
                self.townhall_queens[townhall] = None
            elif not tmp[townhall]:
                self.townhall_queens[townhall] = None
            else:
                exist = False
                for unit in self.units(UnitTypeId.QUEEN):
                    if unit.tag == tmp[townhall].tag:
                        exist = True
                        self.townhall_queens[townhall] = unit
                        break
                if not exist:
                    self.townhall_queens[townhall] = None

        for townhall in self.townhalls:
            if not self.townhall_queens[townhall]:
                for unit in self.units(UnitTypeId.QUEEN):
                    if unit not in self.townhall_queens.values():
                        self.townhall_queens[townhall] = unit
                        break
            if self.townhall_queens[townhall]:
                queen = self.townhall_queens[townhall]
                if queen.distance_to(townhall) > 7:
                    queen.move(townhall)
                elif queen.energy > 25 and self.supply_left > 0:
                    queen(AbilityId.EFFECT_INJECTLARVA, townhall)

        for unit in self.units(UnitTypeId.QUEEN):
            ability = await self.get_available_abilities(unit)
            if unit not in self.townhall_queens.values() and unit.is_idle and AbilityId.BUILD_CREEPTUMOR_QUEEN in ability:
                unit(AbilityId.BUILD_CREEPTUMOR_QUEEN, self.find_creep(unit.position))

        # Since python-sc2 doesn't support morphing Zerglings into Banelings, we manually kill 2 Zerglings and spawn 1 Baneling instead.
        if self.structures(UnitTypeId.BANELINGNEST).exists:
            if self.units(UnitTypeId.ZERGLING).amount >= 2:
                zergling = self.units(UnitTypeId.ZERGLING).random
                await self.client.debug_kill_unit([zergling.tag])
                await self.client.debug_kill_unit([self.units(UnitTypeId.ZERGLING).random.tag])
                await self.client.debug_create_unit([
                    [UnitTypeId.BANELING, 1, zergling.position, self.player_id]
                ])
        else:
            if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.vespene > 50:
                if 207 not in self.action_queue:
                    self.action_queue.appendleft(207)

    def find_creep(self, target):
        for _ in range(100):
            for _ in range(100):
                target_cand = target.towards_with_random_angle(self.enemy_start_locations[0], 3, math.pi / 2).rounded
                if self.game_info.pathing_grid[target_cand]:
                    break
            if self.has_creep(target_cand):
                target = target_cand
                continue
            else:
                break
        return target

    async def action_build(self, tgt):
        action_id = self.action.r_dict[tgt]
        step = 6 if action_id <= 22 else 4 if action_id <= 30 else 3
        if tgt == 'Lair':
            townhall = sorted([townhall for townhall in self.structures(UnitTypeId.HATCHERY)], key=lambda x: x.distance_to(self.start_location))[0]
            if not townhall.is_idle:
                return self.record_failure(action_id, 'not_afford')
            townhall.build(UnitTypeId.LAIR)
            return self.record_succeed(action_id)
        elif tgt == 'Hive':
            lairs = self.structures(UnitTypeId.LAIR).ready
            for lair in lairs:
                if lair.is_idle:
                    lair.build(UnitTypeId.HIVE)
                    return self.record_succeed(action_id)
            return self.record_failure(action_id, 'fail_hive')
        elif tgt == 'GreaterSpire':
            spires = self.structures(UnitTypeId.SPIRE).ready
            if not spires:
                return self.record_failure(action_id, 'not_ready')
            abilities = await self.get_available_abilities(spires)
            for spire in spires:
                if spire.is_idle:
                    if AbilityId.UPGRADETOGREATERSPIRE_GREATERSPIRE not in abilities:
                        self.record_failure(action_id, 'no_ability')
                spire.build(UnitTypeId.GREATERSPIRE)
                return self.record_succeed(action_id)
        elif tgt in ['SpineCrawler', 'SporeCrawler']:
            nexuses = sorted([townhall for townhall in self.townhalls], key=lambda x: x.distance_to(self.enemy_start_locations[0]))[:2]
            ramps = [sorted(MAP_RAMPS, key=lambda ramp: ramp.distance_to(nexus))[:2] for nexus in nexuses]
            ramps = set([ramp for _ramp in ramps for ramp in _ramp if self.has_creep(ramp.position)])
            if not ramps:
                return self.record_failure(action_id, 'no_creep')
            ramp = min(ramps, key=lambda r: r.distance_to(self.enemy_start_locations[0]))
            res = await self.build(UnitTypeId[tgt.upper()], near=self.find_creep(ramp.position))
            if res:
                self.record_succeed(action_id)
            else:
                self.record_failure(action_id, 'fail')
        elif tgt in ['SpawningPool', 'RoachWarren', 'BanelingNest', 'HydraliskDen']:
            worker_candidates = self.workers.filter(lambda worker: (worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)
            place_postion = self.start_location.position + Point2((self.location * 10, 0))
            placement_position = await self.find_placement(UnitTypeId[tgt.upper()], near=place_postion, placement_step=step)
            if not placement_position:
                return self.record_failure(action_id, 'fail')
            build_worker = worker_candidates.closest_to(placement_position)
            build_worker.build(UnitTypeId[tgt.upper()], placement_position)
            self.record_succeed(action_id)
        else:
            townhall = sorted([townhall for townhall in self.townhalls], key=lambda x: x.distance_to(self.enemy_start_locations[0]))[0]
            placement_position = await self.find_placement(UnitTypeId[tgt.upper()], near=townhall.position, placement_step=step)
            if not placement_position:
                return self.record_failure(action_id, 'fail')
            res = await self.build(UnitTypeId[tgt.upper()], near=placement_position)
            if res:
                self.record_succeed(action_id)
            else:
                self.record_failure(action_id, 'fail')

    async def handle_action_100(self):
        await self.handle_train('Drone', [])

    async def handle_action_101(self):
        await self.handle_train('Overlord', [])

    async def handle_action_102(self):
        await self.handle_train('Zergling', ['SpawningPool'])

    async def handle_action_103(self):
        await self.handle_train('Queen', ['Hatchery', 'SpawningPool'])

    async def handle_action_104(self):
        await self.handle_train('Roach', ['RoachWarren'])

    async def handle_action_105(self):
        # not supported by python-sc2
        return

    async def handle_action_106(self):
        await self.handle_train('Ravager', ['RoachWarren'], 'Roach')

    async def handle_action_107(self):
        await self.handle_train('Overseer', ['Lair'], 'Overlord')

    async def handle_action_108(self):
        await self.handle_train('Hydralisk', ['HydraliskDen'])

    async def handle_action_109(self):
        await self.handle_train('Mutalisk', ['Spire'])

    async def handle_action_110(self):
        await self.handle_train('Corruptor', ['Spire'])

    async def handle_action_111(self):
        await self.handle_train('Infestor', ['Spire'])

    async def handle_action_112(self):
        await self.handle_train('SwarmHostMP', ['EvolutionChamber'])

    async def handle_action_113(self):
        await self.handle_train('LurkerMP', ['LurkerDenMP'], 'Hydralisk')

    async def handle_action_114(self):
        await self.handle_train('Viper', ['Hive'])

    async def handle_action_115(self):
        await self.handle_train('BroodLord', ['GreaterSpire'], 'Corruptor')

    async def handle_action_116(self):
        await self.handle_train('Ultralisk', ['UltraliskCavern'])

    async def handle_action_200(self):
        await self.handle_build('Hatchery', [])

    async def handle_action_201(self):
        await self.handle_build('Extractor', [])

    async def handle_action_202(self):
        await self.handle_build('Lair', ['SpawningPool'])

    async def handle_action_203(self):
        await self.handle_build('Hive', ['Lair', 'InfestationPit'])

    async def handle_action_204(self):
        await self.handle_build('SpawningPool', [])

    async def handle_action_205(self):
        await self.handle_build('EvolutionChamber', [])

    async def handle_action_206(self):
        await self.handle_build('RoachWarren', ['SpawningPool'])

    async def handle_action_207(self):
        await self.handle_build('BanelingNest', ['SpawningPool'])

    async def handle_action_208(self):
        await self.handle_build('SpineCrawler', ['SpawningPool'])

    async def handle_action_209(self):
        await self.handle_build('SporeCrawler', ['SpawningPool'])

    async def handle_action_210(self):
        await self.handle_build('HydraliskDen', ['Lair'])

    async def handle_action_211(self):
        await self.handle_build('InfestationPit', ['Lair'])

    async def handle_action_212(self):
        await self.handle_build('LurkerDenMP', ['HydraliskDen'])

    async def handle_action_213(self):
        await self.handle_build('Spire', ['Lair'])

    async def handle_action_214(self):
        return

    async def handle_action_215(self):
        await self.handle_build('UltraliskCavern', ['Hive'])

    async def handle_action_216(self):
        await self.handle_build('GreaterSpire', ['Hive'])

    async def handle_action_300(self):
        await self.handle_research('ZergMeleeWeaponsLevel1', ['EvolutionChamber'])

    async def handle_action_301(self):
        await self.handle_research('ZergMeleeWeaponsLevel2', ['EvolutionChamber', 'Lair'])

    async def handle_action_302(self):
        await self.handle_research('ZergMeleeWeaponsLevel3', ['EvolutionChamber', 'Hive'])

    async def handle_action_303(self):
        await self.handle_research('ZergMissileWeaponsLevel1', ['EvolutionChamber'])

    async def handle_action_304(self):
        await self.handle_research('ZergMissileWeaponsLevel2', ['EvolutionChamber', 'Lair'])

    async def handle_action_305(self):
        await self.handle_research('ZergMissileWeaponsLevel3', ['EvolutionChamber', 'Hive'])

    async def handle_action_306(self):
        await self.handle_research('ZergGroundArmorsLevel1', ['EvolutionChamber'])

    async def handle_action_307(self):
        await self.handle_research('ZergGroundArmorsLevel2', ['EvolutionChamber', 'Lair'])

    async def handle_action_308(self):
        await self.handle_research('ZergGroundArmorsLevel3', ['EvolutionChamber', 'Hive'])

    async def handle_action_309(self):
        await self.handle_research('ZergFlyerWeaponsLevel1', ['Spire'])

    async def handle_action_310(self):
        await self.handle_research('ZergFlyerWeaponsLevel2', ['Spire'])

    async def handle_action_311(self):
        await self.handle_research('ZergFlyerWeaponsLevel3', ['Spire', 'Hive'])

    async def handle_action_312(self):
        await self.handle_research('ZergFlyerArmorsLevel1', ['Spire'])

    async def handle_action_313(self):
        await self.handle_research('ZergFlyerArmorsLevel2', ['Spire'])

    async def handle_action_314(self):
        await self.handle_research('ZergFlyerArmorsLevel3', ['Spire', 'Hive'])

    async def handle_action_315(self):
        await self.handle_research('Burrow', ['Hatchery'])

    async def handle_action_316(self):
        await self.handle_research('overlordspeed', ['Hatchery'])

    async def handle_action_317(self):
        await self.handle_research('zerglingmovementspeed', ['SpawningPool'])

    async def handle_action_318(self):
        await self.handle_research('zerglingattackspeed', ['SpawningPool', 'Hive'])

    async def handle_action_319(self):
        await self.handle_research('GlialReconstitution', ['RoachWarren', 'Lair'])

    async def handle_action_320(self):
        await self.handle_research('TunnelingClaws', ['RoachWarren', 'Lair'])

    async def handle_action_321(self):
        await self.handle_research('CentrificalHooks', ['BanelingNest', 'Lair'])

    async def handle_action_322(self):
        await self.handle_research('EvolveMuscularAugments', ['HydraliskDen'])

    async def handle_action_323(self):
        await self.handle_research('EvolveGroovedSpines', ['HydraliskDen'])

    async def handle_action_324(self):
        await self.handle_research('NeuralParasite', ['InfestationPit'])

    async def handle_action_325(self):
        await self.handle_research('DiggingClaws', ['LurkerDenMP', 'Hive'])

    async def handle_action_326(self):
        await self.handle_research('LurkerRange', ['LurkerDenMP', 'Hive'])

    async def handle_action_327(self):
        await self.handle_research('ChitinousPlating', ['UltraliskCavern'])

    async def handle_action_328(self):
        await self.handle_research('AnabolicSynthesis', ['UltraliskCavern'])

    async def distribute_workers(self, resource_ratio: float = 2) -> None:
        if not self.mineral_field or not self.workers or not self.townhalls.ready:
            return
        worker_pool = self.workers.idle
        bases = self.townhalls.ready
        gas_buildings = self.gas_buildings.ready

        deficit_mining_places = []

        for mining_place in bases | gas_buildings:
            difference = mining_place.surplus_harvesters
            if not difference:
                continue
            if mining_place.has_vespene:
                local_workers = self.workers.filter(
                    lambda unit: unit.order_target == mining_place.tag
                    or (unit.is_carrying_vespene and unit.order_target == bases.closest_to(mining_place).tag)
                )
                if difference > 0:
                    for worker in local_workers[:difference]:
                        worker_pool.append(worker)
            else:
                local_minerals_tags = {
                    mineral.tag for mineral in self.mineral_field if mineral.distance_to(mining_place) <= 8
                }
                local_workers = self.workers.filter(
                    lambda unit: unit.order_target in local_minerals_tags
                    or (unit.is_carrying_minerals and unit.order_target == mining_place.tag)
                )
                if difference > 0:
                    for worker in local_workers[:difference]:
                        worker_pool.append(worker)
            if difference < 0:
                deficit_mining_places += [mining_place for _ in range(-difference)]

        mining_place_tags = {base.tag for base in self.townhalls.ready}
        local_minerals_tags = {
            mineral.tag
            for base in bases
            for mineral in self.mineral_field
            if mineral.distance_to(base) <= 8
        }

        local_workers = self.workers.filter(
            lambda unit: unit.order_target in local_minerals_tags
            or (unit.is_carrying_minerals and unit.order_target in mining_place_tags)
        )
        gas_worker = sum(place.assigned_harvesters for place in self.gas_buildings)
        gas_ready = sum(place.is_ready for place in self.gas_buildings)
        max_gas_workers = gas_ready * 3
        if gas_ready > 0 and max_gas_workers - gas_worker > 1:
            needed = max_gas_workers - gas_worker
            worker_pool.extend(local_workers[:needed])

        for worker in worker_pool:
            if deficit_mining_places:
                vespene_places = [place for place in deficit_mining_places if place.vespene_contents]
                if vespene_places:
                    current_place = min(vespene_places, key=lambda place: place.distance_to(worker))
                else:
                    current_place = min(deficit_mining_places, key=lambda place: place.distance_to(worker))
                deficit_mining_places.remove(current_place)
                if current_place.vespene_contents:
                    worker.gather(current_place)
                else:
                    local_minerals = (
                        mineral for mineral in self.mineral_field if mineral.distance_to(current_place) <= 8
                    )
                    target_mineral = max(local_minerals, key=lambda mineral: mineral.mineral_contents, default=None)
                    if target_mineral:
                        worker.gather(target_mineral)
