import utils
import random
from bot import HIMA
from constants import MAP_RAMPS
from sc2.position import Point2
from sc2.ids.buff_id import BuffId
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId


class Terran_Bot(HIMA):
    def __init__(self, args):
        self.warning_range = 10
        self.scout_unit = UnitTypeId.SCV
        self.troop = utils.Troop(self)
        super().__init__(args)

    async def unit_attack(self, units):
        for unit in units:
            ability = await self.get_available_abilities(unit)
            enemy_units = self.enemy_units.filter(lambda u: 'Changeling' not in u.name)
            if not enemy_units:
                continue
            enemy, distance = (enemy_units + self.enemy_structures).closest_to(unit), (enemy_units + self.enemy_structures).closest_distance_to(unit)
            unit.attack(enemy)
            if unit.type_id == UnitTypeId.MARINE:
                if not unit.has_buff(BuffId.STIMPACK) and unit.shield_health_percentage > 0.5:
                    ability = await self.get_available_abilities(unit)
                    if AbilityId.EFFECT_STIM_MARINE in ability:
                        unit(AbilityId.EFFECT_STIM_MARINE)
            elif unit.type_id == UnitTypeId.MARAUDER:
                if not unit.has_buff(BuffId.STIMPACKMARAUDER) and unit.shield_health_percentage > 0.5:
                    ability = await self.get_available_abilities(unit)
                    if AbilityId.EFFECT_STIM_MARAUDER in ability:
                        unit(AbilityId.EFFECT_STIM_MARAUDER)
            elif unit.type_id == UnitTypeId.REAPER:
                if distance < 5:
                    ability = await self.get_available_abilities(unit)
                    if AbilityId.KD8CHARGE_KD8CHARGE in ability:
                        unit(AbilityId.KD8CHARGE_KD8CHARGE, self.enemy_units.closest_to(unit).position)
            elif unit.type_id == UnitTypeId.VIKINGFIGHTER:
                air_units = self.enemy_units.filter(lambda u: u.is_flying and u.distance_to(unit) < 9)
                if air_units:
                    unit(AbilityId.MORPH_VIKINGASSAULTMODE)
            elif unit.type_id == UnitTypeId.VIKINGASSAULT:
                air_units = self.enemy_units.filter(lambda u: u.is_flying and u.distance_to(unit) < 9)
                if not air_units:
                    unit(AbilityId.MORPH_VIKINGFIGHTERMODE)
            elif unit.type_id == UnitTypeId.RAVEN:
                if distance < 10:
                    ability = await self.get_available_abilities(unit)
                    if AbilityId.EFFECT_ANTIARMORMISSILE in ability:
                        unit(AbilityId.EFFECT_ANTIARMORMISSILE, self.enemy_units.closest_to(unit).position)
            elif unit.type_id == UnitTypeId.LIBERATOR:
                air_units = self.enemy_units.filter(lambda u: u.is_flying and u.distance_to(unit) < 5)
                if not air_units:
                    unit(AbilityId.MORPH_LIBERATORAGMODE, self.enemy_units.closest_to(unit))
            elif unit.type_id == UnitTypeId.LIBERATORAG:
                air_units = self.enemy_units.filter(lambda u: u.is_flying and u.distance_to(unit) < 5)
                if air_units or distance > 5:
                    unit(AbilityId.MORPH_LIBERATORAAMODE)
            elif unit.type_id == UnitTypeId.BATTLECRUISER:
                if distance < 10:
                    ability = await self.get_available_abilities(unit)
                    if AbilityId.YAMATO_YAMATOGUN in ability:
                        unit(AbilityId.YAMATO_YAMATOGUN, self.enemy_units.closest_to(unit))

    async def handle_train(self, tgt, prerequisites):
        action_id = self.action.r_dict[tgt]
        if not self.check_condition(action_id, tgt, prerequisites):
            return
        if tgt == 'SCV':
            building = [gate for gate in self.townhalls if gate.is_ready]
            sorted_buildings = sorted(building, key=lambda g: (g.add_on_tag == 0, len(g.orders)))
            gate = sorted_buildings[0]
            if len(gate.orders) == 5:
                if not [structure for structure in self.townhalls if structure.build_progress != 1]:
                    return self.record_failure(action_id, 'not_exist', 200)
                else:
                    return self.record_failure(action_id, 'not_ready')
            if gate.train(UnitTypeId.SCV):
                return self.record_succeed(action_id)
            else:
                return self.record_failure(action_id, 'fail_train')

        id = self.action.r_dict[prerequisites[0]]
        building = [gate for gate in self.structures(UnitTypeId[prerequisites[0].upper()]) if gate.is_ready]
        sorted_buildings = sorted(building, key=lambda g: (g.add_on_tag == 0, len(g.orders)))
        gate = sorted_buildings[0]
        num = 8 if gate.has_reactor else 5
        if len(gate.orders) == num:
            if not [structure for structure in self.structures(UnitTypeId[prerequisites[0].upper()]) if structure.build_progress != 1]:
                return self.record_failure(action_id, 'not_exist', id)
            else:
                return self.record_failure(action_id, 'not_ready', id)
        if gate.train(UnitTypeId[tgt.upper()]):
            return self.record_succeed(action_id)
        else:
            return self.record_failure(action_id, 'fail_train')

    async def original_building_position_search(self, base_position, max_distance=15):
        candidate_positions = []
        for d in range(1, max_distance + 1):
            for dx in range(-d, d + 1):
                for dy in range(-d, d + 1):
                    candidate_positions.append(Point2((base_position[0] + dx, base_position[1] + dy)))
        random.shuffle(candidate_positions)
        for pos in candidate_positions:
            if self.is_position_valid_for_building(pos):
                return pos
        return None

    def is_position_valid_for_building(self, position):
        map_size = self.game_info.map_size
        if (
            position.x < 25 or
            position.y < 25 or
            position.x > map_size.x - 25 or
            position.y > map_size.y - 25
        ):
            return False
        for structure in self.structures:
            if structure in self.structures({UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT, UnitTypeId.REFINERY, UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS}):
                if position.distance_to(structure.position) < 8:
                    return False
            else:
                if position.distance_to(structure.position) < 3:
                    return False
        for resource in self.resources:
            if position.distance_to(resource.position) < 5:
                return False
        for location in self.expansion_locations_list:
            if position.distance_to(location) < 10:
                return False
        for location in MAP_RAMPS:
            if position.distance_to(location) < 5:
                return False
        return True

    def find_best_base_for_building(self, building_type):
        base_positions = [base.position for base in self.townhalls]
        building_positions = [building.position for building in self.structures(building_type)]
        base_building_counts = {}
        for base_pos in base_positions:
            count = sum(1 for building_pos in building_positions if base_pos.distance_to(building_pos) <= 12)
            base_building_counts[base_pos] = count
        sorted_bases = sorted(base_building_counts.keys(), key=lambda base: base_building_counts[base])
        return sorted_bases

    async def action_build(self, tgt):
        action_id = self.action.r_dict[tgt]
        if tgt in ['SupplyDepot', 'EngineeringBay', 'FusionCore']:
            if tgt == 'SupplyDepot':
                if self.supply_cap == 200:
                    return self.record_failure(action_id, 'supply_200')
                pending_supply = self.already_pending(UnitTypeId.COMMANDCENTER) * 15 + self.already_pending(UnitTypeId.SUPPLYDEPOT) * 8
                if self.supply_left > 30 - pending_supply:
                    return self.record_failure(action_id, 'supply_many')
            res = await self.build(UnitTypeId[tgt.upper()], near=utils.MAP_POINTS[self.start_position])
            if res:
                self.record_succeed(action_id)
            else:
                self.record_failure(action_id, 'fail')
        elif tgt in ['MissileTurret', 'SensorTower']:
            nexuses = sorted(self.townhalls, key=lambda base: base.distance_to(self.enemy_start_locations[0]))[:2]
            ramps = [sorted(MAP_RAMPS, key=lambda ramp: ramp.distance_to(nexus))[:2] for nexus in nexuses]
            ramps = set([ramp for _ramp in ramps for ramp in _ramp])
            ramp = random.choice(list(ramps))
            for ramp in self.game_info.map_ramps[MAP_RAMPS.index(ramp)].points:
                res = await self.build(UnitTypeId[tgt.upper()], near=ramp)
                if res:
                    self.record_succeed(action_id)
                    return
            self.record_failure(action_id, 'no_ramp')
        elif tgt == 'PlanetaryFortress':
            target = self.structures(UnitTypeId.COMMANDCENTER).random
            if target(AbilityId.UPGRADETOPLANETARYFORTRESS_PLANETARYFORTRESS):
                self.record_succeed(action_id)
                return
            self.record_failure(action_id, 'fail')
        elif 'Reactor' in tgt:
            for building in self.structures(UnitTypeId[tgt[:-7].upper()]).ready.idle:
                if building.has_add_on:
                    continue
                building.build(UnitTypeId[tgt.upper()])
                self.record_succeed(action_id)
            self.record_failure(action_id, 'not_ready')
        elif 'Techlab' in tgt:
            for building in self.structures(UnitTypeId[tgt[:-7].upper()]).ready.idle:
                if building.has_add_on:
                    continue
                building.build(UnitTypeId[tgt.upper()])
                self.record_succeed(action_id)
            self.record_failure(action_id, 'not_ready')
        else:
            best_base = self.find_best_base_for_building(UnitTypeId[tgt.upper()])[0]
            best_position = await self.original_building_position_search(best_base)
            if not best_position:
                for base in self.townhalls:
                    if base.position != best_base.position:
                        best_position = await self.original_building_position_search(base.position)
                        if best_position:
                            break
            if not best_position:
                return self.record_failure(action_id, 'no_space')

            res = await self.build(UnitTypeId[tgt.upper()], near=best_position)
            if res:
                self.record_succeed(action_id)
            else:
                self.record_failure(action_id, 'fail')

    async def lift_building(self, building):
        if AbilityId.BUILD_TECHLAB_FACTORY in await self.get_available_abilities(building):
            building.build(UnitTypeId.FACTORYTECHLAB)
        else:
            building(AbilityId.LIFT, queue=False)
    
    async def land_building(self, building):
        if AbilityId.LAND in [i.ability.id for i in building.orders]:
            return
        for offset in range(4, 12):
            pos = building.position.towards(self.start_location, offset)
            pos = await self.find_placement(UnitTypeId.FACTORY, near=pos, placement_step=1)
            if pos and await self.can_place(UnitTypeId.FACTORY, pos):
                building(AbilityId.LAND, pos)
                return
            
    async def race_specific_tactic(self):
        if self.structures(UnitTypeId.ARMORY).exists:
            for unit in self.units(UnitTypeId.HELLION):
                unit(AbilityId.MORPH_HELLBAT)
        elif self.structures(UnitTypeId.FACTORY).exists and not self.already_pending(UnitTypeId.ARMORY):
            await self.handle_build('Armory', ['Factory'])
        for unit in self.units(UnitTypeId.VIKINGFIGHTER):
            unit(AbilityId.MORPH_VIKINGASSAULTMODE)
        for fac in self.structures(UnitTypeId.FACTORY).ready.idle:
            if not fac.has_techlab:
                await self.lift_building(fac)
        for fac in self.structures(UnitTypeId.FACTORYFLYING):
            await self.land_building(fac)
        if self.structures(UnitTypeId.ORBITALCOMMAND).exists:
            for oc in self.structures(UnitTypeId.ORBITALCOMMAND).ready.filter(lambda x: x.energy >= 50):
                mfs = self.mineral_field.closer_than(20, oc)
                if mfs:
                    mf = max(mfs, key=lambda x: x.mineral_contents)
                    oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf)
        elif self.structures(UnitTypeId.BARRACKS).exists:
            target = self.townhalls[0]
            target(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)
                    
    async def handle_action_100(self):
        await self.handle_train('SCV', [])

    async def handle_action_101(self):
        # We manually train MULE whenever possible.
        return

    async def handle_action_102(self):
        await self.handle_train('Marine', ['Barracks'])

    async def handle_action_103(self):
        await self.handle_train('Reaper', ['Barracks', 'Refinery'])

    async def handle_action_104(self):
        await self.handle_train('Marauder', ['Barracks', 'BarracksTechLab'])

    async def handle_action_105(self):
        await self.handle_train('Ghost', ['Barracks', 'BarracksTechLab', 'GhostAcademy'])

    async def handle_action_106(self):
        await self.handle_train('Hellion', ['Factory'])

    async def handle_action_107(self):
        await self.handle_train('WidowMine', ['Factory'])

    async def handle_action_108(self):
        await self.handle_train('Cyclone', ['Factory', 'FactoryTechLab'])

    async def handle_action_109(self):
        await self.handle_train('SiegeTank', ['Factory', 'FactoryTechLab'])

    async def handle_action_110(self):
        await self.handle_train('Thor', ['Factory', 'FactoryTechLab', 'Armory'])

    async def handle_action_111(self):
        await self.handle_train('VikingFighter', ['Starport'])

    async def handle_action_112(self):
        await self.handle_train('Medivac', ['Starport'])

    async def handle_action_113(self):
        await self.handle_train('Liberator', ['Starport'])

    async def handle_action_114(self):
        await self.handle_train('Banshee', ['Starport', 'StarportTechLab'])

    async def handle_action_115(self):
        await self.handle_train('Raven', ['Starport', 'StarportTechLab'])

    async def handle_action_116(self):
        await self.handle_train('Battlecruiser', ['Starport', 'StarportTechLab', 'FusionCore'])

    async def handle_action_200(self):
        await self.handle_build('CommandCenter', [])

    async def handle_action_201(self):
        await self.handle_build('Refinery', [])

    async def handle_action_202(self):
        # We manually upgrade to OrbitalCommand right after Barrack is ready.
        return

    async def handle_action_203(self):
        await self.handle_build('PlanetaryFortress', ['CommandCenter', 'EngineeringBay'])

    async def handle_action_204(self):
        await self.handle_build('Barracks', ['SupplyDepot'])

    async def handle_action_205(self):
        await self.handle_build('Factory', ['Refinery', 'Barracks'])

    async def handle_action_206(self):
        await self.handle_build('Starport', ['Refinery', 'Factory'])

    async def handle_action_207(self):
        await self.handle_build('BarracksReactor', ['Barracks'])

    async def handle_action_208(self):
        await self.handle_build('BarracksTechLab', ['Barracks'])

    async def handle_action_209(self):
        await self.handle_build('FactoryReactor', ['Factory'])

    async def handle_action_210(self):
        # We manually build Factory and then attach a Techlab.
        return

    async def handle_action_211(self):
        await self.handle_build('StarportReactor', ['Starport'])

    async def handle_action_212(self):
        await self.handle_build('StarportTechLab', ['Starport'])

    async def handle_action_213(self):
        await self.handle_build('SupplyDepot', [])

    async def handle_action_214(self):
        await self.handle_build('EngineeringBay', [])

    async def handle_action_215(self):
        # We don't use it.
        return

    async def handle_action_216(self):
        await self.handle_build('MissileTurret', ['EngineeringBay'])

    async def handle_action_217(self):
        await self.handle_build('SensorTower', ['EngineeringBay'])

    async def handle_action_218(self):
        await self.handle_build('GhostAcademy', ['Barracks'])

    async def handle_action_219(self):
        # We manually build Factory and then build Armory.
        return

    async def handle_action_220(self):
        await self.handle_build('FusionCore', ['Starport'])

    async def handle_action_300(self):
        await self.handle_research('TerranInfantryWeaponsLevel1', ['EngineeringBay'])

    async def handle_action_301(self):
        await self.handle_research('TerranInfantryWeaponsLevel2', ['EngineeringBay', 'Armory'])

    async def handle_action_302(self):
        await self.handle_research('TerranInfantryWeaponsLevel3', ['EngineeringBay', 'Armory'])

    async def handle_action_303(self):
        await self.handle_research('TerranInfantryArmorsLevel1', ['EngineeringBay'])

    async def handle_action_304(self):
        await self.handle_research('TerranInfantryArmorsLevel2', ['EngineeringBay', 'Armory'])

    async def handle_action_305(self):
        await self.handle_research('TerranInfantryArmorsLevel3', ['EngineeringBay', 'Armory'])

    async def handle_action_306(self):
        await self.handle_research('TerranVehicleWeaponsLevel1', ['Armory'])

    async def handle_action_307(self):
        await self.handle_research('TerranVehicleWeaponsLevel2', ['Armory'])

    async def handle_action_308(self):
        await self.handle_research('TerranVehicleWeaponsLevel3', ['Armory'])

    async def handle_action_309(self):
        await self.handle_research('TerranShipWeaponsLevel1', ['Armory'])

    async def handle_action_310(self):
        await self.handle_research('TerranShipWeaponsLevel2', ['Armory'])

    async def handle_action_311(self):
        await self.handle_research('TerranShipWeaponsLevel3', ['Armory'])

    async def handle_action_312(self):
        await self.handle_research('TerranVehicleAndShipArmorsLevel1', ['Armory'])

    async def handle_action_313(self):
        await self.handle_research('TerranVehicleAndShipArmorsLevel2', ['Armory'])

    async def handle_action_314(self):
        await self.handle_research('TerranVehicleAndShipArmorsLevel3', ['Armory'])

    async def handle_action_315(self):
        await self.handle_research('TerranBuildingArmor', ['EngineeringBay'])

    async def handle_action_316(self):
        await self.handle_research('HiSecAutoTracking', ['EngineeringBay'])

    async def handle_action_317(self):
        await self.handle_research('Stimpack', ['BarracksTechLab'])

    async def handle_action_318(self):
        await self.handle_research('ShieldWall', ['BarracksTechLab'])

    async def handle_action_319(self):
        await self.handle_research('PunisherGrenades', ['BarracksTechLab'])

    async def handle_action_320(self):
        await self.handle_research('PersonalCloaking', ['GhostAcademy'])

    async def handle_action_321(self):
        await self.handle_research('SmartServos', ['FactoryTechLab'])

    async def handle_action_322(self):
        await self.handle_research('HighCapacityBarrels', ['FactoryTechLab'])

    async def handle_action_323(self):
        await self.handle_research('DrillClaws', ['FactoryTechLab'])

    async def handle_action_324(self):
        await self.handle_research('CycloneLockOnDamageUpgrade', ['FactoryTechLab'])

    async def handle_action_325(self):
        await self.handle_research('MedivacIncreaseSpeedBoost', ['FusionCore'])

    async def handle_action_326(self):
        await self.handle_research('LiberatorAGRangeUpgrade', ['FusionCore'])

    async def handle_action_327(self):
        await self.handle_research('BansheeCloak', ['StarportTechLab'])

    async def handle_action_328(self):
        await self.handle_research('BansheeSpeed', ['StarportTechLab'])

    async def handle_action_329(self):
        await self.handle_research('InterferenceMatrix', ['StarportTechLab'])

    async def handle_action_330(self):
        await self.handle_research('BattlecruiserEnableSpecializations', ['FusionCore'])

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
