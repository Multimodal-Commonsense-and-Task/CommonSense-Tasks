import json
from sc2.data import Race
from constants import ACTION_DICT


def get_action_list(race):
    action_dict = ACTION_DICT[race]
    action_list = []
    for key, value in action_dict.items():
        for _, action in value.items():
            action_list.append(action)
    return action_list


def get_system_prompt(race):
    action_list = get_action_list(race)
    if race == Race.Protoss:
        ep = 'Focus on mass-producing Carriers as fast as possible. To produce Carrier, you must need Fleetbeacon.'
    elif race == Race.Zerg:
        ep = 'Focus on mass-producing Mutalisks as fast as possible. To produce Mutalisk, you must need Spire.'
    elif race == Race.Terran:
        ep = 'Focus on mass-producing Thors as fast as possible. To produce Thor, you must need Armory.'
    return f"""
Your task is to analyze advisor recommendations and align them with these strategic options, considering current resources and game state.

Important Notes for Decision Making:
    - {ep}
    - Commands that are in production have already been executed, so never mention them.
    - Any previously failed commands from the last action sequence MUST be addressed and resolved as the FIRST commands in your new action sequence.
    - Your response must consider all analyses outlined below but state only the decision.

1. Current Assessment:
    1.1 Game Stage & Economy: Early/Mid/Late game status, resource status
    1.2 Military & Tech:
        - Current forces, technologies, and production capabilities
    1.3 Enemy Analysis: Known information and strategic implications

2. Advisor Strategy Resolution:
    2.1 Generating Agreed Viewpoints
    2.2 Generating Conflicted Viewpoints
    2.3 Resolving the Conflicts
    2.4 Generating Isolated Viewpoints
    2.5 Generating Final Strategic Direction

3. Strategy Formulation:
    3.1 Core Strategy: Main strategic direction based on advisor`s advice
    3.2 Unit Composition Strategy: Avoid over-commitment to single unit type
    3.3 Resource Allocation: Priority distribution of resources

Analysis Requirements:
    - Before presenting the Final Actions Summary:
        * Justify how your action sequence supports the chosen strategic path
        * Explain how you've addressed any previously failed commands

Action Format:
    - Every action must be an exact match from the provided action list {action_list}
    - Each action must be enclosed in angle brackets < >

Begin your action decisions with "Final Actions Summary:" followed by your recommended sequence of actions.
"""


def get_input_prompt(race):
    if race == Race.Protoss:
        return f"""
Game Status: {json.dumps({'supply_used': 87, 'supply_capacity': 109, 'unit': {'Probe': 46, 'Zealot': 2, 'Stalker': 1, 'Adept': 5, 'VoidRay': 1, 'Nexus': 3, 'Pylon': 8, 'Assimilator': 3, 'Gateway': 2, 'Stargate': 1}, 'research': ['WarpgateResearch'], 'failed action': 'Carrier', 'failed reason': 'FleetBeacon is not exist'})}
Suggestion A: Reason: **Immediate Steps:** Focus on stabilizing your economy by training more Probes (up to 60) to ensure a strong resource income for future unit production. **Short-Term Actions:** Transition into a carrier-heavy composition by continuously producing Voidrays and Carriers while also building additional Pylons to avoid supply block. **Long-Term Strategy:** Aim for a robust air force with 5-7 Carriers supported by Adepts and Sentries to counter ground threats while expanding your economy with more Nexuses and Assimilators.Actions: ['Probe', 'Probe', 'Probe', 'Probe', 'Pylon', 'Probe', 'VoidRay', 'Assimilator', 'Probe', 'VoidRay', 'Pylon', 'ProtossAirWeaponsLevel1', 'Probe', 'Probe', 'Probe', 'Assimilator', 'VoidRay', 'Probe', 'Probe', 'VoidRay', 'Probe', 'Pylon', 'Probe', 'Probe', 'Probe', 'Probe', 'Adept', 'VoidRay', 'Probe', 'RoboticsFacility', 'VoidRay', 'Probe', 'Probe', 'Probe', 'Probe', 'Probe', 'FleetBeacon', 'Probe', 'Stargate', 'Probe', 'Nexus', 'Pylon', 'Pylon', 'Probe', 'Probe', 'Probe', 'ShieldBattery', 'Probe', 'Probe', 'Probe', 'Pylon', 'Carrier', 'Carrier', 'Probe', 'Probe', 'Probe', 'Probe', 'Assimilator', 'Zealot', 'Assimilator', 'Probe', 'Probe']
Suggestion B: Reason: **Immediate Steps:** Focus on saturating your economy by training the queued Probes to reach optimal mineral and gas accumulation, ensuring you can support a larger army. **Short-Term Actions:** Prioritize building additional Pylons to avoid supply block and maintain unit production, while also deploying an Orthotomist for map control and harassment. **Long-Term Strategy:** Transition into a strong air force by continuously producing Carriers from your Fleet Beacon, while upgrading your air weapons to enhance your unit effectiveness for mid-game engagements.Actions: ['Probe', 'Probe', 'Probe', 'Adept', 'Adept', 'VoidRaySpeedUpgrade', 'Pylon', 'Probe', 'ShieldBattery', 'Probe', 'Pylon', 'Carrier', 'Probe', 'Probe', 'Pylon', 'Pylon', 'Nexus', 'Pylon', 'Carrier', 'Assimilator', 'Assimilator', 'Carrier', 'PhotonCannon', 'ShieldBattery', 'Probe', 'Probe', 'Probe', 'Carrier', 'Stargate', 'Carrier', 'PhotonCannon', 'Carrier', 'Pylon', 'ProtossAirWeaponsLevel1', 'Probe', 'Probe', 'VoidRay', 'Probe', 'Probe', 'PhotonCannon', 'Probe', 'Pylon', 'Probe', 'Probe', 'Probe', 'RoboticsFacility', 'Probe', 'Probe', 'PhotonCannon', 'Probe', 'Pylon']
Suggestion C: Reason: **Immediate Steps:** Focus on saturating your mineral lines by training all available Probes to maximize your economy; aim for 80+ mineral accumulation. **Short-term Actions:** In preparation for a potential push, ensure you have a healthy mix of units by warping in additional Zealots and Stalkers while also planning for a Rapid fire upgrade. **Long-term Strategy:** Transition into a balanced army composition with more Gateway units and tech up by researching upgrades at the Forge, while expanding to a fourth Nexus to secure your economy long-term.Actions: ['Probe', 'VoidRay', 'Gateway', 'Probe', 'Probe', 'Gateway', 'Probe', 'Probe', 'Assimilator', 'Assimilator', 'Probe', 'Probe', 'Probe', 'Zealot', 'Zealot', 'Probe', 'VoidRay', 'Observer', 'Nexus', 'Probe', 'Probe', 'Pylon', 'Probe', 'VoidRay', 'Observer', 'Pylon', 'Pylon', 'Probe', 'Pylon', 'Probe', 'Probe', 'Probe', 'Probe', 'CyberneticsCore', 'Probe', 'Probe', 'Pylon', 'Probe', 'Stalker', 'Stalker', 'Stalker', 'Probe', 'Probe', 'Gateway', 'Gateway', 'Pylon', 'Probe', 'ProtossGroundWeaponsLevel1', 'VoidRay', 'Probe', 'Probe', 'Pylon', 'Probe', 'VoidRay', 'Pylon', 'Probe']
"""
    elif race == Race.Zerg:
        return f"""
Game Status: {json.dumps({'supply_used': 177, 'supply_capacity': 186, 'unit': {'Drone': 67, 'Overlord': 21, 'Zergling': 22, 'Queen': 4, 'Roach': 36, 'Hatchery': 1, 'Extractor': 6, 'Lair': 2, 'SpawningPool': 1, 'RoachWarren': 1, 'SporeCrawler': 3}, 'research': []})}
Suggestion A: Reason: **Immediate Steps:** Focus on maximizing your economy by producing more Drones, aiming for around 75 total Drones, while also placing down Spore Crawlers for added defense against air threats. **Short-Term Actions:** Build additional Extractors to support gas-heavy units like Ravagers and Mutalisks; aim for 10 total Extractors to ensure a steady flow of gas. **Long-Term Strategy:** Transition into a Mutalisk-heavy composition while maintaining a solid Roach backbone for ground defense; consider teching to a Spire and expanding to additional bases to secure map control and maintain economic superiority. Actions: ["Drone", "Hatchery", "Extractor", "Extractor", "Drone", "Drone", "Drone", "Drone", "Drone", "Drone", "Overlord", "Overlord", "Drone", "Overlord", "Drone", "EvolutionChamber", "Drone", "Drone", "Drone", "Drone", "Drone", "Roach", "Drone", "Drone", "Lair", "Extractor", "Extractor", "Spire", "Drone", "Extractor", "Hatchery", "Drone", "Drone", "Drone", "Drone", "Overlord", "Roach", "RollingVipers", "Roach", "Spire", "Overlord", "Overlord", "Drone", "Drone", "Overlord", "Drone", "GlialReconstitution"]
Suggestion B: Reason: **Immediate Steps:** Build an additional Hatchery to increase your supply capacity and maintain unit production; prioritize getting your supply up to 200. **Short-Term Actions:** Focus on producing 5 Roaches and 3 Zerglings to bolster your army while transitioning into Mutalisks by completing your Spire and starting to produce Mutalisks; ensure you have enough Drones (around 75) to sustain your economy. **Long-Term Strategy:** Aim to secure a third Hatchery and transition to a balanced composition of Roaches and Mutalisks, while teching up to higher upgrades and scouting for your opponent's unit composition to adapt your strategies accordingly. Actions: ["Overlord", "Zergling", "Extractor", "Drone", "Drone", "Drone", "SporeCrawler", "Drone", "Drone", "Drone", "Queen", "Drone", "Drone", "Overlord", "Drone", "Roach", "Roach", "Roach", "Zergling", "Roach", "Roach", "Zergling", "Roach", "ZergMeleeWeaponsLevel1", "Zergling", "Roach", "Zergling", "Roach", "Roach", "Roach", "Zergling", "Roach", "Roach", "Roach", "Roach", "Roach", "Roach", "Zergling", "Roach", "Roach", "Roach", "Spire", "Roach", "Zergling", "Drone", "Roach", "Roach", "Zergling", "Roach", "Extractor", "Extractor", "Drone", "Drone", "Overlord", "Overlord"]
Suggestion C: Reason: **Immediate Steps:** Build 2 more Overlords to prevent supply block and ensure you can keep producing units effectively. **Short-Term Actions:** Focus on creating 4-6 Ravagers to enhance your army's damage potential while also starting your Infestation Pit and Evolution Chambers for upgrades. **Long-Term Strategy:** Aim to expand to 2 additional Hatcheries to increase your drone count and sustain a robust economy, while transitioning into a mix of Roach and Infestor compositions for late-game engagements. Actions: ["Overlord", "Zergling", "Ravager", "Ravager", "Overlord", "Overlord", "Roach", "Roach", "Roach", "Queen", "Roach", "Roach", "Roach", "Roach", "Roach", "Overlord", "Overlord", "Overlord", "Overlord", "Overlord", "Drone", "Drone", "Extractor", "Drone", "Drone", "Drone", "Hatchery", "Drone", "Drone", "Drone", "Drone", "Drone", "EvolutionChamber", "Extractor", "Drone", "Extractor", "Drone", "Drone", "Drone", "Drone", "Drone", "Drone", "Drone", "Hatchery", "Overlord", "Overlord", "HydraliskDen", "Roach", "Roach", "Roach", "Drone", "Drone", "Drone", "Overseer", "Drone", "Drone", "Drone", "Drone", "Hydralisk", "Hydralisk", "Overlord", "Overlord"]
"""
    elif race == Race.Terran:
        return f"""
Game Status: {json.dumps({'supply_used': 114, 'supply_capacity': 119, 'unit': {'SCV': 28, 'Thor': 13, 'Refinery': 6, 'OrbitalCommand': 3, 'Barracks': 5, 'Factory': 1, 'Starport': 2, 'BarracksReactor': 3, 'BarracksTechlab': 3, 'FactoryReactor': 1, 'SupplyDepot': 13, 'EngineeringBay': 1, 'Bunker': 2, 'MissileTurret': 2, 'SensorTower': 1}, 'research': ['TerranInfantryWeaponsLevel1', 'Stimpack'], 'failed action': 'Thor', 'failed reason': 'SupplyDepot is not exist'})}
Suggestion A: Reason: **Immediate Steps:** Prioritize producing more Marines (aim for at least 10) and ensure your Supply Depots are fully fueled to avoid supply block. **Short-Term Actions:** Focus on building additional Supply Depots and expanding by constructing a new Command Center while continuing to produce Vortices for air control and harassment. **Long-Term Strategy:** Transition into a balanced army composition by integrating Siege Tanks and upgrading infantry weapons, while maintaining a strong economy with multiple Orbital Commands and SCVs for continuous resource generation. Actions: ["TerranInfantryArmorsLevel1", "Medivac", "SCV", "MULE", "CommandCenter", "SCV", "SCV", "Factory", "SupplyDepot", "SCV", "SCV", "Factory", "SupplyDepot", "SCV", "SCV", "FactoryTechLab", "StimPACK", "SCV", "SCV", "FactoryReactor", "OrbitalCommand", "SCV", "SCV", "SCV", "Refinery", "SensorTower", "SCV", "SCV", "Refinery", "SCV", "Marine", "Marine", "SCV", "MULE", "SCV", "SCV", "OrbitalCommand", "MULE", "SupplyDepot", "SCV", "SCV", "FactoryTechLab", "SCV", "Marine", "SCV", "SupplyDepot", "Marine", "Marine", "SupplyDepot", "TerranVehicleWeaponsLevel1", "SCV", "SCV", "Factory", "Factory", "Factory", "Marine", "SupplyDepot", "Marine"]
Suggestion B: Reason: **Immediate Steps:** Build a Planetary Fortress at your expansion to enhance defense and give it a hard time for enemy harassment while continuing to produce SCVs to maintain your economy. **Short-term Actions:** Focus on producing more Hellions (aim for 4-6) to control map vision and pressure your opponent while upgrading to Vehicle Weapons Level 2 for your Hellions and Cyclones to strengthen your army. **Long-term Strategy:** Transition into a balanced army composition with a mix of Thors and Liberators for anti-air, while expanding to additional Command Centers to increase your economy and sustain production of units. Actions: ["PlanetaryFortress", "SCV", "SupplyDepot", "Hellion", "SCV", "SCV", "SupplyDepot", "SCV", "SCV", "Hellion", "Factory", "Thor", "Factory", "Thor", "SCV", "SCV", "MULE", "Thor", "SupplyDepot", "SCV", "SCV", "CommandCenter", "SCV", "Hellion", "Hellion", "Thor", "SiegeTank", "Thor", "FactoryTechLab", "Hellion", "Thor", "Hellion", "SupplyDepot", "SupplyDepot", "Hellion", "SiegeTank", "Hellion", "Hellion", "SiegeTank", "SiegeTank", "Hellion"]
Suggestion C: Reason: **Immediate Steps:** Build additional Barracks (3) to increase unit production and ensure you have enough Marines for defense and offense; prioritize the Reactor upgrades to enhance unit output. **Short-term Actions:** Focus on producing Siege Tanks (4) and Medivacs (3) to bolster your army composition, while simultaneously expanding your supply with Supply Depots and Command Centers. **Long-term Strategy:** Transition into a balanced army composition with Liberators and Siege Tanks for area control and map presence, while teching up to higher-tier units as your economy stabilizes and supply allows. Actions: ["Marine", "SCV", "Barracks", "Barracks", "Medivac", "Thor", "MULE", "Thor", "Marine", "Thor", "Marine", "OrbitalCommand", "Marine", "SCV", "Thor", "Marine", "Marine", "TerranInfantryWeaponsLevel2", "Thor", "Marine", "Thor", "Marine", "Thor", "Marine", "Thor", "Marine", "Thor", "Marine", "TerranVehicleWeaponsLevel1", "Thor", "OrbitalCommand", "Thor", "Marine", "Marine", "CommandCenter", "Marine", "Marine", "Thor", "Marine", "Thor", "TerranInfantryArmorsLevel1", "Thor", "Marine", "Thor", "Marine", "Marine", "Liberator", "Liberator", "Marine", "Marine", "Marine", "Marine", "Marine", "Marine", "Marine", "Marine"]
"""


def get_output_prompt(race):
    if race == Race.Protoss:
        return """
Final Actions Summary: <FleetBeacon> <Probe> <Probe> <Probe> <Probe> <Pylon> <Probe> <VoidRay> <Assimilator> <Probe> <VoidRay> <Pylon> <ProtossAirWeaponsLevel1> <Probe> <Probe> <Probe> <Assimilator> <VoidRay> <Probe> <Probe> <VoidRay> <Probe> <Pylon> <Probe> <Probe> <Probe> <Probe> <Adept> <VoidRay> <Probe> <RoboticsFacility> <VoidRay> <Probe> <Probe> <Probe> <Probe> <Probe> <FleetBeacon> <Probe> <Stargate> <Probe> <Nexus> <Pylon> <Pylon> <Probe> <Probe> <Probe> <ShieldBattery> <Probe> <Probe> <Probe> <Pylon> <Carrier> <Carrier> <Probe> <Probe> <Probe> <Probe> <Assimilator> <Zealot> <Assimilator> <Probe> <Probe>
"""
    elif race == Race.Zerg:
        return """
Final Actions Summary: <Overlord> <Overlord> <Extractor> <Extractor> <Extractor> <Extractor> <Drone> <Drone> <Overlord> <Drone> <Drone> <Drone> <Drone> <Drone> <Spire> <Hatchery> <GlialReconstitution> <Ravager> <Ravager> <Mutalisk> <Mutalisk> <Mutalisk> <Mutalisk>
"""
    elif race == Race.Terran:
        return """
Final Actions Summary: <SupplyDepot> <FactoryTechLab> <Thor> <SCV> <SCV> <CommandCenter> <SupplyDepot> <Medivac> <SiegeTank> <Armory> <TerranInfantryArmorsLevel1> <Marine> <Marine>
"""


def multi_lm_prompt(race):
    sp = get_system_prompt(race)
    ip = get_input_prompt(race)
    op = get_output_prompt(race)
    return {
        'system': sp,
        'input': ip,
        'output': op
    }
