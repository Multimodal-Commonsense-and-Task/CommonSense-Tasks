action_dict = {
    'TRAIN UNIT': {
        0: 'TRAIN PROBE', 1: 'TRAIN ZEALOT', 2: 'TRAIN ADEPT', 3: 'TRAIN STALKER',
        4: 'TRAIN SENTRY', 5: 'TRAIN HIGHTEMPLAR', 6: 'TRAIN DARKTEMPLAR', 7: 'TRAIN VOIDRAY',
        8: 'TRAIN CARRIER', 9: 'TRAIN TEMPEST', 10: 'TRAIN ORACLE', 11: 'TRAIN PHOENIX',
        12: 'TRAIN MOTHERSHIP', 13: 'TRAIN OBSERVER', 14: 'TRAIN IMMORTAL',
        15: 'TRAIN WARPPRISM', 16: 'TRAIN COLOSSUS', 17: 'TRAIN DISRUPTOR', 18: 'MORPH ARCHON'
    },
    'BUILD STRUCTURE': {
        19: 'BUILD PYLON', 20: 'BUILD ASSIMILATOR', 21: 'BUILD NEXUS',
        22: 'BUILD GATEWAY', 23: 'BUILD CYBERNETICSCORE', 24: 'BUILD FORGE', 
        25: 'BUILD TWILIGHTCOUNCIL', 26: 'BUILD ROBOTICSFACILITY', 27: 'BUILD STARGATE', 
        28: 'BUILD TEMPLARARCHIVE', 29: 'BUILD DARKSHRINE', 30: 'BUILD ROBOTICSBAY',
        31: 'BUILD FLEETBEACON', 32: 'BUILD PHOTONCANNON', 33: 'BUILD SHIELDBATTERY'
    },
    'RESEARCH TECHNIQUE': {
        34: 'RESEARCH WARPGATERESEARCH', 35: 'RESEARCH PROTOSSAIRWEAPONSLEVEL1',
        36: 'RESEARCH PROTOSSAIRWEAPONSLEVEL2', 37: 'RESEARCH PROTOSSAIRWEAPONSLEVEL3',
        38: 'RESEARCH PROTOSSAIRARMORSLEVEL1', 39: 'RESEARCH PROTOSSAIRARMORSLEVEL2',
        40: 'RESEARCH PROTOSSAIRARMORSLEVEL3', 41: 'RESEARCH ADEPTPIERCINGATTACK',
        42: 'RESEARCH BLINKTECH', 43: 'RESEARCH CHARGE', 44: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL1',
        45: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL2', 46: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL3',
        47: 'RESEARCH PROTOSSGROUNDARMORSLEVEL1', 48: 'RESEARCH PROTOSSGROUNDARMORSLEVEL2',
        49: 'RESEARCH PROTOSSGROUNDARMORSLEVEL3', 50: 'RESEARCH PROTOSSSHIELDSLEVEL1',
        51: 'RESEARCH PROTOSSSHIELDSLEVEL2', 52: 'RESEARCH PROTOSSSHIELDSLEVEL3',
        53: 'RESEARCH EXTENDEDTHERMALLANCE', 54: 'RESEARCH GRAVITICDRIVE',
        55: 'RESEARCH OBSERVERGRAVITICBOOSTER', 56: 'RESEARCH PSISTORMTECH',
        57: 'RESEARCH VOIDRAYSPEEDUPGRADE', 58: 'RESEARCH PHOENIXRANGEUPGRADE',
        59: 'RESEARCH TEMPESTGROUNDATTACKUPGRADE'
    },
    'OTHER ACTION': {
        60: 'SCOUTING PROBE', 61: 'SCOUTING OBSERVER', 62: 'SCOUTING ZEALOT',
        63: 'SCOUTING PHOENIX', 64: 'MULTI-ATTACK', 65: 'MULTI-RETREAT',
        66: 'CHRONOBOOST NEXUS', 67: 'CHRONOBOOST CYBERNETICSCORE',
        68: 'CHRONOBOOST TWILIGHTCOUNCIL', 69: 'CHRONOBOOST STARGATE',
        70: 'CHRONOBOOST FORGE', 71: 'EMPTY ACTION'
    }
}

TEXTSTARCRAFT_PROMPT = {
    'system': f"""
You are an AI trained in analyzing and summarizing StarCraft II games. You understand the nuances and strategies of the Protoss race. 
Based on the summaries of multiple rounds in a game, we want you to analyze the game progression in a structured way. Your analysis should include the following aspects:
1. Game Overview: Provide a brief overview of the current situation based on all the rounds.
2. Current Game Stage: Determine the stage of the game based on the information of all rounds. Is it the early game, mid-game, or late game?
3. Our Situation: Describe our current status in terms of:
    3.1 Units and Buildings: Analyze the state of our units and buildings.
    3.2 Economy: Evaluate our economic condition, including resource collection and usage.
    3.3 Technology: Describe the status of our technological research and what technologies we have unlocked so far. Analyze our technology tree, indicating the available and potential upgrades or units.
4. Our Strategy: Infer our potential strategy based on our current situation and the information of all rounds.
5. Enemy's Strategy: Infer the enemy's potential strategy, based on the available information.
6. Key Information: Highlight the most important aspects from all rounds that have significantly influenced the game.

For Protoss, keep an eye on Nexus's energy to Chrono Boost important structures.

Based on the game situation and strategies used by both sides, provide specific suggestions for the following areas:
1.Our Strategy: Propose adjustments to our current strategy to counter the enemy's moves and capitalize on our strengths.
2.Units and Buildings: Offer ways to enhance our unit composition and improve our building layout, suited to the current stage of the game.
3.Economy: Recommend better practices for resource gathering and usage, in line with our strategic needs.
4.Technology: Suggest focused research paths to gain technological advantages, considering our current research status and technology tree.
Lastly, consider the current situation and the suggestions provided, make 5 actionable and specific decisions from the action dictionary{action_dict}. This dictionary comprises four categories of actions: unit production, building construction, technology research, and other actions. Remember to align these decisions with the current stage of the game, and avoid proposing actions that are not currently feasible.
    """,
    'input': r"chunk0:At 06:48 game time, our current StarCraft II situation is as follows:\n\nResources:\n- game_time: 06:48\n- worker_supply: 63\n- mineral: 2480\n- gas: 2972\n- supply_left: 12\n- supply_cap: 99\n- supply_used: 87\n- army_supply: 14\n- base_count: 5\n\nBuildings:\n- base_count: 5\n- pylon_count: 3\n- gas_buildings_count: 9\n- gateway_count: 4\n- Zealot_count: 7\n- planning_worker_count: 3\n\nUnits:\n- base_count: 5\n- pylon_count: 3\n- gas_buildings_count: 9\n- gateway_count: 4\n- Zealot_count: 7\n- planning_worker_count: 3\n\nIn-process:\n- planning_worker_count: 3\n\nEnemy Information:\n- enemy_UnitTypeId.ZERGLING: 1\n\n:At 06:49 game time, our current StarCraft II situation is as follows:...chunk{K_1}",
    'output': """
1. Game Overview: At 06:48 game time, our current situation is promising. We have 63 workers, 2480 minerals, and 2972 gas. Our supply cap is at 99 with 12 supply left, and our army supply is 14. We have established five bases.
2. Current Game Stage: Based on the game time and resource availability, we are in the mid-game stage.
3. Our Situation:
    3.1 Units and Buildings: We have a substantial worker force and five bases. We also have 3 pylons, 9 gas buildings, 4 gateways, and 7 Zealots. There are 3 workers planned for construction.
    3.2 Economy: Our economy is healthy, with plenty of resources. We have an excellent worker supply to maintain and even expand our economy.
    3.3 Technology: We seem to be lagging in technology development. We have Zealots in our army, but there is no evidence of other advanced units or technologies being developed. Furthermore, the absence of a Cybernetics Core suggests that we are currently limited in our capacity to research higher technologies or produce more advanced units
4. Our Strategy: We seem to have a balanced approach, focusing on both economic growth and military power. The high worker count and the existence of Zealots suggest a strategy of resource expansion and defense.
5. Enemy's Strategy: The enemy has at least one Zergling, suggesting they might be at the early stages of army development or employing a Zerg rush strategy.
6. Key Information: The most crucial aspect at this moment is our strong economy, coupled with our reliance on lower-tier military power (Zealots). However, our technological development is lagging, limiting our unit diversity and potentially leaving us vulnerable to more advanced enemy forces. Our future success is contingent on how we manage to leverage our economic advantage to swiftly boost our technological development and diversify our military strength.

Suggestions:
1. Our Strategy: We should focus on enhancing our technological capabilities alongside continuing to develop our army and expanding our economy. Be mindful of possible Zerg rush attacks.
2. Units and Buildings: Given our technological limitations, it's vital to optimally utilize our Zealots. In the meantime, construction of a Cybernetics Core is necessary to unlock more advanced units and technologies. Further, increasing the supply cap by building more pylons may be beneficial.
3. Economy: Efficient resource gathering is crucial. Our resource reserves should support both immediate army development and future technological upgrades. Consider expanding to new resource locations to facilitate this.
4. Technology: The immediate task is to build a Cybernetics Core to enable the research of higher technologies. Following that, based on the game progression and the enemy's strategy, decide on a focused technology path to gain a technological edge.
This revised strategy should help address the current limitations and better prepare us for upcoming challenges.

Decisions:
0: <BUILD CYBERNETICSCORE>
1: <BUILD PYLON>
2: <BUILD FORGE>
3: <RESEARCH WARPGATERESEARCH>
4: <CHRONOBOOST NEXUS>
    """
}

HEP_TEXTSTARCRAFT_PROMPT = {
    'system': f"""
You are an AI trained in analyzing and summarizing StarCraft II games. You understand the nuances and strategies of the Protoss race. 
Based on the summaries of a game, we want you to analyze the game progression in a structured way.

1. Choose Tactic
First, here are some tactics, you should choose one as <current tactic>:
    1. Zealot & STALKER tactic:
        -Key buildings: 
            military: GATEWAY/WARPGATE (6+).
            technolegy: CYBERNETICSCORE(1 and only 1), TWILIGHTCOUNCIL(1 and only 1, CYBERNETICSCORE needed), FORGE(1 and only 1).
        -Key technologies: 
            Necessary: WARPGATERESEARCH(CYBERNETICSCORE needed), CHARGE(TWILIGHTCOUNCIL needed).
            Optimal: PROTOSSGROUNDWEAPONSLEEVEL(1,2, FORGE needed), PROTOSSGROUNDARMORSLEVEL(1,2, FORGE needed), PROTOSSSHIELDSLEEVEL(1,2, FORGE needed).
        -Key forces: ZEALOT: STALKER=1:1 or 2:1, as many as possible.
        -Key timing: 
            At 2 minutes: Complete the construction of the second NEXUS and 2 GATEWAY.
            At 4 minutes: 3+ WARPGATE, 1 CYBERNETICSCORE built and WARPGATERESEARCH research started.
            At 8 minutes: 6+ WARPGATE, 1 CYBERNETICSCORE, 1 TWILIGHTCOUNCIL(CYBERNETICSCORE needed) built and all necessary key technology research completed.
        -Applicable situation: 
            Only in the first 8 minutes of the game.
    2. Carrier tactic:
        -Key buildings: 
            military: STARGATE(6, CYBERNETICSCORE needed)
            technolegy: CYBERNETICSCORE(1 and only 1), FLEETBEACON(1 and only 1, STARGATE and CYBERNETICSCORE needed), FORGE(1 and only 1).
        -Key technologies: 
            Necessary:  WARPGATERESEARCH(CYBERNETICSCORE needed), PROTOSSAIRWEAPONSLEEVEL(1,2, CYBERNETICSCORE needed).
            Optimal: PROTOSSAIRMARMORSLEEVEL(1,2, CYBERNETICSCORE needed), PROTOSSSHIELDSLEEVEL(1,2, FORGE needed), PROTOSSGROUNDARMORSLEVEL(1,2, FORGE needed).
        -Key forces: 
            First 6 minutes, ZEALOT: STALKER=1:1, as many as possible, with little CARRIER.
            After 6 minutes, Mainly CARRIER(STARGATE and FLEETBEACON need), with a small amount of STALKER or ZEALOT.
        -Key timing: 
            At 2 minutes: Complete the construction of the second NEXUS and 2 GATEWAY.
            At 6 minutes: 2 STARGATE, 1 CYBERNETICSCORE, 1 FLEETBEACON, at least 1 CARRIER in trainning.
            At 10 minutes: 6 STARGATE, 1 CYBERNETICSCORE, 1 FLEETBEACON, 6+ CARRIER and necessary technology research completed.
        -Applicable situation: 
            Usually after 6 minutes of the game, or enemies have air units / heavy units.
Choose one as <current tactic> based on the game time and intelligence gathered through reconnaissance.

2. Priority Construction Analysis:
Based on the game situation and <current tactic>, determine if there are any priorities in infrastructure construction:
    2.1 When we have 'more than 16' workers, we need 2 nexus. If we have less than 2 nexus, set BUILD NEXUS as <priority>.
        When we have 'more than 32' workers, we need 3 nexus. If we have less than 3 nexus, set BUILD NEXUS as <priority>.
        When we have 'more than 48' workers, we need 4 nexus. If we have less than 4 nexus, set BUILD NEXUS as <priority>.
        When we have 'more than 64' workers, we need 5 nexus. If we have less than 5 nexus, set BUILD NEXUS as <priority>.
        When we have 'more than 72' workers, we need 6 nexus. If we have less than 6 nexus, set BUILD NEXUS as <priority>.
        When we have 'more than 80' workers, we need 7 nexus. If we have less than 7 nexus, set BUILD NEXUS as <priority>.
    2.2 Before we have 25 workers, we do not need gas building, do not build any ASSIMILATOR!
        When we have 'more than 25' workers, we need 1 gas building. If we have less than 1 gas building, set BUILD ASSIMILATOR as <priority>.
        When we have 'more than 40' workers, we need 3 gas building. If we have less than 3 gas building, set BUILD ASSIMILATOR as <priority>.
        When we have 'more than 60' workers, we need 7 gas building. If we have less than 7 gas building, set BUILD ASSIMILATOR as <priority>.
        When we have 'more than 75' workers, we need 10 gas building. If we have less than 10 gas building, set BUILD ASSIMILATOR as <priority>.
You should first extract relevant data from input information and then analyze all the two items, make sure whether they are <priority>.
If there is no priority according to 2.1, 2.2, set <priority> as NONE.

3. Conventional Construction Planning:
Based on the game situation and <current tactic>, make plans for economic, technology and military development:
    3.1 Technology: Make sure what technolegy building need to be build (if the technolegy building is not exist or not planned) and make sure what technolegy mentioned in <current tactic> need to be upgraded (if the technolegy research is not completed or not started).
    3.2 Economic: Always TRAIN PROBE if 'worker supply' is less than 80. If 'supply left' is less than 4 and there is no PYLON planned, build PYLON.
    3.3 Military: Build military building (if the build not enough according to <current tactic> and situation, but don't build too many), train military unit according to <current tactic> and situation.
    3.4 Scouting: If we have too many probes, dispatch SCOUTING PROBE.
    3.5 Chronoboost: If CARRIER is trainning, CHRONOBOOST STARGATE. CHRONOBOOST one of [CYBERNETICSCORE, TWILIGHTCOUNCIL, FORGE] if technology is researching there. Otherwise we can try CHRONOBOOST NEXUS.
    3.6 Attack: If army supply less than 110 or supply used less than 190, do not raise any attack! Otherwise Attack immediately.

Decisions:
if there is <priority> in 'Priority Construction Analysis', set 1 decision as <priority>, 1 <TRAIN PROBE> and other actions must be <EMPTY ACTION>. For example, if <priority> is BUILD NEXUS:
    <BUILD NEXUS>
    <TRAIN PROBE>
    <EMPTY ACTION>
    <EMPTY ACTION>
    <EMPTY ACTION>
otherwise, set decisions according to 'Conventional Construction Planning', set 1 Technology item, 1~2 Economic item, 2 Military items, 1 Other item. For example, if there is no <priority>:
    <BUILD CYBERNETICSCORE> (Technology item)
    <TRAIN PROBE> (Economic item)
    <TRAIN ZEALOT> (Military item)
    <TRAIN STALKER> (Military item)
    <SCOUTING PROBE> (Other item, Scouting, Chronoboost, or Attack)
When you have technology building and some gas storage, you should add technology study that have not been started. For example, if there is no <priority>:
    <RESEARCH PROTOSSAIRWEAPONSLEVEL1> (Technology item, or research any required but not studied technology)
    <TRAIN PROBE> (Economic item)
    <BUILD PYLON> (Economic item)
    <TRAIN ZEALOT> (Military item)
    <TRAIN STALKER> (Military item)
If you have a lot supply left and lot resources, you can set more Military items and stop use CHRONOBOOST. For example, if there is no <priority>:
    <BUILD FLEETBEACON> (Technology item, or research any required but not studied technology)
    <TRAIN PROBE> (Economic item)
    <BUILD STARGATE> (Military item)
    <TRAIN CARRIER> (Military item)
    <TRAIN ZEALOT> (Military item)

Lastly, consider the current situation and the analyses above, make 10 actionable and specific decisions from the action dictionary{action_dict}. This dictionary comprises four categories of actions: unit production, building construction, technology research, and other actions. Remember to align these decisions with the current stage of the game and <current tactic>, and avoid proposing actions that are not currently feasible.

Give your analysis and decision, do not ask for other data:
    """,
    'input': r"chunk0:At 01:59 game time, our current StarCraft II situation is as follows:\n\nResources:\n- Game time: 01:59\n- Worker supply: 21\n- Mineral: 215\n- Gas: 100\n- Supply left: 10\n- Supply cap: 31\n- Supply used: 21\n\nBuildings:\n- Nexus count: 1\n- Pylon count: 2\n- Gas buildings count: 1\n- Gateway count: 1\n\nUnits:\n- Probe count: 21\n\nPlanning:\n\nPlanning structure:\n- Planning gateway count: 1\n\nEnemy:\n\nUnit:\n- Enemy unittypeid.drone: 1\n\n",
    'output': """
Based on the provided information, Here's a summary of the situation:

1. Choose Tactic
Based on the current game time 01:59, it is within 8 minutes, we should choose Zealot & STALKER tactic. 
<current tactic> = Zealot & STALKER tactic

2. Priority Construction Analysis:
Here are the relative data and analysis for 2.1, 2.2:
2.1 At 01:59 game time, Worker supply: 21, Nexus count: 1, Planning nexus count: 0. 
    According to Current situation, we have more than 15 but less than 35 workers, so we need 2 nexus. While we have 1 nexus, so BUILD NEXUS is <priority>.
2.2 At 01:59 game time, Worker supply: 21, Gas buildings count: 1, Planning gas buildings count: 0
    According to Current situation, we have less than 25 workers, so we do not need gas building. We already have 1 gas building, so BUILD ASSIMILATOR is not <priority>.
<priority> = BUILD NEXUS

3. Conventional Construction Planning:
Here are the analysis for 3.1, 3.2 3.3, 3.4 and 3.5:
3.1 Technology: At 01:59 game time, Technology buildings that have been built: None. Key technologies that start studied or completely studied: None.
    According to <current tactic>, we can consider research WARPGATERESEARCH or build technolegy like CYBERNETICSCORE.
3.2 Economic: At 01:59 game time, Worker supply: 21, Supply left: 11
    Our worker supply is less than 80, we should TRAIN PROBE. Our 'supply left' is more than 4, we do not need more PYLON.
3.3 Military: At 01:59 game time, Gateway count: 1, Warp gate count: 0, Stargate count: 0, Roboticsfacility count: 0
    According to 'key building, military' part of <current tactic>, we need 2 GATEWAY at 2 minutes, we have 1 GATEWAY built and 1 GATEWAY under construction, we need to build 1 more GATEWAY. We should train ZEALOT and STALKER to enhance military strength. 
3.4 Scouting: We have too many probes so that we can try SCOUTING PROBE.
3.5 Chronoboost: Consider that no CARRIER is trainning nor any research is upgrading, we can try CHRONOBOOST NEXUS.
3.6 Attack: At 01:59 game time, Army supply: 0, Supply used: 21
    Army supply 0 is less than 110 and supply used 21 is less than 190, do not raise any attack!

Decisions:
Consider that <priority> = BUILD NEXUS, we set 1 decision as <priority>, 1 <TRAIN PROBE> and other actions as <EMPTY ACTION>.
0: <BUILD NEXUS>
1: <TRAIN PROBE>
2: <EMPTY ACTION>
3: <EMPTY ACTION>
4: <EMPTY ACTION>
    """
}
