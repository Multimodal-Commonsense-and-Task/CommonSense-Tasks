from sc2.position import Point2
from sc2.data import Race, Difficulty
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.unit_typeid import UnitTypeId


ACTION_DICT = {
    Race.Protoss: {
        'Train Unit': {
            100: 'Probe', 101: 'Zealot', 102: 'Sentry', 103: 'Stalker', 104: 'Adept',
            105: 'HighTemplar', 106: 'DarkTemplar', 107: 'Observer', 108: 'WarpPrism', 109: 'Immortal',
            110: 'Colossus', 111: 'Disruptor', 112: 'Phoenix', 113: 'VoidRay', 114: 'Oracle',
            115: 'Carrier', 116: 'Tempest', 117: 'Mothership', 118: 'Archon'
        },
        'Build Structure': {
            200: 'Nexus', 201: 'Assimilator', 202: 'Gateway', 203: 'RoboticsFacility', 204: 'Stargate',
            205: 'Pylon', 206: 'Forge', 207: 'CyberneticsCore', 208: 'PhotonCannon', 209: 'ShieldBattery',
            210: 'TwilightCouncil', 211: 'TemplarArchive', 212: 'DarkShrine', 213: 'FleetBeacon', 214: 'RoboticsBay',
        },
        'Research Technique': {
            300: 'ProtossGroundWeaponsLevel1', 301: 'ProtossGroundWeaponsLevel2', 302: 'ProtossGroundWeaponsLevel3',
            303: 'ProtossGroundArmorsLevel1', 304: 'ProtossGroundArmorsLevel2', 305: 'ProtossGroundArmorsLevel3',
            306: 'ProtossShieldsLevel1', 307: 'ProtossShieldsLevel2', 308: 'ProtossShieldsLevel3',
            309: 'ProtossAirWeaponsLevel1', 310: 'ProtossAirWeaponsLevel2', 311: 'ProtossAirWeaponsLevel3',
            312: 'ProtossAirArmorsLevel1', 313: 'ProtossAirArmorsLevel2', 314: 'ProtossAirArmorsLevel3',
            315: 'WarpGateResearch', 316: 'Charge', 317: 'BlinkTech',
            318: 'AdeptPiercingAttack', 319: 'PsiStormTech', 320: 'DarkTemplarBlinkUpgrade',
            321: 'ObserverGraviticBooster', 322: 'GraviticDrive', 323: 'ExtendedThermalLance',
            324: 'PhoenixRangeUpgrade', 325: 'VoidRaySpeedUpgrade'
        }
    },
    Race.Zerg: {
        'Train Unit': {
            100: 'Drone', 101: 'Overlord', 102: 'Zergling', 103: 'Queen', 104: 'Roach',
            105: 'Baneling', 106: 'Ravager', 107: 'Overseer', 108: 'Hydralisk', 109: 'Mutalisk',
            110: 'Corruptor', 111: 'Infestor', 112: 'SwarmHostMP', 113: 'LurkerMP', 114: 'Viper',
            115: 'BroodLord', 116: 'Ultralisk'
        },
        'Build Structure': {
            200: 'Hatchery', 201: 'Extractor', 202: 'Lair', 203: 'Hive', 204: 'SpawningPool',
            205: 'EvolutionChamber', 206: 'RoachWarren', 207: 'BanelingNest', 208: 'SpineCrawler', 209: 'SporeCrawler',
            210: 'HydraliskDen', 211: 'InfestationPit', 212: 'LurkerDenMP', 213: 'Spire', 214: 'NydusNetwork',
            215: 'UltraliskCavern', 216: 'GreaterSpire'
        },
        'Research Technique': {
            300: 'ZergMeleeWeaponsLevel1', 301: 'ZergMeleeWeaponsLevel2', 302: 'ZergMeleeWeaponsLevel3',
            303: 'ZergMissileWeaponsLevel1', 304: 'ZergMissileWeaponsLevel2', 305: 'ZergMissileWeaponsLevel3',
            306: 'ZergGroundArmorsLevel1', 307: 'ZergGroundArmorsLevel2', 308: 'ZergGroundArmorsLevel3',
            309: 'ZergFlyerWeaponsLevel1', 310: 'ZergFlyerWeaponsLevel2', 311: 'ZergFlyerWeaponsLevel3',
            312: 'ZergFlyerArmorsLevel1', 313: 'ZergFlyerArmorsLevel2', 314: 'ZergFlyerArmorsLevel3',
            315: 'Burrow', 316: 'overlordspeed', 317: 'zerglingmovementspeed',
            318: 'zerglingattackspeed', 319: 'GlialReconstitution', 320: 'TunnelingClaws',
            321: 'CentrificalHooks', 322: 'EvolveMuscularAugments', 323: 'EvolveGroovedSpines',
            324: 'NeuralParasite', 325: 'DiggingClaws', 326: 'LurkerRange',
            327: 'ChitinousPlating', 328: 'AnabolicSynthesis'
        }
    },
    Race.Terran: {
        'Train Unit': {
            100: 'SCV', 101: 'MULE', 102: 'Marine', 103: 'Reaper', 104: 'Marauder',
            105: 'Ghost', 106: 'Hellion', 107: 'WidowMine', 108: 'Cyclone', 109: 'SiegeTank',
            110: 'Thor', 111: 'VikingFighter', 112: 'Medivac', 113: 'Liberator', 114: 'Banshee',
            115: 'Raven', 116: 'Battlecruiser'
        },
        'Build Structure': {
            200: 'CommandCenter', 201: 'Refinery', 202: 'OrbitalCommand', 203: 'PlanetaryFortress', 204: 'Barracks',
            205: 'Factory', 206: 'Starport', 207: 'BarracksReactor', 208: 'BarracksTechLab', 209: 'FactoryReactor',
            210: 'FactoryTechLab', 211: 'StarportReactor', 212: 'StarportTechLab', 213: 'SupplyDepot', 214: 'EngineeringBay',
            215: 'Bunker', 216: 'MissileTurret', 217: 'SensorTower', 218: 'GhostAcademy', 219: 'Armory',
            220: 'FusionCore'
        },
        'Research Technique': {
            300: 'TerranInfantryWeaponsLevel1', 301: 'TerranInfantryWeaponsLevel2', 302: 'TerranInfantryWeaponsLevel3',
            303: 'TerranInfantryArmorsLevel1', 304: 'TerranInfantryArmorsLevel2', 305: 'TerranInfantryArmorsLevel3',
            306: 'TerranVehicleWeaponsLevel1', 307: 'TerranVehicleWeaponsLevel2', 308: 'TerranVehicleWeaponsLevel3',
            309: 'TerranShipWeaponsLevel1', 310: 'TerranShipWeaponsLevel2', 311: 'TerranShipWeaponsLevel3',
            312: 'TerranVehicleAndShipArmorsLevel1', 313: 'TerranVehicleAndShipArmorsLevel2', 314: 'TerranVehicleAndShipArmorsLevel3',
            315: 'TerranBuildingArmor', 316: 'HiSecAutoTracking', 317: 'Stimpack',
            318: 'ShieldWall', 319: 'PunisherGrenades', 320: 'PersonalCloaking',
            321: 'SmartServos', 322: 'HighCapacityBarrels',  323: 'DrillClaws',
            324: 'CycloneLockOnDamageUpgrade', 325: 'MedivacIncreaseSpeedBoost', 326: 'LiberatorAGRangeUpgrade',
            327: 'BansheeCloak', 328: 'BansheeSpeed', 329: 'InterferenceMatrix',
            330: 'BattlecruiserEnableSpecializations'
        }
    }
}

CHRONO_PRIORITY = [
    UnitTypeId.STARGATE,
    UnitTypeId.FLEETBEACON,
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.GATEWAY
]

MAP_RAMPS = [Point2((78, 58)), Point2((96, 68)), Point2((96, 34)), Point2((68, 70)), Point2((62, 134)),
             Point2((120, 50)), Point2((90, 98)), Point2((124, 126)), Point2((40, 134)), Point2((110, 82)),
             Point2((48, 86)), Point2((38, 118)), Point2((118, 34)), Point2((49, 99)), Point2((62, 100)),
             Point2((80, 110)), Point2((63, 85)), Point2((109, 69)), Point2((51, 69)), Point2((95, 83)),
             Point2((107, 99)), Point2((34, 42))]

MAP_POINTS = {
    'A': Point2((30.5, 142.5)),
    'B': Point2((72.5, 144.5)),
    'C': Point2((128.5, 143.5)),
    'D': Point2((28.5, 110.5)),
    'E': Point2((74.5, 120.5)),
    'F': Point2((98.5, 124.5)),
    'G': Point2((131.5, 119.5)),
    'H': Point2((28.5, 79.5)),
    'I': Point2((131.5, 90.5)),
    'J': Point2((28.5, 50.5)),
    'K': Point2((61.5, 45.5)),
    'L': Point2((85.5, 49.5)),
    'M': Point2((131.5, 59.5)),
    'N': Point2((31.5, 26.5)),
    'O': Point2((87.5, 25.5)),
    'P': Point2((129.5, 27.5))
}

DIFFICULTY_LEVELS = [Difficulty.VeryEasy, Difficulty.Easy, Difficulty.Medium, Difficulty.MediumHard, Difficulty.Hard, Difficulty.Harder, Difficulty.VeryHard, Difficulty.CheatVision, Difficulty.CheatMoney, Difficulty.CheatInsane]

ABILITYS = {
    UnitTypeId.ZEALOT: AbilityId.WARPGATETRAIN_ZEALOT, UnitTypeId.ADEPT: AbilityId.TRAINWARP_ADEPT, UnitTypeId.STALKER: AbilityId.WARPGATETRAIN_STALKER,
    UnitTypeId.SENTRY: AbilityId.WARPGATETRAIN_SENTRY, UnitTypeId.HIGHTEMPLAR: AbilityId.WARPGATETRAIN_HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR: AbilityId.WARPGATETRAIN_DARKTEMPLAR,
    UnitTypeId.MOTHERSHIP: AbilityId.NEXUSTRAINMOTHERSHIP_MOTHERSHIP
}

UNITS = {
    Race.Protoss: [UnitTypeId.ZEALOT, UnitTypeId.ADEPT, UnitTypeId.STALKER, UnitTypeId.SENTRY,
                   UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR, UnitTypeId.VOIDRAY, UnitTypeId.CARRIER,
                   UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.PHOENIX, UnitTypeId.MOTHERSHIP,
                   UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS, UnitTypeId.DISRUPTOR, UnitTypeId.ARCHON],
    Race.Zerg: [UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.HYDRALISK, UnitTypeId.MUTALISK,
                UnitTypeId.CORRUPTOR, UnitTypeId.INFESTOR, UnitTypeId.SWARMHOSTMP, UnitTypeId.ULTRALISK,
                UnitTypeId.VIPER, UnitTypeId.BANELING, UnitTypeId.RAVAGER, UnitTypeId.LURKERMP, UnitTypeId.BROODLORD],
    Race.Terran: [UnitTypeId.MARINE, UnitTypeId.REAPER, UnitTypeId.MARAUDER, UnitTypeId.GHOST,
                  UnitTypeId.HELLION, UnitTypeId.HELLIONTANK, UnitTypeId.CYCLONE, UnitTypeId.SIEGETANK,
                  UnitTypeId.THOR, UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT, UnitTypeId.LIBERATOR, UnitTypeId.LIBERATORAG,
                  UnitTypeId.BANSHEE, UnitTypeId.RAVEN, UnitTypeId.BATTLECRUISER]
}

BUILDINGS = {
    Race.Protoss: {UnitTypeId.NEXUS: 71, UnitTypeId.PYLON: 18, UnitTypeId.ASSIMILATOR: 21, UnitTypeId.GATEWAY: 46, UnitTypeId.CYBERNETICSCORE: 36,
                   UnitTypeId.FORGE: 32, UnitTypeId.TWILIGHTCOUNCIL: 36, UnitTypeId.ROBOTICSFACILITY: 46, UnitTypeId.STARGATE: 43, UnitTypeId.TEMPLARARCHIVE: 36,
                   UnitTypeId.DARKSHRINE: 71, UnitTypeId.ROBOTICSBAY: 46, UnitTypeId.FLEETBEACON: 43, UnitTypeId.PHOTONCANNON: 29, UnitTypeId.SHIELDBATTERY: 30},
    Race.Zerg: {UnitTypeId.EXTRACTOR: 21, UnitTypeId.HATCHERY: 71, UnitTypeId.SPAWNINGPOOL: 46, UnitTypeId.BANELINGNEST: 43, UnitTypeId.ROACHWARREN: 39,
                UnitTypeId.HYDRALISKDEN: 29, UnitTypeId.LAIR: 57, UnitTypeId.HIVE: 71, UnitTypeId.EVOLUTIONCHAMBER: 25, UnitTypeId.INFESTATIONPIT: 36,
                UnitTypeId.SPIRE: 71, UnitTypeId.GREATERSPIRE: 71, UnitTypeId.ULTRALISKCAVERN: 46, UnitTypeId.LURKERDENMP: 57, UnitTypeId.SPORECRAWLER: 21, UnitTypeId.SPINECRAWLER: 36},
    Race.Terran: {UnitTypeId.COMMANDCENTER: 71, UnitTypeId.SUPPLYDEPOT: 21, UnitTypeId.REFINERY: 21, UnitTypeId.BARRACKS: 46, UnitTypeId.ENGINEERINGBAY: 25, UnitTypeId.BUNKER: 29,
                  UnitTypeId.FACTORY: 43, UnitTypeId.STARPORT: 36, UnitTypeId.ORBITALCOMMAND: 25, UnitTypeId.PLANETARYFORTRESS: 36, UnitTypeId.BARRACKSREACTOR: 36, UnitTypeId.BARRACKSTECHLAB: 18,
                  UnitTypeId.FACTORYREACTOR: 36, UnitTypeId.FACTORYTECHLAB: 18, UnitTypeId.STARPORTREACTOR: 36, UnitTypeId.STARPORTTECHLAB: 18,
                  UnitTypeId.GHOSTACADEMY: 29, UnitTypeId.ARMORY: 46, UnitTypeId.FUSIONCORE: 46, UnitTypeId.MISSILETURRET: 18, UnitTypeId.SENSORTOWER: 18}
}

RESEARCHS = {
    Race.Protoss: {
        'ProtossGroundWeaponsLevel1': [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1],
        'ProtossGroundWeaponsLevel2': [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2],
        'ProtossGroundWeaponsLevel3': [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3],
        'ProtossGroundArmorsLevel1': [UpgradeId.PROTOSSGROUNDARMORSLEVEL1, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1],
        'ProtossGroundArmorsLevel2': [UpgradeId.PROTOSSGROUNDARMORSLEVEL2, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2],
        'ProtossGroundArmorsLevel3': [UpgradeId.PROTOSSGROUNDARMORSLEVEL3, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3],
        'ProtossShieldsLevel1': [UpgradeId.PROTOSSSHIELDSLEVEL1, AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1],
        'ProtossShieldsLevel2': [UpgradeId.PROTOSSSHIELDSLEVEL2, AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2],
        'ProtossShieldsLevel3': [UpgradeId.PROTOSSSHIELDSLEVEL3, AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3],
        'ProtossAirWeaponsLevel1': [UpgradeId.PROTOSSAIRWEAPONSLEVEL1, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1],
        'ProtossAirWeaponsLevel2': [UpgradeId.PROTOSSAIRWEAPONSLEVEL2, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2],
        'ProtossAirWeaponsLevel3': [UpgradeId.PROTOSSAIRWEAPONSLEVEL3, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3],
        'ProtossAirArmorsLevel1': [UpgradeId.PROTOSSAIRARMORSLEVEL1, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1],
        'ProtossAirArmorsLevel2': [UpgradeId.PROTOSSAIRARMORSLEVEL2, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2],
        'ProtossAirArmorsLevel3': [UpgradeId.PROTOSSAIRARMORSLEVEL3, AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3],
        'WarpGateResearch': [UpgradeId.WARPGATERESEARCH, AbilityId.RESEARCH_WARPGATE],
        'BlinkTech': [UpgradeId.BLINKTECH, AbilityId.RESEARCH_BLINK],
        'Charge': [UpgradeId.CHARGE, AbilityId.RESEARCH_CHARGE],
        'AdeptPiercingAttack': [UpgradeId.ADEPTPIERCINGATTACK, AbilityId.RESEARCH_ADEPTRESONATINGGLAIVES],
        'PsiStormTech': [UpgradeId.PSISTORMTECH, AbilityId.RESEARCH_PSISTORM],
        'DarkTemplarBlinkUpgrade': [UpgradeId.DARKTEMPLARBLINKUPGRADE, AbilityId.RESEARCH_SHADOWSTRIKE],
        'ObserverGraviticBooster': [UpgradeId.OBSERVERGRAVITICBOOSTER, AbilityId.RESEARCH_GRAVITICBOOSTER],
        'GraviticDrive': [UpgradeId.GRAVITICDRIVE, AbilityId.RESEARCH_GRAVITICDRIVE],
        'ExtendedThermalLance': [UpgradeId.EXTENDEDTHERMALLANCE, AbilityId.RESEARCH_EXTENDEDTHERMALLANCE],
        'PhoenixRangeUpgrade': [UpgradeId.PHOENIXRANGEUPGRADE, AbilityId.RESEARCH_PHOENIXANIONPULSECRYSTALS],
        'VoidRaySpeedUpgrade': [UpgradeId.VOIDRAYSPEEDUPGRADE, AbilityId.FLEETBEACONRESEARCH_RESEARCHVOIDRAYSPEEDUPGRADE],
    },
    Race.Zerg: {
        'ZergMeleeWeaponsLevel1': [UpgradeId.ZERGMELEEWEAPONSLEVEL1, AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL1],
        'ZergMeleeWeaponsLevel2': [UpgradeId.ZERGMELEEWEAPONSLEVEL2, AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL2],
        'ZergMeleeWeaponsLevel3': [UpgradeId.ZERGMELEEWEAPONSLEVEL3, AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL3],
        'ZergMissileWeaponsLevel1': [UpgradeId.ZERGMISSILEWEAPONSLEVEL1, AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL1],
        'ZergMissileWeaponsLevel2': [UpgradeId.ZERGMISSILEWEAPONSLEVEL2, AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL2],
        'ZergMissileWeaponsLevel3': [UpgradeId.ZERGMISSILEWEAPONSLEVEL3, AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL3],
        'ZergGroundArmorsLevel1': [UpgradeId.ZERGGROUNDARMORSLEVEL1, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL1],
        'ZergGroundArmorsLevel2': [UpgradeId.ZERGGROUNDARMORSLEVEL2, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL2],
        'ZergGroundArmorsLevel3': [UpgradeId.ZERGGROUNDARMORSLEVEL3, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL3],
        'ZergFlyerWeaponsLevel1': [UpgradeId.ZERGFLYERWEAPONSLEVEL1, AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL1],
        'ZergFlyerWeaponsLevel2': [UpgradeId.ZERGFLYERWEAPONSLEVEL2, AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL2],
        'ZergFlyerWeaponsLevel3': [UpgradeId.ZERGFLYERWEAPONSLEVEL3, AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL3],
        'ZergFlyerArmorsLevel1': [UpgradeId.ZERGFLYERARMORSLEVEL1, AbilityId.RESEARCH_ZERGFLYERARMORLEVEL1],
        'ZergFlyerArmorsLevel2': [UpgradeId.ZERGFLYERARMORSLEVEL2, AbilityId.RESEARCH_ZERGFLYERARMORLEVEL2],
        'ZergFlyerArmorsLevel3': [UpgradeId.ZERGFLYERARMORSLEVEL3, AbilityId.RESEARCH_ZERGFLYERARMORLEVEL3],
        'Burrow': [UpgradeId.BURROW, AbilityId.RESEARCH_BURROW],
        'overlordspeed': [UpgradeId.OVERLORDSPEED, AbilityId.RESEARCH_PNEUMATIZEDCARAPACE],
        'zerglingmovementspeed': [UpgradeId.ZERGLINGMOVEMENTSPEED, AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST],
        'zerglingattackspeed': [UpgradeId.ZERGLINGATTACKSPEED, AbilityId.RESEARCH_ZERGLINGADRENALGLANDS],
        'GlialReconstitution': [UpgradeId.GLIALRECONSTITUTION, AbilityId.RESEARCH_GLIALREGENERATION],
        'TunnelingClaws': [UpgradeId.TUNNELINGCLAWS, AbilityId.RESEARCH_TUNNELINGCLAWS],
        'CentrificalHooks': [UpgradeId.CENTRIFICALHOOKS, AbilityId.RESEARCH_CENTRIFUGALHOOKS],
        'EvolveMuscularAugments': [UpgradeId.EVOLVEMUSCULARAUGMENTS, AbilityId.RESEARCH_MUSCULARAUGMENTS],
        'EvolveGroovedSpines': [UpgradeId.EVOLVEGROOVEDSPINES, AbilityId.RESEARCH_GROOVEDSPINES],
        'NeuralParasite': [UpgradeId.NEURALPARASITE, AbilityId.RESEARCH_NEURALPARASITE],
        'DiggingClaws': [UpgradeId.DIGGINGCLAWS, AbilityId.RESEARCH_ADAPTIVETALONS],
        'LurkerRange': [UpgradeId.LURKERRANGE, AbilityId.LURKERDENRESEARCH_RESEARCHLURKERRANGE],
        'ChitinousPlating': [UpgradeId.CHITINOUSPLATING, AbilityId.RESEARCH_CHITINOUSPLATING],
        'AnabolicSynthesis': [UpgradeId.ANABOLICSYNTHESIS, AbilityId.RESEARCH_ANABOLICSYNTHESIS],
    },
    Race.Terran: {
        'TerranInfantryWeaponsLevel1': [UpgradeId.TERRANINFANTRYWEAPONSLEVEL1, AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1],
        'TerranInfantryWeaponsLevel2': [UpgradeId.TERRANINFANTRYWEAPONSLEVEL2, AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL2],
        'TerranInfantryWeaponsLevel3': [UpgradeId.TERRANINFANTRYWEAPONSLEVEL3, AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL3],
        'TerranInfantryArmorsLevel1': [UpgradeId.TERRANINFANTRYARMORSLEVEL1, AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1],
        'TerranInfantryArmorsLevel2': [UpgradeId.TERRANINFANTRYARMORSLEVEL2, AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL2],
        'TerranInfantryArmorsLevel3': [UpgradeId.TERRANINFANTRYARMORSLEVEL3, AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL3],
        'TerranVehicleWeaponsLevel1': [UpgradeId.TERRANVEHICLEWEAPONSLEVEL1, AbilityId.ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL1],
        'TerranVehicleWeaponsLevel2': [UpgradeId.TERRANVEHICLEWEAPONSLEVEL2, AbilityId.ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL2],
        'TerranVehicleWeaponsLevel3': [UpgradeId.TERRANVEHICLEWEAPONSLEVEL3, AbilityId.ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL3],
        'TerranShipWeaponsLevel1': [UpgradeId.TERRANSHIPWEAPONSLEVEL1, AbilityId.ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL1],
        'TerranShipWeaponsLevel2': [UpgradeId.TERRANSHIPWEAPONSLEVEL2, AbilityId.ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL2],
        'TerranShipWeaponsLevel3': [UpgradeId.TERRANSHIPWEAPONSLEVEL3, AbilityId.ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL3],
        'TerranVehicleAndShipArmorsLevel1': [UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1, AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL1],
        'TerranVehicleAndShipArmorsLevel2': [UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL2, AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL2],
        'TerranVehicleAndShipArmorsLevel3': [UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL3, AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL3],
        'TerranBuildingArmor': [UpgradeId.TERRANBUILDINGARMOR, AbilityId.RESEARCH_TERRANSTRUCTUREARMORUPGRADE],
        'HiSecAutoTracking': [UpgradeId.HISECAUTOTRACKING, AbilityId.RESEARCH_HISECAUTOTRACKING],
        'Stimpack': [UpgradeId.STIMPACK, AbilityId.BARRACKSTECHLABRESEARCH_STIMPACK],
        'ShieldWall': [UpgradeId.SHIELDWALL, AbilityId.RESEARCH_COMBATSHIELD],
        'PunisherGrenades': [UpgradeId.PUNISHERGRENADES, AbilityId.RESEARCH_CONCUSSIVESHELLS],
        'PersonalCloaking': [UpgradeId.PERSONALCLOAKING, AbilityId.RESEARCH_PERSONALCLOAKING],
        'SmartServos': [UpgradeId.SMARTSERVOS, AbilityId.RESEARCH_SMARTSERVOS],
        'HighCapacityBarrels': [UpgradeId.HIGHCAPACITYBARRELS, AbilityId.RESEARCH_INFERNALPREIGNITER],
        'DrillClaws': [UpgradeId.DRILLCLAWS, AbilityId.RESEARCH_DRILLINGCLAWS],
        'CycloneLockOnDamageUpgrade': [UpgradeId.CYCLONELOCKONDAMAGEUPGRADE, AbilityId.RESEARCH_CYCLONELOCKONDAMAGE],
        'MedivacIncreaseSpeedBoost': [UpgradeId.MEDIVACCADUCEUSREACTOR, AbilityId.FUSIONCORERESEARCH_RESEARCHMEDIVACENERGYUPGRADE],
        'LiberatorAGRangeUpgrade': [UpgradeId.LIBERATORAGRANGEUPGRADE, AbilityId.FUSIONCORERESEARCH_RESEARCHBALLISTICRANGE],
        'BansheeCloak': [UpgradeId.BANSHEECLOAK, AbilityId.RESEARCH_BANSHEECLOAKINGFIELD],
        'BansheeSpeed': [UpgradeId.BANSHEESPEED, AbilityId.RESEARCH_BANSHEEHYPERFLIGHTROTORS],
        'InterferenceMatrix': [UpgradeId.INTERFERENCEMATRIX, AbilityId.STARPORTTECHLABRESEARCH_RESEARCHRAVENINTERFERENCEMATRIX],
        'BattlecruiserEnableSpecializations': [UpgradeId.BATTLECRUISERENABLESPECIALIZATIONS, AbilityId.RESEARCH_BATTLECRUISERWEAPONREFIT]
    }
}
