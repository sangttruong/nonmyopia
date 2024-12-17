algos_name = [
    "SR",
    "EI",
    "PI",
    "UCB",
    "KG",
    "MSL",
    "Ours",
]

algos = [
    "qSR",
    "qEI",
    "qPI",
    "qUCB",
    "qKG",
    "HES-TS-20",
    "HES-TS-AM-20",
]

seeds = {
    "qSR": 2,
    "qEI": 2,
    "qPI": 2,
    "qUCB": 2,
    "qKG": 2,
    "HES-TS-20": 2,
    "HES-TS-AM-20": 2,
}

env_names = [
    "NightLight",
]

env_noises = [
    0.0,
]

env_discretizeds = [
    False,
]

cost_functions = [
    "euclidean",
    # "r-spotlight",
]
