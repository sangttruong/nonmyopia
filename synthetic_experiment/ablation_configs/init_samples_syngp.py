algos_name = [
    # "EI",
    # "KG",
    # "MSL",
    "Ours",
]

algos = [
    # "qEI",
    # "qKG",
    # "HES-TS-20",
    "HES-TS-AM-20",
]

seeds = {
    # "qEI": 3,
    # "qKG": 3,
    # "HES-TS-20": 3,
    # "HES-TS-AM-20": 3,
    # "qEI": 5,
    # "qKG": 5,
    # "HES-TS-20": 5,
    "HES-TS-AM-20": 2,
}

env_names = [
    "SynGP",
]

env_noises = [
    0.05,
]

env_discretizeds = [
    False,
]

cost_functions = ["r-spotlight", "euclidean", "manhattan", "non-markovian"]
