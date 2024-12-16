algos_name = [
    "UCB ($\\beta = 0.1$)",
    "UCB ($\\beta = 0.5$)",
    "UCB ($\\beta = 1$)",
    "UCB ($\\beta = 2$)",
    "UCB ($\\beta = 5$)",
    "UCB ($\\beta = 10$)",
    "UCB ($\\beta = 20$)",
    "UCB ($\\beta = 50$)",
    "UCB ($\\beta = 100$)",
    "UCB ($\\beta = 200$)",
    "UCB ($\\beta = 500$)",
    "UCB ($\\beta = 1000$)",
    "Ours"
]

algos = [
    "qUCB-0.1",
    "qUCB-0.5",
    "qUCB-1",
    "qUCB-2",
    "qUCB-5",
    "qUCB-10",
    "qUCB-20",
    "qUCB-50",
    "qUCB-100",
    "qUCB-200",
    "qUCB-500",
    "qUCB-1000",
    "HES-TS-AM-20",
]

seeds = {
    "qUCB-0.1": 2,
    "qUCB-0.5": 3,
    "qUCB-1": 3,
    "qUCB-2": 3,
    "qUCB-5": 3,
    "qUCB-10": 3,
    "qUCB-20": 2,
    "qUCB-50": 3,
    "qUCB-100": 3,
    "qUCB-200": 3,
    "qUCB-500": 3,
    "qUCB-1000": 3,
    "HES-TS-AM-20": 5,
}

env_names = [
    "Ackley",
    # "Ackley4D",
    "Alpine",
    # "Cosine8", 
    # "Hartmann", 
    "HolderTable", 
    "Levy", 
    "StyblinskiTang",
    "SynGP",
]

env_noises = [
    0.05,
]

env_discretizeds = [
    False,
]

cost_functions = [
    "r-spotlight",
]