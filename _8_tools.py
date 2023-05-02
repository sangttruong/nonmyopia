import os
import subprocess
import numpy as np
import pandas as pd
from abc import ABC


class BaseTool(ABC):
    def __init__(self):
        AA = "ACDEFGHIKLMNPQRSTVWY"
        self.AA_to_idx = {aa: i for i, aa in enumerate(AA)}
        self.idx_to_AA = {value: key for key, value in self.AA_to_idx.items()}

    def Energy(self, x):
        """
        x: categorical vector
        """
        raise NotImplementedError


############################
# Black Box Tools
############################


class Absolut(BaseTool):
    def __init__(self, config):
        BaseTool.__init__(self)
        """
        config: dictionary of parameters for BO
            antigen: PDB ID of antigen
            path: path to Absolut installation
            process: Number of CPU processes
            expid: experiment ID
        """
        for key in ["antigen", "path", "process"]:
            assert key in config, f'"{key}" is not defined in config'
        self.config = config
        assert self.config["startTask"] >= 0 and (
            self.config["startTask"] + self.config["process"] < os.cpu_count()
        ), f"{self.config['startTask']} is not a valid cpu"

    def Energy(self, x):
        """
        x: categorical vector (num_Seq x Length)
        """
        x = x.astype("int32")
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # randomise TempCDR3 name in case 2 or more experiments are run in parallel on the same server
        # rand_num = np.random.randint(low=0, high=1000)

        # Change working directory
        current_dir = os.getcwd()
        os.chdir(f"{self.config['path']}")

        sequences = []
        with open(f"TempCDR3_{self.config['antigen']}.txt", "w") as f:
            for i, seq in enumerate(x):
                seq2char = "".join(self.idx_to_AA[aa] for aa in seq)
                line = f"{i + 1}\t{seq2char}\n"
                f.write(line)
                sequences.append(seq2char)

        cmd = [
            "taskset",
            "-c",
            f"{self.config['startTask']}-{self.config['startTask'] + self.config['process']}",
            "./Absolut",
            "repertoire",
            self.config["antigen"],
            f"{self.config['path']}/TempCDR3_{self.config['antigen']}.txt",
            str(self.config["process"]),
        ]

        _ = subprocess.run(
            cmd, capture_output=True, text=False, cwd=self.config["path"]
        )

        data = pd.read_csv(
            os.path.join(
                self.config["path"],
                f"{self.config['antigen']}FinalBindings_Process_1_Of_1.txt",
            ),
            sep="\t",
            skiprows=1,
        )

        # Add an extra column to ensure that ordering will be ok after groupby operation
        data["sequence_idx"] = data.apply(
            lambda row: int(row.ID_slide_Variant.split("_")[0]), axis=1
        )
        energy = data.groupby(by=["sequence_idx"]).min(["Energy"])
        min_energy = energy["Energy"].values

        # Remove all created files and change the working directory to what it was
        for i in range(self.config["process"]):
            os.remove(
                f"{self.config['path']}/TempBindingsFor{self.config['antigen']}_t{i}_Part1_of_1.txt"
            )
        os.remove(f"{self.config['path']}/TempCDR3_{self.config['antigen']}.txt")

        os.remove(
            f"{self.config['path']}/{self.config['antigen']}FinalBindings_Process_1_Of_1.txt"
        )
        os.chdir(current_dir)
        return min_energy, sequences
