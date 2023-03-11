import pickle
import numpy as np



vaccination_bounds = [0.1, 0.9]


class Vaccination:

    def __init__(self, skl_filepath='pa_vaccination_func_gp_00.pkl'):
        """Initialize."""

        # Load vaccination func from pkl
        model = pickle.load(open(skl_filepath, 'rb'))
        self.model = model

    def __call__(self, x_list):
        """Call on (potentially multiple) inputs."""
        x_list = np.array(x_list).reshape(-1, 2)
        y_list = self.model.predict(x_list)
        return y_list
