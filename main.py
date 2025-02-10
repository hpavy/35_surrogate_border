import torch
from run import RunSimulation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Le code se lance sur {device}")


folder_result_name = "1_first_try"  # name of the result folder


# On utilise hyper_param_init uniquement si c'est un nouveau modèle
hyper_param_init = {
    "H": [
        230.67,
    ],
    "ya0": [
        0.0025,
    ],
    "m": 1.57,
    "file": [
        "data_john_2_case_2.csv",
    ],
    "nb_epoch": 1000,
    "save_rate": 10,
    "dynamic_weights": True,
    "lr_weights": 0.1,
    "weight_data": 0.5,
    "weight_border": 0.5,
    "batch_size": 10000,
    "nb_points_pde": 1000000,
    "Re": 100,
    "lr_init": 3e-4,
    "gamma_scheduler": 0.999,
    "nb_layers": 10,
    "nb_neurons": 64,
    "n_pde_test": 5000,
    "n_data_test": 5000,
    "nb_points": 130,
    "x_min": -0.06,
    "x_max": 0.06,
    "y_min": -0.06,
    "y_max": 0.06,
    "t_min": 6.5,
    "nb_period": 20,
    "nb_period_plot": 2,
    "nb_points_close_cylinder": 10,
    "rayon_close_cylinder": 0.0135,
}

param_adim = {"V": 1.0, "L": 0.025, "rho": 1.2}

simu = RunSimulation(hyper_param_init, folder_result_name, param_adim)

simu.run()
