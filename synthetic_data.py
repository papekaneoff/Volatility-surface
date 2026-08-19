import numpy as np
import pandas as pd
from svi_model import calibrate_svi


def generate_synthetic_smile(spot: float, maturity: float, true_params: dict,
                              n_strikes: int = 30, noise_level: float = 0.02) -> pd.DataFrame:

    k = np.linspace(-0.4, 0.4, n_strikes)
    strikes = spot * np.exp(k)

    a, b, rho, m, sigma = true_params["a"], true_params["b"], true_params["rho"], true_params["m"], true_params["sigma"]

    w_true = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
    iv_true = np.sqrt(w_true / maturity)

    iv_noisy = iv_true + np.random.normal(0, noise_level, size=n_strikes)
    iv_noisy = np.clip(iv_noisy, 0.01, None)  # jamais d'IV négative

    return pd.DataFrame({"strike": strikes, "impliedVolatility": iv_noisy})


def generate_training_dataset(n_scenarios: int = 300) -> pd.DataFrame:
    """
    Génère un dataset d'entraînement : pour chaque scénario simulé,
    on stocke les features de marché + les paramètres SVI calibrés dessus.
    """
    rows = []

    for _ in range(n_scenarios):
        spot = np.random.uniform(50, 500)
        maturity = np.random.uniform(0.05, 2.0)

        # On injecte de vraies relations entre maturity et les paramètres SVI,
        # avec un peu de bruit aléatoire par-dessus (le marché n'est jamais parfaitement lisse)

        # 1. rho plus négatif à court terme (skew/peur de crash plus fort),
        #    remonte progressivement vers 0 à long terme
        rho_base = -0.7 + 0.5 * (maturity / 2.0)
        rho = np.clip(rho_base + np.random.normal(0, 0.1), -0.95, -0.05)

        # 2. sigma (courbure) augmente avec la maturité : smiles longs plus larges/lissés
        sigma_base = 0.05 + 0.15 * (maturity / 2.0)
        sigma = np.clip(sigma_base + np.random.normal(0, 0.01), 0.02, 0.4)

        # 3. a (niveau de variance de base) croît avec la maturité,
        #    cohérent avec w = IV^2 * T qui grandit dans le temps
        a_base = 0.005 + 0.03 * (maturity / 2.0)
        a = np.clip(a_base + np.random.normal(0, 0.005), 0.0001, 0.1)

        true_params = {
            "a": a,
            "b": np.random.uniform(0.05, 0.3),   # pas de relation forte connue, on garde aléatoire
            "rho": rho,
            "m": np.random.uniform(-0.1, 0.1),   # idem
            "sigma": sigma,
        }

        smile_df = generate_synthetic_smile(spot, maturity, true_params)

        calibrated = calibrate_svi(
            strikes=smile_df["strike"].values,
            ivs=smile_df["impliedVolatility"].values,
            spot=spot,
            maturity=maturity
        )

        if not calibrated["success"]:
            continue

        # Ce bloc est maintenant au même niveau que le "if" ci-dessus,
        # donc il s'exécute à chaque itération où la calibration a réussi
        rows.append({
            "spot": spot,
            "maturity": maturity,
            "atm_iv": smile_df["impliedVolatility"].iloc[len(smile_df) // 2],
            "a": calibrated["a"],
            "b": calibrated["b"],
            "rho": calibrated["rho"],
            "m": calibrated["m"],
            "sigma": calibrated["sigma"],
            "true_sigma": true_params["sigma"],  # vérité terrain, pour diagnostic uniquement
        })

    df = pd.DataFrame(rows)

    # On retire les calibrations aberrantes (paramètres hors plage physique raisonnable)
    df = df[(df["m"].abs() < 1) & (df["sigma"] < 1) & (df["b"] < 0.5)]

    return df


if __name__ == "__main__":
    dataset = generate_training_dataset(n_scenarios=2000)
    print(dataset.shape)
    print(dataset.head())
    dataset.to_csv("svi_training_data.csv", index=False)
    print("Dataset sauvegardé dans svi_training_data.csv")