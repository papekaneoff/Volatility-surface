import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize


def svi_total_variance(k: np.ndarray, params: tuple) -> np.ndarray:

    a, b, rho, m, sigma = params

    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_loss(params: tuple, k: np.ndarray, w_observed: np.ndarray) -> float:
    """
    Erreur quadratique RELATIVE entre la variance SVI prédite et observée.
    """
    w_predicted = svi_total_variance(k, params)
    relative_error = (w_predicted - w_observed) / w_observed
    return np.sum(relative_error ** 2)


def calibrate_svi(strikes: np.ndarray, ivs: np.ndarray, spot: float, maturity: float,
                   k_min: float = -0.5, k_max: float = 0.5) -> dict:
    """
    Calibre les 5 paramètres SVI pour UNE maturité donnée.
    Ne garde que les strikes dans une fenêtre de log-moneyness liquide [k_min, k_max].
    Calcule aussi la qualité du fit (RMSE en volatilité, erreur relative moyenne),
    pour ne jamais présenter des paramètres calibrés sans indiquer à quel point
    ils reproduisent réellement les données de marché.
    """
    k_all = np.log(strikes / spot)

    mask = (k_all >= k_min) & (k_all <= k_max)
    k = k_all[mask]
    ivs_filtered = ivs[mask]
    w_observed = (ivs_filtered ** 2) * maturity

    initial_guess = (0.05, 0.1, -0.3, 0.0, 0.1)
    bounds = [
        (0.0001, None),
        (0.0001, None),
        (-0.999, 0.999),
        (None, None),
        (0.0001, None),
    ]

    result = minimize(
        svi_loss,
        x0=initial_guess,
        args=(k, w_observed),
        bounds=bounds,
        method="L-BFGS-B"
    )

    a, b, rho, m, sigma = result.x

    # Qualité du fit : on repasse en volatilité implicite (plus interprétable
    # que la variance totale) pour calculer le RMSE et l'erreur relative moyenne
    w_predicted = svi_total_variance(k, result.x)
    iv_predicted = np.sqrt(np.clip(w_predicted, 1e-8, None) / maturity)

    rmse_iv = float(np.sqrt(np.mean((iv_predicted - ivs_filtered) ** 2)))
    relative_rmse = float(rmse_iv / np.mean(ivs_filtered)) if len(ivs_filtered) > 0 else np.nan

    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma, "success": result.success,
            "n_points_used": len(k), "rmse_iv": rmse_iv, "relative_rmse": relative_rmse}


def calibrate_all_maturities(ticker_symbol: str, min_volume: int = 10, k_min: float = -0.5,
                              k_max: float = 0.5) -> list:

    from vol_surface import get_option_chain, get_spot_price, time_to_maturity

    spot = get_spot_price(ticker_symbol)
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    results = []

    for expiration in expirations:
        chain = get_option_chain(ticker_symbol, expiration)
        chain = chain[chain["volume"] >= min_volume]

        maturity = time_to_maturity(expiration)

        if maturity <= 0.001:
            continue

        if len(chain) < 5:
            continue

        params = calibrate_svi(strikes=chain["strike"].values,
                                ivs=chain["impliedVolatility"].values,
                                spot=spot,
                                maturity=maturity,
                                k_min=k_min,
                                k_max=k_max
                                )
        params["expiration"] = expiration
        params["maturity"] = maturity

        results.append(params)

    return results


def plot_svi_fit(strikes, ivs, spot, maturity, params: dict, k_min: float = -0.5, k_max: float = 0.5):
    """
    Compare visuellement les IV observées sur le marché vs la courbe SVI calibrée.
    N'affiche que les points dans la fenêtre [k_min, k_max] réellement utilisée
    pour la calibration : afficher des points hors fenêtre donnerait une image
    trompeuse (la courbe semblerait "rater" des points qu'elle n'a jamais
    essayé de fitter).
    """
    k_all = np.log(strikes / spot)
    mask = (k_all >= k_min) & (k_all <= k_max)
    k_observed = k_all[mask]
    ivs_observed = ivs[mask]

    n_excluded = (~mask).sum()

    # On génère une courbe lisse pour visualiser le fit
    k_smooth = np.linspace(k_min, k_max, 200)
    w_smooth = svi_total_variance(k_smooth, (params["a"], params["b"], params["rho"], params["m"], params["sigma"]))
    iv_smooth = np.sqrt(w_smooth / maturity)

    plt.figure(figsize=(8, 5))
    plt.scatter(k_observed, ivs_observed, color="red", label="Observed IV (used in calibration)")
    plt.plot(k_smooth, iv_smooth, color="blue", label="Calibrated SVI curve")
    plt.xlabel("Log-moneyness (k)")
    plt.ylabel("Implied volatility")
    subtitle = f" ({n_excluded} illiquid points outside [{k_min}, {k_max}] excluded)" if n_excluded > 0 else ""
    plt.title(f"SVI calibration, maturity = {maturity:.4f} years{subtitle}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    all_params = calibrate_all_maturities("AAPL")
    for p in all_params:
        print(p["expiration"], "success:", p["success"], "n_points:", p["n_points_used"], "rho:", round(p["rho"], 3))