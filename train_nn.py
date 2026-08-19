import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def load_dataset(path: str = "svi_training_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train_svi_nn(df: pd.DataFrame):
    features = ["spot", "maturity", "atm_iv"]
    targets = ["a", "b", "rho", "m", "sigma"]

    X = df[features].values
    y = df[targets].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    mse_global = mean_squared_error(y_test, y_pred)
    print(f"Overall MSE: {mse_global:.6f}")

    for i, param_name in enumerate(targets):
        mse_param = mean_squared_error(y_test[:, i], y_pred[:, i])
        print(f"  MSE {param_name}: {mse_param:.6f}")

    for i, param_name in enumerate(targets):
        r2_param = r2_score(y_test[:, i], y_pred[:, i])
        print(f"  R² {param_name}: {r2_param:.4f}")

    r2 = r2_score(y_test, y_pred)
    print(f"Overall R²: {r2:.4f}")

    return model, scaler_X, scaler_y


def predict_svi_params(model, scaler_X, scaler_y, spot: float, maturity: float, atm_iv: float) -> dict:
    """
    Utilise le NN entraîné pour prédire les 5 paramètres SVI
    à partir de features de marché simples. C'est LA fonction
    qui donne le "vrai" output utilisable du modèle.
    """
    import numpy as np

    X_new = np.array([[spot, maturity, atm_iv]])
    X_new_scaled = scaler_X.transform(X_new)

    y_pred_scaled = model.predict(X_new_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    targets = ["a", "b", "rho", "m", "sigma"]
    return dict(zip(targets, y_pred[0]))


def evaluate_on_real_market(ticker_symbol: str, model, scaler_X, scaler_y,
                             min_volume: int = 10, min_points: int = 5) -> pd.DataFrame:
    """
    Compare, échéance par échéance, les paramètres SVI obtenus par calibration
    classique (scipy.optimize, notre "vérité de référence") à ceux prédits par
    le réseau de neurones, sur les VRAIES données de marché d'aujourd'hui.

    Contrairement au R² calculé plus haut (qui mesure la performance sur des
    smiles SYNTHÉTIQUES jamais vus), cette fonction donne une mesure honnête
    de la fiabilité du NN sur des données réelles, celles qu'on utiliserait
    vraiment en pratique.
    """
    import yfinance as yf
    from vol_surface import get_spot_price, get_option_chain, time_to_maturity
    from svi_model import calibrate_svi

    spot = get_spot_price(ticker_symbol)
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    rows = []

    for expiration in expirations:
        chain = get_option_chain(ticker_symbol, expiration)
        chain = chain[chain["volume"] >= min_volume]

        maturity = time_to_maturity(expiration)

        if maturity <= 0.001 or len(chain) < min_points:
            continue

        atm_iv = chain["impliedVolatility"].iloc[len(chain) // 2]

        scipy_params = calibrate_svi(
            strikes=chain["strike"].values,
            ivs=chain["impliedVolatility"].values,
            spot=spot,
            maturity=maturity
        )
        if not scipy_params["success"]:
            continue

        nn_params = predict_svi_params(model, scaler_X, scaler_y, spot, maturity, atm_iv)

        rows.append({
            "expiration": expiration,
            "maturity": maturity,
            "a_scipy": scipy_params["a"], "a_nn": nn_params["a"],
            "b_scipy": scipy_params["b"], "b_nn": nn_params["b"],
            "rho_scipy": scipy_params["rho"], "rho_nn": nn_params["rho"],
            "m_scipy": scipy_params["m"], "m_nn": nn_params["m"],
            "sigma_scipy": scipy_params["sigma"], "sigma_nn": nn_params["sigma"],
        })

    return pd.DataFrame(rows)


def summarize_real_market_errors(comparison_df: pd.DataFrame) -> None:
    """
    Affiche l'erreur absolue moyenne (NN vs scipy) pour chaque paramètre,
    calculée sur TOUTES les échéances réelles, pas un seul exemple isolé.
    C'est la mesure la plus honnête de la fiabilité du NN en pratique.
    """
    print(f"\nComparison over {len(comparison_df)} real AAPL maturities:\n")
    print(f"{'Parameter':<10} {'Mean absolute error':<25}")
    for p in ["a", "b", "rho", "m", "sigma"]:
        mae = (comparison_df[f"{p}_scipy"] - comparison_df[f"{p}_nn"]).abs().mean()
        print(f"{p:<10} {mae:<25.4f}")


if __name__ == "__main__":
    df = load_dataset()
    model, scaler_X, scaler_y = train_svi_nn(df)

    comparison_df = evaluate_on_real_market("AAPL", model, scaler_X, scaler_y)
    summarize_real_market_errors(comparison_df)
    print(comparison_df.round(4))