import yfinance as yf
import pandas as pd
import numpy as np

def get_option_chain(ticker_symbol: str, expiration: str) -> pd.DataFrame:

    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(expiration)

    return chain.calls


from datetime import datetime

def time_to_maturity(expiration: str) -> float:
  
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    today = datetime.now().date()
    days = (exp_date - today).days
    return max(days, 0) / 365


def build_surface_data(ticker_symbol:str, min_volume:int = 10) -> pd.DataFrame :
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    all_rows = []

    for expiration in expirations : 
        chain = ticker.option_chain(expiration)
        calls = chain.calls

        calls = calls[calls["volume"]>= min_volume]

        calls = calls.copy()
        calls["expiration"] = expiration
        calls["time_to_maturity"]= time_to_maturity(expiration)
        all_rows.append(calls[["strike", "time_to_maturity","impliedVolatility","expiration"]])

    return pd.concat(all_rows,ignore_index=True)



from scipy.interpolate import griddata 
def interpolate_surface(surface_df:pd.DataFrame, grid_size: int = 50):
    strikes = surface_df["strike"].values
    maturities = surface_df["time_to_maturity"].values
    ivs = surface_df["impliedVolatility"].values

    strike_axis = np.linspace(strikes.min(), strikes.max(),grid_size)
    maturity_axis = np.linspace(maturities.min(), maturities.max(), grid_size)

    strike_grid, maturity_grid = np.meshgrid(strike_axis,maturity_axis)


    iv_grid = griddata(points=(strikes, maturities), values = ivs,
                       xi=(strike_grid, maturity_grid),
                       method="linear")




    return strike_grid, maturity_grid, iv_grid


import  plotly.graph_objects as go 

def plot_volatility_surface(strike_grid, maturity_grid, iv_grid, title : str = "Volatility Surface"):
    fig = go.Figure(data= [go.Surface(x= strike_grid, y = maturity_grid, z=
                                       iv_grid, colorscale="RdYlGn", reversescale =True ,colorbar=dict(title= "IV"))])
    
    fig.update_layout(title = title, scene = dict(xaxis_title = "Strike", yaxis_title = "Maturity", 
                                                  zaxis_title = "Implied Volatility")
                                                  ,width = 900, height = 700)

    fig.show()


def get_spot_price(ticker_synbol:str) -> float:

    ticker = yf.Ticker(ticker_synbol)
    hist = ticker.history(period = "1d")
    return hist["Close"].iloc[-1]

from scipy.optimize import minimize

def svi_total_variance(k:np.ndarray, params : tuple) -> np.ndarray:

    a, b, rho, m, sigma = params

    return a + b*(rho*(k - m) + np.sqrt((k-m)**2 + sigma**2))


 











if __name__ == "__main__":
    surface_df = build_surface_data("AAPL")
    strike_grid, maturity_grid, iv_grid = interpolate_surface(surface_df)
    plot_volatility_surface(strike_grid, maturity_grid, iv_grid, title="AAPL - Volatility Surface")