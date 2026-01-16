import numpy as np
import pandas as pd
import tidyfinance as tf
from adjustText import adjust_text
from mizani.formatters import percent_format
from plotnine import *


def run() -> None:
    symbols = tf.download_data(
        domain="constituents", index="Dow Jones Industrial Average"
    )
    print(symbols)

    prices_daily = tf.download_data(
        domain="stock_prices",
        symbols=symbols["symbol"].tolist(),
        start_date="2000-01-01",
        end_date="2023-12-31",
    )
    # Group by symbol & filter to those with the most data(by counting adjusted_close entries)
    prices_daily["counts"] = prices_daily.groupby(by="symbol")[
        "adjusted_close"
    ].transform("count")
    # Keep only those with the maximum count
    prices_daily = prices_daily.query("counts == counts.max()").drop(columns=["counts"])
    print(prices_daily)  # Because they may have different IPO dates/delisting dates

    returns_monthly = (
        prices_daily.assign(
            date=prices_daily["date"].dt.to_period("M").dt.to_timestamp()
        )
        .groupby(["symbol", "date"], as_index=False)
        .agg(
            adjusted_close=("adjusted_close", "last")
        )  # Convert daily adjusted close to monthly adjusted close
        .assign(
            ret=lambda x: x.groupby("symbol")["adjusted_close"].pct_change()
        )  # Find percent change in adjusted close to get monthly returns
    )
    print(returns_monthly)

    assets = returns_monthly.groupby("symbol", as_index=False).agg(
        mu=("ret", "mean"), sigma=("ret", "std")
    )  # Estimate expected return mu and volatility sigma
    print(assets)

    assets_figure = (
        ggplot(assets, aes(x="sigma", y="mu", label="symbol"))
        + geom_point()
        + geom_text(adjust_text={"arrowprops": {"arrowstyle": "-"}})
        + scale_x_continuous(labels=percent_format())
        + scale_y_continuous(labels=percent_format())
        + labs(
            x="Volatility",
            y="Expected return",
            title="Expected returns and volatilities of Dow Jones index constituents",
        )
    )
    assets_figure.show()

    # Estimate variance-covariance matrix
    returns_wide = returns_monthly.pivot(
        index="date", columns="symbol", values="ret"
    ).reset_index()
    # converts:
    # date       | symbol | ret |
    # 202x-xx-xx | AAPL   | 0.02 |
    # 202x-xx-xx | MSFT   | 0.03 |
    # 202x-xx-xx | ...    | ...  |
    # to:
    # date(index)| AAPL | MSFT | ... |
    # 202x-xx-xx | 0.02 | 0.03 | ... |
    # 202x-xx-xx | ...  | ...  | ... |

    # drop date column for covariance calculation
    sigma = returns_wide.drop(columns=["date"]).cov()

    sigma_long = sigma.reset_index().melt(
        id_vars="symbol", var_name="symbol_b", value_name="value"
    )
    # converts:
    # AAPL | MSFT | ... |
    # 0.02 | 0.01 | ... |
    # 0.01 | 0.03 | ... |
    # to:
    # symbol | symbol_b | value |
    # AAPL   | AAPL     | 0.02  |
    # AAPL   | MSFT     | 0.01  |
    # MSFT   | AAPL     | 0.01  |

    sigma_long["symbol_b"] = pd.Categorical(
        sigma_long["symbol_b"],
        categories=sigma_long["symbol_b"].unique()[::-1],
        ordered=True,
    )  # For better visualization: reverse the order of symbol_b

    sigma_figure = (
        ggplot(sigma_long, aes(x="symbol", y="symbol_b", fill="value"))
        + geom_tile()
        + labs(
            x="",
            y="",
            fill="(Co-)Variance",
            title="Sample Variance-covariance matrix of Dow Jones index constituents",
        )
        + scale_fill_continuous(labels=percent_format())
        + theme(axis_text_x=element_text(angle=45, hjust=1))
    )
    sigma_figure.show()

    # sigma being the variance-covariance matrix
    iota = np.ones(sigma.shape[0])  # Nx1 vector of ones
    sigma_inv = np.linalg.inv(sigma.values)  # NxN
    omega_mvp = (sigma_inv @ iota) / (iota @ sigma_inv @ iota)
    # numerator: NxN @ Nx1 = Nx1 vector
    # denominator: 1xN @ NxN @ Nx1 = 1x1 scalar
    # numerator / denominator = Nx1 vector
    # This gives the minimum-variance portfolio weights

    assets = assets.assign(omega_mvp=omega_mvp)

    assets["symbol"] = pd.Categorical(
        assets["symbol"],
        categories=assets.sort_values("omega_mvp")["symbol"],
        ordered=True,
    )  # For better visualization: order by omega_mvp
    print(assets)

    omega_figure = (
        ggplot(assets, aes(y="omega_mvp", x="symbol", fill="omega_mvp>0"))
        + geom_col()  # position is symbol and height is omega_mvp
        + coord_flip()
        + scale_y_continuous(labels=percent_format())
        + labs(x="", y="", title="Minimum-variance portfolio weights")
        + theme(legend_position="none")
    )
    omega_figure.show()

    mu = assets["mu"].values
    mu_mvp = omega_mvp @ mu
    sigma_mvp = np.sqrt(omega_mvp @ sigma.values @ omega_mvp)
    summary_mvp = pd.DataFrame(
        {"mu": [mu_mvp], "sigma": [sigma_mvp], "type": ["Minimum-Variance Portfolio"]}
    )
    print(summary_mvp)

    # Efficient Portfolio
    mu_bar = assets["mu"].max()
    C = iota @ sigma_inv @ iota
    D = iota @ sigma_inv @ mu
    E = mu @ sigma_inv @ mu
    lambda_tilde = 2 * (mu_bar - D / C) / (E - (D**2) / C)
    omega_efp = omega_mvp + (lambda_tilde / 2) * (sigma_inv @ mu - D * omega_mvp)
    mu_efp = omega_efp @ mu
    sigma_efp = np.sqrt(omega_efp @ sigma.values @ omega_efp)

    summary_efp = pd.DataFrame(
        {"mu": [mu_efp], "sigma": [sigma_efp], "type": ["Efficient Portfolio"]}
    )

    summaries = pd.concat([assets, summary_mvp, summary_efp], ignore_index=True)

    summaries_figure = (
        ggplot(summaries, aes(x="sigma", y="mu"))
        + geom_point(data=summaries.query("type.isna()"))
        + geom_point(data=summaries.query("type.notna()"), color="#F21A00", size=3)
        + geom_label(aes(label="type"), adjust_text={"arrowprops": {"arrowstyle": "-"}})
        + scale_x_continuous(labels=percent_format())
        + scale_y_continuous(labels=percent_format())
        + labs(
            x="Volatility",
            y="Expected return",
            title="Efficient & minimum-variance portfolios",
        )
    )
    summaries_figure.show()

    efficient_frontier = (
        pd.DataFrame({"a": np.arange(-1, 2.01, 0.01)})
        .assign(
            omega=lambda x: x["a"].map(lambda x: x * omega_efp + (1 - x) * omega_mvp)
        )
        .assign(
            mu=lambda x: x["omega"].map(lambda x: x @ mu),
            sigma=lambda x: x["omega"].map(lambda x: np.sqrt(x @ sigma @ x)),
        )
    )

    summaries = pd.concat([summaries, efficient_frontier], ignore_index=True)

    summaries_figure = (
        ggplot(summaries, aes(x="sigma", y="mu"))
        + geom_point(data=summaries.query("type.isna()"))
        + geom_point(data=summaries.query("type.notna()"), color="#F21A00", size=3)
        + geom_label(aes(label="type"), adjust_text={"arrowprops": {"arrowstyle": "-"}})
        + scale_x_continuous(labels=percent_format())
        + scale_y_continuous(labels=percent_format())
        + labs(
            x="Volatility",
            y="Expected return",
            title="Efficient & minimum-variance portfolios",
        )
    )
    summaries_figure.show()


if __name__ == "__main__":
    run()
