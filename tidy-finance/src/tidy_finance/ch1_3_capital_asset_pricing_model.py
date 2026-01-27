import numpy as np
import pandas as pd
import pandas_datareader as pdr
import statsmodels.formula.api as smf
import tidyfinance as tf
from adjustText import adjust_text
from mizani.formatters import percent_format
from plotnine import *


def run():
    symbols = tf.download_data(
        domain="constituents", index="Dow Jones Industrial Average"
    )

    prices_daily = tf.download_data(
        domain="stock_prices",
        symbols=symbols["symbol"].tolist(),
        start_date="2000-01-01",
        end_date="2023-12-31",
    )

    prices_daily = (
        prices_daily.groupby("symbol")
        .apply(lambda x: x.assign(counts=x["adjusted_close"].dropna().count()))
        .reset_index(drop=True)
        .query("counts == counts.max()")
    )

    returns_monthly = (
        prices_daily.assign(
            date=prices_daily["date"].dt.to_period("M").dt.to_timestamp()
        )
        .groupby(["symbol", "date"], as_index=False)
        .agg(adjusted_close=("adjusted_close", "last"))
        .assign(ret=lambda x: x.groupby("symbol")["adjusted_close"].pct_change())
    )

    risk_free_monthly = (
        tf.download_data(
            "stock_prices",
            symbols="^IRX",
            start_date="2019-10-01",
            end_date="2024-09-30",
        )
        .assign(risk_free=lambda x: (1 + x["adjusted_close"] / 100) ** (1 / 12) - 1)
        .dropna()
    )
    rf = risk_free_monthly["risk_free"].mean()

    assets = returns_monthly.groupby("symbol", as_index=False).agg(
        mu=("ret", "mean"), sigma=("ret", "std")
    )

    sigma = returns_monthly.pivot(index="date", columns="symbol", values="ret").cov()

    mu = returns_monthly.groupby("symbol")["ret"].mean().values

    w_tan = np.linalg.solve(sigma, mu - rf)
    w_tan = w_tan / np.sum(w_tan)

    mu_w = w_tan.T @ mu
    sigma_w = np.sqrt(w_tan.T @ sigma @ w_tan)

    efficient_portfolios = pd.DataFrame(
        [
            {"symbol": "\\omega_{tan}", "mu": mu_w, "sigma": sigma_w},
            {"symbol": "r_f", "mu": rf, "sigma": 0},
        ]
    )

    sharpe_ratio = (mu_w - rf) / sigma_w

    efficient_portfolios_figure = (
        ggplot(efficient_portfolios, aes(x="sigma", y="mu"))
        + geom_point(data=assets)
        + geom_point(data=efficient_portfolios, color="blue")
        + geom_label(
            aes(label="symbol"),
            parse=True,
            adjust_text={"arrowprops": {"arrowstyle": "-"}},
        )
        + scale_x_continuous(labels=percent_format())
        + scale_y_continuous(labels=percent_format())
        + labs(
            x="Volatility",
            y="Expected return",
            title="The efficient frontier with a risk-free asset and Dow index constituents",
        )
        + geom_abline(aes(intercept=rf, slope=sharpe_ratio), linetype="dotted")
    )
    efficient_portfolios_figure.show()

    betas = (sigma @ w_tan) / (w_tan.T @ sigma @ w_tan)
    assets["beta"] = betas.values

    price_of_risk = float(w_tan.T @ mu - rf)

    assets_figure = (
        ggplot(assets, aes(x="beta", y="mu"))
        + geom_point()
        + geom_abline(intercept=rf, slope=price_of_risk)
        + scale_y_continuous(labels=percent_format())
        + labs(x="Beta", y="Expected return", title="Security market line")
        + annotate("text", x=0, y=rf, label="Risk-free")
    )
    assets_figure.show()

    factors_raw = pdr.DataReader(
        name="F-F_Research_Data_5_Factors_2x3",
        data_source="famafrench",
        start="2000-01-01",
        end="2024-09-30",
    )[0]

    factors = (
        factors_raw.divide(100)
        .reset_index(names="date")
        .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
        .rename(str.lower, axis="columns")
        .rename(columns={"mkt-rf": "mkt_excess"})
    )

    returns_excess_monthly = returns_monthly.merge(
        factors, on="date", how="left"
    ).assign(ret_excess=lambda x: x["ret"] - x["rf"])

    def estimate_capm(data):
        model = smf.ols("ret_excess ~ mkt_excess", data=data).fit()
        result = pd.DataFrame(
            {
                "coefficient": ["alpha", "beta"],
                "estimate": model.params.values,
                "t_statistic": model.tvalues.values,
            }
        )
        return result

    capm_results = (
        returns_excess_monthly.groupby("symbol", group_keys=True)
        .apply(estimate_capm)
        .reset_index()
    )

    alphas = capm_results.query("coefficient=='alpha'").assign(
        is_significant=lambda x: np.abs(x["t_statistic"]) >= 1.96
    )

    alphas["symbol"] = pd.Categorical(
        alphas["symbol"],
        categories=alphas.sort_values("estimate")["symbol"],
        ordered=True,
    )

    alphas_figure = (
        ggplot(alphas, aes(y="estimate", x="symbol", fill="is_significant"))
        + geom_col()
        + scale_y_continuous(labels=percent_format())
        + coord_flip()
        + labs(
            x="Estimated asset alphas",
            y="",
            fill="Significant at 95%?",
            title="Estimated CAPM alphas for Dow index constituents",
        )
    )
    alphas_figure.show()
