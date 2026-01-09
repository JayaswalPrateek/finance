import numpy as np
import pandas as pd
import tidyfinance as tf
from mizani.formatters import percent_format
from plotnine import *  # type: ignore


def run():
    prices = tf.download_data(
        domain="stock_prices",
        symbols="AAPL",
        start_date="2000-01-01",
        end_date="2023-12-31",
    )
    print(prices.head().round(3))

    apple_prices_figure = (
        ggplot(prices, aes(y="adjusted_close", x="date"))
        + geom_line()
        + labs(x="", y="", title="Apple stock prices from 2000 to 2023")
    )
    apple_prices_figure.show()

    returns = (
        prices.sort_values("date")
        .assign(ret=lambda x: x["adjusted_close"].pct_change())
        .loc[:, ["symbol", "date", "ret"]]
    )
    returns = returns.dropna()
    print(returns)

    print("Summary statistics for Apple stock returns:")
    print(pd.DataFrame(returns["ret"].describe()).round(3).T)

    print("Yearly summary statistics for Apple stock returns:")
    print((returns["ret"].groupby(returns["date"].dt.year).describe().round(3)))

    quantile_05 = returns["ret"].quantile(0.05)
    apple_returns_figure = (
        ggplot(returns, mapping=aes(x="ret"))
        + geom_histogram(bins=100)
        + geom_vline(aes(xintercept=quantile_05), linetype="dashed")
        + labs(x="", y="", title="Distribution of daily Apple stock returns")
        + scale_x_continuous(labels=percent_format())
    )
    apple_returns_figure.show()

    symbols = tf.download_data(
        domain="constituents", index="Dow Jones Industrial Average"
    )
    prices_daily = tf.download_data(
        domain="stock_prices",
        symbols=symbols["symbol"].tolist(),
        start_date="2000-01-01",
        end_date="2023-12-31",
    )

    prices_figure = (
        ggplot(prices_daily, aes(y="adjusted_close", x="date", color="symbol"))
        + geom_line()
        + scale_x_datetime(date_breaks="5 years", date_labels="%Y")
        + labs(x="", y="", color="", title="Stock prices of DOW index constituents")
        + theme(legend_position="none")
    )
    prices_figure.show()

    returns_daily = (
        prices_daily.assign(
            ret=lambda x: x.groupby("symbol")["adjusted_close"].pct_change()
        )
        .get(["symbol", "date", "ret"])
        .dropna(subset="ret")
    )
    print(returns_daily.groupby("symbol")["ret"].describe().round(3))

    returns_monthly = (
        returns_daily.assign(
            date=returns_daily["date"].dt.to_period("M").dt.to_timestamp()
        )
        .groupby(["symbol", "date"], as_index=False)
        .agg(ret=("ret", lambda x: np.prod(1 + x) - 1))
    )
    apple_daily = returns_daily.query("symbol == 'AAPL'").assign(frequency="Daily")
    apple_monthly = returns_monthly.query("symbol == 'AAPL'").assign(
        frequency="Monthly"
    )
    apple_returns = pd.concat([apple_daily, apple_monthly], ignore_index=True)

    apple_returns_figure = (
        ggplot(apple_returns, aes(x="ret", fill="frequency"))
        + geom_histogram(position="identity", bins=50)
        + labs(
            x="",
            y="",
            fill="Frequency",
            title="Distribution of Apple returns across different frequencies",
        )
        + scale_x_continuous(labels=percent_format())
        + facet_wrap("frequency", scales="free")
        + theme(legend_position="none")
    )
    apple_returns_figure.show()

    trading_volume = (
        prices_daily.assign(
            trading_volume=lambda x: (x["volume"] * x["adjusted_close"]) / 1e9
        )  # Convert to billion USD
        .groupby("date")["trading_volume"]  # Aggregate trading volume by date
        .sum()  # Sum trading volumes across all symbols
        .reset_index()  # Reset index to turn Series back into DataFrame
        .assign(trading_volume_lag=lambda x: x["trading_volume"].shift(periods=1))
        .dropna(subset=["trading_volume_lag"])  # drop row(s) with missing values
        # Add column for lagged trading volume(yesterday's volume) for comparison
    )

    trading_volume_figure = (
        ggplot(trading_volume, aes(x="date", y="trading_volume"))
        + geom_line()
        + scale_x_datetime(date_breaks="5 years", date_labels="%Y")
        + labs(
            x="",
            y="",
            title="Aggregate daily trading volume of DOW index constituents in billion USD",
        )
    )
    trading_volume_figure.show()

    persistence_figure = (
        ggplot(trading_volume, aes(x="trading_volume_lag", y="trading_volume"))
        + geom_point()
        + geom_abline(aes(intercept=0, slope=1), linetype="dashed")
        + labs(
            x="Previous day aggregate trading volume",
            y="Aggregate trading volume",
            title="Persistence in daily trading volume of DOW constituents in billion USD",
        )
    )
    persistence_figure.show()


if __name__ == "__main__":
    run()
