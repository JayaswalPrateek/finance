import pandas as pd
import numpy as np
import tidyfinance as tf

from plotnine import *
from mizani.formatters import percent_format
from adjustText import adjust_text


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
