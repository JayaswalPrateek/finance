import tidyfinance as tf
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


if __name__ == "__main__":
    run()
