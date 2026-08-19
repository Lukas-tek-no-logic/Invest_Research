"""Transaction cost models for simulated broker fees.

Supported brokers:
  ibkr   — Interactive Brokers US stocks
  xtb    — XTB US stocks
  crypto — Generic crypto exchange (Binance / Kraken style)
"""

from __future__ import annotations


def calculate_cost(broker: str, quantity: float, price: float) -> float:
    """Calculate a transaction fee based on broker cost model.

    Args:
        broker: Broker identifier (case-insensitive: "ibkr", "xtb", "crypto").
        quantity: Number of shares / units traded.
        price: Price per share / unit in USD.

    Returns:
        Transaction fee in USD.  Returns 0.0 for unknown brokers.
    """
    broker = broker.lower().strip()
    trade_value = quantity * price

    if broker == "ibkr":
        # Interactive Brokers US stocks: max($1.00, $0.005 × shares)
        return max(1.00, 0.005 * quantity)

    if broker == "xtb":
        # XTB US stocks: max($0.01, 0.08% × trade_value)
        return max(0.01, 0.0008 * trade_value)

    if broker in ("crypto", "binance", "kraken"):
        # Generic crypto: 0.1% × trade_value
        return 0.001 * trade_value

    # Unknown broker — no simulated cost
    return 0.0


def calculate_option_cost(broker: str, contracts: int, legs: int = 1) -> float:
    """Calculate a commission for one options order (all legs of one structure).

    Args:
        broker: Broker identifier (case-insensitive).
        contracts: Contracts per leg.
        legs: Number of legs filled in the order (vertical = 2, condor = 4).

    Returns:
        Commission in USD. Returns 0.0 for unknown brokers.
    """
    broker = broker.lower().strip()

    if broker == "ibkr":
        # IBKR US options: $0.65 per contract, $1.00 minimum per order.
        return max(1.00, 0.65 * contracts * legs)

    # Unknown broker — no simulated cost
    return 0.0
