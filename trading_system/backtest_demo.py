"""Create simulated trading records with mock prices only."""

import csv
from pathlib import Path

from trading_system import config
from trading_system.strategy import calculate_moving_average, generate_signal


BACKTEST_PRICES = [
    73200,
    73500,
    73400,
    73600,
    73300,
    71700,
    72000,
    74400,
    75600,
    73900,
]

FIELD_NAMES = [
    "step",
    "symbol",
    "current_price",
    "moving_average",
    "signal",
    "reason",
    "dry_run",
    "order_status",
]


def build_trade_records(symbol=config.STOCK_SYMBOL):
    """Simulate several steps and keep every signal as a trade record."""
    config.validate_safety_settings()

    records = []
    window = config.MOVING_AVERAGE_WINDOW

    for price_index in range(window - 1, len(BACKTEST_PRICES)):
        visible_prices = BACKTEST_PRICES[: price_index + 1]
        current_price = visible_prices[-1]
        moving_average = calculate_moving_average(visible_prices, window)
        signal, reason = generate_signal(
            current_price,
            moving_average,
            config.THRESHOLD_PERCENT,
        )

        records.append(
            {
                "step": len(records) + 1,
                "symbol": symbol,
                "current_price": current_price,
                "moving_average": round(moving_average, 2),
                "signal": signal,
                "reason": reason,
                "dry_run": config.DRY_RUN,
                "order_status": "SIMULATION ONLY - no real order was placed",
            }
        )

    return records


def save_trade_log_csv(records, output_path):
    """Save the simulated records in a spreadsheet-friendly CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(records)


def save_trade_log_txt(records, output_path):
    """Save the simulated records in a readable text report."""
    lines = [
        "=== Simulated Mock Trading Record ===",
        "Source: mock price data only",
        f"Dry run: {config.DRY_RUN}",
        "Order status: simulation only; no real KIS API order was placed.",
        "",
    ]

    for record in records:
        lines.extend(
            [
                f"Step {record['step']}",
                f"  Symbol: {record['symbol']}",
                f"  Current price: {record['current_price']:,.0f} KRW",
                f"  Moving average: {record['moving_average']:,.2f} KRW",
                f"  Signal: {record['signal']}",
                f"  Reason: {record['reason']}",
                f"  Dry run: {record['dry_run']}",
                f"  Order status: {record['order_status']}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_backtest_demo():
    """Run the mock backtest and save CSV and text trade logs."""
    records = build_trade_records()
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "trade_log.csv"
    txt_path = results_dir / "trade_log.txt"
    save_trade_log_csv(records, csv_path)
    save_trade_log_txt(records, txt_path)

    return "\n".join(
        [
            "=== Mock Trading Backtest Demo ===",
            f"Records created: {len(records)}",
            f"CSV log: {csv_path.relative_to(project_root)}",
            f"Text log: {txt_path.relative_to(project_root)}",
            "Safety: simulation only, dry_run=True, no real order was placed.",
        ]
    )


if __name__ == "__main__":
    print(run_backtest_demo())

