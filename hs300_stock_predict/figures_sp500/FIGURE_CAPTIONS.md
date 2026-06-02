# S&P500 Figure Captions

## SP500_Fig1_dataset_profile

Dataset profile of the large-scale S&P500 panel, including train/test split size, future up-day ratio, and the distribution of observations per ticker.

## SP500_Fig2_direction_task_framework

Overall framework for the 1-day stock-return direction-prediction task. Historical S&P500 panel features are encoded by sequence models, and the sign of the predicted future return is used as the main direction label.

## SP500_Fig3_direction_metrics

Comparison of calibrated test-set direction-prediction metrics, including Directional Accuracy, Binary F1, and Direction AUC.

## SP500_Fig4_training_curves

Training and validation loss curves for the target-only, transfer, and source-pretraining stages.

## SP500_Fig5_direction_confusion

Row-normalized confusion matrices for down/up direction prediction on the out-of-time S&P500 test set.

## SP500_Fig6_prediction_deciles

Actual up-day ratio across prediction deciles. Higher deciles correspond to larger predicted returns.

## SP500_Fig7_market_prediction_curve

Market-level actual versus predicted 1-day return curves on the out-of-time test period. Values are averaged cross-sectionally across S&P500 tickers and smoothed with a 15-day rolling window.

## SP500_Fig8_rolling_direction_accuracy

Thirty-day rolling cross-sectional direction accuracy on the 2024-2025 out-of-time test period, illustrating temporal variation and market-regime drift.

## SP500_Fig9_single_ticker_prediction_examples

Single-stock actual versus predicted return curves for representative large-cap stocks, smoothed with a 10-day rolling window.

## SP500_Fig10_validation_direction_accuracy

Validation directional accuracy during direction-aware P-sLSTM training. The curve highlights the in-distribution improvement achieved by adding a direction-aware loss.

