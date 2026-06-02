# Figure Captions

**Fig. 1. Experimental framework.** Non-financial time-series data are used for source-domain pretraining, and HS300 stock data are used for target-domain fine-tuning and evaluation.

**Fig. 2. Training curves.** Panel (a) shows target-domain training and fine-tuning curves. Panel (b) compares source-domain pretraining curves for Weather-6, ETTm1, and Electricity.

**Fig. 3. Same-distribution holdout metrics.** Comparison of LSTM, P-sLSTM, and P-sLSTM transfer variants under random holdout evaluation on HS300 2017-2018 windows.

**Fig. 4. Out-of-time 2019 metrics.** Evaluation on HS300 2019 data. This setting is harder and reflects temporal distribution shift.

**Fig. 5. Regression metrics across transfer sources.** MAE and RMSE comparisons for P-sLSTM variants, showing the effect of source-domain selection.

**Fig. 6. Prediction quality of the best transfer model.** Panel (a) compares predicted and true `p_change` trajectories. Panel (b) shows the predicted-vs-true scatter distribution.

**Fig. 7. Confusion matrices.** Normalized 6-class confusion matrices for LSTM target-only and P-sLSTM with Weather pretraining.

**Fig. 8. Transfer gain.** Metric gains of each non-financial source-domain transfer setting over P-sLSTM target-only training.

Recommended usage:

```text
Use PDF or SVG for LaTeX/Word when possible.
Use PNG only if the editor does not accept vector graphics.
```
