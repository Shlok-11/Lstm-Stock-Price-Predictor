# Interview questions and answers

1. **What is the target?** The next observed trading day's closing price, using the preceding 20 observed closes. Holidays are not filled as artificial observations.
2. **Why a stacked LSTM?** A first recurrent layer emits a sequence; a second summarizes it before a scalar regression output. Capacity is deliberately small (32 and 16 units).
3. **How do LSTM gates help?** Input, forget and output gates regulate updates to cell state and hidden state, allowing temporal dependencies to persist more effectively than in a basic recurrent network.
4. **Why chronological splits?** Random splitting can train on a future regime and evaluate on the past. This workflow uses 70/15/15 raw-row boundaries.
5. **What is data leakage?** Information unavailable at forecast time influencing features, scaling or selection. The scaler sees training prices only and no window includes its target.
6. **May test windows include earlier test closes?** Yes for rolling one-step evaluation, where yesterday's realized close is known today. No for a multi-step forecast issued at a single starting point.
7. **Why MinMax scaling?** It gives the optimizer a manageable numeric range; it is not an assumption that all future prices stay in that range. Held-out values are not clipped.
8. **How are weights selected?** Adam minimizes scaled training MSE; early stopping restores the weights with lowest validation loss. Test data does not affect the stopping decision.
9. **Why evaluate in original units?** RMSE and MAE then describe price errors. MSE has squared-price units. Scaled errors alone are harder to interpret economically.
10. **What does R² mean?** 1 − SSE/SST on the evaluated targets. It can be negative and is not a percentage of correct prices or a guarantee of profitability.
11. **Why use persistence?** Price levels are strongly autocorrelated, so predicting the previous observed close is a difficult and essential benchmark.
12. **What is directional accuracy here?** The fraction where sign(predicted − previous close) equals sign(actual − previous close). Zero is a separate class, and the flat baseline can score only on unchanged days.
13. **Can good RMSE coexist with poor direction?** Yes. A prediction may be close to the price yet on the wrong side of yesterday's close.
14. **Why no claimed stock prediction accuracy?** This is regression with multiple metrics; a single invented percent hides error, period, units and baseline performance.
15. **What indicates overfitting?** Training loss falls while validation loss worsens. Early stopping limits it, but does not prove generalization.
16. **Is the experiment reproducible?** Data bytes, seed, model, dependency lock and split are recorded. Hardware and numerical library differences may still affect exact values.
17. **Why not tune repeatedly on the test set?** Repeated choices adapt to it and invalidate its role as unseen evidence. Use training/validation or nested walk-forward procedures instead.
18. **How would you improve evaluation?** Walk-forward backtests, multiple seeds, more stocks, alternative targets, uncertainty intervals and predefined model comparisons; report failures as well as wins.
19. **How would deployment differ?** Persist the scaler/model, validate incoming data and training range drift, enforce as-of timestamps, monitor errors, and retrain under a defined policy.
20. **Can this model support investment decisions?** This demonstration does not establish trading value. Costs, slippage, timing, risk, corporate actions and robust out-of-sample backtests must be addressed separately.
