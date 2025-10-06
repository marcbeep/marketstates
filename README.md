# Machine Learning Models to predict Market States

> An exploratory repository for developing a model to predict the future of market states from their past. Adapted from [WunderChallenge](https://wunder-challenge.io).

The goal is to predict the next market state vector based on the sequence of states that came before it.
It's a sequence modeling problem.
You're given the market's history up to a certain point, and you need to forecast what happens next.

## The data

The dataset is a single table in Parquet format, containing multiple independent sequences.
Each row in the table represents a single market state at a specific step in a sequence. The table has **N + 3** columns:

- `seq_ix`: An ID for the sequence. When this number changes, you're starting a new, completely independent sequence.
- `step_in_seq`: The step number within a sequence (from 0 to 999).
- `need_prediction`: A boolean that’s `True` if we need a prediction from you for the _next_ step, and `False` otherwise.
- **N feature columns**: The remaining `N` columns are the anonymized numeric features that describe the market state.

### The sequences

Each sequence is exactly **1000 steps** long.

> **Note:**
> The first 100 steps (0-99) of every sequence are for warm-up. The model can use them to build context, but predictions won't be scored here. The score comes from predictions for steps 100 to 998.

Because each sequence is independent, you must reset your model’s internal state whenever you see a new `seq_ix`.

You can also rely on two key facts about the data ordering:

- **Within a sequence**, all steps are ordered by time.
- **The sequences themselves** are randomly shuffled, so `seq_ix` and `seq_ix + 1` are not related.

> **Tip: Create a validation set**
> Since all the sequences are independent and shuffled, you can create a reliable local validation set by splitting the sequences. For example, you could use the first 80% of the sequences for training and the remaining 20% for validation. You can split them by `seq_ix`.

## Evaluation and metrics

Predictions are evaluated using the **R²** (coefficient of determination) score.
For each feature _i_, the score is calculated as:
R²ᵢ = 1 - Σ(y_true - y_pred)² / Σ(y_true - y_mean)²
The final score is the average of the R² scores across all N features.
A higher R² score is better!

### Ideas

> - Recurrent models like **LSTM** or **GRU**.
> - Attention-based models like the **Transformer**.
> - Newer architectures like **Mamba-2**.
