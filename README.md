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

## Submission

### Evaluation and metrics

Predictions are evaluated using the **R²** (coefficient of determination) score.
For each feature _i_, the score is calculated as:
R²ᵢ = 1 - Σ(y_true - y_pred)² / Σ(y_true - y_mean)²
The final score is the average of the R² scores across all N features.
A higher R² score is better!

### Required structure

Your `solution.py` must define a class named `PredictionModel`. This class must have a `predict` method with the following signature:

```python
import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Initialize your model, internal state, etc.
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Your logic here.
        if not data_point.need_prediction:
            return None

        # When a prediction is needed, return a numpy array of length N.
        prediction = np.zeros(data_point.state.shape) # Replace with your model's output
        return prediction
```

The `data_point` object passed to your `predict` method is an instance of the `DataPoint` class, which has the following attributes:

- `seq_ix: int`: The ID for the current sequence.
- `step_in_seq: int`: The step number within the sequence.
- `need_prediction: bool`: Whether a prediction is required for this point.
- `state: np.ndarray`: The current market state vector of N features.

Your `predict` method should:

- Return `None` when `need_prediction` is `False`.
- Return a `numpy.ndarray` of shape `(N,)` when `need_prediction` is `True`.
- Remember to manage and reset the model's state for each new sequence (`seq_ix`).

### Packaging your solution

Your solution might include more than just `solution.py` (e.g., model weight files, helper Python modules, configs). Make sure to include all necessary files in your zip archive.

You can create the zip archive from your solution directory with a command like this:

```bash
# From inside your solution folder (e.g., my_awesome_solution/)
# This command zips up everything in the current directory.
zip -r ../solution.zip .
```

> **Note:** Make sure `solution.py` is at the root level inside the zip archive, not inside another folder.

### Ideas

> - Recurrent models like **LSTM** or **GRU**.
> - Attention-based models like the **Transformer**.
> - Newer architectures like **Mamba-2**.
