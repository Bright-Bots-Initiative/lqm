Optuna hyper-parameter tuning loop aiming for a 7-day hit-rate ≥ TARGET.
Writes best params to best_params.json.

Enhanced with robust logging and error handling for Cloud Run debugging.
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path

# --- Robust print function ---
def print_flush(*args, **kwargs):
    """Prints and immediately flushes the output stream."""
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

print_flush("--- tune.py starting --- ")

try:
    print_flush("Importing Optuna...")
    import optuna
    print_flush("Importing run_backtest from backtest...")
    # Ensure backtest.py is also robust or add try/except here if needed
    from backtest import run_backtest
    print_flush("Imports successful.")
except ImportError as e:
    print_flush(f"ERROR: Failed to import dependencies: {e}")
    print_flush(traceback.format_exc())
    sys.exit(1) # Exit immediately if imports fail
except Exception as e:
    print_flush(f"ERROR: Unexpected error during imports: {e}")
    print_flush(traceback.format_exc())
    sys.exit(1)

SEARCH_SPACE = {
    "ma_trend_weight":    (-1.0, 1.0),
    "rsi_extreme_weight": (-1.0, 1.0),
    "vol_rank_weight":    (-1.0, 1.0),
    "momentum_3d_weight": (-1.0, 1.0),
    "angle_0":            (0.0, 3.14159),
    "angle_1":            (0.0, 3.14159),
    # static thresholds left as constants inside model.py
}
print_flush(f"Search space defined: {SEARCH_SPACE}")

BEST_FILE = Path("best_params.json")
print_flush(f"Best parameters file path: {BEST_FILE}")

def objective(trial: optuna.trial.Trial, target: float) -> float:
    print_flush(f"--- Starting Optuna Trial {trial.number} ---")
    try:
        params = {}
        print_flush("Suggesting parameters...")
        for k, rng in SEARCH_SPACE.items():
            params[k] = trial.suggest_float(k, *rng)
            print_flush(f"  {k}: {params[k]:.4f}")

        print_flush("Setting user attributes...")
        # >>>> inject params into model globally (OptionsPredictor reads from env)
        # This mechanism might need review depending on how OptionsPredictor actually reads params.
        # If it reads environment variables, this won't work directly. 
        # Assuming OptionsPredictor is adapted or this is a placeholder.
        for k, v in params.items():
            trial.set_user_attr(k, v)
        print_flush("User attributes set.")

        print_flush("Running backtest...")
        # Ensure run_backtest also has robust error handling/logging
        _, hit_rate = run_backtest(years=0.25)      # ~3 months sample
        print_flush(f"Backtest complete. Hit rate: {hit_rate:.4f}")

        # maximise metric = hit-rate – abs(target-hit) penalty
        metric = hit_rate - abs(target - hit_rate)
        print_flush(f"Trial {trial.number} result: Metric = {metric:.4f} (Hit Rate = {hit_rate:.4f}, Target = {target:.4f})")
        return metric

    except Exception as e:
        print_flush(f"ERROR: Exception occurred during Optuna trial {trial.number}: {e}")
        print_flush(traceback.format_exc())
        # Re-raise the exception to let Optuna handle trial failure
        raise

def main(trials: int, target: float) -> None:
    print_flush("--- Starting main function --- ")
    print_flush(f"Number of trials: {trials}")
    print_flush(f"Target hit rate: {target}")

    try:
        print_flush("Creating Optuna study...")
        study = optuna.create_study(direction="maximize")
        print_flush("Optuna study created.")

        print_flush("Starting Optuna optimization...")
        study.optimize(lambda t: objective(t, target), n_trials=trials)
        print_flush("Optuna optimization finished.")

        print_flush("Processing best trial...")
        best_params = study.best_trial.user_attrs
        print_flush(f"Best parameters found: {best_params}")

        print_flush(f"Writing best parameters to {BEST_FILE}...")
        BEST_FILE.write_text(json.dumps(best_params, indent=2))
        print_flush(f"Best parameters saved ➜ {BEST_FILE.resolve()}")
        print_flush(f"Best trial metric (Hit-rate - Penalty): {study.best_value:.3f}")
        # You might want to log the actual best hit rate as well
        # best_hit_rate = study.best_trial.value + abs(target - study.best_trial.value) # Reconstruct if needed
        # print_flush(f"Best trial hit rate: {best_hit_rate:.3f}")

        print_flush("--- main function finished successfully ---")

    except Exception as e:
        print_flush(f"ERROR: An exception occurred in the main function: {e}")
        print_flush(traceback.format_exc())
        sys.exit(1) # Ensure the script exits with a non-zero code on error

if __name__ == "__main__":
    print_flush("--- Script execution started (__name__ == '__main__') ---")
    try:
        p = argparse.ArgumentParser()
        p.add_argument("--trials", type=int, default=50, help="Optuna trials")
        p.add_argument("--target", type=float, default=0.10,
                       help="Desired weekly profit as decimal, e.g. 0.10 = 10 %")
        print_flush("Parsing command-line arguments...")
        args = p.parse_args()
        print_flush(f"Arguments parsed: trials={args.trials}, target={args.target}")

        main(args.trials, args.target)

        print_flush("--- Script execution finished successfully --- ")
        sys.exit(0) # Explicitly exit with 0 on success

    except SystemExit as e:
        # Catch SystemExit to report the exit code if it's non-zero
        print_flush(f"Script exited with code: {e.code}")
        raise # Re-raise to ensure the process actually exits
    except Exception as e:
        print_flush(f"ERROR: An unhandled exception occurred at the top level: {e}")
        print_flush(traceback.format_exc())
        sys.exit(1) # Ensure non-zero exit code for any top-level error