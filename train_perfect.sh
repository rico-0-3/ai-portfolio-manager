#!/bin/bash
################################################################################
# PERFECT MODEL TRAINING - Local Training Script
################################################################################
#
# Trains MetaModel with ALL advanced techniques locally.
# Requires: GPU with CUDA support (recommended) or CPU (slower)
#
# Usage:
#   ./train_perfect.sh                    # Train top 50 tickers with optimization
#   ./train_perfect.sh --tickers AAPL,MSFT,GOOGL  # Custom tickers
#   ./train_perfect.sh --no-optimize      # Skip Optuna (faster, less accurate)
#   ./train_perfect.sh --parallel 4       # Train 4 tickers in parallel
#
################################################################################

set -e  # Exit on error

# Default values (optimized for 5-day predictions with 1y training)
TICKERS=""
TOP50="true"
OPTIMIZE="true"
PARALLEL="1"
PERIOD="10y"          # Fetch 10 years for good local cache
ROLLING_WINDOW="true" # Enable rolling window by default
WINDOW_YEARS="2"      # Use only most recent 2 years for training → AV AUC ~0.70
HOLDOUT_MONTHS="3"    # 0 months holdout for 2y training (12 cycles of 5-day predictions)
OUTPUT="data/models/pretrained_perfect"
DISABLE_HOLDOUT="false"

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --tickers)
            TICKERS="$2"
            TOP50="false"
            shift 2
            ;;
        --top50)
            TOP50="true"
            shift
            ;;
        --no-optimize)
            OPTIMIZE="false"
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --period)
            PERIOD="$2"
            shift 2
            ;;
        --no-rolling-window)
            ROLLING_WINDOW="false"
            shift
            ;;
        --window-years)
            WINDOW_YEARS="$2"
            shift 2
            ;;
        --holdout-months)
            HOLDOUT_MONTHS="$2"
            shift 2
            ;;
        --disable-holdout)
            DISABLE_HOLDOUT="true"
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tickers AAPL,MSFT,GOOGL   Train specific tickers (comma-separated)"
            echo "  --top50                     Train on S&P 500 top 50 (default)"
            echo "  --no-optimize               Skip hyperparameter optimization (faster)"
            echo "  --parallel N                Train N tickers in parallel (default: 1)"
            echo "  --period PERIOD             Data period: 1y, 2y, 3y, 5y, 10y, max (default: 10y)"
            echo "  --no-rolling-window         Disable rolling window (uses all data)"
            echo "  --window-years N            Rolling window size in years (default: 1)"
            echo "  --holdout-months N          Holdout period in months (default: 3)"
            echo "  --disable-holdout           Train on entire dataset (no final holdout)"
            echo "  --output DIR                Output directory (default: data/models/pretrained_perfect)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Best Practices (5-day predictions):"
            echo "  - 1y training → 3 months holdout (12 cycles, default)"
            echo "  - 2y training → 6 months holdout (balanced)"
            echo "  - 5y+ training → 12 months holdout (robust)"
            echo "  - For quick tests: --period 2y (skip rolling window, direct 2y)"
            echo "  - For max data: --period 10y --no-rolling-window (high AV AUC ~0.92)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # DEFAULT: 1y training, 3mo holdout"
            echo "  $0 --tickers AAPL,MSFT,GOOGL         # Train 3 custom tickers"
            echo "  $0 --parallel 4 --no-optimize        # Fast training, 4 parallel"
            echo "  $0 --window-years 2 --holdout-months 6   # 2y training, 6mo holdout"
            echo "  $0 --window-years 5 --holdout-months 12  # 5y training, 12mo holdout"
            echo "  $0 --disable-holdout                 # Train on full dataset (no holdout)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Error: Could not activate virtual environment. Please run 'source venv/bin/activate' manually."
        exit 1
    fi
fi

# Build command
CMD="python3 train_perfect_colab.py"

if [ "$TOP50" = "true" ]; then
    CMD="$CMD --top50"
elif [ -n "$TICKERS" ]; then
    CMD="$CMD --tickers $TICKERS"
else
    echo "Error: Must specify --tickers or --top50"
    exit 1
fi

CMD="$CMD --period $PERIOD"
CMD="$CMD --output $OUTPUT"

if [ "$OPTIMIZE" = "true" ]; then
    CMD="$CMD --optimize"
fi

if [ "$PARALLEL" -gt 1 ]; then
    CMD="$CMD --parallel-tickers $PARALLEL"
fi

if [ "$ROLLING_WINDOW" = "true" ]; then
    CMD="$CMD --rolling-window --window-years $WINDOW_YEARS"
fi

# Add holdout parameter(s)
if [ "$DISABLE_HOLDOUT" = "true" ]; then
    CMD="$CMD --disable-holdout"
else
    CMD="$CMD --holdout-months $HOLDOUT_MONTHS"
fi

# Print summary
echo ""
echo "=========================================="
echo "  PERFECT MODEL TRAINING"
echo "=========================================="
echo ""

if [ "$TOP50" = "true" ]; then
    echo "Tickers: S&P 500 Top 50"
else
    echo "Tickers: $TICKERS"
fi

echo "Period: $PERIOD"
if [ "$ROLLING_WINDOW" = "true" ]; then
    echo "Rolling Window: Enabled (most recent $WINDOW_YEARS years)"
else
    echo "Rolling Window: Disabled (use all $PERIOD data)"
fi
if [ "$DISABLE_HOLDOUT" = "true" ]; then
    echo "Holdout: DISABLED (train + validate on full dataset)"
else
    echo "Holdout: $HOLDOUT_MONTHS months (~$((HOLDOUT_MONTHS * 21)) trading days)"
fi
if [ "$OPTIMIZE" = "true" ]; then
    echo "Optimization: Enabled (100 Optuna trials)"
else
    echo "Optimization: Disabled"
fi
echo "Parallel: $PARALLEL tickers"
echo "Output: $OUTPUT"
echo ""

echo "Note: Training can be resumed if interrupted"
echo ""

# Confirm
printf "Start training? (y/N) "
read -r reply
case "$reply" in
    [Yy]*) ;;
    *)
        echo "Training cancelled"
        exit 0
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT"
mkdir -p logs

# Run training with logging
LOGFILE="logs/train_perfect_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Starting training..."
echo "Logs: $LOGFILE"
echo "=========================================="
echo ""

# Run with output to both terminal and log
$CMD 2>&1 | tee "$LOGFILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  TRAINING COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Models saved to: $OUTPUT"
    echo "Training log: $LOGFILE"
    echo ""
    echo "Next steps:"
    echo "  1. Verify models: ls -lh $OUTPUT"
    echo "  2. Update orchestrator to use: $OUTPUT"
    echo "  3. Test predictions: ./predict.sh AAPL MSFT GOOGL"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "  TRAINING FAILED!"
    echo "=========================================="
    echo ""
    echo "Check log for errors: $LOGFILE"
    echo "Training can be resumed - run the same command again"
    echo ""
    exit 1
fi
