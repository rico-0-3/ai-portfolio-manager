#!/bin/sh
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

# Default values
TICKERS=""
TOP50="true"
OPTIMIZE="true"
PARALLEL="1"
PERIOD="10y"
OUTPUT="data/models/pretrained_perfect"

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
            echo "  --period PERIOD             Data period: 5y, 10y, max (default: 10y)"
            echo "  --output DIR                Output directory (default: data/models/pretrained_perfect)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Train top 50, optimize, sequential"
            echo "  $0 --tickers AAPL,MSFT,GOOGL         # Train 3 custom tickers"
            echo "  $0 --parallel 4 --no-optimize        # Fast training, 4 parallel"
            echo "  $0 --period max --parallel 8         # Max data, 8 parallel"
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
    echo "Warning: No virtual environment detected"
    echo "Activate venv first: source venv/bin/activate"
    printf "Continue anyway? (y/N) "
    read -r reply
    case "$reply" in
        [Yy]*) ;;
        *) exit 1 ;;
    esac
fi

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print('GPU Available:', torch.cuda.is_available())" 2>/dev/null || echo "No GPU detected, using CPU"

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
if [ "$OPTIMIZE" = "true" ]; then
    echo "Optimization: Enabled (100 Optuna trials)"
else
    echo "Optimization: Disabled"
fi
echo "Parallel: $PARALLEL tickers"
echo "Output: $OUTPUT"
echo ""

# Estimate time
if [ "$OPTIMIZE" = "true" ]; then
    echo "Estimated time: ~1-1.5 hours per ticker"
else
    echo "Estimated time: ~15-20 minutes per ticker"
fi
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
