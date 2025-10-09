#!/bin/bash
# AI Portfolio Manager - Prediction Script
# Takes stock tickers and returns complete portfolio allocation with predictions

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
BUDGET=""
PERIOD="2y"
RISK_PROFILE=""
USE_ML="true"
USE_PRETRAINED="true"
FINETUNE_DAYS="30"
OUTPUT_FILE=""

# Help function
show_help() {
    cat << EOF
Usage: ./predict.sh [TICKER1 TICKER2 ...] [OPTIONS]

Analyze stocks and generate optimal portfolio allocation with ML predictions.
Uses ALL optimization methods combined based on your risk profile.

ARGUMENTS:
    TICKER1 TICKER2 ...    Stock ticker symbols (e.g., AAPL MSFT GOOGL)
                          If omitted, uses top 50 US stocks automatically

OPTIONS:
    --budget NUM           Initial investment budget (default: from config.yaml)
    --period PERIOD        Historical data period: 1y, 2y, 5y (default: 2y)
    --risk PROFILE         Risk profile: low, medium, high (default: medium)
                          low = conservative (focus on risk parity + CVaR)
                          medium = balanced (mix of all methods)
                          high = aggressive (focus on max returns)
    --no-ml               Disable ML predictions
    --no-pretrained       Disable pretrained models (train from scratch)
    --finetune-days NUM   Days for fine-tuning pretrained models (default: 30)
    --output FILE         Save results to JSON file
    -h, --help            Show this help message

EXAMPLES:
    # Quick start - analyze top 50 US stocks with medium risk
    ./predict.sh

    # Analyze specific stocks
    ./predict.sh AAPL MSFT GOOGL

    # Conservative portfolio with \$50k
    ./predict.sh AAPL MSFT GOOGL AMZN TSLA --budget 50000 --risk low

    # Aggressive portfolio on top 50 stocks
    ./predict.sh --risk high --budget 100000

    # Save results to file
    ./predict.sh AAPL MSFT --output results/portfolio.json

    # Use 5 years of data
    ./predict.sh --period 5y

EOF
    exit 0
}

# Parse arguments
TICKERS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --period)
            PERIOD="$2"
            shift 2
            ;;
        --risk)
            RISK_PROFILE="$2"
            shift 2
            ;;
        --no-ml)
            USE_ML="false"
            shift
            ;;
        --no-pretrained)
            USE_PRETRAINED="false"
            shift
            ;;
        --finetune-days)
            FINETUNE_DAYS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            TICKERS+=("$1")
            shift
            ;;
    esac
done

# Check if tickers provided - if not, use top 50 US stocks
if [ ${#TICKERS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No tickers provided. Using top 50 US stocks...${NC}"
    # Top 50 US stocks by market cap (mix of sectors for diversification)
    TICKERS=(
        # Tech Giants
        AAPL MSFT GOOGL AMZN NVDA META TSLA
        # Finance
        JPM BAC WFC GS MS C V MA
        # Healthcare
        UNH JNJ PFE ABBV MRK LLY TMO
        # Consumer
        WMT HD PG KO PEP COST MCD NKE
        # Industrial
        BA CAT HON UPS GE RTX
        # Energy
        XOM CVX COP SLB
        # Communication
        DIS NFLX CMCSA T VZ
        # Tech/Semiconductor
        AVGO INTC AMD QCOM ORCL CSCO
        # Others
        ABT ADBE CRM NOW
    )
    echo -e "${GREEN}Selected ${#TICKERS[@]} diversified stocks across sectors${NC}"
fi

# Set default risk profile if not provided
if [ -z "$RISK_PROFILE" ]; then
    RISK_PROFILE="medium"
fi

# Header
source venv/bin/activate 2>/dev/null || true
echo ""
echo "================================================================================"
echo -e "              ${BLUE}AI PORTFOLIO MANAGER - PREDICTION ENGINE${NC}"
echo "================================================================================"
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run ./setup.sh first"
    exit 1
fi

echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Create Python script
TEMP_SCRIPT=$(mktemp /tmp/predict_XXXXXX.py)

# Get project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cat > "$TEMP_SCRIPT" << PYTHON_SCRIPT
import sys
import json
from pathlib import Path

# Add project directory to path
project_dir = Path("$PROJECT_DIR")
sys.path.insert(0, str(project_dir))

from src.orchestrator import PortfolioOrchestrator

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('tickers', nargs='+')
    parser.add_argument('--budget', type=float, default=None)
    parser.add_argument('--period', default='2y')
    parser.add_argument('--risk', default=None)
    parser.add_argument('--use-ml', action='store_true')
    parser.add_argument('--use-pretrained', action='store_true')
    parser.add_argument('--finetune-days', type=int, default=30)
    parser.add_argument('--output', default='')

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PortfolioOrchestrator()

    # Update config if provided
    if args.budget:
        orchestrator.config.config['portfolio']['initial_budget'] = args.budget

    if args.risk:
        orchestrator.config.config['optimization']['risk_profile'] = args.risk

    # Run pipeline (uses all optimization methods automatically)
    results = orchestrator.run_full_pipeline(
        tickers=args.tickers,
        period=args.period,
        use_ml_predictions=args.use_ml,
        use_pretrained=args.use_pretrained,
        finetune_days=args.finetune_days
    )

    # Print results
    orchestrator.print_results(results)

    # Save to file if requested
    if args.output:
        # Convert to JSON-serializable format
        output_data = {
            'tickers': args.tickers,
            'budget': args.budget,
            'weights': results['weights'],
            'allocation': results['allocation'],
            'metrics': results['metrics'],
            'risk_metrics': results['risk_metrics'],
            'sentiment_scores': results['sentiment_scores'],
            'latest_prices': results['latest_prices']
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to: {args.output}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
PYTHON_SCRIPT

# Build Python command
PYTHON_CMD="python3 $TEMP_SCRIPT ${TICKERS[*]}"

if [ -n "$BUDGET" ]; then
    PYTHON_CMD="$PYTHON_CMD --budget $BUDGET"
fi

PYTHON_CMD="$PYTHON_CMD --period $PERIOD"

if [ -n "$RISK_PROFILE" ]; then
    PYTHON_CMD="$PYTHON_CMD --risk $RISK_PROFILE"
fi

if [ "$USE_ML" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --use-ml"
fi

if [ "$USE_PRETRAINED" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --use-pretrained --finetune-days $FINETUNE_DAYS"
fi

if [ -n "$OUTPUT_FILE" ]; then
    PYTHON_CMD="$PYTHON_CMD --output $OUTPUT_FILE"
fi

# Run Python script
echo -e "\n${YELLOW}Running portfolio analysis...${NC}"
echo ""

eval $PYTHON_CMD

EXIT_CODE=$?

# Cleanup
rm -f "$TEMP_SCRIPT"
deactivate

# Footer
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}                    PREDICTION COMPLETED SUCCESSFULLY${NC}"
    echo "================================================================================"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo -e "${RED}                    PREDICTION FAILED${NC}"
    echo "================================================================================"
    echo ""
    echo "Please check the error messages above"
    echo ""
fi

exit $EXIT_CODE
