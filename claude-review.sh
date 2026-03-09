#!/bin/bash

# Claude Code Local Review Script
# Adapted for causal-optimizer - Causally-informed experiment optimization engine
# Supports structured prompts, focus areas, and multiple output modes

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
FOCUS_AREAS=""
MODEL=""
POST_COMMENT=true
OUTPUT_MODE="comment"
DRY_RUN=false
MAX_DIFF_LINES=500  # Maximum diff lines to include for review

# Get current branch to return to later
ORIGINAL_BRANCH=$(git branch --show-current)

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] [PR_NUMBER]"
    echo "  PR_NUMBER: Optional PR number to review (defaults to current PR)"
    echo ""
    echo "Options:"
    echo "  --focus AREA        Focus review on specific area:"
    echo "                      security, performance, testing, causal, engine,"
    echo "                      types, algorithms, architecture, style"
    echo "  --model MODEL       Use specific Claude model"
    echo "  --save-file         Save review to file instead of posting as comment (default: post comment)"
    echo "  --draft-comment     Post review as draft PR comment"
    echo "  --max-diff-lines N  Maximum diff lines to include (default: 500, 0 = no limit)"
    echo "  --dry-run          Show what would be reviewed without calling Claude"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                               # Review current PR and post as comment"
    echo "  $0 54                           # Review PR #54 and post as comment"
    echo "  $0 --focus causal 54            # Focus on causal inference correctness"
    echo "  $0 --focus engine 54            # Focus on engine loop integration"
    echo "  $0 --focus algorithms 54        # Focus on algorithm correctness"
    echo "  $0 --focus types 54             # Focus on type safety and data models"
    echo "  $0 --save-file 54               # Save review to file instead of posting"
    echo "  $0 --draft-comment 54           # Post as draft PR comment"
    echo "  $0 --max-diff-lines 1000 54     # Include up to 1000 diff lines"
    echo "  $0 --max-diff-lines 0 54        # Include full diff (no limit)"
    echo "  $0 --dry-run 54                 # Preview what would be reviewed"
    exit 1
}

# Check dependencies
check_dependencies() {
    local missing_deps=()

    if ! command -v gh &> /dev/null; then
        missing_deps+=("GitHub CLI (gh) - https://cli.github.com/")
    fi

    if ! command -v claude &> /dev/null; then
        missing_deps+=("Claude Code - https://docs.anthropic.com/en/docs/claude-code")
    fi

    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq - https://jqlang.github.io/jq/")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing dependencies:${NC}"
        printf '  - %s\n' "${missing_deps[@]}"
        echo ""
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
}

check_dependencies

# Verify script is run from repository root
if [ ! -f "pyproject.toml" ] || [ ! -d "causal_optimizer" ]; then
    echo -e "${RED}Error: Must be run from causal-optimizer repository root${NC}"
    echo "Expected to find pyproject.toml and causal_optimizer/ directory"
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --focus)
            FOCUS_AREAS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --save-file)
            POST_COMMENT=false
            OUTPUT_MODE="file"
            shift
            ;;
        --draft-comment)
            POST_COMMENT=true
            OUTPUT_MODE="draft-comment"
            shift
            ;;
        --max-diff-lines)
            MAX_DIFF_LINES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
        *)
            if [[ $1 =~ ^[0-9]+$ ]]; then
                PR_NUM=$1
            else
                echo -e "${RED}Error: Invalid PR number: $1${NC}"
                usage
            fi
            shift
            ;;
    esac
done

# Get PR number if not provided
if [ -z "$PR_NUM" ]; then
    PR_NUM=$(gh pr view --json number -q .number 2>/dev/null || echo "")
    if [ -z "$PR_NUM" ]; then
        echo -e "${RED}Error: Not currently on a PR branch${NC}"
        echo "Please specify a PR number or checkout a PR branch"
        usage
    fi
fi

# Validate PR exists
if ! gh pr view "$PR_NUM" > /dev/null 2>&1; then
    echo -e "${RED}Error: PR #$PR_NUM not found${NC}"
    exit 1
fi

# Helper function to detect engine/loop files
has_engine_files() {
    gh pr diff "$PR_NUM" --name-only | grep -E "(engine/|loop\.py)" > /dev/null 2>&1
}

# Helper function to detect causal/graph files
has_causal_files() {
    gh pr diff "$PR_NUM" --name-only | grep -E "(discovery/|pomis|causal|graph)" > /dev/null 2>&1
}

# Helper function to detect algorithm files
has_algorithm_files() {
    gh pr diff "$PR_NUM" --name-only | grep -E "(estimator/|predictor/|designer/|evolution/|optimizer/)" > /dev/null 2>&1
}

# Helper function to detect type/model files
has_type_files() {
    gh pr diff "$PR_NUM" --name-only | grep -E "(types\.py|domain_adapters/)" > /dev/null 2>&1
}

# Helper function to detect test files
has_test_files() {
    gh pr diff "$PR_NUM" --name-only | grep -E "^tests/" > /dev/null 2>&1
}

# Helper function to count significant changes
count_significant_changes() {
    local additions=$(gh pr view "$PR_NUM" --json additions -q .additions)
    local deletions=$(gh pr view "$PR_NUM" --json deletions -q .deletions)
    echo $((additions + deletions))
}

# Helper function to create intelligent diff summary
create_diff_summary() {
    local pr_num="$1"
    local max_lines="$2"

    if [ "$max_lines" -eq 0 ]; then
        # No limit - include full diff
        gh pr diff "$pr_num"
        return
    fi

    local full_diff
    full_diff=$(gh pr diff "$pr_num")
    local diff_line_count
    diff_line_count=$(echo "$full_diff" | wc -l | tr -d ' ')

    if [ "$diff_line_count" -le "$max_lines" ]; then
        # Diff is within limits - include it all
        echo "$full_diff"
    else
        # Diff is too long - create intelligent summary
        echo "### Large Diff Summary (${diff_line_count} lines total, showing first ${max_lines} lines)"
        echo ""
        echo "\`\`\`diff"
        echo "$full_diff" | head -n "$max_lines"
        echo ""
        echo "... (diff truncated - ${diff_line_count} total lines, showing first ${max_lines})"
        local repo_owner
        local repo_name
        repo_owner=$(gh repo view --json owner -q '.owner.login')
        repo_name=$(gh repo view --json name -q '.name')
        echo "Full diff available at: https://github.com/${repo_owner}/${repo_name}/pull/${pr_num}/files"
        echo "\`\`\`"
    fi
}

# Helper function to generate review prompt based on focus and file types
generate_review_prompt() {
    local base_prompt="Please review this pull request for causal-optimizer, a causally-informed experiment optimization engine, and provide feedback on:
- Code quality and Python best practices
- Potential bugs or issues
- Performance considerations
- Test coverage
- Type safety and dataclass/Pydantic usage
- Correct use of numpy, scipy, scikit-learn, and networkx APIs

Be constructive and helpful in your feedback. This is a Python library (Python 3.10+) using hatch for build, ruff for linting (100 char line length), and pytest for testing.

### Project Architecture
The engine runs an optimization loop with phases (exploration -> optimization -> exploitation):
- engine/loop.py: Main ExperimentEngine orchestrator
- designer/: Factorial/LHS design and fANOVA screening
- estimator/: Bootstrap effect estimation for keep/discard decisions
- predictor/: Off-policy surrogate prediction (observation-intervention tradeoff)
- optimizer/: Suggestion strategies, POMIS computation
- evolution/: MAP-Elites diversity archive
- discovery/: Causal graph learning (PC algorithm)
- validator/: Sensitivity analysis for causal findings
- domain_adapters/: Marketing and ML training templates
- types.py: CausalGraph, SearchSpace, ExperimentResult, ExperimentLog

### Key Concepts
- CausalGraph supports directed edges (X -> Y) and bidirected edges (X <-> Y for unobserved confounders)
- POMIS (Possibly Optimal Minimal Intervention Sets) prunes which variable subsets to experiment with
- The engine decides what to try next using causal inference, Bayesian optimization, and DoE
- Modules are built but many are not yet wired into the engine loop (integration is in progress)"

    local additional_prompt=""

    # Add engine-specific prompts if relevant files are detected
    if has_engine_files || [[ "$FOCUS_AREAS" == *"engine"* ]]; then
        additional_prompt="${additional_prompt}

For engine loop code, also review:
- Phase transition logic (exploration -> optimization -> exploitation)
- Correct integration of sub-modules (screening, effect estimation, off-policy prediction)
- Proper use of ExperimentLog for history tracking
- Keep/discard decision logic and statistical rigor
- Thread safety if any async patterns are used"
    fi

    # Add causal-specific prompts if relevant files are detected
    if has_causal_files || [[ "$FOCUS_AREAS" == *"causal"* ]]; then
        additional_prompt="${additional_prompt}

For causal inference code, also review:
- CausalGraph operations: ancestors, descendants, c-components, do-operator, subgraph
- Bidirected edge handling for unobserved confounders
- POMIS algorithm correctness (MUCT fixed-point, IB computation, recursive enumeration)
- Graph surgery (do-operator) must remove ALL incoming edges to intervened variables
- Topological ordering assumptions and DAG validation
- Correctness of causal discovery (PC algorithm) integration"
    fi

    # Add algorithm-specific prompts if relevant files are detected
    if has_algorithm_files || [[ "$FOCUS_AREAS" == *"algorithms"* ]]; then
        additional_prompt="${additional_prompt}

For algorithm code, also review:
- Statistical correctness (bootstrap CIs, p-values, effect sizes)
- Numerical stability (log-space computations, division by zero guards)
- fANOVA screening: correct variance decomposition
- Off-policy prediction: surrogate model fit quality and uncertainty calibration
- MAP-Elites: archive update rules, behavioral descriptor computation
- Surrogate model: RandomForest fallback when Ax/BoTorch not available
- Latin Hypercube sampling: space-filling properties"
    fi

    # Add type/model-specific prompts if relevant files are detected
    if has_type_files || [[ "$FOCUS_AREAS" == *"types"* ]]; then
        additional_prompt="${additional_prompt}

For types and data models, also review:
- CausalGraph immutability and consistency (edges match nodes)
- SearchSpace variable bounds validation
- ExperimentResult and ExperimentLog correctness
- Domain adapter causal graphs: do edges match real-world causal relationships?
- Backward compatibility of type changes (other modules depend on these)"
    fi

    # Add focus area specific prompts
    case "$FOCUS_AREAS" in
        security)
            additional_prompt="${additional_prompt}

Focus specifically on security concerns:
- Input validation for parameters and variable names
- No arbitrary code execution from user-provided functions
- Safe handling of numpy/scipy operations (no unchecked array indexing)
- Dependency security (pinned versions, trusted sources)"
            ;;
        performance)
            additional_prompt="${additional_prompt}

Focus specifically on performance:
- Numpy vectorization vs Python loops
- Unnecessary copies of large arrays/dataframes
- Graph algorithm complexity (ancestors/descendants should be O(V+E))
- Bootstrap resampling efficiency
- Surrogate model fitting overhead
- Memory usage for large experiment logs"
            ;;
        testing)
            additional_prompt="${additional_prompt}

Focus specifically on testing:
- Test coverage for edge cases (empty graphs, single variable, no experiments)
- Statistical tests: are assertions accounting for randomness? (use seeds or tolerances)
- Integration tests: does the full loop work end-to-end?
- Benchmark correctness: do known POMIS match expected values?
- Parametrized tests for multiple graph structures
- Mock vs real computation tradeoffs"
            ;;
        style)
            additional_prompt="${additional_prompt}

Focus specifically on code style:
- Ruff compliance (100 char line length, select rules: E, F, I, N, W, UP, B, SIM, TCH)
- Type hints on all public functions (mypy strict mode)
- Dataclass usage for value types
- Import ordering (stdlib, third-party, local)
- Docstring quality for public APIs
- Naming conventions (snake_case functions, PascalCase classes)"
            ;;
        architecture)
            additional_prompt="${additional_prompt}

Focus specifically on architecture:
- Module boundaries and dependency direction (types <- engine <- modules)
- Are modules properly decoupled? Can they be tested independently?
- Is the engine loop's integration of sub-modules clean?
- Domain adapter extensibility
- Configuration vs hardcoded values
- Separation of algorithm logic from orchestration"
            ;;
    esac

    echo "${base_prompt}${additional_prompt}"
}

# Get comprehensive PR info
PR_INFO=$(gh pr view "$PR_NUM" --json title,author,baseRefName,headRefName,additions,deletions,changedFiles,commits)
PR_TITLE=$(echo "$PR_INFO" | jq -r .title)
PR_AUTHOR=$(echo "$PR_INFO" | jq -r .author.login)
PR_BRANCH=$(echo "$PR_INFO" | jq -r .headRefName)
PR_BASE_BRANCH=$(echo "$PR_INFO" | jq -r .baseRefName)
PR_ADDITIONS=$(echo "$PR_INFO" | jq -r .additions)
PR_DELETIONS=$(echo "$PR_INFO" | jq -r .deletions)
PR_CHANGED_FILES=$(echo "$PR_INFO" | jq -r .changedFiles)
PR_COMMITS=$(echo "$PR_INFO" | jq -r '.commits | length')

echo -e "${GREEN}Reviewing PR #$PR_NUM: $PR_TITLE${NC}"
echo -e "Author: $PR_AUTHOR"
echo -e "Branch: $PR_BRANCH -> $PR_BASE_BRANCH"
echo -e "Changes: ${GREEN}+$PR_ADDITIONS${NC} ${RED}-$PR_DELETIONS${NC} lines across $PR_CHANGED_FILES files"
echo -e "Commits: $PR_COMMITS"

# Show focus area if specified
if [ -n "$FOCUS_AREAS" ]; then
    echo -e "Focus: ${BLUE}$FOCUS_AREAS${NC}"
fi

echo ""

# Dry run mode - show what would be reviewed
if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}DRY RUN MODE - Preview of review context:${NC}"
    echo ""
    echo "Files to be reviewed:"
    gh pr diff "$PR_NUM" --name-only | sed 's/^/  - /'
    echo ""
    echo "Generated prompt:"
    echo "$(generate_review_prompt)" | sed 's/^/  /'
    echo ""
    echo -e "${YELLOW}Diff handling: Max lines set to $MAX_DIFF_LINES${NC}"
    echo -e "${YELLOW}Use without --dry-run to perform actual review${NC}"
    exit 0
fi

# Checkout PR if not already on it
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$PR_BRANCH" ]; then
    echo -e "${YELLOW}Checking out PR branch...${NC}"
    gh pr checkout "$PR_NUM"
fi

# Generate the review prompt
REVIEW_PROMPT=$(generate_review_prompt)

# Prepare context information with intelligent diff handling
echo -e "${BLUE}Preparing PR context (max diff lines: $MAX_DIFF_LINES)...${NC}"

PR_CONTEXT="
### PR Context
- **Title:** $PR_TITLE
- **Author:** $PR_AUTHOR
- **Branch:** $PR_BRANCH -> $PR_BASE_BRANCH
- **Additions:** $PR_ADDITIONS lines
- **Deletions:** $PR_DELETIONS lines
- **Files Changed:** $PR_CHANGED_FILES
- **Commits:** $PR_COMMITS

### Repository Context
causal-optimizer is a Python library that implements a causally-informed experiment optimization engine.
It decides what to try next using causal inference, Bayesian optimization, and design of experiments.

- **causal_optimizer/engine/**: Main optimization loop with phase transitions
- **causal_optimizer/designer/**: Factorial/LHS design and fANOVA screening
- **causal_optimizer/estimator/**: Bootstrap effect estimation
- **causal_optimizer/predictor/**: Off-policy surrogate prediction
- **causal_optimizer/optimizer/**: Suggestion strategies, POMIS computation
- **causal_optimizer/evolution/**: MAP-Elites diversity archive
- **causal_optimizer/discovery/**: Causal graph learning
- **causal_optimizer/validator/**: Sensitivity analysis
- **causal_optimizer/domain_adapters/**: Marketing and ML training templates
- **causal_optimizer/types.py**: Core types (CausalGraph, SearchSpace, ExperimentResult)
- **Standards**: Python 3.10+, ruff (100 char), type hints required, dataclasses + Pydantic

### Key Patterns
- CausalGraph with directed + bidirected edges for unobserved confounders
- POMIS for pruning intervention search space
- Engine loop phases: exploration (LHS) -> optimization (surrogate) -> exploitation (local search)
- Modules are designed to be independently testable

### Files in this PR:
\`\`\`
$(gh pr diff "$PR_NUM" --name-only)
\`\`\`

### Code Changes:
$(create_diff_summary "$PR_NUM" "$MAX_DIFF_LINES")
"

# Execute review based on output mode
case "$OUTPUT_MODE" in
    "comment"|"draft-comment")
        echo -e "${YELLOW}Running Claude review and posting to PR...${NC}"

        # Create temporary file for review
        TEMP_FILE=$(mktemp)

        # Add context and prompt to temp file
        cat > "$TEMP_FILE" << EOF
$PR_CONTEXT

---

$REVIEW_PROMPT
EOF

        # Run Claude and capture output

        if claude chat < "$TEMP_FILE" > "${TEMP_FILE}.output" 2>&1; then
            # Prepare comment body with header (exclude full context to save space)
            COMMENT_FILE=$(mktemp)
            cat > "$COMMENT_FILE" << EOF
# Claude Code Review

## Review Feedback

$(cat "${TEMP_FILE}.output")

---
*Review generated by Claude Local PR Review Tool for causal-optimizer*
EOF

            # Post comment
            if [ "$OUTPUT_MODE" = "draft-comment" ]; then
                gh pr comment "$PR_NUM" --body-file "$COMMENT_FILE" --draft
                echo -e "${GREEN}Review posted as draft PR comment${NC}"
            else
                gh pr comment "$PR_NUM" --body-file "$COMMENT_FILE"
                echo -e "${GREEN}Review posted as PR comment${NC}"
            fi

            # Show summary
            echo ""
            echo "Review Summary:"
            echo "---------------"
            head -n 20 "${TEMP_FILE}.output"
            echo "..."

            rm -f "$COMMENT_FILE"
        else
            echo -e "${RED}Review failed${NC}"
            echo "Error output:"
            cat "${TEMP_FILE}.output"
        fi

        rm -f "$TEMP_FILE" "${TEMP_FILE}.output"
        ;;

    "file")
        # Create output filename
        DATE=$(date +%Y%m%d_%H%M)
        OUTPUT_DIR="reviews/manual"
        FOCUS_SUFFIX=""
        if [ -n "$FOCUS_AREAS" ]; then
            FOCUS_SUFFIX="-${FOCUS_AREAS}"
        fi
        OUTPUT_FILE="$OUTPUT_DIR/pr-${PR_NUM}${FOCUS_SUFFIX}-${DATE}.md"

        # Ensure output directory exists
        mkdir -p "$OUTPUT_DIR"

        echo -e "${YELLOW}Running Claude review and saving to file...${NC}"

        # Create header for the review file (include full context for local files)
        cat > "$OUTPUT_FILE" << EOF
# Claude Code Review: PR #$PR_NUM

**Title:** $PR_TITLE
**Author:** $PR_AUTHOR
**Date:** $(date +"%Y-%m-%d %H:%M:%S")
**Branch:** $PR_BRANCH -> $PR_BASE_BRANCH
**Focus:** ${FOCUS_AREAS:-"General review"}

$PR_CONTEXT

---

## Review Prompt Used

$REVIEW_PROMPT

---

## Claude Review Output

EOF

        # Create temp file with context and prompt
        TEMP_FILE=$(mktemp)
        cat > "$TEMP_FILE" << EOF
$PR_CONTEXT

---

$REVIEW_PROMPT
EOF

        # Run Claude and append to file
        if claude chat < "$TEMP_FILE" >> "$OUTPUT_FILE" 2>&1; then
            echo -e "${GREEN}Review completed successfully${NC}"
            echo -e "${GREEN}Saved to: $OUTPUT_FILE${NC}"

            # Show summary
            echo ""
            echo "Review Summary:"
            echo "---------------"
            # Extract first few lines of review output
            tail -n +25 "$OUTPUT_FILE" | head -n 20
            echo "..."
            echo ""
            echo -e "${YELLOW}Full review saved to: $OUTPUT_FILE${NC}"

            # Add token usage estimate
            if command -v wc &> /dev/null; then
                WORD_COUNT=$(wc -w < "$OUTPUT_FILE")
                TOKEN_ESTIMATE=$((WORD_COUNT * 4 / 3))
                echo -e "${BLUE}Estimated tokens used: ~$TOKEN_ESTIMATE${NC}"
            fi
        else
            echo -e "${RED}Review failed${NC}"
            echo "Check $OUTPUT_FILE for error details"
        fi

        rm -f "$TEMP_FILE"
        ;;
esac

# Return to original branch if we switched
if [ "$CURRENT_BRANCH" != "$PR_BRANCH" ] && [ -n "$ORIGINAL_BRANCH" ]; then
    echo ""
    echo -e "${YELLOW}Returning to branch: $ORIGINAL_BRANCH${NC}"
    git checkout "$ORIGINAL_BRANCH"
fi

# Add helpful tips based on output mode
echo ""
case "$OUTPUT_MODE" in
    "comment"|"draft-comment")
        echo -e "${GREEN}Next steps:${NC}"
        echo "  - Review the posted comment on GitHub"
        echo "  - Address any issues raised in the review"
        echo "  - Run: uv run ruff check causal_optimizer/ && uv run pytest tests/"
        echo "  - Create follow-up issues if needed: gh issue create --title \"...\" --body \"...\""
        ;;
    "file")
        echo -e "${GREEN}Next steps:${NC}"
        echo "  - Review the saved file: $OUTPUT_FILE"
        echo "  - Extract concerns/issues for follow-up"
        echo "  - Run: uv run ruff check causal_optimizer/ && uv run pytest tests/"
        echo "  - Create GitHub issues: gh issue create --title \"...\" --body \"...\""
        ;;
esac

echo ""
echo -e "${BLUE}Claude Review Script for causal-optimizer v1.0${NC}"
echo -e "For help: $0 --help"
