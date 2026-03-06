#!/usr/bin/env bash
# =============================================================================
# Testa CI/CD localmente — reproduz os mesmos jobs do GitHub Actions
# =============================================================================
# Uso:
#   ./scripts/test_ci_locally.sh           # Todos os jobs
#   ./scripts/test_ci_locally.sh lint       # Apenas lint
#   ./scripts/test_ci_locally.sh imports    # Apenas imports
#   ./scripts/test_ci_locally.sh tests      # Apenas testes
#   ./scripts/test_ci_locally.sh ml         # Apenas validação ML
#   ./scripts/test_ci_locally.sh quality    # Apenas qualidade de código
#   ./scripts/test_ci_locally.sh docker     # Apenas build Docker
#   ./scripts/test_ci_locally.sh all        # Todos (incluindo Docker)
# =============================================================================

set -uo pipefail

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Contadores
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

# Timer
START_TIME=$(date +%s)

# =============================================================================
# Funções auxiliares
# =============================================================================

header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

step() {
    echo -e "\n${YELLOW}▸ $1${NC}"
}

pass() {
    echo -e "  ${GREEN}✓ $1${NC}"
    ((PASSED++))
}

fail() {
    echo -e "  ${RED}✗ $1${NC}"
    ((FAILED++))
    FAILURES+=("$1")
}

skip() {
    echo -e "  ${YELLOW}⊘ $1 (pulado)${NC}"
    ((SKIPPED++))
}

run_step() {
    local description="$1"
    shift
    step "$description"
    if "$@" 2>&1 | tail -5; then
        pass "$description"
    else
        fail "$description"
    fi
}

# Garante que estamos na raiz do projeto
cd "$(dirname "$0")/.."

JOB="${1:-ci}"

# =============================================================================
# Job 1: Lint (ci.yml)
# =============================================================================
job_lint() {
    header "🔍 Job 1: Lint"

    step "Ruff (linter)"
    if poetry run ruff check src/ api/ tests/ scripts/ 2>&1; then
        pass "Ruff"
    else
        fail "Ruff"
    fi

    step "Black (formatação)"
    if poetry run black --check src/ api/ tests/ scripts/ 2>&1; then
        pass "Black"
    else
        fail "Black"
    fi

    step "isort (ordenação de imports)"
    if poetry run isort --check-only src/ api/ tests/ scripts/ 2>&1; then
        pass "isort"
    else
        fail "isort"
    fi
}

# =============================================================================
# Job 2: Imports & Ambiente (ci.yml)
# =============================================================================
job_imports() {
    header "📦 Job 2: Imports & Ambiente"

    step "Verificar sintaxe Python"
    if poetry run python -c "
import ast, pathlib, sys

errors = []
for pattern in ['src/**/*.py', 'api/**/*.py', 'tests/**/*.py', 'scripts/**/*.py']:
    for py_file in pathlib.Path('.').glob(pattern):
        if '__pycache__' in str(py_file):
            continue
        try:
            with open(py_file) as f:
                ast.parse(f.read(), filename=str(py_file))
        except SyntaxError as e:
            errors.append(f'{py_file}:{e.lineno} - {e.msg}')

if errors:
    print('Erros de sintaxe encontrados:')
    for err in errors:
        print(f'  {err}')
    sys.exit(1)
print('Sintaxe OK em todos os arquivos')
" 2>&1; then
        pass "Sintaxe Python"
    else
        fail "Sintaxe Python"
    fi

    step "Importar módulos src"
    if poetry run python -c "
import sys
sys.path.insert(0, '.')

modules = [
    'src.config', 'src.enums', 'src.logger', 'src.utils',
    'src.preprocessing', 'src.feature_engineering', 'src.pipeline',
    'src.evaluate', 'src.inference', 'src.interpret',
    'src.monitoring', 'src.optimization', 'src.notebook_utils',
]

failed = []
for mod in modules:
    try:
        __import__(mod)
    except Exception as e:
        failed.append((mod, str(e)))

if failed:
    for mod, err in failed:
        print(f'  FALHA: {mod}: {err}')
    sys.exit(1)
print(f'Todos os {len(modules)} módulos src importados com sucesso')
" 2>&1; then
        pass "Imports src"
    else
        fail "Imports src"
    fi

    step "Importar módulos da API"
    if poetry run python -c "
import sys
sys.path.insert(0, '.')

modules = [
    'api.core.config', 'api.core.logging',
    'api.schemas.prediction', 'api.schemas.drift',
    'api.services.model_service',
    'api.routes.health', 'api.routes.prediction',
    'api.routes.interpret', 'api.routes.drift',
    'api.main',
]

failed = []
for mod in modules:
    try:
        __import__(mod)
    except Exception as e:
        failed.append((mod, str(e)))

if failed:
    for mod, err in failed:
        print(f'  FALHA: {mod}: {err}')
    sys.exit(1)
print(f'Todos os {len(modules)} módulos da API importados com sucesso')
" 2>&1; then
        pass "Imports API"
    else
        fail "Imports API"
    fi

    step "Consistência pyproject.toml / poetry.lock"
    if poetry check --lock 2>&1; then
        pass "Poetry check"
    else
        fail "Poetry check"
    fi
}

# =============================================================================
# Job 3: Testes (ci.yml)
# =============================================================================
job_tests() {
    header "🧪 Job 3: Testes"

    step "Executar testes com cobertura (threshold 70%)"
    if poetry run pytest tests/ \
        -v \
        --tb=short \
        --cov=src \
        --cov=api \
        --cov-report=term-missing \
        --cov-fail-under=70 2>&1; then
        pass "Testes + Cobertura ≥70%"
    else
        fail "Testes + Cobertura"
    fi
}

# =============================================================================
# Job 4: Validação ML (ci.yml)
# =============================================================================
job_ml() {
    header "🤖 Job 4: Validação ML"

    step "Validar classes e enums do pipeline"
    if poetry run python -c "
from src.pipeline import ChurnPipeline
from src.optimization import HyperparameterOptimizer, OptimizationConfig
from src.enums import ModelType, DriftSeverity
from src.evaluate import evaluate
from src.feature_engineering import FeatureEngineer, AdvancedFeatureEngineer
print('Pipeline ML: todas as classes importadas')
" 2>&1; then
        pass "Classes ML"
    else
        fail "Classes ML"
    fi

    step "Validar detecção de drift (PSI)"
    if poetry run python -c "
import numpy as np
from src.monitoring import population_stability_index

np.random.seed(42)
baseline = np.random.normal(0, 1, 1000)
similar  = np.random.normal(0, 1, 1000)
drifted  = np.random.normal(1, 1, 1000)

psi_ok    = population_stability_index(baseline, similar)
psi_drift = population_stability_index(baseline, drifted)

assert psi_ok < 0.15, f'PSI sem drift deveria ser < 0.15, obteve {psi_ok:.4f}'
assert psi_drift > 0.1, f'PSI com drift deveria ser > 0.1, obteve {psi_drift:.4f}'
print(f'PSI sem drift: {psi_ok:.4f} | com drift: {psi_drift:.4f} — OK')
" 2>&1; then
        pass "Detecção de drift"
    else
        fail "Detecção de drift"
    fi

    step "Validar normalização de métricas"
    if poetry run python -c "
from src.utils import normalize_metrics_keys

raw = {'f1_score': 0.57, 'roc_auc': 0.83, 'auprc': 0.79,
       'accuracy': 0.80, 'precision': 0.65, 'recall': 0.52}
normalized = normalize_metrics_keys(raw)
for key in ['f1_score', 'roc_auc', 'auprc', 'accuracy', 'precision', 'recall']:
    assert key in normalized, f'Métrica {key} não encontrada'
print('Normalização de métricas: OK')
" 2>&1; then
        pass "Métricas"
    else
        fail "Métricas"
    fi
}

# =============================================================================
# Job 5: Qualidade de código (code-quality.yml)
# =============================================================================
job_quality() {
    header "📊 Job 5: Qualidade de Código"

    step "Verificar funções grandes (>50 linhas)"
    poetry run python -c "
import ast, pathlib

class FunctionAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.large = []
        self.file = None
    def visit_FunctionDef(self, node):
        lines = node.end_lineno - node.lineno
        if lines > 50:
            self.large.append((self.file, node.name, lines))
        self.generic_visit(node)

analyzer = FunctionAnalyzer()
for py_file in pathlib.Path('src').glob('**/*.py'):
    if '__pycache__' not in str(py_file):
        analyzer.file = str(py_file)
        with open(py_file) as f:
            try:
                tree = ast.parse(f.read())
                analyzer.visit(tree)
            except SyntaxError:
                pass

if analyzer.large:
    print(f'{len(analyzer.large)} função(ões) grande(s):')
    for file, name, lines in sorted(analyzer.large, key=lambda x: -x[2]):
        print(f'  {file}: {name}() — {lines} linhas')
else:
    print('Todas as funções têm tamanho adequado')
" 2>&1
    pass "Funções grandes (informativo)"

    step "Verificar complexidade ciclomática"
    poetry run python -c "
import ast, pathlib

class ComplexityChecker(ast.NodeVisitor):
    def __init__(self):
        self.results = []
        self.file = None
    def visit_FunctionDef(self, node):
        c = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                  ast.BoolOp, ast.Assert)):
                c += 1
        if c > 10:
            self.results.append((self.file, node.name, c))
        self.generic_visit(node)

checker = ComplexityChecker()
for py_file in pathlib.Path('src').glob('**/*.py'):
    if '__pycache__' not in str(py_file):
        checker.file = str(py_file)
        with open(py_file) as f:
            try:
                checker.visit(ast.parse(f.read()))
            except SyntaxError:
                pass

if checker.results:
    print(f'{len(checker.results)} função(ões) com alta complexidade (>10):')
    for file, name, c in sorted(checker.results, key=lambda x: -x[2]):
        print(f'  {file}: {name}() — complexidade {c}')
else:
    print('Complexidade ciclomática OK')
" 2>&1
    pass "Complexidade ciclomática (informativo)"
}

# =============================================================================
# Job 6: Build Docker (ci.yml)
# =============================================================================
job_docker() {
    header "🐳 Job 6: Build Docker"

    if ! command -v docker &>/dev/null; then
        skip "Docker não instalado"
        return
    fi

    step "Build imagem production"
    if docker build --target production -t churn-api:local . 2>&1 | tail -3; then
        pass "Docker build (production)"
    else
        fail "Docker build (production)"
    fi

    step "Build imagem dashboard"
    if docker build --target dashboard -t churn-dashboard:local . 2>&1 | tail -3; then
        pass "Docker build (dashboard)"
    else
        fail "Docker build (dashboard)"
    fi

    step "Verificar imagens"
    docker images churn-api:local --format "  api:        {{.Repository}}:{{.Tag}} — {{.Size}}"
    docker images churn-dashboard:local --format "  dashboard:  {{.Repository}}:{{.Tag}} — {{.Size}}"
    pass "Imagens Docker verificadas"
}

# =============================================================================
# Relatório final
# =============================================================================
report() {
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))

    header "📋 Relatório Final"
    echo ""
    echo -e "  ${GREEN}✓ Passou:  ${PASSED}${NC}"
    echo -e "  ${RED}✗ Falhou:  ${FAILED}${NC}"
    echo -e "  ${YELLOW}⊘ Pulado:  ${SKIPPED}${NC}"
    echo -e "  ⏱ Tempo:   ${MINS}m${SECS}s"

    if [ ${#FAILURES[@]} -gt 0 ]; then
        echo ""
        echo -e "${RED}  Falhas:${NC}"
        for f in "${FAILURES[@]}"; do
            echo -e "    ${RED}✗ $f${NC}"
        done
    fi

    echo ""
    if [ "$FAILED" -eq 0 ]; then
        echo -e "${GREEN}${BOLD}  ✅ CI local passou — pronto para push!${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}${BOLD}  ❌ CI local falhou — corrija antes de fazer push.${NC}"
        echo ""
        return 1
    fi
}

# =============================================================================
# Execução
# =============================================================================

echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║          CI/CD Local — Churn Prediction                 ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

case "$JOB" in
    lint)
        job_lint
        ;;
    imports)
        job_imports
        ;;
    tests|test)
        job_tests
        ;;
    ml)
        job_ml
        ;;
    quality)
        job_quality
        ;;
    docker)
        job_docker
        ;;
    all)
        job_lint
        job_imports
        job_tests
        job_ml
        job_quality
        job_docker
        ;;
    ci|"")
        job_lint
        job_imports
        job_tests
        job_ml
        job_quality
        ;;
    *)
        echo "Uso: $0 [lint|imports|tests|ml|quality|docker|all|ci]"
        exit 1
        ;;
esac

report
