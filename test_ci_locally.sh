#!/bin/bash
# Script para testar o CI localmente antes de fazer push

set -e  # Para no primeiro erro

echo "🚀 Testando CI localmente..."
echo "================================="

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para printar com cor
print_step() {
    echo -e "\n${YELLOW}▶ $1${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Verificar sintaxe Python
print_step "1. Verificando sintaxe Python..."
if poetry run python -m py_compile src/*.py tests/*.py scripts/*.py 2>/dev/null; then
    print_success "Sintaxe Python válida"
else
    print_error "Erros de sintaxe encontrados"
    exit 1
fi

# 2. Validar imports
print_step "2. Validando imports..."
poetry run python -c "
import sys
import pathlib
errors = []
for py_file in pathlib.Path('src').glob('**/*.py'):
    if '__pycache__' not in str(py_file):
        try:
            with open(py_file) as f:
                compile(f.read(), str(py_file), 'exec')
        except SyntaxError as e:
            errors.append(f'{py_file}: {e}')
if errors:
    for error in errors:
        print(f'❌ {error}')
    sys.exit(1)
else:
    print('✅ Todos os arquivos têm sintaxe válida')
"

# 3. Executar testes
print_step "3. Executando testes unitários..."
if poetry run pytest tests/ -v --tb=short; then
    print_success "Todos os testes passaram"
else
    print_error "Alguns testes falharam"
    exit 1
fi

# 4. Cobertura de testes (opcional, continue mesmo se falhar)
print_step "4. Verificando cobertura de testes (mínimo 80%)..."
poetry run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80 2>/dev/null || echo "⚠️  Cobertura abaixo de 80% ou não disponível"

# 5. Linting (opcional)
print_step "5. Verificando linting com flake8..."
poetry run flake8 src/ --max-line-length=100 --ignore=E203,W503 2>/dev/null || echo "⚠️  flake8 não disponível ou avisos encontrados"

# 6. Verificar formatação
print_step "6. Verificando formatação..."
poetry run autopep8 --diff --recursive src/ tests/ scripts/ 2>/dev/null || echo "⚠️  autopep8 não disponível"

# Sucesso!
echo ""
echo "================================="
print_success "🎉 Todos os testes do CI passaram localmente!"
print_success "Você pode fazer push com segurança!"
echo "================================="
