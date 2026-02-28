#!/usr/bin/env bash
# ================================================================
# CORAL ASTROBIO v3.0 — Setup & Run
# Ejecutar en la carpeta raíz del proyecto: bash setup.sh
# ================================================================
set -e

echo ""
echo "  ◈ CORAL ASTROBIO v3.0 — Setup"
echo "  ─────────────────────────────"
echo ""

# ── Python check ──────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "  ERROR: Python 3 no encontrado. Instala Python 3.9+"
  exit 1
fi
PY=$(python3 --version)
echo "  ✓ $PY"

# ── Dependencias ──────────────────────────────────────────────
echo "  Instalando dependencias..."
pip install \
  flask \
  supabase \
  numpy \
  pandas \
  scipy \
  matplotlib \
  seaborn \
  reportlab \
  requests \
  python-dotenv \
  --quiet

echo "  ✓ Dependencias instaladas"

# ── Cargar .env ───────────────────────────────────────────────
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "  ✓ .env cargado"
else
  echo "  WARN: .env no encontrado — copia .env.example como .env"
fi

# ── Verificar conexión Supabase ───────────────────────────────
echo ""
echo "  Verificando conexión Supabase..."
python3 -c "
import os
os.environ.setdefault('SUPABASE_URL', '${SUPABASE_URL:-}')
os.environ.setdefault('SUPABASE_ANON_KEY', '${SUPABASE_ANON_KEY:-}')
import sys
sys.path.insert(0,'.')
from memory.supabase_client import CoralMemory
mem = CoralMemory()
if mem.online:
    stats = mem.get_stats()
    print(f'  ✓ Supabase online')
    print(f'    coral: {stats.get(\"coral_entries\",\"?\")} entradas')
    print(f'    astrobio: {stats.get(\"astrobio_entries\",\"?\")} entradas')
    print(f'    total: {stats.get(\"total\",\"?\")} entradas')
else:
    print('  WARN: Supabase offline — la herramienta funciona pero sin memoria persistente')
" 2>/dev/null || echo "  WARN: Error verificando Supabase"

# ── Arrancar API ──────────────────────────────────────────────
PORT=${API_PORT:-5050}
echo ""
echo "  ◈ Arrancando API en http://localhost:${PORT}"
echo "  Abre astrobio_v3.html en tu browser."
echo "  Ctrl+C para detener."
echo ""
python3 api.py
