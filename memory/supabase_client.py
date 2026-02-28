"""
CORAL ASTROBIO — Módulo Supabase
Adaptado al esquema real de la tabla memory_entries.

Esquema (tabla: memory_entries):
  id              uuid PK
  ia_author       enum  → 'claude', 'gpt4', etc.
  entry_type      enum  → 'fact', 'analysis', 'insight', ...
  field_key       text  → clave semántica jerárquica
  field_value     text  → contenido
  confidence_score double
  schema_hash     text
  is_superseded   boolean  → versionado: True = reemplazada
  parent_entry_id uuid nullable
  embedding       vector (pgvector)
  created_at      timestamptz
  tema            text nullable  ← campo nuevo para separar proyectos
  subtema         text nullable  ← granularidad dentro del tema
  path            text nullable  ← ruta jerárquica legible

Separación de proyectos:
  tema='coral'     → Memoria general del Proyecto Coral
  tema='astrobio'  → Herramienta CORAL ASTROBIO
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

# ── Constantes ──────────────────────────────────────────────────
TABLE          = "memory_entries"
TEMA_CORAL     = "coral"
TEMA_ASTROBIO  = "astrobio"
IA_AUTHOR      = "claude"
SCHEMA_VERSION = "astrobio_v2.0"

# Valores permitidos por los enums de Supabase
# Ajusta si tus enums tienen nombres distintos
ENTRY_TYPES = {
    "analysis": "analysis",
    "fact":     "fact",
    "insight":  "insight",
    "tool":     "tool",
}

try:
    from supabase import create_client, Client
    SUPABASE_OK = True
except ImportError:
    SUPABASE_OK = False

try:
    import openai
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False


class CoralMemory:
    """
    Memoria persistente del Proyecto Coral.
    Separa entradas por tema='coral' vs tema='astrobio'.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        openai_key: Optional[str] = None,
    ):
        self.url = supabase_url or os.getenv("SUPABASE_URL", "")
        self.key = (supabase_key
                    or os.getenv("SUPABASE_ANON_KEY", "")
                    or os.getenv("SUPABASE_SERVICE_KEY", ""))
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY", "")
        self.client: Optional[Any] = None
        self.online = False
        self._schema_hash = hashlib.md5(SCHEMA_VERSION.encode()).hexdigest()[:8]
        self._connect()

    # ── Conexión ────────────────────────────────────────────────
    def _connect(self):
        if not SUPABASE_OK:
            print("[CoralMemory] supabase-py no instalado → pip install supabase")
            return
        if not self.url or not self.key:
            print("[CoralMemory] Sin credenciales (SUPABASE_URL / SUPABASE_ANON_KEY)")
            return
        try:
            self.client = create_client(self.url, self.key)
            self.client.table(TABLE).select("id").limit(1).execute()
            self.online = True
            print(f"[CoralMemory] ✓ {self.url[:48]}...")
        except Exception as e:
            print(f"[CoralMemory] Error: {e}")

    # ── Embedding ───────────────────────────────────────────────
    def _embed(self, text: str) -> Optional[List[float]]:
        if not OPENAI_OK or not self.openai_key:
            return None
        try:
            c = openai.OpenAI(api_key=self.openai_key)
            r = c.embeddings.create(model="text-embedding-3-small", input=text[:8000])
            return r.data[0].embedding
        except Exception as e:
            print(f"[CoralMemory] embed error: {e}")
            return None

    def _key(self, *parts: str) -> str:
        """Genera field_key. Ej: 'astrobio.lhs1140b.bayesiano'"""
        return ".".join(p.lower().replace(" ", "_") for p in parts)

    def _path(self, *parts: str) -> str:
        """Genera path. Ej: 'astrobio/lhs1140b/analisis'"""
        return "/".join(p.lower().replace(" ", "_") for p in parts)

    # ── GUARDAR ANÁLISIS DE PLANETA ─────────────────────────────
    def save_planet_analysis(
        self,
        planet_name: str,
        analysis: Dict,
        supersede: bool = True,
    ) -> Dict:
        """
        Guarda el análisis completo de un planeta.
        tema='astrobio', subtema='exoplaneta'.

        Crea 4 entradas encadenadas por parent_entry_id:
          raíz      → resumen + embedding semántico
          bayesiano → odds, consilience, verdict
          ptsc      → transición dominante, score continuo
          causal    → dirección, lag, non_geological
        """
        if not self.online:
            return {"status": "offline", "message": "Supabase no disponible"}

        b = analysis.get("bayesian", {})
        t = analysis.get("transition", {})
        c = analysis.get("causal", {})
        p = analysis.get("planet", {})
        safe = planet_name.lower().replace(" ", "_")

        # Marcar entradas anteriores como superseded
        if supersede:
            try:
                self.client.table(TABLE).update({"is_superseded": True}) \
                    .eq("tema", TEMA_ASTROBIO) \
                    .eq("is_superseded", False) \
                    .like("field_key", f"astrobio.{safe}%") \
                    .execute()
            except Exception as e:
                print(f"[CoralMemory] supersede sin efecto (falta política UPDATE — ver supabase_setup.sql): {e}")

        # Texto de resumen para embedding semántico
        summary = (
            f"Análisis astrobiológico de {planet_name}. "
            f"Tipo: {p.get('type','—')}, T_eq {p.get('teq_k','—')} K, "
            f"TSM {p.get('tsm','—')}. "
            f"Veredicto bayesiano: {b.get('verdict','—')}, "
            f"odds={b.get('odds_ratio',0):.3f}, "
            f"consilience={b.get('consilience_score',0):.3f}. "
            f"Transición dominante: {t.get('dominant_name','—')} "
            f"(score={t.get('dominant_score',0):.2f}). "
            f"Causalidad: {c.get('causal_direction','—')}, "
            f"no-geológico={c.get('non_geological',0):.3f}."
        )

        # ─ Entrada raíz ─
        root = self._record(
            field_key=self._key("astrobio", safe, "analisis"),
            field_value=json.dumps({
                "planeta": {k: p.get(k) for k in
                            ["name","type","teq_k","dist_pc","tsm","jwst","hz","spectype","period_days"]},
                "bayesiano": {k: b.get(k) for k in
                              ["odds_ratio","bio_probability","abio_probability","consilience_score","verdict","paradigm_shift","false_negative_risk"]},
                "transicion": {k: t.get(k) for k in
                               ["dominant_idx","dominant_label","dominant_name","continuous_state","false_negative_reduction","detectability_window_ma"]},
                "causal": {k: c.get(k) for k in
                           ["causal_direction","non_geological","phase_synchrony","causal_lag_days","info_entropy","effective_dim"]},
                "resumen": summary,
            }, ensure_ascii=False),
            entry_type=ENTRY_TYPES["analysis"],
            confidence=float(b.get("consilience_score", 0.5)),
            tema=TEMA_ASTROBIO,
            subtema="exoplaneta",
            path=self._path("astrobio", safe, "analisis"),
            embedding=self._embed(summary),
        )
        root_r = self._insert(root)
        root_id = root_r.get("id")

        # ─ Subentradas ─
        subs = [
            dict(
                field_key=self._key("astrobio", safe, "bayesiano"),
                field_value=(
                    f"odds={b.get('odds_ratio',0):.4f} "
                    f"p_bio={b.get('bio_probability',0):.4f} "
                    f"consilience={b.get('consilience_score',0):.4f} "
                    f"verdict={b.get('verdict','—')} "
                    f"paradigm_shift={b.get('paradigm_shift',False)}"
                ),
                entry_type=ENTRY_TYPES["fact"],
                confidence=float(b.get("consilience_score", 0.5)),
                subtema="bayesiano",
                path=self._path("astrobio", safe, "bayesiano"),
            ),
            dict(
                field_key=self._key("astrobio", safe, "ptsc"),
                field_value=(
                    f"dominante={t.get('dominant_label','—')} "
                    f"nombre={t.get('dominant_name','—')} "
                    f"score={t.get('dominant_score',0):.3f} "
                    f"continuo={t.get('continuous_state',0):.2f} "
                    f"fn_reducidos={t.get('false_negative_reduction',0):.1f}%"
                ),
                entry_type=ENTRY_TYPES["fact"],
                confidence=float(t.get("dominant_score", 0.5)),
                subtema="transicion",
                path=self._path("astrobio", safe, "ptsc"),
            ),
            dict(
                field_key=self._key("astrobio", safe, "causal"),
                field_value=(
                    f"direccion={c.get('causal_direction','—')} "
                    f"lag={c.get('causal_lag_days',0):.1f}d "
                    f"no_geologico={c.get('non_geological',0):.3f} "
                    f"sincronía={c.get('phase_synchrony',0):.3f} "
                    f"entropia={c.get('info_entropy',0):.3f}"
                ),
                entry_type=ENTRY_TYPES["fact"],
                confidence=float(c.get("non_geological", 0.3)),
                subtema="causal",
                path=self._path("astrobio", safe, "causal"),
            ),
        ]

        sub_ids = []
        for s in subs:
            rec = self._record(**s, tema=TEMA_ASTROBIO, parent_id=root_id)
            sub_ids.append(self._insert(rec).get("id"))

        print(f"[CoralMemory] ✓ Guardado [{TEMA_ASTROBIO}]: {planet_name}")
        return {
            "status": "ok",
            "root_id": root_id,
            "sub_ids": [i for i in sub_ids if i],
            "tema": TEMA_ASTROBIO,
            "subtema": "exoplaneta",
            "planet": planet_name,
            "field_key": self._key("astrobio", safe, "analisis"),
        }

    # ── GUARDAR NOTA CORAL ──────────────────────────────────────
    def save_coral_note(
        self,
        title: str,
        content: str,
        subtema: str = "general",
        confidence: float = 0.8,
    ) -> Dict:
        """
        Guarda una entrada en tema='coral' (memoria general).
        Compatible con el formato existente de la memoria Coral.
        """
        if not self.online:
            return {"status": "offline"}

        safe = title.lower().replace(" ", "_")[:40]
        embedding = self._embed(content)
        rec = self._record(
            field_key=self._key("coral", subtema, safe),
            field_value=content,
            entry_type=ENTRY_TYPES["insight"],
            confidence=confidence,
            tema=TEMA_CORAL,
            subtema=subtema,
            path=self._path("coral", subtema, safe),
            embedding=embedding,
        )
        r = self._insert(rec)
        status = "ok" if r.get("id") else "error"
        if status == "ok":
            print(f"[CoralMemory] ✓ Guardado [{TEMA_CORAL}]: {title}")
        return {"status": status, "id": r.get("id"), "tema": TEMA_CORAL}

    # ── LEER ────────────────────────────────────────────────────
    def list_by_tema(
        self,
        tema: str = TEMA_ASTROBIO,
        subtema: Optional[str] = None,
        limit: int = 50,
        include_superseded: bool = False,
    ) -> List[Dict]:
        if not self.online:
            return []
        try:
            q = (self.client.table(TABLE)
                 .select("id,field_key,field_value,entry_type,ia_author,confidence_score,tema,subtema,path,created_at,is_superseded")
                 .eq("tema", tema)
                 .eq("is_superseded", include_superseded if include_superseded else False))
            if subtema:
                q = q.eq("subtema", subtema)
            return (q.order("created_at", desc=True).limit(limit).execute()).data or []
        except Exception as e:
            print(f"[CoralMemory] list error: {e}")
            return []

    def get_planet_history(self, planet_name: str) -> List[Dict]:
        """Historial completo (incluye versiones superseded) de un planeta."""
        if not self.online:
            return []
        safe = planet_name.lower().replace(" ", "_")
        try:
            return (self.client.table(TABLE)
                    .select("*")
                    .eq("tema", TEMA_ASTROBIO)
                    .like("field_key", f"astrobio.{safe}%")
                    .order("created_at", desc=True)
                    .execute()).data or []
        except Exception as e:
            print(f"[CoralMemory] history error: {e}")
            return []

    def search_similar(
        self,
        query: str,
        tema: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.65,
    ) -> List[Dict]:
        """
        Búsqueda semántica pgvector. Requiere RPC 'match_memory_entries'.
        Fallback automático a búsqueda de texto si RPC no existe.
        """
        if not self.online:
            return []
        embedding = self._embed(query)
        if embedding:
            try:
                params = {
                    "query_embedding": embedding,
                    "match_threshold": threshold,
                    "match_count": limit,
                }
                if tema:
                    params["filter_tema"] = tema
                r = self.client.rpc("match_memory_entries", params).execute()
                if r.data:
                    return r.data
            except Exception:
                pass  # Fallback silencioso
        # Fallback texto
        try:
            q = (self.client.table(TABLE)
                 .select("id,field_key,field_value,entry_type,tema,subtema,confidence_score,created_at")
                 .ilike("field_value", f"%{query}%")
                 .eq("is_superseded", False))
            if tema:
                q = q.eq("tema", tema)
            return (q.order("created_at", desc=True).limit(limit).execute()).data or []
        except Exception as e:
            print(f"[CoralMemory] text search error: {e}")
            return []

    # ── STATS ───────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        if not self.online:
            return {"online": False}
        try:
            def count(t, s=None):
                q = (self.client.table(TABLE)
                     .select("id", count="exact")
                     .eq("tema", t)
                     .eq("is_superseded", False))
                if s:
                    q = q.eq("subtema", s)
                return q.execute().count or 0

            coral_n    = count(TEMA_CORAL)
            astrobio_n = count(TEMA_ASTROBIO)

            # Desglose subtemas en astrobio
            subtemas = {}
            for st in ["exoplaneta", "bayesiano", "transicion", "causal"]:
                n = count(TEMA_ASTROBIO, st)
                if n:
                    subtemas[st] = n

            return {
                "online":           True,
                "table":            TABLE,
                "namespace_field":  "tema",       # para compatibilidad con la API
                "coral_value":      TEMA_CORAL,
                "astrobio_value":   TEMA_ASTROBIO,
                "coral_entries":    coral_n,
                "astrobio_entries": astrobio_n,
                "total":            coral_n + astrobio_n,
                "astrobio_subtemas": subtemas,
            }
        except Exception as e:
            return {"online": False, "error": str(e)}

    # ── INTERNOS ────────────────────────────────────────────────
    def _record(
        self,
        field_key: str,
        field_value: str,
        entry_type: str,
        confidence: float,
        tema: str,
        subtema: str,
        path: str,
        embedding: Optional[List[float]] = None,
        parent_id: Optional[str] = None,
    ) -> Dict:
        rec = {
            "ia_author":       IA_AUTHOR,
            "entry_type":      entry_type,
            "field_key":       field_key[:500],
            "field_value":     field_value[:10000],
            "confidence_score": round(min(max(float(confidence), 0.0), 1.0), 4),
            "schema_hash":     self._schema_hash,
            "is_superseded":   False,
            "tema":            tema,
            "subtema":         subtema,
            "path":            path[:500],
            "created_at":      datetime.now(timezone.utc).isoformat(),
        }
        if embedding:
            rec["embedding"] = embedding
        if parent_id:
            rec["parent_entry_id"] = parent_id
        return rec

    def _insert(self, record: Dict) -> Dict:
        try:
            r = self.client.table(TABLE).insert(record).execute()
            return r.data[0] if r.data else {}
        except Exception as e:
            print(f"[CoralMemory] insert error: {e}")
            return {"error": str(e)}
