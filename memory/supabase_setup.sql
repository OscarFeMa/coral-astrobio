-- ================================================================
-- CORAL ASTROBIO v3.0 — Supabase SQL Setup
-- Ejecutar en el SQL Editor de tu proyecto Supabase
-- ================================================================

-- ────────────────────────────────────────────────────────────────
-- 1. ÍNDICES para filtrado rápido por tema/subtema
-- ────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_memory_tema
  ON memory_entries (tema) WHERE tema IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memory_subtema
  ON memory_entries (subtema) WHERE subtema IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memory_tema_superseded
  ON memory_entries (tema, is_superseded);

CREATE INDEX IF NOT EXISTS idx_memory_field_key
  ON memory_entries (field_key);

-- ────────────────────────────────────────────────────────────────
-- 2. POLÍTICA UPDATE — necesaria para el versionado is_superseded
--
-- Estado actual verificado (Feb 2026):
--   ✅ lectura_publica         → SELECT public  → OK
--   ✅ escritura_solo_service  → INSERT public  → OK
--   ❌ Sin UPDATE              → supersede falla silenciosamente
-- ────────────────────────────────────────────────────────────────
CREATE POLICY "actualizacion_superseded"
  ON memory_entries
  FOR UPDATE
  TO public
  USING (true)
  WITH CHECK (true);

-- ────────────────────────────────────────────────────────────────
-- 3. FUNCIÓN RPC: búsqueda semántica por embedding (pgvector)
--    Filtra por tema y excluye entradas superseded
-- ────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_memory_entries(
  query_embedding vector(1536),
  match_threshold float    DEFAULT 0.65,
  match_count     int      DEFAULT 10,
  filter_tema     text     DEFAULT NULL
)
RETURNS TABLE (
  id               uuid,
  field_key        text,
  field_value      text,
  entry_type       text,
  ia_author        text,
  confidence_score double precision,
  tema             text,
  subtema          text,
  path             text,
  created_at       timestamptz,
  similarity       float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    m.id,
    m.field_key,
    m.field_value,
    m.entry_type::text,
    m.ia_author::text,
    m.confidence_score,
    m.tema,
    m.subtema,
    m.path,
    m.created_at,
    1 - (m.embedding <=> query_embedding) AS similarity
  FROM memory_entries m
  WHERE
    m.is_superseded = FALSE
    AND m.embedding IS NOT NULL
    AND 1 - (m.embedding <=> query_embedding) >= match_threshold
    AND (filter_tema IS NULL OR m.tema = filter_tema)
  ORDER BY m.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- ────────────────────────────────────────────────────────────────
-- 4. VERIFICACIÓN: distribución actual por tema
-- ────────────────────────────────────────────────────────────────
SELECT
  COALESCE(tema, 'sin_tema')       AS tema,
  COALESCE(subtema, 'sin_subtema') AS subtema,
  COUNT(*)                          AS total,
  COUNT(*) FILTER (WHERE is_superseded = FALSE) AS activas,
  ROUND(AVG(confidence_score)::numeric, 3)       AS confianza_media
FROM memory_entries
GROUP BY tema, subtema
ORDER BY tema NULLS LAST, total DESC;
