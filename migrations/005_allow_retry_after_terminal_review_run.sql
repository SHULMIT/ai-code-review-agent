-- Allow a "Run Code Review" retry for the same PR/head SHA once the
-- previous attempt has reached a terminal state (failed/completed/skipped),
-- while still blocking a genuine concurrent duplicate (an active
-- pending/running row for the same identity).
--
-- 001 created review_identity as a plain column-level UNIQUE constraint -
-- global and permanent, with no notion of "terminal vs. active". That is
-- replaced here with a partial unique index scoped to only the active
-- statuses, so create_pending()'s existing "INSERT ... ON CONFLICT DO
-- NOTHING" (no explicit conflict target - Postgres matches it against any
-- applicable unique constraint or index) transparently starts respecting
-- the new semantics with no application code change required.
--
-- The constraint is looked up by which column it actually constrains
-- (pg_constraint.conkey), not assumed by name, matching 004's approach.
DO $$
DECLARE
    existing_constraint_name text;
BEGIN
    SELECT con.conname INTO existing_constraint_name
    FROM pg_constraint con
    JOIN pg_attribute att
      ON att.attrelid = con.conrelid AND att.attnum = ANY (con.conkey)
    WHERE con.conrelid = 'review_runs'::regclass
      AND con.contype = 'u'
      AND att.attname = 'review_identity';

    IF existing_constraint_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE review_runs DROP CONSTRAINT %I', existing_constraint_name);
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS review_runs_active_identity_idx
    ON review_runs (review_identity)
    WHERE status IN ('pending', 'running');
