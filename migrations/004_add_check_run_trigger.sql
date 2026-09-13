ALTER TABLE review_runs
    ADD COLUMN IF NOT EXISTS check_run_id BIGINT;

-- Widen the trigger CHECK constraint to allow the new on-demand Check Run
-- action trigger, alongside the three existing triggers. 001 created this
-- constraint inline on the column without an explicit name, so rather than
-- assuming PostgreSQL's default column-check-constraint naming convention
-- held, this looks up whatever the trigger column's CHECK constraint is
-- actually named in this database - by which column it constrains
-- (pg_constraint.conkey), not by string-matching its rendered definition,
-- since PostgreSQL normalizes "IN (...)" to "= ANY (ARRAY[...])" internally
-- and would not literally contain the substring "IN" to match against -
-- and drops it by that real name before re-adding it under a fixed,
-- predictable name.
DO $$
DECLARE
    existing_constraint_name text;
BEGIN
    SELECT con.conname INTO existing_constraint_name
    FROM pg_constraint con
    JOIN pg_attribute att
      ON att.attrelid = con.conrelid AND att.attnum = ANY (con.conkey)
    WHERE con.conrelid = 'review_runs'::regclass
      AND con.contype = 'c'
      AND att.attname = 'trigger';

    IF existing_constraint_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE review_runs DROP CONSTRAINT %I', existing_constraint_name);
    END IF;
END $$;

ALTER TABLE review_runs
    ADD CONSTRAINT review_runs_trigger_check CHECK (
        trigger IN (
            'pull_request.opened', 'pull_request.synchronize',
            'issue_comment.review', 'check_run.requested_action'
        )
    );
