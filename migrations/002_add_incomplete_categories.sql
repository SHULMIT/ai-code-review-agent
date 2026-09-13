ALTER TABLE review_runs
    ADD COLUMN IF NOT EXISTS incomplete_categories JSONB NOT NULL DEFAULT '[]'::jsonb;
