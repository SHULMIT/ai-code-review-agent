ALTER TABLE review_runs
    ADD COLUMN IF NOT EXISTS last_review_id BIGINT;
