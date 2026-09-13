CREATE TABLE IF NOT EXISTS review_runs (
    id UUID PRIMARY KEY,
    github_delivery_id TEXT NOT NULL UNIQUE,
    review_identity TEXT NOT NULL UNIQUE,
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL CHECK (pr_number > 0),
    installation_id BIGINT NOT NULL CHECK (installation_id > 0),
    trigger TEXT NOT NULL CHECK (
        trigger IN ('pull_request.opened', 'pull_request.synchronize', 'issue_comment.review')
    ),
    head_sha TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    findings JSONB,
    architecture_score SMALLINT CHECK (architecture_score BETWEEN 0 AND 100),
    code_quality_score SMALLINT CHECK (code_quality_score BETWEEN 0 AND 100),
    production_readiness_score SMALLINT CHECK (
        production_readiness_score BETWEEN 0 AND 100
    ),
    recommendation TEXT,
    errors JSONB NOT NULL DEFAULT '[]'::jsonb,
    incomplete_categories JSONB NOT NULL DEFAULT '[]'::jsonb
);
