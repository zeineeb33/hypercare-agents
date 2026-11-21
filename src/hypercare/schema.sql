-- Hypercare Agents: SQLite schema.
-- Domain tables hold the (synthetic) pricing platform data; system tables hold what the agents produce.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- domain: reference data

CREATE TABLE users (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE,
    full_name   TEXT NOT NULL,
    role        TEXT NOT NULL,
    email       TEXT NOT NULL
);

CREATE TABLE pricing_groups (
    pg_id               INTEGER PRIMARY KEY,
    pg_code             TEXT NOT NULL UNIQUE,
    pg_name             TEXT NOT NULL,
    service_line_code   TEXT NOT NULL,
    currency            TEXT NOT NULL DEFAULT 'USD'
);

CREATE TABLE routes (
    route_id    INTEGER PRIMARY KEY,
    pol_code    TEXT NOT NULL,          -- port (or inland point) of loading
    pol_name    TEXT NOT NULL,
    pod_code    TEXT NOT NULL,          -- port of discharge
    pod_name    TEXT NOT NULL,
    via_port    TEXT,                   -- gateway port for inland origins
    trade_lane  TEXT NOT NULL,
    pg_id       INTEGER NOT NULL REFERENCES pricing_groups(pg_id),
    is_inland   INTEGER NOT NULL DEFAULT 0 CHECK (is_inland IN (0, 1))
);

CREATE TABLE commodities (
    commodity_id    INTEGER PRIMARY KEY,
    commodity_code  TEXT NOT NULL UNIQUE,
    commodity_name  TEXT NOT NULL,
    family          TEXT NOT NULL,
    is_hazardous    INTEGER NOT NULL DEFAULT 0 CHECK (is_hazardous IN (0, 1))
);

CREATE TABLE charges (
    charge_code     TEXT PRIMARY KEY,
    description     TEXT NOT NULL,
    charge_type     TEXT NOT NULL CHECK (charge_type IN ('SURCHARGE', 'INLAND')),
    is_active       INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE inland_rates (
    inland_point    TEXT NOT NULL,      -- e.g. FRLIL
    gateway_port    TEXT NOT NULL,      -- e.g. FRLEH
    charge_code     TEXT NOT NULL REFERENCES charges(charge_code),
    amount_20       REAL NOT NULL,
    amount_40       REAL NOT NULL,
    valid_from      TEXT NOT NULL,
    valid_to        TEXT NOT NULL,
    PRIMARY KEY (inland_point, gateway_port, charge_code, valid_from)
);

-- ---------------------------------------------------------------- domain: pricing

CREATE TABLE price_lists (
    pricelist_id        INTEGER PRIMARY KEY,
    pricelist_code      TEXT NOT NULL UNIQUE,
    pg_id               INTEGER NOT NULL REFERENCES pricing_groups(pg_id),
    charges_included    INTEGER NOT NULL DEFAULT 0 CHECK (charges_included IN (0, 1)),  -- 1 = all-in
    created_by          INTEGER REFERENCES users(user_id)
);

CREATE TABLE price_lines (
    priceline_id    INTEGER PRIMARY KEY,
    pricelist_id    INTEGER NOT NULL REFERENCES price_lists(pricelist_id),
    line_ref        INTEGER NOT NULL,
    route_id        INTEGER NOT NULL REFERENCES routes(route_id),
    commodity_id    INTEGER NOT NULL REFERENCES commodities(commodity_id),
    currency        TEXT NOT NULL DEFAULT 'USD',
    rate_20st       REAL NOT NULL,
    rate_40st       REAL NOT NULL,
    rate_40hc       REAL NOT NULL,
    valid_from      TEXT NOT NULL,
    valid_to        TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('ACTIVE', 'DRAFT', 'EXPIRED')),
    has_soc         INTEGER NOT NULL DEFAULT 0 CHECK (has_soc IN (0, 1)),   -- shipper-owned container
    has_haz         INTEGER NOT NULL DEFAULT 0 CHECK (has_haz IN (0, 1)),   -- hazardous cargo accepted
    UNIQUE (pricelist_id, line_ref)
);

CREATE TABLE price_line_surcharges (
    priceline_id    INTEGER NOT NULL REFERENCES price_lines(priceline_id),
    charge_code     TEXT NOT NULL REFERENCES charges(charge_code),
    amount_20st     REAL NOT NULL,
    amount_40st     REAL NOT NULL,
    amount_40hc     REAL NOT NULL,
    PRIMARY KEY (priceline_id, charge_code)
);

CREATE TABLE voyages (
    voyage_id       INTEGER PRIMARY KEY,
    voyage_code     TEXT NOT NULL UNIQUE,
    service_code    TEXT NOT NULL,
    vessel_name     TEXT NOT NULL,
    departure_date  TEXT NOT NULL,
    arrival_date    TEXT NOT NULL,
    route_id        INTEGER NOT NULL REFERENCES routes(route_id)
);

CREATE TABLE quotes (
    quote_id        INTEGER PRIMARY KEY,
    quote_ref       TEXT NOT NULL UNIQUE,
    priceline_id    INTEGER NOT NULL REFERENCES price_lines(priceline_id),
    customer_name   TEXT NOT NULL,
    equipment_size  TEXT NOT NULL CHECK (equipment_size IN ('20ST', '40ST', '40HC')),
    quantity        INTEGER NOT NULL,
    quoted_rate     REAL NOT NULL,      -- per container, as shown to the customer
    quote_date      TEXT NOT NULL,
    valid_until     TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('PENDING', 'ACCEPTED', 'EXPIRED', 'BOOKED')),
    channel         TEXT NOT NULL DEFAULT 'QUOTEPORT'
);

CREATE TABLE bookings (
    booking_id      INTEGER PRIMARY KEY,
    booking_ref     TEXT NOT NULL UNIQUE,
    quote_id        INTEGER NOT NULL REFERENCES quotes(quote_id),
    voyage_id       INTEGER REFERENCES voyages(voyage_id),
    booking_date    TEXT NOT NULL,
    status          TEXT NOT NULL,
    equipment_size  TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    total_amount    REAL NOT NULL
);

-- ---------------------------------------------------------------- support tickets

CREATE TABLE tickets (
    ticket_id       INTEGER PRIMARY KEY,
    ticket_ref      TEXT NOT NULL UNIQUE,
    category        TEXT NOT NULL,
    subcategory     TEXT NOT NULL,
    priority        TEXT NOT NULL DEFAULT 'MEDIUM',
    status          TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'IN_PROGRESS', 'RESOLVED')),
    description     TEXT NOT NULL,
    priceline_id    INTEGER REFERENCES price_lines(priceline_id),
    route_id        INTEGER REFERENCES routes(route_id),
    reported_by     INTEGER REFERENCES users(user_id),
    created_at      TEXT NOT NULL,
    related_doc     TEXT,
    scenario_id     TEXT,
    source          TEXT NOT NULL DEFAULT 'SEED'
);

-- ---------------------------------------------------------------- knowledge base

CREATE TABLE kb_documents (
    doc_id          INTEGER PRIMARY KEY,
    file_name       TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    category        TEXT NOT NULL,
    version         TEXT NOT NULL,
    effective_date  TEXT NOT NULL,
    source_type     TEXT NOT NULL,      -- DOC, VIDEO_TRANSCRIPT
    content         TEXT NOT NULL
);

CREATE TABLE kb_proposals (
    proposal_id         INTEGER PRIMARY KEY,
    created_at          TEXT NOT NULL,
    subcategory         TEXT NOT NULL,
    related_ticket_ids  TEXT NOT NULL,  -- JSON array
    target_document     TEXT NOT NULL,
    identified_gap      TEXT NOT NULL,
    proposed_title      TEXT NOT NULL,
    proposed_content    TEXT NOT NULL,
    rationale           TEXT NOT NULL,
    sources             TEXT NOT NULL,  -- JSON array
    checks              TEXT NOT NULL,  -- JSON array of pre-publication checks
    status              TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED')),
    reviewed_by         TEXT,
    reviewed_at         TEXT,
    review_comment      TEXT
);

CREATE TABLE kb_versions (
    version_id      INTEGER PRIMARY KEY,
    file_name       TEXT NOT NULL,
    version_number  INTEGER NOT NULL,
    content         TEXT NOT NULL,
    change_note     TEXT NOT NULL,
    proposal_id     INTEGER REFERENCES kb_proposals(proposal_id),
    created_by      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    is_current      INTEGER NOT NULL DEFAULT 1,
    UNIQUE (file_name, version_number)
);

-- ---------------------------------------------------------------- observability

CREATE TABLE agent_traces (
    trace_id        INTEGER PRIMARY KEY,
    execution_id    TEXT NOT NULL,
    ticket_id       INTEGER,
    created_at      TEXT NOT NULL,
    agent           TEXT NOT NULL,
    action          TEXT NOT NULL,
    input_summary   TEXT,
    output_summary  TEXT,
    sources         TEXT,               -- JSON array
    decision        TEXT,
    status          TEXT NOT NULL,      -- OK, ERROR, SKIPPED
    duration_ms     INTEGER NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0
);

CREATE INDEX idx_traces_execution ON agent_traces(execution_id);
CREATE INDEX idx_tickets_subcategory ON tickets(subcategory, status);
CREATE INDEX idx_price_lines_route ON price_lines(route_id, status);
