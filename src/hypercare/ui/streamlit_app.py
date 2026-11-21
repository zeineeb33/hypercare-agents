"""Hypercare Agents: Streamlit console.

Run with ``hypercare serve`` (or ``streamlit run src/hypercare/ui/streamlit_app.py``).
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from hypercare.agents.maintenance import ReviewError, review_proposal
from hypercare.agents.schemas import TicketRun
from hypercare.config import DATA_DIR
from hypercare.db import query, query_one, transaction, utcnow
from hypercare.llm import LLMError
from hypercare.seed import build_database
from hypercare.services import Services, open_services

st.set_page_config(
    page_title="Hypercare Agents", page_icon=":material/support_agent:", layout="wide"
)

SCENARIOS = [
    ("S1", "Documentation", "HAZ cargo and SOC lines", "ANSWER", [1, 2]),
    ("S2", "Price gap", "Quote vs price line", "ANSWER", [3, 4, 5]),
    ("S3", "Unsafe to answer", "DRAFT or overlapping lines", "ESCALATE", [6, 7, 8]),
    ("S4", "Knowledge gap", "Recurring, undocumented", "DETECT_KB_GAP", [9, 10, 11, 12, 13]),
    ("S5", "Clarification", "No reference given", "ASK_CLARIFICATION", [23, 24]),
    ("S6", "Video", "Answer from a video", "ANSWER", [25]),
]
DECISION_STYLE = {
    "ANSWER": ("green", ":material/check_circle:"),
    "ESCALATE": ("red", ":material/error:"),
    "DETECT_KB_GAP": ("orange", ":material/lightbulb:"),
    "ASK_CLARIFICATION": ("blue", ":material/help:"),
}
PIPELINE = [
    ("Classification", "Route the ticket"),
    ("Knowledge agent", "Search and grounded answer"),
    ("Data investigation", "SQL on quotes, lines, routes"),
    ("Pattern detection", "Recurring and undocumented?"),
    ("Resolution", "Decide and draft the reply"),
    ("Verification", "6 deterministic checks"),
]


@st.cache_resource
def services() -> Services:
    return open_services()


def svc() -> Services:
    return services()


def df(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """st.dataframe renders a list of records directly."""
    return rows


# ---------------------------------------------------------------- sidebar


def sidebar(s: Services) -> None:
    last: TicketRun | None = st.session_state.get("last_run")
    done = 6 if last else 0
    with st.sidebar:
        st.markdown("### :material/support_agent: Hypercare Agents")
        st.caption(
            "Multi-agent support for the TideRate rollout at Northwind Shipping (fictional)."
        )
        mode = s.settings.mode
        if mode == "replay":
            st.info(
                "**Replay mode.** LLM answers were recorded from real runs; retrieval, SQL and "
                "verification run live. New tickets need live mode.",
                icon=":material/replay:",
            )
        else:
            st.success(f"**Live mode** · `{s.settings.model}`", icon=":material/bolt:")
        st.markdown("**Pipeline**")
        st.progress(done / 6)
        for i, (name, desc) in enumerate(PIPELINE):
            icon = ":material/check_circle:" if i < done else ":material/radio_button_unchecked:"
            st.markdown(f"{icon} **{name}**  \n:gray[{desc}]")
        if last:
            color, icon = DECISION_STYLE.get(str(last.final.decision), ("gray", ""))
            st.markdown(f"Last: **{last.ticket_ref}** → :{color}[{last.final.decision}]")
        st.divider()
        if mode == "live":
            new_ticket_form(s)
        if st.button(":material/restart_alt: Reset demo data", use_container_width=True):
            s.conn.close()
            build_database(s.settings.db_path, s.settings.seed_dir, s.settings.kb_dir)
            services.clear()
            st.session_state.clear()
            st.rerun()


def new_ticket_form(s: Services) -> None:
    with st.expander(":material/add: New ticket"), st.form("new_ticket", clear_on_submit=True):
        category = st.selectbox("Category", ["PRICING_ISSUE", "TECHNICAL", "DOCUMENTATION"])
        subcategory = st.text_input("Subcategory", "RATE_MISMATCH").strip().upper()
        description = st.text_area("Description", placeholder="Describe the problem…")
        if st.form_submit_button("Create") and description.strip():
            row = query_one(s.conn, "SELECT COALESCE(MAX(ticket_id), 0) + 1 AS n FROM tickets")
            ticket_id = int(row["n"]) if row else 1
            with transaction(s.conn):
                s.conn.execute(
                    "INSERT INTO tickets (ticket_id, ticket_ref, category, subcategory, "
                    "description, created_at, source) VALUES (?, ?, ?, ?, ?, ?, 'UI')",
                    (
                        ticket_id,
                        f"HC-2026-{ticket_id:04d}",
                        category,
                        subcategory or "OTHER",
                        description.strip(),
                        utcnow(),
                    ),
                )
            st.session_state["selected_ticket"] = ticket_id
            st.toast(f"Ticket HC-2026-{ticket_id:04d} created")


# ---------------------------------------------------------------- ticket console


def show_run(run: TicketRun) -> None:
    decision = str(run.final.decision)
    color, icon = DECISION_STYLE.get(decision, ("gray", ""))
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        c1.markdown(f"### {icon} :{color}[{decision}]")
        if run.verification and run.verification.overridden and run.proposed:
            c1.markdown(
                f":red-badge[OVERRIDE] The verification agent changed "
                f"`{run.proposed.decision}` → `{decision}`"
            )
        c2.metric("Time", f"{run.duration_ms / 1000:.1f} s")
        checks = run.verification.checks if run.verification else []
        c3.metric("Checks", f"{sum(c.passed for c in checks)}/{len(checks)}")
        c4.metric("Cost", f"${run.cost_usd:.3f}")

    st.markdown("#### Reply")
    reply = run.final.response
    {"ANSWER": st.success, "ESCALATE": st.error, "DETECT_KB_GAP": st.warning}.get(
        decision, st.info
    )(reply)
    if run.final.missing_information and decision == "ASK_CLARIFICATION":
        st.markdown("\n".join(f"- {m}" for m in run.final.missing_information))
    if run.final.sources:
        st.caption("Sources: " + ", ".join(f"`{x}`" for x in run.final.sources))
    if run.verification and run.verification.overridden and run.proposed:
        with st.expander("Reply proposed by the resolution agent (blocked)"):
            st.write(run.proposed.response)

    left, right = st.columns(2)
    with left:
        with st.expander(":material/verified: Verification agent", expanded=True):
            for c in checks:
                if c.passed:
                    st.markdown(f":green-badge[PASS] **{c.name}** :gray[{c.detail}]")
                else:
                    tag = "BLOCK" if c.overrides else "WARN"
                    st.markdown(f":red-badge[{tag}] **{c.name}** — {c.detail}")
        if run.knowledge:
            with st.expander(":material/search: Knowledge agent", expanded=True):
                if run.knowledge.queries:
                    st.caption("Queries: " + " · ".join(run.knowledge.queries))
                for p in run.knowledge.passages:
                    st.markdown(f"- `{p.file_name}` > {p.section} :gray-badge[v{p.version}]")
                if run.knowledge.answer:
                    a = run.knowledge.answer
                    st.markdown(
                        f"Confidence **{a.confidence}** · insufficient evidence: "
                        f"**{a.insufficient_evidence}**"
                    )
                if run.knowledge.error:
                    st.warning(run.knowledge.error)
    with right:
        if run.data:
            with st.expander(":material/query_stats: Data investigation agent", expanded=True):
                d = run.data
                for line in d.price_lines:
                    st.markdown(
                        f"**{line['pricelist_code']} line {line['line_ref']}** · "
                        f"{line['pol_name']} → {line['pod_name']} · `{line['status']}` · 40HC "
                        f"{line['rate_40hc']:.0f} (all-in {line['all_in_40hc']:.0f}) · valid "
                        f"{line['valid_from']} → {line['valid_to']}"
                    )
                for q in d.quotes:
                    st.caption(
                        f"Quote {q['quote_ref']}: {q['quoted_rate']:.0f} {q['equipment_size']} "
                        f"· {q['status']} · valid until {q['valid_until']}"
                    )
                for b in d.bookings:
                    st.caption(
                        f"Booking {b['booking_ref']}: {b['quantity']} x {b['equipment_size']}"
                        f" = {b['total_amount']:.0f}"
                    )
                if d.overlapping_lines:
                    st.markdown(f":orange-badge[{len(d.overlapping_lines)} overlapping lines]")
                if d.inland:
                    found = d.inland["pre_carriage_rate_found"]
                    st.markdown(
                        f"Inland {d.inland['inland_point_name']} via {d.inland['gateway_port']}: "
                        + (":green[pre-carriage rate found]" if found else ":red[no INL00 rate]")
                    )
                st.caption(f"{len(d.similar_open_tickets)} similar open ticket(s)")
        if run.pattern:
            with st.expander(":material/pattern: Pattern detection"):
                st.write(
                    f"{run.pattern.open_tickets_in_subcategory} open tickets in {run.subcategory} "
                    f"(threshold {run.pattern.threshold}) · knowledge gap: **{run.pattern.kb_gap}**"
                )
    with st.expander(":material/timeline: Execution trace"):
        st.dataframe(
            df([s.model_dump() for s in run.steps]), hide_index=True, use_container_width=True
        )
    with st.expander(":material/code: Full JSON"):
        st.json(json.loads(run.model_dump_json()))


def ticket_console(s: Services) -> None:
    st.markdown("#### Reference scenarios")
    cols = st.columns(len(SCENARIOS))
    for col, (sid, title, desc, expected, ids) in zip(cols, SCENARIOS, strict=True):
        with col.container(border=True):
            color, _ = DECISION_STYLE[expected]
            st.markdown(f"**{sid} · {title}**  \n:gray[{desc}]")
            st.markdown(f":{color}-badge[{expected}]")
            if st.button("Select", key=f"sc_{sid}", use_container_width=True):
                st.session_state["selected_ticket"] = ids[0]

    tickets = query(
        s.conn,
        "SELECT ticket_id, ticket_ref, category, subcategory, status, priority, scenario_id, "
        "description FROM tickets ORDER BY ticket_id",
    )
    labels = {
        t["ticket_id"]: f"{t['ticket_ref']} · {t['subcategory']}"
        + (f" · {t['scenario_id']}" if t["scenario_id"] else "")
        for t in tickets
    }
    ids = list(labels)
    current = st.session_state.get("selected_ticket", ids[0])
    left, right = st.columns([1, 2])
    with left:
        selected = st.selectbox(
            "Ticket",
            ids,
            index=ids.index(current) if current in ids else 0,
            format_func=lambda i: labels[i],
        )
        st.session_state["selected_ticket"] = selected
        t = next(t for t in tickets if t["ticket_id"] == selected)
        st.markdown(
            f":gray-badge[{t['category']}] :gray-badge[{t['subcategory']}] "
            f":blue-badge[{t['priority']}] :violet-badge[{t['status']}]"
        )
        run_clicked = st.button(
            ":material/play_arrow: Run the agents", type="primary", use_container_width=True
        )
    with right, st.container(border=True):
        st.caption(t["ticket_ref"])
        st.write(t["description"])

    if run_clicked:
        try:
            with st.spinner("Agents at work…"):
                st.session_state["last_run"] = s.orchestrator().run(int(selected))
        except LLMError as e:
            st.error(str(e))
    run: TicketRun | None = st.session_state.get("last_run")
    if run and run.ticket_id == selected:
        st.divider()
        show_run(run)


# ---------------------------------------------------------------- knowledge base


def knowledge_tab(s: Services) -> None:
    docs = query(
        s.conn,
        "SELECT file_name, title, category, version, source_type, "
        "length(content) AS characters FROM kb_documents ORDER BY category, file_name",
    )
    st.markdown(
        f"#### {len(docs)} documents · {len(s.index.passages)} indexed sections · "
        f"{'hybrid (BM25 + embeddings)' if s.index.hybrid else 'BM25'} retrieval"
    )
    st.dataframe(df(docs), hide_index=True, use_container_width=True)
    q = st.text_input(
        "Search the knowledge base", placeholder="e.g. inland pre-carriage, DRAFT line"
    )
    if q:
        for hit in s.index.search(q, k=5):
            with st.expander(f"{hit.score:.2f} · {hit.passage.file_name} > {hit.passage.section}"):
                st.markdown(hit.passage.text)


# ---------------------------------------------------------------- self-healing KB


def self_healing_tab(s: Services) -> None:
    st.markdown("#### Self-healing knowledge base")
    st.caption(
        "The maintenance agent looks for subcategories with many open tickets that the "
        "documentation cannot answer, drafts a new section, tests it (does the updated document "
        "answer the question?) and waits for a human review."
    )
    if st.button(":material/build: Run the maintenance agent", type="primary"):
        try:
            with st.spinner("Detecting gaps and drafting updates…"):
                report = s.maintenance().run()
            st.success(
                f"{len(report.proposals)} new proposal(s); covered by the documentation: "
                f"{', '.join(report.covered) or '-'}"
            )
        except LLMError as e:
            st.error(str(e))

    proposals = query(s.conn, "SELECT * FROM kb_proposals ORDER BY proposal_id DESC")
    if not proposals:
        st.info("No proposal yet.")
    for p in proposals:
        badge = {"APPROVED": "green", "PENDING": "orange", "REJECTED": "red"}[p["status"]]
        with st.container(border=True):
            st.markdown(
                f":{badge}-badge[{p['status']}] **#{p['proposal_id']} · {p['proposed_title']}**"
                f" → `{p['target_document']}` · {len(json.loads(p['related_ticket_ids']))} tickets"
            )
            st.caption(f"Gap: {p['identified_gap']}")
            st.markdown(p["proposed_content"])
            for c in json.loads(p["checks"]):
                mark = ":green-badge[PASS]" if c["passed"] else ":red-badge[FAIL]"
                st.markdown(f"{mark} **{c['name']}** :gray[{c['detail']}]")
            if p["status"] == "PENDING":
                reviewer = st.text_input("Reviewer", "kb.reviewer", key=f"rev_{p['proposal_id']}")
                a, b, _ = st.columns([1, 1, 4])
                for col, label, decision in ((a, "Approve", "APPROVED"), (b, "Reject", "REJECTED")):
                    if col.button(label, key=f"{decision}_{p['proposal_id']}"):
                        try:
                            review_proposal(s.conn, int(p["proposal_id"]), decision, reviewer)
                            s.refresh_index()
                            st.rerun()
                        except ReviewError as e:
                            st.error(str(e))
    versions = query(
        s.conn,
        "SELECT file_name, version_number, is_current, change_note, created_by, "
        "created_at FROM kb_versions ORDER BY file_name, version_number DESC",
    )
    if versions:
        st.markdown("#### Version history")
        st.dataframe(df(versions), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- evaluation and system


def evaluation_tab() -> None:
    path = DATA_DIR / "eval" / "report.json"
    if not path.exists():
        st.info("No evaluation report yet. Run `hypercare eval` to produce one.")
        return
    report = json.loads(path.read_text())
    st.markdown(f"#### Evaluation · {report['cases']} cases · `{report['model']}`")
    c = st.columns(5)
    c[0].metric("Decision accuracy", f"{report['decision_accuracy']:.0%}")
    c[1].metric("Cites expected doc", f"{(report['source_accuracy'] or 0):.0%}")
    c[2].metric("States expected facts", f"{(report['fact_accuracy'] or 0):.0%}")
    c[3].metric("Median latency", f"{report['median_latency_s']} s")
    c[4].metric("Cost / ticket", f"${report['cost_per_ticket_usd']:.3f}")
    rows = [
        {k: r[k] for k in ("ticket_ref", "note", "expected", "proposed", "actual", "decision_ok")}
        for r in report["results"]
    ]
    st.dataframe(df(rows), hide_index=True, use_container_width=True)


def row_count(s: Services, table: str) -> int:
    if not table.isidentifier():  # names come from sqlite_master; refuse anything else
        return 0
    row = query_one(s.conn, f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608
    return int(row["n"]) if row else 0


def system_tab(s: Services) -> None:
    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Recent agent traces")
        traces = query(
            s.conn,
            "SELECT execution_id, ticket_id, agent, action, status, decision, "
            "duration_ms, cost_usd FROM agent_traces ORDER BY trace_id DESC LIMIT 60",
        )
        st.dataframe(df(traces), hide_index=True, use_container_width=True, height=380)
    with right:
        st.markdown("#### Tables")
        tables = query(s.conn, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        counts = [{"table": str(t["name"]), "rows": row_count(s, str(t["name"]))} for t in tables]
        st.dataframe(df(counts), hide_index=True, use_container_width=True, height=380)
    st.caption(
        f"Model `{s.settings.model}` · effort `{s.settings.effort}` · mode `{s.settings.mode}`"
    )


def main() -> None:
    s = svc()
    sidebar(s)
    st.title(":material/support_agent: Hypercare Agents")
    st.markdown(
        ":blue-badge[Multi-agent] :violet-badge[Self-healing KB] "
        ":green-badge[Independent verification] :gray-badge[Claude API]"
    )
    st.caption("Handle hypercare tickets faster and consistently, without ever inventing a price.")
    stats = (
        query_one(
            s.conn,
            "SELECT (SELECT COUNT(*) FROM tickets) AS tickets, (SELECT COUNT(*) FROM kb_documents) AS docs, "
            "(SELECT COUNT(*) FROM kb_proposals WHERE status = 'PENDING') AS pending, "
            "(SELECT COUNT(*) FROM agent_traces) AS traces",
        )
        or {}
    )
    cols = st.columns(4)
    for col, (label, key) in zip(
        cols,
        [
            ("Tickets", "tickets"),
            ("KB documents", "docs"),
            ("Pending KB proposals", "pending"),
            ("Agent traces", "traces"),
        ],
        strict=True,
    ):
        col.metric(label, stats.get(key, 0), border=True)

    tabs = st.tabs(
        [
            ":material/confirmation_number: Tickets",
            ":material/menu_book: Knowledge base",
            ":material/build: Self-healing KB",
            ":material/analytics: Evaluation",
            ":material/settings: System",
        ]
    )
    with tabs[0]:
        ticket_console(s)
    with tabs[1]:
        knowledge_tab(s)
    with tabs[2]:
        self_healing_tab(s)
    with tabs[3]:
        evaluation_tab()
    with tabs[4]:
        system_tab(s)


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
