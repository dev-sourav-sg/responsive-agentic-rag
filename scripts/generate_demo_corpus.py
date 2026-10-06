from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from textwrap import wrap
import hashlib
import html
import re

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = ROOT / "data" / "demo_corpus" / "documents"
HTML_DIR = ROOT / "data" / "demo_corpus" / "websites"

PDF_COUNT = 50
HTML_COUNT = 30


DOMAINS = [
    ("payments", "Payment Processing"),
    ("settlement", "Merchant Settlement"),
    ("reconciliation", "Reconciliation"),
    ("chargebacks", "Chargebacks and Disputes"),
    ("merchant", "Merchant Operations"),
    ("risk", "Risk and Compliance"),
    ("support", "Merchant Support"),
    ("security", "Security Operations"),
    ("processor", "Processor Integrations"),
    ("operations", "Platform Operations"),
]


OWNERS = [
    "Payments Engineering",
    "Settlement Operations",
    "Reconciliation Engineering",
    "Risk & Compliance",
    "Merchant Operations",
    "Platform Reliability",
]


APPROVALS = [
    "approved",
    "approved",
    "approved",
    "pending",
    "draft",
]


TOPICS = {
    "payments": [
        "authorization processing",
        "payment capture",
        "payment reversal",
        "payment retry policy",
        "transaction lifecycle",
    ],
    "settlement": [
        "settlement SLA",
        "merchant payout scheduling",
        "settlement exception handling",
        "funding reconciliation",
        "settlement cut-off windows",
    ],
    "reconciliation": [
        "daily reconciliation",
        "transaction matching",
        "reconciliation exceptions",
        "source completeness",
        "reconciliation reporting",
    ],
    "chargebacks": [
        "chargeback intake",
        "dispute evidence",
        "representment timelines",
        "chargeback reconciliation",
        "dispute escalation",
    ],
    "merchant": [
        "merchant onboarding",
        "merchant profile changes",
        "merchant operational support",
        "merchant configuration",
        "merchant lifecycle",
    ],
    "risk": [
        "risk controls",
        "compliance review",
        "transaction monitoring",
        "approval governance",
        "policy exceptions",
    ],
    "support": [
        "support escalation",
        "merchant incident handling",
        "support response SLA",
        "case ownership",
        "operational communications",
    ],
    "security": [
        "credential rotation",
        "access review",
        "security incident response",
        "service authentication",
        "audit evidence",
    ],
    "processor": [
        "processor onboarding",
        "processor file ingestion",
        "processor outage handling",
        "processor reconciliation",
        "processor certification",
    ],
    "operations": [
        "operational readiness",
        "incident management",
        "batch processing",
        "service monitoring",
        "business continuity",
    ],
}


def stable_int(value: str) -> int:
    """Return a deterministic integer derived from a string."""
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:4],
        "big",
    )


def clean_slug(value: str) -> str:
    """Convert text into a filesystem-safe slug."""
    return re.sub(
        r"[^a-z0-9]+",
        "-",
        value.lower(),
    ).strip("-")


def metadata_for(
    domain_key: str,
    index: int,
    source_kind: str,
) -> dict[str, str]:
    """Generate deterministic enterprise-style metadata."""
    source_hash = stable_int(
        f"{source_kind}:{domain_key}:{index}"
    )

    approval = APPROVALS[
        source_hash % len(APPROVALS)
    ]

    authority = {
        "approved": 0.90,
        "pending": 0.72,
        "draft": 0.50,
    }[approval]

    version = (
        f"{1 + (index % 3)}.{index % 4}"
    )

    published = (
        date(2025, 1, 10)
        + timedelta(days=source_hash % 600)
    )

    modified = (
        published
        + timedelta(days=index % 30)
    )

    return {
        "Document ID": (
            f"DEMO-{domain_key.upper()}-{index:03d}"
        ),
        "Version": version,
        "Owner": OWNERS[
            source_hash % len(OWNERS)
        ],
        "Approval Status": approval,
        "Publication Date": published.isoformat(),
        "Last Modified": modified.isoformat(),
        "Authority": f"{authority:.2f}",
        "Source Type": source_kind,
    }


def paragraphs(
    domain_key: str,
    title: str,
    index: int,
) -> list[str]:
    """Generate realistic but deterministic policy content."""
    domain_name = dict(DOMAINS)[domain_key]

    topics = TOPICS[domain_key]

    topic = topics[
        index % len(topics)
    ]

    next_topic = topics[
        (index + 1) % len(topics)
    ]

    settlement_sla = (
        4 if index % 4 else 5
    )

    reconciliation_sla = (
        24 if index % 3 else 48
    )

    retry_limit = (
        3 if index % 5 else 5
    )

    support_hours = (
        4 if index % 2 else 8
    )

    return [
        (
            f"{title} defines the operating policy for "
            f"{domain_name.lower()} and provides the controlled "
            f"reference for {topic}. This document is part of "
            f"the synthetic enterprise knowledge corpus used "
            f"for retrieval evaluation."
        ),
        (
            f"Scope. The policy applies to internal engineering, "
            f"operations, and support teams that participate in "
            f"{topic}. Processing must preserve the transaction "
            f"identifier, source provenance, policy version, "
            f"and applicable ownership metadata."
        ),
        (
            f"Standard processing. Requests are validated before "
            f"execution. A transaction or case must be associated "
            f"with an approved source record before it can move "
            f"to the next lifecycle stage. Automated processing "
            f"should be preferred, while manual intervention "
            f"requires an auditable reason."
        ),
        (
            f"Service level. The standard {topic} target is "
            f"{settlement_sla} business days for settlement-related "
            f"processing and {reconciliation_sla} hours for "
            f"reconciliation or operational follow-up. Exceptions "
            f"must be recorded with the responsible owner."
        ),
        (
            f"Exception handling. A failed operation may be retried "
            f"up to {retry_limit} times when the failure is transient. "
            f"Permanent validation failures must not be retried "
            f"indefinitely and should be routed to the appropriate "
            f"exception workflow."
        ),
        (
            f"Monitoring. Teams should monitor completion rate, "
            f"failure rate, processing latency, unresolved "
            f"exceptions, and age of the oldest outstanding case. "
            f"Alerts should include source identifier and policy "
            f"version so operators can reproduce the issue."
        ),
        (
            f"Escalation. Support cases should receive an initial "
            f"operational response within {support_hours} business "
            f"hours. Material customer or financial impact must be "
            f"escalated to the domain owner and risk/compliance "
            f"representatives."
        ),
        (
            f"Related policy. The same controls also apply to "
            f"{next_topic}. When two sources appear to conflict, "
            f"the approved source with the latest applicable "
            f"version takes precedence over a draft or deprecated "
            f"source."
        ),
    ]


def write_pdf(
    path: Path,
    title: str,
    metadata: dict[str, str],
    body: list[str],
) -> None:
    """Write a synthetic enterprise policy as a PDF."""
    document = canvas.Canvas(
        str(path),
        pagesize=LETTER,
    )

    width, height = LETTER

    del width

    y = height - 48

    document.setFont(
        "Helvetica-Bold",
        16,
    )

    document.drawString(
        48,
        y,
        title,
    )

    y -= 28

    document.setFont(
        "Helvetica",
        9,
    )

    for key, value in metadata.items():
        document.drawString(
            48,
            y,
            f"{key}: {value}",
        )

        y -= 13

    y -= 12

    document.setFont(
        "Helvetica",
        10,
    )

    for paragraph in body:
        lines = wrap(
            paragraph,
            width=92,
        )

        for line in lines:
            if y < 54:
                document.showPage()

                y = height - 54

                document.setFont(
                    "Helvetica",
                    10,
                )

            document.drawString(
                48,
                y,
                line,
            )

            y -= 14

        y -= 8

    document.save()


def write_html(
    path: Path,
    title: str,
    metadata: dict[str, str],
    body: list[str],
) -> None:
    """Write a synthetic website snapshot."""
    metadata_rows = "\n".join(
        (
            f"<tr>"
            f"<th>{html.escape(key)}</th>"
            f"<td>{html.escape(value)}</td>"
            f"</tr>"
        )
        for key, value in metadata.items()
    )

    paragraphs_html = "\n".join(
        f"<p>{html.escape(paragraph)}</p>"
        for paragraph in body
    )

    document = f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(title)}</title>
</head>
<body>
    <article>
        <h1>{html.escape(title)}</h1>

        <table>
            {metadata_rows}
        </table>

        <section>
            {paragraphs_html}
        </section>
    </article>
</body>
</html>
"""

    path.write_text(
        document,
        encoding="utf-8",
    )


def main() -> None:
    """Generate the complete controlled demo corpus."""
    PDF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HTML_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only files generated by this script.
    for directory, suffix in (
        (PDF_DIR, ".pdf"),
        (HTML_DIR, ".html"),
    ):
        for path in directory.glob(
            f"demo_*{suffix}"
        ):
            path.unlink()

    domains = [
        key
        for key, _ in DOMAINS
    ]

    domain_names = dict(DOMAINS)

    # ---------------------------------------------------------
    # Generate PDF corpus
    # ---------------------------------------------------------

    for index in range(
        1,
        PDF_COUNT + 1,
    ):
        domain_key = domains[
            (index - 1) % len(domains)
        ]

        domain_name = domain_names[
            domain_key
        ]

        title = (
            f"{domain_name} "
            f"Operating Policy "
            f"{index:03d}"
        )

        metadata = metadata_for(
            domain_key=domain_key,
            index=index,
            source_kind="document",
        )

        body = paragraphs(
            domain_key=domain_key,
            title=title,
            index=index,
        )

        # -----------------------------------------------------
        # Deliberate conflict scenarios.
        # These are useful for testing authority-aware ranking.
        # -----------------------------------------------------

        if index == 7:
            body.insert(
                4,
                (
                    "Conflict test policy: this approved document "
                    "states that settlement processing should target "
                    "3 business days for the synthetic legacy flow."
                ),
            )

        if index == 8:
            body.insert(
                4,
                (
                    "Conflict test policy: this draft document "
                    "states a 7 business day target. This lower-"
                    "authority draft should not override the "
                    "approved policy."
                ),
            )

        filename = (
            f"demo_{index:03d}_"
            f"{clean_slug(domain_name)}.pdf"
        )

        write_pdf(
            path=PDF_DIR / filename,
            title=title,
            metadata=metadata,
            body=body,
        )

    # ---------------------------------------------------------
    # Generate website corpus
    # ---------------------------------------------------------

    for index in range(
        1,
        HTML_COUNT + 1,
    ):
        domain_key = domains[
            (index - 1) % len(domains)
        ]

        domain_name = domain_names[
            domain_key
        ]

        title = (
            f"{domain_name} "
            f"Knowledge Portal "
            f"Page {index:03d}"
        )

        metadata = metadata_for(
            domain_key=domain_key,
            index=index + 100,
            source_kind="website",
        )

        body = paragraphs(
            domain_key=domain_key,
            title=title,
            index=index + 100,
        )

        filename = (
            f"demo_{index:03d}_"
            f"{clean_slug(domain_name)}.html"
        )

        write_html(
            path=HTML_DIR / filename,
            title=title,
            metadata=metadata,
            body=body,
        )

    print()
    print("Demo corpus generation complete.")
    print(f"PDF documents : {PDF_COUNT}")
    print(f"HTML websites : {HTML_COUNT}")
    print(f"Total sources : {PDF_COUNT + HTML_COUNT}")
    print()
    print(f"Documents: {PDF_DIR}")
    print(f"Websites : {HTML_DIR}")


if __name__ == "__main__":
    main()