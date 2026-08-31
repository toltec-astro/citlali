#!/usr/bin/env python3
"""Verify shared-source identity and rendered SCI-FLT-MATCHED draft content.

This is a build-consistency check only. It is not scientific validation or an
implementation-conformity result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


COMMON = [
    "notation.tex",
    "definitions.tex",
    "equations.tex",
    "assumptions.tex",
    "requirements.tex",
    "edge_cases.tex",
]
REQ_IDS = [f"SCI-FLT-MATCHED-REQ-{i:03d}" for i in range(1, 40)]
PRED_IDS = [f"SCI-FLT-MATCHED-PRED-{i:03d}" for i in range(1, 19)]
AO_OPTIONS = {
    *[f"SCI-FLT-MATCHED-AO-001-{x}" for x in "ABCD"],
    *[f"SCI-FLT-MATCHED-AO-002-{x}" for x in "ABC"],
    *[f"SCI-FLT-MATCHED-AO-003-{x}" for x in "ABCDE"],
    *[f"SCI-FLT-MATCHED-AO-004-{x}" for x in "ABC"],
    *[f"SCI-FLT-MATCHED-AO-005-{x}" for x in "ABC"],
    *[f"SCI-FLT-MATCHED-AO-006-{x}" for x in "ABC"],
}
PACKET_HASHES = {
    "AUTHOR_PACKET_MANIFEST.md": "255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8",
    "SCOPE_BRIEF.md": "69e8db020cd8569fe94dd18ef2012c13691fe5b3a2b188201bfc96cc31fa13c3",
    "AUTHOR_SUPERSESSION_COVER.md": "198ec9c50d5c5dc8cb8e68794dd5034a6aca7aaebf81521726ac371ad9bfc1c3",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md": "a5d91666a13ea06e39af4356b07a1ef316219559a19847fc81460c9c067446d6",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md": "d7235655bbce52be28714bb637a9d003f1f5bc9947d310bf61d3e5877d6feb9d",
    "AUTHOR_OPERATOR_STATE_AND_PRODUCT_TAXONOMY.md": "4e47f825d54645dc6e82b34ed4848e0bb656fc385edd61f744ca43c6124eb980",
    "AUTHOR_BOUNDARIES.md": "273fa218eb0ff9610e433c99d8f4d30f5a3abe653da375dfb17781148fbcce34",
    "REQUIRED_AUTHORED_OPTION_SETS.md": "bad539a66199090ba9e32bf22100da708cb3730b02650e0401bdeeee01330cfe",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_relative(path: Path, package: Path) -> str:
    return str(path.resolve().relative_to(package.resolve()))


def rendered_text(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = re.sub(r"\s+", " ", text)
    # TeX permits a visual line break immediately after ID hyphens. PDF text
    # extractors expose that break as whitespace; canonicalize it for identity
    # checks without changing ordinary prose.
    text = re.sub(r"-\s+", "-", text)
    return text, len(reader.pages)


def source_ids(text: str) -> tuple[set[str], set[str], set[str]]:
    req = {f"SCI-FLT-MATCHED-REQ-{x}" for x in re.findall(r"\\Req\{(\d{3})\}", text)}
    pred = {f"SCI-FLT-MATCHED-PRED-{x}" for x in re.findall(r"\\Pred\{(\d{3})\}", text)}
    ao = {
        f"SCI-FLT-MATCHED-AO-{family}-{alt}"
        for family, alt in re.findall(r"\\AOpt\{(\d{3})\}\{([A-Z])\}", text)
    }
    return req, pred, ao


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    package = args.package.resolve()
    src = package / "src"
    pdf_dir = package / "pdf"
    views = {
        "scientific": src / "scientific-rationale.tex",
        "engineering": src / "engineering-conformance.tex",
    }
    pdfs = {
        "scientific": pdf_dir / "SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf",
        "engineering": pdf_dir / "SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf",
    }

    errors: list[str] = []
    observed_packet_hashes = {name: digest(package / name) for name in PACKET_HASHES}
    for name, expected in PACKET_HASHES.items():
        if observed_packet_hashes[name] != expected:
            errors.append(
                f"approved packet object changed: {name}: "
                f"{observed_packet_hashes[name]} != {expected}"
            )
    expected_imports = [f"common/{name}" for name in COMMON]
    for label, path in views.items():
        text = path.read_text(encoding="utf-8")
        imports = re.findall(r"\\input\{([^}]+)\}", text)
        if imports != expected_imports:
            errors.append(f"{label}: shared imports differ: {imports!r}")

    common_text = "\n".join(
        (src / "common" / name).read_text(encoding="utf-8") for name in COMMON
    )
    req, pred, ao = source_ids(common_text)
    expected_req, expected_pred = set(REQ_IDS), set(PRED_IDS)
    if req != expected_req:
        errors.append(f"source requirement ID mismatch: {sorted(req ^ expected_req)}")
    if pred != expected_pred:
        errors.append(f"source prediction ID mismatch: {sorted(pred ^ expected_pred)}")
    if ao != AO_OPTIONS:
        errors.append(f"source option ID mismatch: {sorted(ao ^ AO_OPTIONS)}")

    crosswalk_text = (package / "CROSSWALK.md").read_text(encoding="utf-8")
    ledger_text = (package / "SCIENTIFIC_OWNER_DECISION_LEDGER.md").read_text(
        encoding="utf-8"
    )
    missing_crosswalk = [
        item
        for item in [*REQ_IDS, *PRED_IDS, *sorted(AO_OPTIONS)]
        if item not in crosswalk_text
    ]
    if missing_crosswalk:
        errors.append(f"crosswalk stable IDs missing: {missing_crosswalk}")
    ledger_ids = [f"SCI-FLT-MATCHED-SODL-{i:03d}" for i in range(1, 18)]
    missing_ledger = [item for item in ledger_ids if item not in ledger_text]
    if missing_ledger:
        errors.append(f"owner-decision ledger IDs missing: {missing_ledger}")

    pdf_report: dict[str, object] = {}
    for label, path in pdfs.items():
        if not path.is_file():
            errors.append(f"{label}: PDF missing: {path}")
            continue
        text, pages = rendered_text(path)
        missing_req = [x for x in REQ_IDS if x not in text]
        missing_pred = [x for x in PRED_IDS if x not in text]
        missing_ao = [x for x in sorted(AO_OPTIONS) if x not in text]
        for phrase in [
            "STAGE B DRAFT",
            "No numerical route",
            "unselected",
            "implementation conformity",
            "achieved performance",
            "readiness",
            "production",
        ]:
            if phrase.lower() not in text.lower():
                errors.append(f"{label}: rendered nonclaim phrase missing: {phrase}")
        if missing_req:
            errors.append(f"{label}: rendered requirements missing: {missing_req}")
        if missing_pred:
            errors.append(f"{label}: rendered predictions missing: {missing_pred}")
        if missing_ao:
            errors.append(f"{label}: rendered options missing: {missing_ao}")
        pdf_report[label] = {
            "path": package_relative(path, package),
            "pages": pages,
            "sha256": digest(path),
            "requirements_found": len(REQ_IDS) - len(missing_req),
            "predictions_found": len(PRED_IDS) - len(missing_pred),
            "option_alternatives_found": len(AO_OPTIONS) - len(missing_ao),
        }

    log_report: dict[str, object] = {}
    fatal_log_patterns = re.compile(
        r"error:|undefined control sequence|overfull \\hbox|missing character|"
        r"reference .* undefined|citation .* undefined",
        re.IGNORECASE,
    )
    for label, log_name in {
        "scientific": "scientific-rationale.log",
        "engineering": "engineering-conformance.log",
    }.items():
        log_path = package / "build" / log_name
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        fatal_lines = [
            line for line in log_text.splitlines() if fatal_log_patterns.search(line)
        ]
        if fatal_lines:
            errors.append(f"{label}: unexpected build-log lines: {fatal_lines}")
        log_report[label] = {
            "path": package_relative(log_path, package),
            "overfull_boxes": len(re.findall(r"Overfull \\hbox", log_text)),
            "underfull_boxes": len(re.findall(r"Underfull \\hbox", log_text)),
            "unexpected_lines": fatal_lines,
        }

    result = {
        "status": "PASS" if not errors else "FAIL",
        "scope": "source/PDF build consistency only; not scientific validation or implementation conformity",
        "shared_import_order": expected_imports,
        "approved_packet_sha256": observed_packet_hashes,
        "shared_module_sha256": {
            name: digest(src / "common" / name) for name in COMMON
        },
        "source_counts": {
            "requirements": len(req),
            "predictions": len(pred),
            "option_families": 6,
            "option_alternatives": len(ao),
            "crosswalk_stable_ids": len(REQ_IDS) + len(PRED_IDS) + len(AO_OPTIONS),
            "owner_decision_questions": len(ledger_ids),
        },
        "pdfs": pdf_report,
        "build_logs": log_report,
        "errors": errors,
    }
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
