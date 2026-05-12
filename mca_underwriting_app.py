"""
MCA Underwriting Analyzer - recurring position finder version

Run:
  pip install streamlit pandas numpy python-dateutil openpyxl pdfplumber pymupdf pillow
  streamlit run mca_underwriting_app_confidence_engine.py

Optional OCR for scanned PDFs:
  pip install pytesseract
  Install Tesseract locally if you need scanned/image-only PDFs.
"""

from __future__ import annotations

import re
from io import BytesIO
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import fitz
except Exception:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# -----------------------------
# Tunable MCA / revenue rules
# -----------------------------

MCA_RULES = [
    # ===== Known lenders from your sample PDFs =====
    ("IRUKA CAPITAL", ["IRUKA CAPITAL", "ACHPAYMENTIRUKA", "ACH PAYMENT IRUKA"]),
    ("MCA Servicing Co", ["MCA SERVICING", "DR MCA SERVICING CO", "MCA SERVICING CO"]),
    ("River Advance", ["RIVER ADVANCE", "1728-RO RIVER ADVANCE", "RIVER ADV"]),
    ("Slate Advance", ["SLATE ADVANCE", "PAYMENT19 SLATE ADVANCE", "SLATE ADV"]),
    ("Stage Advance", ["STAGE ADVANCE"]),
    ("TrueAdvance", ["TRUEADVANCE", "TRUE ADVANCE"]),
    ("ZLUR Funding", ["ZLURFUNDING", "ZLUR FUNDING"]),
    ("CFG Merchant Solutions", ["CFGMS", "8446623467 CFGMS", "CFG MERCHANT"]),
    ("Fundbox", ["FUNDBOX", "ADV DEBIT FUNDBOX", "FUNDBOX INC"]),
    ("First Source Advance", ["FIRST SOURCE ADV", "DEBIT FIRST SOURCE ADV", "FIRST SOURCE ADVANCE"]),
    ("OD Capital", ["OD CAPITAL", "RAY-ECCHEREN OD CAPITAL"]),
    ("Highland Hill Capital", ["HIGHLAND HILL", "9549478724 HIGHLAND HILL"]),
    ("Targeted Lending", ["TARGETED LENDING"]),
    ("Advantage Leasing", ["ADVANTAGE LEAS", "ADVANTAGE LEASING"]),
    ("Riverside Capital", ["RIVERSIDE CAPITA", "RIVERSIDE CAPITAL"]),
    ("MEF", ["AUTH PAYMENMEF", "MEF 2022-A", "AUTH PAYMENT MEF"]),

    # ===== Common MCA / revenue based finance companies =====
    ("Rapid Finance", ["RAPID FINANCE", "RAPIDFUNDING", "RAPID FIN"]),
    ("OnDeck", ["ONDECK", "ON DECK", "ONDK", "ONDECK ACH"]),
    ("Forward Financing", ["FORWARD FINANCING", "FORWARDFIN", "FWD FIN"]),
    ("CAN Capital", ["CAN CAPITAL", "CANCAP", "CAN CAPITAL MERCHANT"]),
    ("National Funding", ["NATIONAL FUNDING", "NATL FUNDING"]),
    ("Credibly", ["CREDIBLY", "CERES", "ADVANCIAL"]),
    ("Liberis", ["LIBERIS"]),
    ("Kapitus", ["KAPITUS", "KAPITAL"]),
    ("PayPal Working Capital", ["PAYPAL WC", "PAYPAL WORKING", "PPWC", "PAYPAL WORKING CAPITAL"]),
    ("Shopify Capital", ["SHOPIFY CAPITAL"]),
    ("Square Capital", ["SQUARE CAPITAL", "SQ CAPITAL", "BLOCK CAPITAL"]),
    ("Stripe Capital", ["STRIPE CAPITAL"]),
    ("Amazon Lending", ["AMAZON LENDING"]),
    ("Par Funding", ["PAR FUNDING", "PARFUNDING"]),
    ("Yellowstone Capital", ["YELLOWSTONE", "YELLOW STONE"]),
    ("Everest Business Funding", ["EVEREST", "EVEREST BUSINESS"]),
    ("Pearl Capital", ["PEARL CAPITAL"]),
    ("IOU Financial", ["IOU FINANCIAL"]),
    ("Biz2Credit", ["BIZ2CREDIT"]),
    ("Lendini", ["LENDINI"]),
    ("Expansion Capital Group", ["EXPANSION CAPITAL", "ECG FUNDING", "EXPANSION FUNDING"]),
    ("Greenbox Capital", ["GREENBOX", "GREEN BOX CAPITAL"]),
    ("United Capital Source", ["UNITED CAPITAL SOURCE"]),
    ("Delta Bridge Funding", ["DELTA BRIDGE"]),
    ("Fora Financial", ["FORA FINANCIAL"]),
    ("BHG Financial", ["BHG", "BANKERS HEALTHCARE"]),
    ("FinTap", ["FINTAP"]),
    ("Bluevine", ["BLUEVINE"]),
    ("Torro", ["TORRO", "TORRO FUNDING"]),
    ("Velocity Capital", ["VELOCITY CAPITAL"]),
    ("FundKite", ["FUNDKITE"]),
    ("CashBuoy", ["CASHBUOY"]),
    ("Revenued", ["REVENUED"]),
    ("Working Capital Funding", ["WORKING CAPITAL"]),
    ("Libertas Funding", ["LIBERTAS"]),
    ("Idea Financial", ["IDEA FINANCIAL"]),
    ("PIRS Capital", ["PIRS CAPITAL"]),
    ("Champion Funding", ["CHAMPION FUNDING"]),
    ("Flash Funding", ["FLASH FUNDING"]),
    ("Knight Capital", ["KNIGHT CAPITAL"]),
    ("Premium Merchant Funding", ["PMF", "PREMIUM MERCHANT"]),
    ("TVT Capital", ["TVT CAPITAL"]),
    ("Fundfi Merchant Funding", ["FUNDFI"]),
    ("Fox Capital", ["FOX CAPITAL"]),
    ("Advance Funds Network", ["AFN FUNDING", "ADVANCE FUNDS NETWORK"]),
    ("United Funding", ["UNITED FUNDING"]),
    ("Quick Bridge Funding", ["QUICKBRIDGE", "QUICK BRIDGE"]),
    ("1West", ["1WEST"]),
    ("Eagle Crest Funding", ["EAGLE CREST"]),
    ("Momentum Funding", ["MOMENTUM FUNDING"]),
    ("SmartBiz", ["SMARTBIZ"]),
    ("Breakout Capital", ["BREAKOUT CAPITAL"]),
    ("Headway Capital", ["HEADWAY CAPITAL"]),
    ("LoanMe", ["LOANME"]),
    ("Funding Circle", ["FUNDING CIRCLE"]),

    # ===== Loans / leases that are useful in MCA underwriting =====
    ("SBA Loan", ["SBA EIDL", "PAYMENT SBA", "SBA LOAN"]),
    ("AmEx Business Financing", ["AMEX EPAYMENT", "AMEX BUSINESS", "AMERICAN EXPRESS BUSINESS"]),


    ("JPM Advance", ["JPM ADVANCE", "JPMADVANCE"]),
    ("Insight Capital", ["INSIGHT CAPITAL", "INSIGHTCAPITAL"]),
    ("Seamless Funding", ["SEAMLESS FUNDING", "SEAMLESSFUNDING"]),
    ("NAV Kapital", ["NAV KAPITAL", "NAV KAPITAL LLC", "NAVKAPITAL"]),
    ("UFCE Funding", ["UFCE", "UFCE/ 8449090040", "8449090040"]),
    ("FDM001", ["FDM001"]),
    ("Small Business Advance", ["SMALL BUSINESS A", "SMALL BUSINESS ADV", "SMALLBUSINESSA"]),
    ("Headway Capital", ["HEADWAYCAPITAL", "HEADWAY CAPITAL", "HWCRCVBLS23", "HEADWAY"]),

    # ===== Generic fallback - keep last because it is broad =====
    ("General Funding / Capital", [
        "MERCHANT CASH", "ACH FUNDING", "DAILY REMITTANCE", "BUSINESS LOAN",
        "WORKING CAPITAL", "LOAN PAYMENT", "FUNDING", "CAPITAL", "ADVANCE"
    ]),
]

MCA_FALSE_POSITIVES = [
    # Banks / cards / common non-MCA merchants
    "BANK OF AMERICA", "CHASE", "WELLS FARGO", "CAPITAL ONE", "CAPITALONE",
    "AMEX", "AMERICAN EXPRESS", "GOOGLE", "VISA", "MASTERCARD", "DISCOVER",
    "CHECKCARD", "PURCHASE", "COSTCO", "ATT", "ADT", "COMCAST", "INSURANCE",
    "ADVANCE AUTO", "ADVANCED AUTO", "LOAN ACCT",

    # Payroll / HR / taxes
    "PAYROLL", "ADP", "PAYCHEX", "GUSTO", "VENSURE", "TAX", "IRS",

    # Transfers / P2P / internal money movement
    "ONLINE TRANSFER", "MOBILE TRANSFER", "INTERNET TRANSFER", "TRANSFER TO",
    "WIRE TRANSFER FROM", "WIRE TRANSFER FEE", "ZELLE", "VENMO", "CASHAPP", "CASH APP",

    # Logistics / trucking / factoring / carrier settlement descriptors
    "TRIUMPHPAY", "RXO", "TOTAL QUALITY", "TQL", "ARMSTRONG", "WORLDWIDE EXPRESS",
    "ITS LOGISTICS", "ENGLAND LOGISTIC", "SCOTLYNN", "ARL LOGISTICS", "JAKEBRAKE",
    "RELAY PAYMENTS", "LIMITED LOGISTIC", "EPES LOGISTIC", "NOLAN TRANSPORT",
    "HTS LOGISTICS", "COMMODITY TRANSP", "EXPRESS FREIGHT", "DESERT ROSE TRANSPORT",
]

# Industry/context words prevent the engine from treating normal operations as MCA debt.
# This keeps the system broad: it does not need every company name; it learns from descriptors.
OPERATIONAL_CONTEXT_WORDS = [
    "PAYROLL", "INVOICE", "SUPPLIER", "VENDOR", "LEASE", "RENT", "INSURANCE", "TAX",
    "TRANSFER", "ZELLE", "VENMO", "CASHAPP", "CHECKCARD", "PURCHASE", "POS", "CARD",
    "FUEL", "GAS", "UTILITY", "PHONE", "INTERNET", "COMCAST", "ATT", "ADT",
    "TRIUMPHPAY", "TRUCK", "TRUCKING", "TRANSPORT", "TRANSPORTATION", "LOGISTIC", "LOGISTICS",
    "FREIGHT", "CARRIER", "LOAD", "DISPATCH", "FACTOR", "FACTORING", "BROKER",
]

# Strong funding language. A recurring debit normally needs one of these OR a direct known lender.
TRUE_MCA_HINTS = [
    "MCA", "MERCHANT CASH", "DAILY REMITTANCE", "REMITTANCE", "PURCHASED RECEIVABLES",
    "PURCHASED RECEIVABLE", "FUTURE RECEIVABLE", "FUTURE RECEIVABLES", "REVENUE BASED",
    "REVENUE-BASED", "RBF", "ACH DEBIT", "ACH PAYMENT", "ACH PMT", "ACH WITHDRAWAL",
    "ACHPAYMENT", "ADV DEBIT", "DAILY ACH", "WEEKLY ACH", "SERVICING", "WORKING CAPITAL",
    "BUSINESS FUNDING", "MERCHANT FUNDING", "CAPITAL ADVANCE", "LOAN PAYMENT", "LOAN PMT",
]

# Strong keywords mean the memo itself sounds like an MCA or repayment product.
MCA_STRONG_KEYWORDS = [
    "MCA", "MERCHANT CASH", "DAILY REMITTANCE", "FUTURE RECEIVABLE", "PURCHASED RECEIVABLE",
    "ACH REMIT", "REMITTANCE", "BUSINESS FUNDING", "WORKING CAPITAL", "CAPITAL ADVANCE",
    "MERCHANT FUNDING", "REVENUE BASED", "REVENUE-BASED", "RBF", "ADV DEBIT",
]

# Company-name hints are not enough alone. They need recurrence + payment language + consistency.
MCA_COMPANY_HINTS = [
    "ADVANCE", "CAPITAL", "FUNDING", "FUNDER", "FINANCING", "FINANCE",
    "SERVICING", "RECEIVABLES", "HOLDINGS", "COLLECTIONS", "MERCHANT",
]

# Payment hints show the debit is a repayment/remittance, not just a vendor name.
MCA_PAYMENT_HINTS = [
    "ACH DEBIT", "ACH PAYMENT", "ACH PMT", "ACH WITHDRAWAL", "ACHPAYMENT",
    "DEBIT", "PAYMENT", "DAILY", "WEEKLY", "REPAYMENT", "LOAN PMT", "LOAN PAYMENT",
    "WITHDRAWAL", "REMIT", "REMITTANCE",
]

# Words that are broad and dangerous by themselves. They need more evidence before flagging.
BROAD_FUNDING_WORDS = ["CAPITAL", "FUNDING", "ADVANCE", "LOAN", "PAYMENT"]

REVENUE_HINTS = [
    "DEPOSIT", "CREDIT", "RTP DEPOSIT", "ZELLE", "SQUARE", "STRIPE", "SHOPIFY",
    "INVOICES PAID", "WIRE TRANSFER FROM", "ACH COLLEC", "CORP COLL", "INTUITPMTS",
    "INTUIT PYMT", "EDI PAYMTS", "MERCHANT SERVICES", "CARD SETTLEMENT", "BATCH",
    "CPC-CLIENT", "STATEFARM", "RESTORATIO", "OLYMPIASALESGEN", "SALE",
]

NON_REVENUE_HINTS = [
    "MOBILE TRANSFER", "INTERNET TRANSFER", "TRANSFER TO", "WIRE TRANSFER FEE",
    "LOAN", "ADVANCE", "CAPITAL", "PAYROLL", "ACH PMT", "PAYMENT", "VISA", "CHECK",
    "REFUND", "REVERSAL", "RETURNED", "OVERDRAFT", "FEE",
]

DEBIT_WORDS = ["DEBIT", "WITHDRAWAL", "WITHDRAWALS", "CHECKS", "CHECK", "PAYMENT", "PAID OUT", "MONEY OUT"]
CREDIT_WORDS = ["CREDIT", "DEPOSIT", "DEPOSITS", "ADDITION", "ADDITIONS", "MONEY IN"]
BALANCE_WORDS = ["BALANCE", "RUNNING", "LEDGER"]

DATE_RE = re.compile(r"(?P<date>\b\d{1,2}\s*[/-]\s*\d{1,2}(?:\s*[/-]\s*\d{2,4})?\b)")
MONEY_RE = re.compile(r"(?<!\w)\(?-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})\)?(?!\w)|(?<!\w)-?\$?\d+\.\d{2}(?!\w)")


def clean_money(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").replace("$", "").strip()
    if text in {"", "-", "nan", "None", ".00"}:
        return np.nan
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
    text = text.strip("()").lstrip("-")
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return np.nan
    number = float(m.group())
    return -number if negative else number


def money(x) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"${x:,.0f}"


def pct(x) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.0%}"


def normalize_text(text: str) -> str:
    text = str(text or "").upper()
    text = text.replace("WITHDRAWALSTOTALING", "WITHDRAWALS TOTALING")
    text = text.replace("WITHDRAWALS*TOTALING", "WITHDRAWALS TOTALING")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_description(desc: str) -> str:
    text = normalize_text(desc)
    text = re.sub(r"\bW\d+\b|\bNC\d+\b|\bCOP\d+\b|\bG-[A-Z0-9]+\b|\bBCI[A-Z0-9]+\b", " ", text)
    text = re.sub(r"\bRPP\d+\b|\bTRACE\b|\bID\b|\bCARD\b|#", " ", text)
    text = re.sub(r"\d{5,}", " ", text)
    text = re.sub(r"[^A-Z0-9 &./-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_statement_year(text: str) -> int | None:
    years = re.findall(r"\b20\d{2}\b", text or "")
    if years:
        return int(Counter(years).most_common(1)[0][0])
    short_years = re.findall(r"\b\d{1,2}/\d{1,2}/(\d{2})\b", text or "")
    if short_years:
        return 2000 + int(Counter(short_years).most_common(1)[0][0])
    return None


def parse_date(value, default_year: int):
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    m = DATE_RE.search(text)
    if not m:
        return pd.NaT
    date_text = m.group("date")
    parts = re.split(r"\s*[/-]\s*", date_text)
    try:
        month, day = int(parts[0]), int(parts[1])
        year = int(parts[2]) if len(parts) > 2 else default_year
        if year < 100:
            year += 2000
        return pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        return pd.NaT


def parse_statement_summary(text: str) -> dict:
    """Extract statement-level totals from many common bank layouts.

    The parser intentionally supports both old-style "TOTALING" summaries and
    modern table summaries like Bank of America/Chase/Wells lines.
    """
    clean = normalize_text(text)
    summary = {}

    # Old FirstBank style and generic count + amount lines
    patterns = [
        ("deposit_count", "statement_deposits", r"(\d+)\s+DEPOSITS? (?:AND OTHER )?(?:ADDITIONS|CREDITS)?\s*TOTALING\D*([\d,]+\.\d{2})"),
        ("withdrawal_count", "statement_withdrawals", r"(\d+)\s+(?:CHECKS? AND OTHER )?WITHDRAWALS?\s*TOTALING\D*([\d,]+\.\d{2})"),
        ("deposit_count", "statement_deposits", r"(\d+)\s+DEPOSITS/CREDITS\s+([\d,]+\.\d{2})"),
        ("withdrawal_count", "statement_withdrawals", r"(\d+)\s+CHECKS/DEBITS\s+([\d,]+\.\d{2})"),
    ]
    for count_key, amount_key, pat in patterns:
        m = re.search(pat, clean)
        if m:
            summary[count_key] = int(m.group(1))
            summary[amount_key] = abs(clean_money(m.group(2)))

    # Bank of America / Chase / Wells style summary rows without "totaling"
    amount_patterns = {
        "statement_deposits": [
            r"DEPOSITS? AND OTHER CREDITS\s+\$?(-?[\d,]+\.\d{2})",
            r"TOTAL DEPOSITS? AND OTHER CREDITS\s+\$?(-?[\d,]+\.\d{2})",
            r"DEPOSITS?\s+\$?(-?[\d,]+\.\d{2})",
            r"TOTAL CREDITS?\s+\$?(-?[\d,]+\.\d{2})",
        ],
        "statement_withdrawals": [
            r"WITHDRAWALS? AND OTHER DEBITS\s+\$?-?([\d,]+\.\d{2})",
            r"TOTAL WITHDRAWALS? AND OTHER DEBITS\s+\$?-?([\d,]+\.\d{2})",
            r"TOTAL DEBITS?\s+\$?-?([\d,]+\.\d{2})",
        ],
        "statement_checks": [
            r"\bCHECKS\s+\$?-?([\d,]+\.\d{2})",
            r"TOTAL CHECKS\s+\$?-?([\d,]+\.\d{2})",
        ],
        "service_fees": [
            r"SERVICE FEES\s+\$?-?([\d,]+\.\d{2})",
            r"TOTAL SERVICE FEES\s+\$?-?([\d,]+\.\d{2})",
        ],
    }
    for key, pats in amount_patterns.items():
        for pat in pats:
            m = re.search(pat, clean)
            if m:
                summary[key] = abs(clean_money(m.group(1)))
                break

    count_patterns = {
        "deposit_count": r"# OF DEPOSITS/CREDITS:\s*(\d+)",
        "withdrawal_count": r"# OF WITHDRAWALS/DEBITS:\s*(\d+)",
        "days_in_cycle": r"# OF DAYS IN CYCLE:\s*(\d+)",
    }
    for key, pat in count_patterns.items():
        m = re.search(pat, clean)
        if m:
            summary[key] = int(m.group(1))

    scalar_patterns = {
        "min_balance": r"MINIMUM BALANCE(?: OF)?\D*([\d,]+\.\d{2})",
        "opening_balance": r"(?:BEGINNING|OPENING|PREVIOUS) BALANCE(?: ON [A-Z]+ \d{1,2}, \d{4})?\s+\$?([\d,]+\.\d{2})",
        "closing_balance": r"(?:ENDING|CLOSING) BALANCE(?: ON [A-Z]+ \d{1,2}, \d{4})?\s+\$?([\d,]+\.\d{2})",
        "avg_ledger_balance": r"AVG(?:ERAGE)? LEDGER BALANCE:?\s*\$?([\d,]+(?:\.\d{2})?)",
        "avg_collected_balance": r"AVG(?:ERAGE)? COLLECTED BALANCE:?\s*\$?([\d,]+(?:\.\d{2})?)",
    }
    for key, pat in scalar_patterns.items():
        m = re.search(pat, clean)
        if m:
            summary[key] = clean_money(m.group(1))
    return summary


def extract_pdf_text(data: bytes) -> str:
    if pdfplumber is None:
        return ""
    chunks = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
    return "\n".join(chunks)


def extract_pdf_tables(data: bytes, default_year: int) -> pd.DataFrame:
    if pdfplumber is None:
        return pd.DataFrame()
    rows = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "intersection_tolerance": 5,
                }) or []
                if not tables:
                    tables = page.extract_tables() or []
            except Exception:
                tables = []
            for table in tables:
                rows.extend(rows_from_table(table, default_year, page_no))
    return pd.DataFrame(rows)


def rows_from_table(table, default_year: int, page_no: int) -> list[dict]:
    out = []
    if not table or len(table) < 2:
        return out
    raw = [["" if c is None else str(c).replace("\n", " ").strip() for c in r] for r in table]
    header_idx = None
    for i, r in enumerate(raw[:5]):
        joined = normalize_text(" ".join(r))
        if "DATE" in joined and any(w in joined for w in ["DESCRIPTION", "DETAIL", "TRANSACTION", "MEMO"]):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0
    headers = [normalize_text(c) for c in raw[header_idx]]
    body = raw[header_idx + 1:]

    def col_score(names):
        best = None
        best_score = 0
        for idx, h in enumerate(headers):
            score = sum(1 for n in names if n in h)
            if score > best_score:
                best, best_score = idx, score
        return best

    date_col = col_score(["DATE", "POSTED"])
    desc_col = col_score(["DESCRIPTION", "DETAIL", "TRANSACTION", "MEMO", "NAME"])
    amount_col = col_score(["AMOUNT"])
    debit_col = col_score(DEBIT_WORDS)
    credit_col = col_score(CREDIT_WORDS)
    balance_col = col_score(BALANCE_WORDS)

    for r in body:
        joined = " ".join(r).strip()
        if not joined or not DATE_RE.search(joined):
            continue
        date_val = parse_date(r[date_col] if date_col is not None and date_col < len(r) else joined, default_year)
        if pd.isna(date_val):
            continue
        description = r[desc_col] if desc_col is not None and desc_col < len(r) else joined
        amount = np.nan
        section = "unknown"

        if debit_col is not None and debit_col < len(r) and not pd.isna(clean_money(r[debit_col])):
            amount = -abs(clean_money(r[debit_col])); section = "debit"
        if credit_col is not None and credit_col < len(r) and not pd.isna(clean_money(r[credit_col])):
            amount = abs(clean_money(r[credit_col])); section = "credit"
        if pd.isna(amount) and amount_col is not None and amount_col < len(r):
            amount = clean_money(r[amount_col])
            section = "debit" if amount < 0 else "credit"
        if pd.isna(amount):
            money_cells = [(i, clean_money(c)) for i, c in enumerate(r) if not pd.isna(clean_money(c))]
            if not money_cells:
                continue
            # Prefer a money cell that is not balance. If only one exists, use it.
            candidates = [(i, v) for i, v in money_cells if i != balance_col]
            i, amount = candidates[-1] if candidates else money_cells[0]
            amount = float(amount)
            amount = -abs(amount) if any(w in normalize_text(" ".join(headers[max(0, i-1):i+1])) for w in DEBIT_WORDS) else amount
            section = "debit" if amount < 0 else "credit"

        # Avoid making running balance the transaction amount.
        if balance_col is not None and len([c for c in r if MONEY_RE.search(c or "")]) == 1:
            continue

        out.append({
            "date": date_val,
            "description": re.sub(r"\s+", " ", description).strip(),
            "amount": float(amount),
            "section": section,
            "source_line": joined,
            "source": f"table page {page_no}",
        })
    return out


def ocr_pdf_text_if_available(data: bytes) -> str:
    if fitz is None or pytesseract is None or Image is None:
        return ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        chunks = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.open(BytesIO(pix.tobytes("png")))
            chunks.append(pytesseract.image_to_string(img))
        return "\n".join(chunks)
    except Exception:
        return ""


def infer_section(line_upper: str, current: str | None) -> str | None:
    if any(w in line_upper for w in ["DEPOSITS", "CREDITS", "ADDITIONS", "MONEY IN"]):
        return "credit"
    if any(w in line_upper for w in ["WITHDRAWALS", "DEBITS", "CHECKS", "MONEY OUT", "PAYMENTS"]):
        return "debit"
    if any(w in line_upper for w in ["DAILY BALANCE", "ENDING BALANCE", "END OF STATEMENT", "RECONCILIATION", "SUMMARY"]):
        return None
    return current


def parse_text_line(line: str, default_year: int, section: str | None) -> dict | None:
    upper_line = normalize_text(line)
    if any(skip in upper_line for skip in ["MONTHLY FEE WILL BEGIN", "IMPORTANT INFORMATION", "CHANGE IN TERMS", "ACCOUNT NUMBER", "STATEMENT DATE"]):
        return None
    m = DATE_RE.search(line)
    if not m:
        return None
    date_value = parse_date(m.group("date"), default_year)
    if pd.isna(date_value):
        return None
    rest = (line[:m.start()] + " " + line[m.end():]).strip()
    money_matches = list(MONEY_RE.finditer(rest))
    if not money_matches:
        return None
    amounts = [clean_money(mm.group()) for mm in money_matches]
    amounts = [a for a in amounts if not pd.isna(a)]
    if not amounts:
        return None

    # Common statement lines are: date desc debit credit balance OR date desc amount balance.
    # Use section when known. Otherwise infer from signs/keywords.
    amount = amounts[-1]
    if len(amounts) >= 2:
        # Last number is often running balance, transaction amount is often second-to-last.
        amount = amounts[-2]
    if section == "debit":
        amount = -abs(amount)
    elif section == "credit":
        amount = abs(amount)
    else:
        upper = normalize_text(rest)
        if any(w in upper for w in ["WITHDRAWAL", "DEBIT", "PAYMENT", "CHECK", "ACH DEBIT", "PURCHASE"]):
            amount = -abs(amount)
        elif any(w in upper for w in ["DEPOSIT", "CREDIT", "ACH CREDIT", "RTP", "WIRE FROM"]):
            amount = abs(amount)
        elif str(money_matches[-1].group()).strip().startswith("-") or str(money_matches[-1].group()).strip().startswith("("):
            amount = -abs(amount)

    amt_match = money_matches[-2] if len(money_matches) >= 2 else money_matches[-1]
    description = (rest[:amt_match.start()] + " " + rest[amt_match.end():]).strip(" -|*")
    description = MONEY_RE.sub(" ", description)
    description = re.sub(r"\s+", " ", description).strip()
    if len(description) < 2:
        return None
    return {
        "date": date_value,
        "description": description,
        "amount": float(amount),
        "section": "debit" if amount < 0 else "credit",
        "source_line": line,
        "source": "text line",
    }


def parse_pdf_transactions_from_text(text: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    statement = parse_statement_summary(text)
    default_year = extract_statement_year(text) or pd.Timestamp.today().year
    daily_balances = parse_daily_balances(text, default_year)
    rows = []
    section = None
    current = ""

    def flush():
        nonlocal current
        if current:
            tx = parse_text_line(current, default_year, section)
            if tx:
                rows.append(tx)
        current = ""

    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        upper = normalize_text(line)
        new_section = infer_section(upper, section)
        if new_section != section and not DATE_RE.search(line):
            flush()
            section = new_section
            continue
        if "DATE" in upper and any(w in upper for w in ["DESCRIPTION", "AMOUNT", "BALANCE"]):
            continue
        if DATE_RE.search(line):
            flush()
            current = line
        elif current:
            current += " " + line
    flush()
    tx = pd.DataFrame(rows)
    if not tx.empty:
        tx = tx.drop_duplicates(subset=["date", "description", "amount"]).reset_index(drop=True)
    return tx, statement, daily_balances


def parse_daily_balances(text: str, default_year: int) -> pd.DataFrame:
    balances = []
    in_daily = False
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        upper = normalize_text(line)
        if "DAILY" in upper and "BALANCE" in upper:
            in_daily = True
            continue
        if in_daily and any(x in upper for x in ["END OF STATEMENT", "RECONCILIATION", "ACCOUNT SUMMARY"]):
            break
        if not in_daily:
            continue
        for m in re.finditer(r"(\d{1,2})[/-](\d{1,2})(?:[/-]\d{2,4})?\s+(-?[\d,]+\.\d{2})", line):
            try:
                balances.append({
                    "day": pd.Timestamp(year=default_year, month=int(m.group(1)), day=int(m.group(2))).date(),
                    "balance": clean_money(m.group(3)),
                })
            except Exception:
                pass
    return pd.DataFrame(balances)


def combine_parsed_transactions(*frames: pd.DataFrame) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame(columns=["date", "description", "amount", "section", "source_line", "source"])
    tx = pd.concat(valid, ignore_index=True)
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
    tx = tx.dropna(subset=["date", "amount"])
    tx["description"] = tx["description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    tx["section"] = np.where(tx["amount"] < 0, "debit", "credit")
    tx["dedupe_desc"] = tx["description"].apply(normalize_description)
    tx = tx.sort_values(["date", "amount", "dedupe_desc"]).drop_duplicates(
        subset=["date", "amount", "dedupe_desc"], keep="first"
    )
    return tx.drop(columns=["dedupe_desc"]).sort_values("date").reset_index(drop=True)


def read_pdf(uploaded_file) -> tuple[pd.DataFrame, dict, pd.DataFrame, str]:
    data = uploaded_file.read()
    text = extract_pdf_text(data)
    default_year = extract_statement_year(text) or pd.Timestamp.today().year

    table_tx = extract_pdf_tables(data, default_year)
    text_tx, statement, daily_balances = parse_pdf_transactions_from_text(text)
    tx = combine_parsed_transactions(table_tx, text_tx)
    method_parts = []
    if not table_tx.empty:
        method_parts.append(f"tables={len(table_tx)}")
    if not text_tx.empty:
        method_parts.append(f"text={len(text_tx)}")

    if len(tx) < 5:
        ocr_text = ocr_pdf_text_if_available(data)
        if ocr_text.strip():
            ocr_tx, ocr_statement, ocr_balances = parse_pdf_transactions_from_text(ocr_text)
            tx = combine_parsed_transactions(tx, ocr_tx)
            statement = {**statement, **ocr_statement}
            if daily_balances.empty and not ocr_balances.empty:
                daily_balances = ocr_balances
            if not ocr_tx.empty:
                method_parts.append(f"ocr={len(ocr_tx)}")

    if len(tx) < 5:
        raise ValueError(
            "Could not parse enough transactions. Try exporting CSV/XLSX from the bank portal, "
            "or install OCR/Tesseract if this is a scanned PDF."
        )

    return tx, statement, daily_balances, (
        f"Parsed {len(tx)} transactions ({', '.join(method_parts) or 'text'}); "
        f"summary fields found: {list(statement.keys())}"
    )


def read_csv_excel(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    df = pd.read_csv(BytesIO(data)) if name.endswith(".csv") else pd.read_excel(BytesIO(data))
    cols = {str(c).lower(): c for c in df.columns}
    date_col = next((c for k, c in cols.items() if "date" in k or "posted" in k), None)
    desc_col = next((c for k, c in cols.items() if any(x in k for x in ["desc", "memo", "merchant", "name", "details"])), None)
    amt_col = next((c for k, c in cols.items() if "amount" in k), None)
    debit_col = next((c for k, c in cols.items() if any(x in k for x in ["debit", "withdrawal", "money out"])), None)
    credit_col = next((c for k, c in cols.items() if any(x in k for x in ["credit", "deposit", "money in"])), None)
    if not date_col or not desc_col or not (amt_col or debit_col or credit_col):
        raise ValueError("CSV/XLSX needs date, description, and either amount or debit/credit columns.")
    if amt_col:
        amounts = df[amt_col].apply(clean_money)
    else:
        debits = df[debit_col].apply(clean_money) if debit_col else pd.Series(np.nan, index=df.index)
        credits = df[credit_col].apply(clean_money) if credit_col else pd.Series(np.nan, index=df.index)
        amounts = credits.fillna(0) - debits.fillna(0).abs()
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "description": df[desc_col].astype(str),
        "amount": pd.to_numeric(amounts, errors="coerce"),
    }).dropna(subset=["date", "amount"])
    out["section"] = np.where(out["amount"] >= 0, "credit", "debit")
    out["source_line"] = out["description"]
    out["source"] = "csv/excel"
    return out, {}, pd.DataFrame(), "CSV/Excel loaded"



def has_operational_context(text: str) -> bool:
    norm = normalize_description(text)
    return any(word in norm for word in OPERATIONAL_CONTEXT_WORDS)


def has_true_mca_language(text: str) -> bool:
    norm = normalize_description(text)
    return any(word in norm for word in TRUE_MCA_HINTS)


def detect_mca_funder(desc: str) -> str | None:
    """Return a known MCA/funding name from the transaction description.

    This is intentionally conservative. Broad words like CAPITAL/FUNDING/ADVANCE
    are not enough when the descriptor looks operational (trucking, payroll,
    transfers, vendors, card purchases). That prevents daily-vendor false positives.
    """
    norm = normalize_description(desc)

    # Direct known lender match, except for the final broad generic fallback.
    for funder, keys in MCA_RULES:
        if funder == "General Funding / Capital":
            continue
        if any(key in norm for key in keys):
            if any(fp in norm for fp in MCA_FALSE_POSITIVES):
                return None
            return funder

    # Generic fallback must have funding language and must not look operational.
    generic_keys = []
    for funder, keys in MCA_RULES:
        if funder == "General Funding / Capital":
            generic_keys = keys
            break

    if any(key in norm for key in generic_keys):
        if has_operational_context(norm):
            return None
        # Avoid calling a company MCA just because it contains CAPITAL.
        if any(word in norm for word in ["MERCHANT CASH", "DAILY REMITTANCE", "WORKING CAPITAL", "ACH FUNDING"]):
            return "General Funding / Capital"
        if sum(1 for word in BROAD_FUNDING_WORDS if word in norm) >= 2:
            return "General Funding / Capital"

    return None


def canonical_funder_key(desc: str) -> str:
    """Normalize descriptions so recurring debit positions group together.

    This version is intentionally more aggressive for MCA underwriting. It removes
    ACH plumbing words, IDs, trace numbers, class codes, dates, and amount noise so
    descriptors like "Withdrawal ACH JPM ADVANCE TYPE: ONLINE PMT ID: 1016207445 CCD"
    all collapse into "JPM ADVANCE".
    """
    raw = normalize_description(desc)

    funder = detect_mca_funder(raw)
    if funder:
        return funder

    text = raw

    noise_phrases = [
        "WITHDRAWAL ACH", "EXTERNAL WITHDRAWAL", "DESCRIPTIVE WITHDRAWAL",
        "ACH WITHDRAWAL", "ACH DEBIT", "ACH PAYMENT", "ACH PMT", "ACHPAYMENT",
        "TYPE ONLINE PMT", "TYPE ACH PMT", "TYPE RETRY PYMT", "TYPE DEBIT",
        "TYPE PAYMENT", "TYPE WEB PAY", "TYPE EFT DEBIT", "ENTRY CLASS CODE",
        "CLASS CODE", "CCD", "WEB", "PPD", "CTX", "RPP", "TRACE", "TRACER",
        "ONLINE PMT", "RETRY PYMT", "ACH", "PMT", "PYMT", "PAYMENT", "DEBIT",
        "WITHDRAWAL", "TYPE", "ID", "CO ID", "INDN", "DES",
    ]
    for phrase in noise_phrases:
        text = text.replace(phrase, " ")

    text = re.sub(r"\b[A-Z]*\d{4,}[A-Z0-9-]*\b", " ", text)
    text = re.sub(r"\b\d{3,}\b", " ", text)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", " ", text)
    text = re.sub(r"[^A-Z0-9 &./-]+", " ", text)

    # Preserve common lender phrases that include generic words.
    special_patterns = [
        ("JPM ADVANCE", ["JPM", "ADVANCE"]),
        ("INSIGHT CAPITAL", ["INSIGHT", "CAPITAL"]),
        ("SEAMLESS FUNDING", ["SEAMLESS", "FUNDING"]),
        ("NAV KAPITAL", ["NAV", "KAPITAL"]),
        ("SMALL BUSINESS ADVANCE", ["SMALL", "BUSINESS"]),
        ("HEADWAY CAPITAL", ["HEADWAY"]),
        ("UFCE FUNDING", ["UFCE"]),
        ("FDM001", ["FDM"]),
    ]
    for name, required in special_patterns:
        if all(part in raw for part in required) or all(part in text for part in required):
            return name

    stop_words = {
        "THE", "AND", "INC", "LLC", "LTD", "CO", "COMPANY", "CORP", "CORPORATION",
        "BANK", "NA", "ONLINE", "WEB", "PORTAL", "BILL", "BILLING", "BATCH",
        "BUSINESS", "SMALL", "ACH", "PMT", "PYMT", "PAYMENT", "DEBIT", "TYPE",
        "ID", "CODE", "ENTRY", "CLASS"
    }
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]

    if not words:
        return raw[:50]

    return " ".join(words[:4])


def is_revenue(desc: str, amount: float) -> bool:
    if amount <= 0:
        return False
    norm = normalize_description(desc)
    if any(bad in norm for bad in NON_REVENUE_HINTS):
        return False
    return any(hint in norm for hint in REVENUE_HINTS)


def infer_frequency(dates: pd.Series) -> str:
    d = pd.to_datetime(dates).sort_values().drop_duplicates()
    if len(d) == 1:
        return "monthly/one seen"
    gaps = d.diff().dt.days.dropna()
    median_gap = float(gaps.median()) if len(gaps) else 999
    # MCA daily ACH normally appears every business day, so gaps of 1-3 days are normal.
    if len(d) >= 7 and median_gap <= 3:
        return "daily"
    if 5 <= median_gap <= 9:
        return "weekly"
    if 12 <= median_gap <= 18:
        return "biweekly"
    if 25 <= median_gap <= 35:
        return "monthly"
    if len(d) >= 4 and median_gap <= 4:
        return "daily-ish"
    return "irregular"


def expected_monthly_count(freq: str, observed_count: int) -> float:
    freq = str(freq).lower()
    if "daily" in freq:
        return 21.0
    if "weekly" in freq and "bi" not in freq:
        return 4.33
    if "biweekly" in freq:
        return 2.16
    if "monthly" in freq:
        return 1.0
    return max(float(observed_count), 1.0)


def mca_score_for_group(descs: pd.Series, amounts: pd.Series, dates: pd.Series) -> tuple[int, str]:
    """Confidence-based MCA / funding position detector.

    This intentionally does NOT rely on a single keyword. It scores behavior:
    known lender + funding language + repayment language + repetition + frequency
    + payment consistency, then subtracts for payroll, transfers, cards, logistics,
    utilities, and other operating activity.

    Return: (0-100 score, reasons string)
    """
    desc_list = descs.astype(str).tolist()
    text = normalize_description(" ".join(desc_list[:30]))
    count = int(len(amounts))
    abs_amounts = amounts.abs().astype(float)
    avg_amount = float(abs_amounts.mean()) if count else 0.0
    std_dev = float(abs_amounts.std(ddof=0)) if count else 0.0
    variation = (std_dev / avg_amount) if avg_amount > 0 else 999.0
    freq = infer_frequency(dates)

    known_name = detect_mca_funder(text)
    known = known_name is not None

    has_strong_mca_keyword = any(k in text for k in MCA_STRONG_KEYWORDS)
    has_company_hint = any(k in text for k in MCA_COMPANY_HINTS)
    has_payment_hint = any(k in text for k in MCA_PAYMENT_HINTS)
    operational_context = has_operational_context(text)

    score = 0
    reasons = []

    # 1) Known lender is the strongest signal, but still subject to obvious false positives.
    if known:
        score += 60
        reasons.append(f"known lender: {known_name}")

    # 2) True MCA language. This catches unknown lenders whose memo says what it is.
    if has_strong_mca_keyword:
        score += 35
        reasons.append("strong MCA/funding language")

    # 3) Generic company words only count when paired with repayment language.
    if has_company_hint and has_payment_hint:
        score += 20
        reasons.append("funding-style company name plus payment language")
    elif has_company_hint:
        score += 5
        reasons.append("funding-style company word only")

    # 4) Repetition and frequency. Recurrence alone is never enough.
    if count >= 15:
        score += 25
        reasons.append("heavy repeated debits")
    elif count >= 5:
        score += 15
        reasons.append("recurring debits")
    elif count >= 3:
        score += 8
        reasons.append("some repeated debits")

    if "daily" in str(freq).lower():
        score += 20
        reasons.append("daily ACH pattern")
    elif str(freq).lower() in {"weekly", "biweekly"}:
        score += 10
        reasons.append(f"{freq} pattern")

    # 5) MCA payments are often fixed or near-fixed.
    if count >= 3 and avg_amount > 0:
        if variation <= 0.08:
            score += 20
            reasons.append("fixed payment amount")
        elif variation <= 0.15:
            score += 10
            reasons.append("mostly fixed payment amount")

    # 6) Operating context should knock down unknowns. Known lenders are not removed here,
    # because some descriptors include words like PAYMENT or LOAN.
    if operational_context and not known:
        score -= 40
        reasons.append("operating/vendor/payroll/industry context")

    if any(fp in text for fp in MCA_FALSE_POSITIVES) and not known:
        score -= 45
        reasons.append("false-positive keyword")

    # 7) Transfers, card purchases, payroll, and logistics/factoring should almost never be MCA.
    hard_operating_words = [
        "ONLINE TRANSFER", "MOBILE TRANSFER", "INTERNET TRANSFER", "TRANSFER TO", "WIRE TRANSFER",
        "ZELLE", "VENMO", "CASHAPP", "CASH APP", "PAYROLL", "CHECKCARD", "PURCHASE", "VISA",
        "MASTERCARD", "AMEX", "TRIUMPHPAY", "LOGISTICS", "TRANSPORT", "TRANSPORTATION", "FREIGHT",
        "CARRIER", "FUEL", "GAS", "INSURANCE", "TAX", "IRS", "ADP", "PAYCHEX", "VENSURE",
    ]
    if any(w in text for w in hard_operating_words) and not known:
        score -= 50
        reasons.append("hard operating/transfer/card/payroll filter")

    # 8) One-off big wires or reversals are usually not active positions.
    if avg_amount > 15000 and "irregular" in str(freq).lower() and count < 4:
        score -= 35
        reasons.append("large irregular debit")
    if "REVERSAL" in text and count < 3:
        score -= 50
        reasons.append("one-off reversal/payoff")

    # 9) Final safeguards so the engine scales across industries.
    # No lender + no funding/payment language means recurrence cannot become MCA by itself.
    if not known and not has_strong_mca_keyword and not (has_company_hint and has_payment_hint):
        score = min(score, 35)
        reasons.append("capped: recurrence without funding/payment language")

    score = max(0, min(100, int(score)))
    if score >= 80:
        reasons.append("confidence=HIGH")
    elif score >= 60:
        reasons.append("confidence=MEDIUM")
    elif score >= 40:
        reasons.append("confidence=LOW/REVIEW")
    else:
        reasons.append("confidence=IGNORE")

    return score, ", ".join(dict.fromkeys([r for r in reasons if r]))


# -----------------------------
# Universal recurring-position engine
# -----------------------------

POSITION_CATEGORIES = {
    "MCA / Funding": [
        "MCA", "MERCHANT CASH", "DAILY REMITTANCE", "REMITTANCE", "FUTURE RECEIVABLE",
        "PURCHASED RECEIVABLE", "REVENUE BASED", "RBF", "ADVANCE", "WORKING CAPITAL",
        "BUSINESS FUNDING", "MERCHANT FUNDING", "CAPITAL ADVANCE", "SERVICING", "RECEIVABLE",
    ],
    "Term Loan / Credit": [
        "LOAN", "LOANPYMT", "LOAN PMT", "LOAN PAYMENT", "SOFI", "AMEX EPAYMENT",
        "APPLECARD", "APPLE CARD", "DISCOVER", "ALLY", "CAPITAL ONE", "LENDING",
    ],
    "Equipment Lease": [
        "LEASE", "LEASING", "LEASE SERVICES", "FINANCIAL PACIFIC", "FINOVA", "EQUIPMENT",
    ],
    "Merchant Processor / Fees": [
        "WORLDPAY", "MERCHANTSERVCS", "MERCHANT SERVICES", "CARD SETTLEMENT", "STRIPE",
        "SQUARE", "INTUIT", "TRAN FEE", "PROCESSING", "BATCH", "CHERRY - FUNDING",
    ],
    "Payroll / HR": [
        "ADP", "PAYROLL", "PAYCHEX", "GUSTO", "VENSURE", "DEFINTI", "DEFINITI",
    ],
    "Marketing / Advertising": [
        "GOOGLE", "GOOGLE ADS", "FACEBK", "FACEBOOK", "META", "33 MILE RADIUS",
        "NETWORX", "HOUZZ", "EL OCAL", "E LOCAL", "INQUIRLY", "EXPERTISE MARKETPLACE",
    ],
    "Insurance": [
        "INSURANCE", "FREEDOM LIFE", "LIBERTY MUTUAL", "FLBLUE", "BLUE CROSS", "PINCACOL", "PINNACOL",
    ],
    "Utilities / Telecom": [
        "FPL", "COMCAST", "SPECTRUM", "ATT", "AT&T", "CENTURYLINK", "LUMEN", "BLACK HILLS", "WASTE",
    ],
    "Rent / Real Estate": [
        "RENT", "LANDLORD", "MALLS LLC", "PROPERTY", "SELF STORAGE", "STORAGE",
    ],
    "Tax / Government": [
        "IRS", "TAX", "DEPT OF", "STATE OF", "CITY OF", "ASSESSME", "EFTPS",
    ],
    "Internal Transfer / Owner Draw": [
        "ONLINE TRANSFER", "MOBILE TRANSFER", "INTERNET TRANSFER", "TRANSFER TO", "TRANSFER FROM",
        "WIRE OUT", "WIRE IN", "DOMESTIC WIRE", "ZELLE", "VENMO", "CASHAPP", "CASH APP",
    ],
    "Vendor / Supplier": [
        "DENTAL", "PATTERSON", "ALIGN TECHNOLOGY", "DC DENTAL", "ATLANTA DENTAL", "SUPPLY",
        "SUPPLIER", "VENDOR", "LOWES", "HOME DEPOT", "AMAZON", "EBAY", "PAYPAL",
    ],
}

POSITION_EXCLUDE_WORDS = [
    "REFUND", "REVERSAL", "RETURNED", "REJECTED", "FEE WAIVER", "BEGINNING BALANCE", "ENDING BALANCE",
]


def classify_position_category(desc: str) -> str:
    text = normalize_description(desc)
    known = detect_mca_funder(text)
    if known:
        return "MCA / Funding"
    for category, keys in POSITION_CATEGORIES.items():
        if any(k in text for k in keys):
            return category
    return "Recurring Debit / Review"


def position_score_for_group(descs: pd.Series, amounts: pd.Series, dates: pd.Series) -> tuple[int, str, str]:
    """Find recurring MCA/funding positions and other obligations.

    This engine does not require an exact lender match. It scores:
    repeated debit count, daily/weekly cadence, fixed amount, ACH/CCD/RPP language,
    funding-style company names, and known lender names.
    """
    desc_list = descs.astype(str).tolist()
    raw_text = normalize_description(" ".join(desc_list[:40]))
    count = int(len(amounts))
    abs_amounts = amounts.abs().astype(float)
    avg_amount = float(abs_amounts.mean()) if count else 0.0
    std_dev = float(abs_amounts.std(ddof=0)) if count else 0.0
    variation = (std_dev / avg_amount) if avg_amount > 0 else 999.0
    freq = infer_frequency(dates)

    known_name = detect_mca_funder(raw_text)
    known_mca = known_name is not None

    funding_words = [
        "MCA", "ADVANCE", "CAPITAL", "FUNDING", "FUNDER", "FINANCING", "FINANCE",
        "RECEIVABLE", "RECEIVABLES", "RCVBLS", "HOLDINGS", "KAPITAL", "SERVICING",
        "MERCHANT", "REMIT", "REMITTANCE", "DAILY", "WORKING CAPITAL", "BUSINESS FUNDING",
        "JPM ADVANCE", "INSIGHT CAPITAL", "SEAMLESS FUNDING", "NAV KAPITAL", "UFCE", "FDM"
    ]
    ach_words = [
        "ACH", "CCD", "RPP", "DEBIT", "WITHDRAWAL", "PAYMENT", "PMT", "PYMT",
        "ONLINE PMT", "DAILY", "WEEKLY", "RETRY PYMT", "WEB PAY"
    ]
    operating_words = [
        "PAYROLL", "ADP", "PAYCHEX", "VENSURE", "GUSTO", "ZELLE", "VENMO", "CASHAPP",
        "CASH APP", "ONLINE TRANSFER", "MOBILE TRANSFER", "INTERNET TRANSFER",
        "TRANSFER TO", "WIRE FEE", "WIRE OUT", "CHECKCARD", "DEBIT CARD", "POS",
        "PURCHASE", "CARD", "LOWES", "HOME DEPOT", "AMAZON", "APPLE.COM", "SPOTIFY",
        "NETFLIX", "GOOGLE", "FACEBK", "COMCAST", "ATT", "INSURANCE", "ALFA MUTUAL",
        "NORTHWESTERN", "PROG FREEDOM", "TRACTOR SUPPLY", "AGRICREDIT", "GM FINANCIAL",
        "TD AUTO FINANCE", "SHEFFIELD", "INTUIT TRAN FEE", "TRAN FEE", "MERCHANT SERV",
        "CPS MERCHANT", "INTUIT "
    ]

    has_funding_word = any(w in raw_text for w in funding_words)
    has_ach_word = any(w in raw_text for w in ach_words)
    has_operating_word = any(w in raw_text for w in operating_words)

    category = classify_position_category(raw_text)
    if known_mca or (has_funding_word and has_ach_word):
        category = "MCA / Funding"
    elif "LOAN" in raw_text or "FINANCING" in raw_text:
        category = "Term Loan / Credit"

    score = 0
    reasons = []

    if any(w in raw_text for w in POSITION_EXCLUDE_WORDS):
        score -= 60
        reasons.append("reversal/refund/rejected/balance line")

    if known_mca:
        score += 45
        reasons.append(f"known MCA/funding lender: {known_name}")

    if has_funding_word:
        score += 25
        reasons.append("funding-style descriptor")

    if has_ach_word:
        score += 15
        reasons.append("ACH/debit/payment language")

    if count >= 15:
        score += 35
        reasons.append("heavy repeated debits")
    elif count >= 8:
        score += 28
        reasons.append("strong recurring debits")
    elif count >= 5:
        score += 20
        reasons.append("recurring debits")
    elif count >= 3:
        score += 12
        reasons.append("some repetition")
    elif count == 2:
        score += 5
        reasons.append("two occurrences")

    if "daily" in str(freq).lower():
        score += 25
        reasons.append("daily/business-day pattern")
    elif str(freq).lower() in {"weekly", "biweekly", "monthly"}:
        score += 14
        reasons.append(f"{freq} pattern")

    if count >= 2 and avg_amount > 0:
        if variation <= 0.03:
            score += 25
            reasons.append("fixed amount")
        elif variation <= 0.08:
            score += 18
            reasons.append("near-fixed amount")
        elif variation <= 0.20:
            score += 8
            reasons.append("mostly fixed amount")

    if count >= 5 and "daily" in str(freq).lower() and variation <= 0.10 and has_ach_word:
        score += 25
        reasons.append("hidden position pattern: daily fixed ACH")

    if has_operating_word and not known_mca and not has_funding_word:
        score -= 45
        reasons.append("operational/vendor/payroll/card context")

    if any(w in raw_text for w in ["DEBIT CARD", "CHECKCARD", "POINT OF SALE", "POS", "PURCHASE"]) and count < 10:
        score -= 30
        reasons.append("card/POS activity")

    if avg_amount > 15000 and count < 3:
        score -= 25
        reasons.append("large one-off debit")

    if not known_mca and not has_funding_word and category == "Recurring Debit / Review":
        if not (count >= 5 and "daily" in str(freq).lower() and variation <= 0.10 and has_ach_word):
            score = min(score, 50)
            reasons.append("capped: recurring but no funding language")

    score = max(0, min(100, int(score)))
    return score, ", ".join(dict.fromkeys([r for r in reasons if r])), category


def build_positions(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    debits = tx[tx["amount"] < 0].copy()
    empty_cols = [
        "Position", "Position Category", "Frequency", "Avg Debit", "Est Monthly", "# Debits",
        "Total Debited", "First Date", "Last Date", "Confidence Score", "Confidence", "Why Flagged"
    ]
    if debits.empty:
        return pd.DataFrame(columns=empty_cols), tx.assign(mca_funder=None, is_mca_debit=False, mca_score=0, position_category=None)

    debits["position_key"] = debits["description"].apply(canonical_funder_key)
    rows = []
    for key, g in debits.groupby("position_key", dropna=False):
        score, why, category = position_score_for_group(g["description"], g["amount"], g["date"])
        if score < 45:
            continue
        known_name = detect_mca_funder(" ".join(g["description"].astype(str)))
        freq = infer_frequency(g["date"])
        avg_debit = float(g["amount"].abs().mean())
        est_monthly = avg_debit * expected_monthly_count(freq, len(g))
        rows.append({
            "Position": known_name or str(key).title(),
            "Position Category": category,
            "Frequency": freq,
            "Avg Debit": round(avg_debit, 2),
            "Est Monthly": round(est_monthly, 2),
            "# Debits": int(len(g)),
            "Total Debited": round(float(g["amount"].abs().sum()), 2),
            "First Date": pd.to_datetime(g["date"]).min(),
            "Last Date": pd.to_datetime(g["date"]).max(),
            "Confidence Score": int(score),
            "Confidence": "HIGH" if score >= 80 else "MEDIUM" if score >= 60 else "LOW / REVIEW",
            "Why Flagged": why,
            "position_key": key,
        })

    positions = pd.DataFrame(rows)
    tx = tx.copy()
    if positions.empty:
        tx["mca_funder"] = None
        tx["is_mca_debit"] = False
        tx["mca_score"] = 0
        tx["position_category"] = None
        return pd.DataFrame(columns=empty_cols), tx

    key_to_name = dict(zip(positions["position_key"], positions["Position"]))
    key_to_score = dict(zip(positions["position_key"], positions["Confidence Score"]))
    key_to_category = dict(zip(positions["position_key"], positions["Position Category"]))
    tx["position_key"] = np.where(tx["amount"] < 0, tx["description"].apply(canonical_funder_key), None)
    tx["mca_funder"] = tx["position_key"].map(key_to_name)
    tx["position_category"] = tx["position_key"].map(key_to_category)
    tx["mca_score"] = tx["position_key"].map(key_to_score).fillna(0).astype(int)
    tx["is_mca_debit"] = (tx["amount"] < 0) & tx["mca_funder"].notna()
    positions = positions.sort_values(
        by=["# Debits", "Confidence Score", "Est Monthly"],
        ascending=[False, False, False]
    ).drop(columns=["position_key"])
    return positions, tx.drop(columns=["position_key"], errors="ignore")

def build_report(tx: pd.DataFrame, statement: dict, daily_balances: pd.DataFrame):
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    tx = tx.dropna(subset=["date"])
    tx["month"] = tx["date"].dt.to_period("M").astype(str)
    months = max(1, tx["month"].nunique()) if not tx.empty else 1

    positions, tx = build_positions(tx)
    tx["is_revenue"] = tx.apply(lambda row: is_revenue(row["description"], row["amount"]), axis=1)

    revenue = tx[tx["is_revenue"]]
    revenue_by_month = revenue.groupby("month")["amount"].sum() if not revenue.empty else pd.Series(dtype=float)
    revenue_total = float(statement.get("statement_deposits") or revenue["amount"].sum() or 0.0)

    mca = tx[(tx["is_mca_debit"]) & (tx.get("position_category", "") == "MCA / Funding")]
    mca_total_actual = abs(float(mca["amount"].sum())) if not mca.empty else 0.0
    if not positions.empty and "Position Category" in positions.columns:
        mca_est_monthly = float(positions.loc[positions["Position Category"] == "MCA / Funding", "Est Monthly"].sum())
        total_position_monthly = float(positions["Est Monthly"].sum())
    else:
        mca_est_monthly = 0.0
        total_position_monthly = 0.0

    all_months = sorted(tx["month"].unique()) if not tx.empty else []
    month_deposits = pd.DataFrame({
        "Month": all_months,
        "Deposits": [float(revenue_by_month.get(m, revenue_total / months if months else 0)) for m in all_months],
        "# deposits": [int(revenue[revenue["month"] == m].shape[0]) for m in all_months],
    })

    if not daily_balances.empty and "balance" in daily_balances:
        min_balance = float(daily_balances["balance"].min())
        avg_daily_balance = float(daily_balances["balance"].mean())
        negative_days = int((daily_balances["balance"] < 0).sum())
    else:
        min_balance = statement.get("min_balance", np.nan)
        avg_daily_balance = statement.get("avg_ledger_balance", np.nan)
        negative_days = 0 if pd.isna(min_balance) or min_balance >= 0 else np.nan

    debt_to_revenue = (mca_est_monthly / (revenue_total / months)) if revenue_total else 0.0
    summary = {
        "Avg Monthly Deposits": revenue_total / months,
        "Statement Deposits / Revenue Used": revenue_total,
        "Statement Withdrawals": statement.get("statement_withdrawals"),
        "Avg Daily Balance": avg_daily_balance,
        "Min Balance": min_balance,
        "Negative Days": negative_days,
        "Months Analyzed": months,
        "MCA Monthly Debits": mca_est_monthly,
        "Estimated Monthly Positions": total_position_monthly,
        "MCA Actual Debited": mca_total_actual,
        "Debt-to-Revenue": debt_to_revenue,
        "Detected Positions": int(positions.shape[0]),
        "Existing Funders": int(positions[positions["Position Category"] == "MCA / Funding"].shape[0]) if not positions.empty and "Position Category" in positions.columns else 0,
        "Transactions Parsed": int(tx.shape[0]),
        "Statement Deposit Count": statement.get("deposit_count", np.nan),
        "Statement Withdrawal Count": statement.get("withdrawal_count", np.nan),
    }

    red_flags = []
    if summary["Existing Funders"] >= 6:
        red_flags.append("Severe stacking risk: 6+ active funding/MCA positions detected.")
    elif summary["Existing Funders"] >= 3:
        red_flags.append("Stacking risk: multiple active MCA/funding positions detected.")
    if debt_to_revenue >= 0.25:
        red_flags.append("High MCA debt-to-revenue load based on estimated monthly debits.")
    elif debt_to_revenue >= 0.15:
        red_flags.append("Moderate MCA debt-to-revenue load based on estimated monthly debits.")
    if not pd.isna(min_balance) and min_balance < 1000:
        red_flags.append("Low minimum balance during statement period.")
    if not red_flags:
        red_flags.append("No major automated red flags detected; manual review still required.")
    return summary, positions, month_deposits, red_flags, tx




# -----------------------------
# Underwriting risk engine
# -----------------------------

UNDERWRITING_RULES = {
    "max_nsf": 5,
    "min_avg_daily_balance": 5000,
    "max_negative_days": 3,
    "max_mca_percentage": 0.18,
    "max_stacked_positions": 4,
    "max_transfer_dependency": 0.30,
    "max_revenue_drop": 0.25,
}


def safe_number(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def calculate_underwriting_risk(tx: pd.DataFrame, summary: dict, funders: pd.DataFrame, daily_balances: pd.DataFrame) -> dict:
    """Create a simple MCA underwriting score from statements, balances, NSFs, stacking, and cash-flow stress."""
    findings = []
    score = 0

    work = tx.copy()
    if "description" not in work:
        work["description"] = ""
    descriptions = work["description"].astype(str).str.upper()

    # NSF / overdraft detection. Avoid false positives like "ID:NSF" inside ACH metadata.
    nsf_regex = r"\b(NSF FEE|NON[- ]?SUFFICIENT|OVERDRAFT(?: FEE)?|RETURN(?:ED)? (?:ITEM|ACH|CHECK|PAYMENT)|ACH RETURN)\b"
    nsf_mask = descriptions.str.contains(nsf_regex, regex=True, na=False) & ~descriptions.str.contains(r"ID:NSF", regex=True, na=False)
    nsf_count = int(nsf_mask.sum())
    if nsf_count > UNDERWRITING_RULES["max_nsf"]:
        findings.append(f"High NSF/returned-item activity detected ({nsf_count}).")
        score += 25
    elif nsf_count > 0:
        findings.append(f"Some NSF/returned-item activity detected ({nsf_count}); review manually.")
        score += 8

    # Balance health
    avg_balance = safe_number(summary.get("Avg Daily Balance"), np.nan)
    min_balance = safe_number(summary.get("Min Balance"), np.nan)
    if not pd.isna(avg_balance) and avg_balance < UNDERWRITING_RULES["min_avg_daily_balance"]:
        findings.append(f"Low average daily balance ({money(avg_balance)}).")
        score += 20
    if not pd.isna(min_balance) and min_balance < 1000:
        findings.append(f"Low minimum balance ({money(min_balance)}).")
        score += 15

    # Negative days
    negative_days = safe_number(summary.get("Negative Days"), 0)
    if negative_days > UNDERWRITING_RULES["max_negative_days"]:
        findings.append(f"Too many negative balance days ({int(negative_days)}).")
        score += 20

    # Funding stacking should count true/likely funding positions, not every recurring bill/vendor.
    active_positions = 0
    if funders is not None and not funders.empty:
        if "Position Category" in funders.columns:
            funding_mask = funders["Position Category"].astype(str).isin(["MCA / Funding", "Term Loan / Credit", "Equipment Lease"])
            if "Confidence Score" in funders.columns:
                funding_mask = funding_mask & (pd.to_numeric(funders["Confidence Score"], errors="coerce").fillna(0) >= 60)
            active_positions = int(funding_mask.sum())
        else:
            active_positions = int(len(funders))
    if active_positions >= UNDERWRITING_RULES["max_stacked_positions"]:
        findings.append(f"Heavy funding/loan stacking detected ({active_positions} likely debt positions).")
        score += 30
    elif active_positions >= 2:
        findings.append(f"Multiple funding/loan positions detected ({active_positions}).")
        score += 12

    # MCA load
    revenue = safe_number(summary.get("Avg Monthly Deposits"), 0)
    mca_debits = safe_number(summary.get("MCA Monthly Debits"), 0)
    debt_ratio = mca_debits / revenue if revenue > 0 else 0
    if debt_ratio > UNDERWRITING_RULES["max_mca_percentage"]:
        findings.append(f"High MCA debit load ({debt_ratio:.0%} of average monthly deposits).")
        score += 30
    elif debt_ratio > 0.10:
        findings.append(f"Moderate MCA debit load ({debt_ratio:.0%} of average monthly deposits).")
        score += 12

    # Transfer dependency. We calculate it, but only score it heavily when it appears
    # to be internal/P2P funding rather than normal industry settlement activity.
    transfer_keywords = ["ONLINE TRANSFER", "MOBILE TRANSFER", "INTERNET TRANSFER", "TRANSFER TO", "ZELLE", "VENMO", "CASHAPP", "CASH APP"]
    transfer_mask = descriptions.str.contains("|".join(map(re.escape, transfer_keywords)), na=False)
    transfer_volume = float(work.loc[transfer_mask, "amount"].abs().sum()) if "amount" in work else 0.0
    transfer_dependency = transfer_volume / revenue if revenue > 0 else 0
    if transfer_dependency > UNDERWRITING_RULES["max_transfer_dependency"]:
        findings.append(f"High transfer/internal-payment activity ({transfer_dependency:.0%} of average monthly deposits); review context manually.")
        # Smaller penalty because many industries legitimately use transfers/Zelle for operations.
        score += 6

    # Revenue volatility / seasonality
    if "month" in work.columns and "is_revenue" in work.columns:
        rev_by_month = work[work["is_revenue"]].groupby("month")["amount"].sum()
        if len(rev_by_month) >= 2 and rev_by_month.max() > 0:
            revenue_drop = 1 - (rev_by_month.min() / rev_by_month.max())
            if revenue_drop > UNDERWRITING_RULES["max_revenue_drop"]:
                findings.append(f"Revenue volatility/seasonality detected ({revenue_drop:.0%} drop from high month to low month).")
                score += 15
        else:
            revenue_drop = 0
    else:
        revenue_drop = 0

    if not findings:
        findings.append("No major automated underwriting red flags detected; manual review still required.")

    score = int(max(0, min(100, score)))
    if score <= 20:
        grade = "A"
    elif score <= 40:
        grade = "B"
    elif score <= 60:
        grade = "C"
    elif score <= 80:
        grade = "D"
    else:
        grade = "F"

    return {
        "risk_score": score,
        "risk_grade": grade,
        "findings": findings,
        "nsf_count": nsf_count,
        "transfer_dependency": transfer_dependency,
        "debt_ratio": debt_ratio,
        "active_positions": active_positions,
        "revenue_drop": revenue_drop,
    }


# =============================================================
# EXTRA ROBUST PDF PARSER PATCH
# Added for US Bank / scanned-looking PDFs / month-name dates.
# This block intentionally overrides earlier parser functions.
# =============================================================

MONTH_MAP = {
    "JAN": 1, "JANUARY": 1,
    "FEB": 2, "FEBRUARY": 2,
    "MAR": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4,
    "MAY": 5,
    "JUN": 6, "JUNE": 6,
    "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
    "OCT": 10, "OCTOBER": 10,
    "NOV": 11, "NOVEMBER": 11,
    "DEC": 12, "DECEMBER": 12,
}

# Accept 10/31, 10-31, 10 31, Oct 31, October 31, 31 Oct, etc.
DATE_RE = re.compile(
    r"(?P<date>\b(?:"
    r"\d{1,2}\s*[/-]\s*\d{1,2}(?:\s*[/-]\s*\d{2,4})?"
    r"|\d{1,2}\s+\d{1,2}(?:\s+\d{2,4})?"
    r"|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\.?\s+\d{1,2}(?:,?\s+\d{2,4})?"
    r"|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\.?\s*(?:\d{2,4})?"
    r")\b)",
    re.IGNORECASE,
)


def parse_date(value, default_year: int):
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(value).strip().replace(".", "")
    m = DATE_RE.search(text)
    if not m:
        return pd.NaT
    d = re.sub(r"\s+", " ", m.group("date").strip().replace(",", ""))
    parts = re.split(r"\s*[/-]\s*|\s+", d)
    try:
        # Numeric dates: 10/31/2024, 10-31, or 10 31
        if parts[0].isdigit():
            if len(parts) >= 2 and parts[1].isalpha():
                day = int(parts[0])
                month = MONTH_MAP[parts[1][:3].upper()]
                year = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else default_year
            else:
                month = int(parts[0])
                day = int(parts[1])
                year = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else default_year
        else:
            month = MONTH_MAP[parts[0][:3].upper()]
            day = int(parts[1])
            year = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else default_year
        if year < 100:
            year += 2000
        return pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        return pd.NaT


def extract_pdf_text(data: bytes) -> str:
    """Extract text using pdfplumber first, then PyMuPDF as a second digital-text engine."""
    chunks = []
    if pdfplumber is not None:
        try:
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages:
                    chunks.append(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
        except Exception:
            pass

    plumber_text = "\n".join(chunks).strip()

    fitz_chunks = []
    if fitz is not None:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            for page in doc:
                # sort=True helps bank statement lines stay in reading order.
                fitz_chunks.append(page.get_text("text", sort=True) or "")
        except Exception:
            pass

    fitz_text = "\n".join(fitz_chunks).strip()

    # Use whichever extraction gave more usable characters, but keep both if they differ.
    if len(fitz_text) > len(plumber_text) * 1.25:
        return fitz_text + "\n" + plumber_text
    return plumber_text + "\n" + fitz_text


def infer_section(line_upper: str, current: str | None) -> str | None:
    u = normalize_text(line_upper)
    # US Bank and many others use these phrases.
    if any(w in u for w in [
        "DEPOSITS", "OTHER DEPOSITS", "CREDITS", "OTHER CREDITS", "ADDITIONS",
        "MONEY IN", "ELECTRONIC CREDITS", "ACH CREDITS", "CREDIT ACTIVITY"
    ]):
        return "credit"
    if any(w in u for w in [
        "WITHDRAWALS", "OTHER WITHDRAWALS", "DEBITS", "CHECKS", "MONEY OUT",
        "PAYMENTS", "ELECTRONIC DEBITS", "ACH DEBITS", "DEBIT ACTIVITY",
        "CARD AND ELECTRONIC WITHDRAWALS"
    ]):
        return "debit"
    if any(w in u for w in [
        "DAILY BALANCE", "ENDING BALANCE", "END OF STATEMENT", "RECONCILIATION",
        "SUMMARY", "BALANCE SUMMARY", "ACCOUNT SUMMARY"
    ]):
        return None
    return current


def _looks_like_noise_line(line: str) -> bool:
    u = normalize_text(line)
    noise = [
        "PAGE ", "MEMBER FDIC", "ACCOUNT NUMBER", "STATEMENT PERIOD", "CUSTOMER SERVICE",
        "IMPORTANT INFORMATION", "CHANGE IN TERMS", "PRIVACY", "WWW.", "HTTP",
        "BEGINNING BALANCE", "ENDING BALANCE", "TOTAL DEPOSITS", "TOTAL WITHDRAWALS",
        "DATE DESCRIPTION", "DESCRIPTION AMOUNT", "BALANCE"
    ]
    return any(n in u for n in noise)


def parse_text_line(line: str, default_year: int, section: str | None) -> dict | None:
    """Very flexible single-line parser for US Bank, Chase, Wells, BOA, FirstBank, etc."""
    if not line or _looks_like_noise_line(line):
        return None

    m = DATE_RE.search(line)
    if not m:
        return None

    date_value = parse_date(m.group("date"), default_year)
    if pd.isna(date_value):
        return None

    rest = (line[:m.start()] + " " + line[m.end():]).strip()
    money_matches = list(MONEY_RE.finditer(rest))
    if not money_matches:
        return None

    money_items = []
    for mm in money_matches:
        val = clean_money(mm.group())
        if not pd.isna(val):
            money_items.append((mm, float(val)))
    if not money_items:
        return None

    upper = normalize_text(rest)

    # If a line has multiple money numbers, the final one is often running balance.
    # Choose the most likely transaction amount.
    if len(money_items) >= 2:
        chosen_mm, amount = money_items[-2]
    else:
        chosen_mm, amount = money_items[-1]

    # Some statements show columns: withdrawals deposits balance. In that case choose
    # the non-balance amount by section when we can.
    if len(money_items) >= 3 and section in {"debit", "credit"}:
        chosen_mm, amount = money_items[0] if section == "debit" else money_items[1]

    debit_clues = [
        "WITHDRAWAL", "DEBIT", "PAYMENT", "CHECK", "ACH DEBIT", "PURCHASE",
        "POS", "VISA", "MASTERCARD", "AUTOPAY", "BILL PAY", "FEE", "TRANSFER TO"
    ]
    credit_clues = [
        "DEPOSIT", "CREDIT", "ACH CREDIT", "RTP", "WIRE FROM", "TRANSFER FROM",
        "REMOTE ONLINE DEPOSIT", "MERCHANT", "SETTLEMENT", "BATCH"
    ]

    if section == "debit":
        amount = -abs(amount)
    elif section == "credit":
        amount = abs(amount)
    else:
        if any(w in upper for w in debit_clues):
            amount = -abs(amount)
        elif any(w in upper for w in credit_clues):
            amount = abs(amount)
        else:
            raw_amt = chosen_mm.group().strip()
            if raw_amt.startswith("-") or (raw_amt.startswith("(") and raw_amt.endswith(")")):
                amount = -abs(amount)
            # Default: keep positive if we truly cannot infer.

    description = (rest[:chosen_mm.start()] + " " + rest[chosen_mm.end():]).strip(" -|*")
    # Remove other money values from description, especially balances.
    description = MONEY_RE.sub(" ", description)
    description = re.sub(r"\s+", " ", description).strip()

    if len(description) < 2 or description.isdigit():
        return None

    return {
        "date": date_value,
        "description": description,
        "amount": float(amount),
        "section": "debit" if amount < 0 else "credit",
        "source_line": line,
        "source": "text line enhanced",
    }


def parse_pdf_transactions_from_text(text: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    statement = parse_statement_summary(text)
    default_year = extract_statement_year(text) or pd.Timestamp.today().year
    daily_balances = parse_daily_balances(text, default_year)
    rows = []
    section = None
    current = ""

    def flush():
        nonlocal current
        if current:
            tx = parse_text_line(current, default_year, section)
            if tx:
                rows.append(tx)
        current = ""

    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        upper = normalize_text(line)
        next_section = infer_section(upper, section)
        if next_section != section and not DATE_RE.search(line):
            flush()
            section = next_section
            continue
        if "DATE" in upper and any(w in upper for w in ["DESCRIPTION", "AMOUNT", "BALANCE", "WITHDRAWAL", "DEPOSIT"]):
            continue
        if DATE_RE.search(line):
            flush()
            current = line
        elif current:
            # continuation lines such as card numbers, recurring purchase, reference IDs
            current += " " + line
    flush()

    tx = pd.DataFrame(rows)
    if not tx.empty:
        tx = tx.drop_duplicates(subset=["date", "description", "amount"]).reset_index(drop=True)
    return tx, statement, daily_balances


def read_pdf(uploaded_file) -> tuple[pd.DataFrame, dict, pd.DataFrame, str]:
    data = uploaded_file.read()
    text = extract_pdf_text(data)
    default_year = extract_statement_year(text) or pd.Timestamp.today().year

    table_tx = extract_pdf_tables(data, default_year)
    text_tx, statement, daily_balances = parse_pdf_transactions_from_text(text)
    tx = combine_parsed_transactions(table_tx, text_tx)

    method_parts = []
    if not table_tx.empty:
        method_parts.append(f"tables={len(table_tx)}")
    if not text_tx.empty:
        method_parts.append(f"text={len(text_tx)}")

    # OCR fallback if installed.
    if len(tx) < 5:
        ocr_text = ocr_pdf_text_if_available(data)
        if ocr_text.strip():
            ocr_tx, ocr_statement, ocr_balances = parse_pdf_transactions_from_text(ocr_text)
            tx = combine_parsed_transactions(tx, ocr_tx)
            statement = {**statement, **ocr_statement}
            if daily_balances.empty and not ocr_balances.empty:
                daily_balances = ocr_balances
            if not ocr_tx.empty:
                method_parts.append(f"ocr={len(ocr_tx)}")

    text_len = len((text or "").strip())
    if len(tx) < 5:
        if text_len < 300:
            raise ValueError(
                "This PDF appears to be scanned/image-only or locked, so there is almost no extractable text. "
                "Install Tesseract OCR, then rerun the app, or upload the bank CSV/XLSX export. "
                "On Windows: install Tesseract from UB Mannheim, then pip install pytesseract."
            )
        raise ValueError(
            f"Only parsed {len(tx)} transaction rows from {text_len:,} text characters. "
            "This bank layout is unusual. Try the bank CSV/XLSX export, or send me the original statement PDF so I can add a bank-specific rule."
        )

    return tx, statement, daily_balances, (
        f"Parsed {len(tx)} transactions ({', '.join(method_parts) or 'enhanced text'}; text chars={text_len:,}); "
        f"summary fields found: {list(statement.keys())}"
    )



# -----------------------------
# Streamlit UI - focused office version
# -----------------------------


def filter_positions_for_underwriting(positions: pd.DataFrame, min_confidence: int, show_operational: bool) -> pd.DataFrame:
    """Underwriting-focused position filter.

    Always keeps likely debt/funding positions, uses confidence for everything else,
    hides operational noise by default, and sorts by highest # Debits first.
    """
    positions_view = filter_positions_for_underwriting(positions, min_confidence, show_operational)
    if positions_view.empty:
        return positions_view

    debt_category_mask = (
        positions_view["Position Category"]
        .astype(str)
        .str.contains("MCA|Funding|Loan|Credit|Advance|Kapital|Capital", case=False, na=False)
    )

    confidence_mask = (
        pd.to_numeric(positions_view["Confidence Score"], errors="coerce")
        .fillna(0)
        >= min_confidence
    )

    positions_view = positions_view[debt_category_mask | confidence_mask]

    if not show_operational:
        operational_mask = (
            positions_view["Position Category"]
            .astype(str)
            .str.contains(
                "Operational|Payroll|Marketing|Insurance|Utilities|Telecom|Rent|Tax|Vendor|Supplier|Transfer|Owner Draw|Processor|Fees",
                case=False,
                na=False,
            )
        )
        positions_view = positions_view[~operational_mask]

    return positions_view.sort_values(
        by=["# Debits", "Confidence Score", "Est Monthly"],
        ascending=[False, False, False],
    )

st.set_page_config(page_title="MCA Position Finder", layout="wide")
st.title("MCA Position Finder")
st.caption("Focused underwriting view: Risk Findings + Detected Recurring Positions.")

uploaded_files = st.file_uploader(
    "Upload PDF bank statements, CSV, or Excel",
    type=["pdf", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

with st.sidebar:
    st.header("View settings")
    min_confidence = st.slider("Minimum position confidence", 0, 100, 0, 5)
    show_operational = st.checkbox("Show operational recurring payments", value=False)
    show_transactions = st.checkbox("Show underlying position transactions", value=False)
    st.divider()
    st.caption("Default view hides normal operating noise so underwriters can focus on debt/funding positions.")

FUNDING_CATEGORIES = ["MCA / Funding", "Term Loan / Credit", "Equipment Lease", "Recurring Debit / Review"]
OPERATIONAL_CATEGORIES = [
    "Payroll / HR", "Merchant Processor / Fees", "Marketing / Advertising", "Insurance",
    "Utilities / Telecom", "Rent / Real Estate", "Tax / Government",
    "Internal Transfer / Owner Draw", "Vendor / Supplier",
]

if uploaded_files:
    frames, daily_frames, statements = [], [], []
    for file in uploaded_files:
        try:
            if file.name.lower().endswith(".pdf"):
                frame, statement, daily_balances, note = read_pdf(file)
            else:
                frame, statement, daily_balances, note = read_csv_excel(file)
            frame["file_name"] = file.name
            frames.append(frame)
            statements.append(statement)
            if not daily_balances.empty:
                daily_balances["file_name"] = file.name
                daily_frames.append(daily_balances)
            st.success(f"{file.name}: {note}")
        except Exception as exc:
            st.error(f"{file.name}: {exc}")

    if frames:
        tx = pd.concat(frames, ignore_index=True).sort_values("date")
        combined_statement = {}
        for s in statements:
            combined_statement.update(s)
        all_daily_balances = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
        summary, positions, month_deposits, red_flags, classified_tx = build_report(tx, combined_statement, all_daily_balances)
        risk = calculate_underwriting_risk(classified_tx, summary, positions, all_daily_balances)

        # Filter positions for office workflow.
        display_positions = positions.copy()
        if not display_positions.empty:
            display_positions["Confidence Score"] = pd.to_numeric(display_positions["Confidence Score"], errors="coerce").fillna(0).astype(int)
            display_positions = display_positions[display_positions["Confidence Score"] >= min_confidence]
            if not show_operational:
                display_positions = display_positions[display_positions["Position Category"].isin(FUNDING_CATEGORIES)]

            priority = {
                "MCA / Funding": 0,
                "Term Loan / Credit": 1,
                "Equipment Lease": 2,
                "Recurring Debit / Review": 3,
                "Merchant Processor / Fees": 4,
                "Payroll / HR": 5,
                "Internal Transfer / Owner Draw": 6,
            }
            display_positions["_priority"] = display_positions["Position Category"].map(priority).fillna(9)
            display_positions = display_positions.sort_values(
                ["_priority", "Confidence Score", "Est Monthly", "# Debits"],
                ascending=[True, False, False, False],
            ).drop(columns=["_priority"], errors="ignore")

        # Top metrics only.
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Grade", risk["risk_grade"])
        m2.metric("Risk Score", risk["risk_score"])
        m3.metric("Funding Positions", str(summary.get("Existing Funders", 0)))
        m4.metric("MCA Debt-to-Revenue", pct(summary.get("Debt-to-Revenue", 0)))

        # Risk Findings first.
        st.subheader("Risk Findings")
        if risk["findings"]:
            for item in risk["findings"]:
                if risk["risk_score"] >= 60:
                    st.error(item)
                elif risk["risk_score"] >= 30:
                    st.warning(item)
                else:
                    st.info(item)
        for flag in red_flags:
            if flag not in risk["findings"] and "No major" not in flag:
                st.warning(flag)

        # Detected Recurring Positions second.
        st.subheader("Detected Recurring Positions")
        if display_positions.empty:
            st.info("No positions matched the current confidence/category filters. Lower the confidence slider or enable operational recurring payments.")
        else:
            cols = [
                "Position", "Position Category", "Confidence", "Confidence Score", "Frequency",
                "Avg Debit", "Est Monthly", "# Debits", "Total Debited", "First Date", "Last Date", "Why Flagged"
            ]
            cols = [c for c in cols if c in display_positions.columns]
            st.dataframe(display_positions[cols], use_container_width=True, hide_index=True)

            st.download_button(
                "Download detected positions",
                display_positions.to_csv(index=False).encode("utf-8"),
                "detected_positions.csv",
                "text/csv",
            )

        # Optional drill-down only when wanted.
        if show_transactions:
            st.subheader("Underlying Position Transactions")
            tx_cols = ["file_name", "date", "mca_funder", "position_category", "description", "amount", "mca_score", "source"]
            tx_cols = [c for c in tx_cols if c in classified_tx.columns]
            position_tx = classified_tx[classified_tx["is_mca_debit"]].copy()
            if not show_operational and "position_category" in position_tx.columns:
                position_tx = position_tx[position_tx["position_category"].isin(FUNDING_CATEGORIES)]
            if "mca_score" in position_tx.columns:
                position_tx = position_tx[pd.to_numeric(position_tx["mca_score"], errors="coerce").fillna(0) >= min_confidence]
            st.dataframe(position_tx[tx_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Download classified transactions",
                classified_tx.to_csv(index=False).encode("utf-8"),
                "classified_transactions.csv",
                "text/csv",
            )
else:
    st.info("Upload one or more bank statements to begin.")
