import re


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_response_text(resp) -> str:
    """
    Best-effort extraction for OpenAI-compatible responses where some models
    (e.g., reasoning models) may not populate message.content.
    """
    try:
        choice0 = resp.choices[0]
    except Exception:
        return ""

    # 1) Standard chat content
    msg = getattr(choice0, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    itype = str(item.get("type", "")).strip().lower()
                    if "reasoning" in itype or "think" in itype:
                        continue
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt)
                else:
                    itype = str(getattr(item, "type", "")).strip().lower()
                    if "reasoning" in itype or "think" in itype:
                        continue
                    txt = getattr(item, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt)
            joined = "\n".join(parts).strip()
            if joined:
                return joined

    # 2) Legacy text completion shape
    txt = getattr(choice0, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt
    return ""


def normalize_money(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\bRs\.\s*", "₹ ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bINR\s*", "₹ ", text, flags=re.IGNORECASE)
    return text


def dedupe_section_heading(text: str, section: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").strip()
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    body = [ln for ln in lines if ln.lower() != section.lower()]
    out = section + "\n" + "\n".join(body)
    return re.sub(r"\n{3,}", "\n\n", out).strip()
