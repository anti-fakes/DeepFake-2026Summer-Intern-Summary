import re
import sys
from pathlib import Path


# ------------------------------------------------------------
# Load CoreaSpeech N2gk+
# ------------------------------------------------------------
COREASPEECH_ROOT = Path("/home/user/CoreaSpeech")
COREASPEECH_SRC = COREASPEECH_ROOT / "src"

if str(COREASPEECH_SRC) not in sys.path:
    sys.path.insert(0, str(COREASPEECH_SRC))

from module.data_conditioning.normalization import N2gkPlus


_n2gk = None


def get_n2gk():
    global _n2gk

    if _n2gk is None:
        _n2gk = N2gkPlus(natural=True)

    return _n2gk


# ------------------------------------------------------------
# Unit verbalization
# ------------------------------------------------------------

# Canonical unit name -> Korean spoken form
UNIT_KO = {
    # length
    "mm": "밀리미터",
    "cm": "센티미터",
    "m": "미터",
    "km": "킬로미터",

    # mass
    "mg": "밀리그램",
    "g": "그램",
    "kg": "킬로그램",

    # time
    "ms": "밀리초",
    "s": "초",
    "min": "분",
    "h": "시간",

    # frequency
    "hz": "헤르츠",
    "khz": "킬로헤르츠",
    "mhz": "메가헤르츠",
    "ghz": "기가헤르츠",

    # bytes
    "b": "바이트",
    "kb": "킬로바이트",
    "mb": "메가바이트",
    "gb": "기가바이트",
    "tb": "테라바이트",

    # bits / transfer rate
    "bps": "비피에스",
    "kbps": "킬로비피에스",
    "mbps": "메가비피에스",
    "gbps": "기가비피에스",

    # electrical
    "v": "볼트",
    "w": "와트",
    "kw": "킬로와트",
}


def _expand_compound_units_ko(text: str) -> str:
    """
    Expand number + unit expressions before N2gk+.

    Examples:
        5km      -> 5킬로미터
        24GB     -> 24기가바이트
        3.5GHz   -> 3.5기가헤르츠
        30km/h   -> 30킬로미터 퍼 시간
        100MB/s  -> 100메가바이트 퍼 초
    """

    # --------------------------------------------------------
    # number + UNIT / UNIT
    # --------------------------------------------------------
    compound_pattern = re.compile(
        r'(?P<num>\d+(?:[.,]\d+)?)\s*'
        r'(?P<num_unit>[A-Za-z]+)\s*/\s*'
        r'(?P<den_unit>[A-Za-z]+)',
        flags=re.IGNORECASE,
    )

    def compound_replacer(match):
        num = match.group("num")
        numerator = match.group("num_unit").lower()
        denominator = match.group("den_unit").lower()

        if numerator not in UNIT_KO or denominator not in UNIT_KO:
            return match.group(0)

        return (
            f"{num}{UNIT_KO[numerator]} "
            f"퍼 {UNIT_KO[denominator]}"
        )

    text = compound_pattern.sub(compound_replacer, text)

    # --------------------------------------------------------
    # number + UNIT
    # --------------------------------------------------------
    simple_pattern = re.compile(
        r'(?P<num>\d+(?:[.,]\d+)?)\s*'
        r'(?P<unit>[A-Za-z]+)',
        flags=re.IGNORECASE,
    )

    def simple_replacer(match):
        num = match.group("num")
        unit = match.group("unit").lower()

        if unit not in UNIT_KO:
            return match.group(0)

        return f"{num}{UNIT_KO[unit]}"

    return simple_pattern.sub(simple_replacer, text)


def normalize_korean_text(text: str) -> str:
    """
    Korean TN used by X-Voice.

    1) Expand structured unit expressions.
    2) Run CoreaSpeech N2gk+ for Korean numeral/counter normalization.
    """

    text = _expand_compound_units_ko(text)

    normalizer = get_n2gk()
    text = normalizer(text)

    # Cosmetic whitespace cleanup only.
    text = re.sub(r"\s+", " ", text).strip()

    return text
