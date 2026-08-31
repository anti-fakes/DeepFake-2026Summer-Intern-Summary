import sys
import os

import torch
from tqdm import tqdm
from jiwer import compute_measures
import unicodedata
import editdistance

import zhconv
from funasr import AutoModel
import whisper

from x_voice.eval.text_normalizer_original import TextNormalizer


# ============================================================
# Arguments
# ============================================================

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3]  # zh, en, ja, ko, ...

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# ASR model
# ============================================================

def load_en_model(faster=False, asr_ckpt_dir=""):

    if asr_ckpt_dir == "":
        asr_ckpt_dir = "large-v3"

    if faster:
        from faster_whisper import WhisperModel

        print(
            f"[INFO] Using asr model: "
            f"faster whisper {asr_ckpt_dir}"
        )

        model = WhisperModel(
            asr_ckpt_dir,
            device="cuda",
            compute_type="float16",
        )

    else:
        print(
            f"[INFO] Using asr model: "
            f"whisper {asr_ckpt_dir}"
        )

        model = whisper.load_model(
            asr_ckpt_dir
        ).to(device)

        model.eval()

    return model


def load_zh_model(asr_ckpt_dir=""):

    if asr_ckpt_dir == "":
        asr_ckpt_dir = "paraformer-zh"

    model = AutoModel(
        model=asr_ckpt_dir,
        disable_update=True,
    )

    return model


# ============================================================
# Text normalization
# ============================================================

def normalize_one(hypo, truth, normalizer):

    # --------------------------------------------------------
    # 1. Unicode normalization
    # --------------------------------------------------------

    truth = unicodedata.normalize(
        "NFKC",
        truth,
    )

    hypo = unicodedata.normalize(
        "NFKC",
        hypo,
    )


    # --------------------------------------------------------
    # 2. Text normalization
    #
    # NeMo occasionally fails on some Japanese sentences.
    # In that case, keep only that sentence in raw form and
    # continue the original evaluation pipeline.
    # --------------------------------------------------------

    if normalizer is not None:

        try:
            truth = normalizer.normalize(truth)

        except Exception as e:

            print(
                "\n[WARN] Reference normalization failed"
            )
            print(
                "REF :",
                repr(truth),
            )
            print(
                "ERR :",
                repr(e),
            )
            print(
                "[WARN] Using original reference text."
            )


        try:
            hypo = normalizer.normalize(hypo)

        except Exception as e:

            print(
                "\n[WARN] Hypothesis normalization failed"
            )
            print(
                "HYP :",
                repr(hypo),
            )
            print(
                "ERR :",
                repr(e),
            )
            print(
                "[WARN] Using original hypothesis text."
            )


    # --------------------------------------------------------
    # 3. Character-level evaluation
    #
    # X-Voice evaluates these languages character-by-character.
    # Korean/Japanese/Chinese/Thai WER is therefore effectively
    # CER-like.
    # --------------------------------------------------------

    if lang[-2:] in [
        "zh",
        "ja",
        "ko",
        "th",
    ]:

        truth = " ".join(
            [x for x in truth]
        )

        hypo = " ".join(
            [x for x in hypo]
        )


    # --------------------------------------------------------
    # 4. Lowercase
    # --------------------------------------------------------

    truth = truth.lower()
    hypo = hypo.lower()


    # --------------------------------------------------------
    # 5. Remove punctuation / symbols
    # --------------------------------------------------------

    def clean_special_chars(text):

        cleaned = "".join(
            ch
            for ch in text
            if unicodedata.category(ch)[0]
            not in ("P", "S")
        )

        # collapse repeated spaces
        return " ".join(
            cleaned.split()
        )


    truth = clean_special_chars(
        truth
    )

    hypo = clean_special_chars(
        hypo
    )


    return truth, hypo


# ============================================================
# Sentence-level WER
# ============================================================

def process_one(hypo, truth):

    measures = compute_measures(
        truth,
        hypo,
    )

    ref_list = truth.split(" ")

    wer = measures["wer"]

    subs = (
        measures["substitutions"]
        / len(ref_list)
    )

    dele = (
        measures["deletions"]
        / len(ref_list)
    )

    inse = (
        measures["insertions"]
        / len(ref_list)
    )

    print(wer)

    return (
        truth,
        hypo,
        wer,
        subs,
        dele,
        inse,
    )


# ============================================================
# Run ASR
# ============================================================

def run_asr(
    wav_res_text_path,
    res_path,
    normalize_text=False,
    faster=False,
    whole=False,
):


    # --------------------------------------------------------
    # Load ASR model
    # --------------------------------------------------------

    if lang[-2:] in [
        "zh",
        "hard_zh",
    ]:

        asr_ckpt_dir = ""

        model = load_zh_model(
            asr_ckpt_dir
        )

    else:

        asr_ckpt_dir = ""

        model = load_en_model(
            faster,
            asr_ckpt_dir,
        )


    # --------------------------------------------------------
    # Load wav/reference pairs
    # --------------------------------------------------------

    params = []

    with open(
        wav_res_text_path,
        "r",
        encoding="utf-8",
    ) as f:

        lines = f.readlines()


    for line in lines:

        line = line.strip()

        parts = line.split("|")


        if len(parts) == 2:

            wav_res_path, text_ref = parts


        elif len(parts) == 3:

            wav_res_path, \
            wav_ref_path, \
            text_ref = parts


        elif len(parts) == 4:

            wav_res_path, \
            _, \
            text_ref, \
            wav_ref_path = parts


        else:

            raise NotImplementedError(
                f"Unsupported line format: {line}"
            )


        if not os.path.exists(
            wav_res_path
        ):
            continue


        params.append(
            (
                wav_res_path,
                text_ref,
            )
        )


    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    fout = open(
        res_path,
        "w",
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Text normalizer
    # --------------------------------------------------------

    if normalize_text:

        normalizer = TextNormalizer(
            language=lang[-2:]
        )

    else:

        normalizer = None


    # --------------------------------------------------------
    # ASR loop
    # --------------------------------------------------------

    for wav_res_path, text_ref in tqdm(
        params
    ):

        try:

            # =================================================
            # Chinese
            # =================================================

            if lang[-2:] in [
                "zh",
                "hard_zh",
            ]:

                res = model.generate(
                    input=wav_res_path,
                    batch_size_s=300,
                )

                transcription = res[0][
                    "text"
                ]


            # =================================================
            # Other languages
            # =================================================

            else:

                if faster:

                    segments, _ = (
                        model.transcribe(
                            wav_res_path,
                            beam_size=5,
                            language=lang[-2:],
                        )
                    )

                    transcription = ""

                    for segment in segments:

                        transcription += (
                            " "
                            + segment.text
                        )


                else:

                    result = model.transcribe(
                        wav_res_path,
                        language=lang[-2:],
                    )

                    transcription = result[
                        "text"
                    ].strip()


        except Exception as e:

            print(
                "[WARN] ASR failed:",
                wav_res_path,
            )

            print(e)

            continue


        # ----------------------------------------------------
        # Simplified Chinese conversion
        # ----------------------------------------------------

        if "zh" in lang:

            transcription = zhconv.convert(
                transcription,
                "zh-cn",
            )


        # ----------------------------------------------------
        # Normalize reference / hypothesis
        # ----------------------------------------------------

        truth_normed, hypo_normed = (
            normalize_one(
                transcription,
                text_ref,
                normalizer,
            )
        )


        # ----------------------------------------------------
        # Per-sentence mode
        # ----------------------------------------------------

        if not whole:

            (
                truth,
                hypo,
                wer,
                subs,
                dele,
                inse,
            ) = process_one(
                hypo_normed,
                truth_normed,
            )


            fout.write(
                f"{wav_res_path}\t"
                f"{wer}\t"
                f"{truth}\t"
                f"{hypo}\t"
                f"{inse}\t"
                f"{dele}\t"
                f"{subs}\n"
            )

            fout.flush()


        # ----------------------------------------------------
        # Whole evaluation mode
        # ----------------------------------------------------

        else:

            h_list = hypo_normed.split()
            r_list = truth_normed.split()

            word = len(r_list)

            score = editdistance.eval(
                h_list,
                r_list,
            )

            wer_per_sen = (
                score / word
                if word > 0
                else float("inf")
            )


            fout.write(
                f"{wav_res_path}\t"
                f"{score}\t"
                f"{word}\t"
                f"{wer_per_sen}\t"
                f"{truth_normed}\t"
                f"{hypo_normed}\n"
            )

            fout.flush()


    fout.close()


# ============================================================
# Main
# ============================================================

run_asr(
    wav_res_text_path,
    res_path,
    normalize_text=True,
    faster=False,
    whole=True,
)