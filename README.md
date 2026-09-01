# DeepFake-2026Summer-Intern-Summary (LEE Minseo)

> **2026 하계 연구연수**
>
> **Research Period:** 2026.07.01 – 2026.08.31

본 연구는 [X-Voice](https://github.com/sunnyxrxrx/X-Voice)를 기반으로 진행하였으며, **한국어 Zero-shot Cross-lingual Voice Cloning 성능 향상**을 목표로 합니다.

Audio Deepfake 및 Voice Cloning 기술을 조사하고, X-Voice의 공개 Stage 1 모델을 기반으로 한국어 음성 데이터 Fine-tuning과 Multilingual Data Replay를 수행했습니다. 또한 한국어 생성 과정에서 발생하는 발음 오류의 원인을 Text Frontend 관점에서 분석하고 개선했습니다.

---

## 1. Audio Deepfake & Voice Cloning Survey

### 1.1 Audio Deepfake

**Audio Deepfake**는 AI 기반 음성 생성 및 변환 기술을 이용하여 실제 사람이 발화하지 않은 음성을 생성하거나, 기존 음성을 다른 화자의 음성처럼 변조하는 기술입니다.

Audio Deepfake 생성 방식은 크게 Speech Synthesis(TTS)와 Voice Conversion(VC)으로 구분할 수 있습니다.

| Category | Input | Description |
|---|---|---|
| **Speech Synthesis (TTS)** | Text | 입력 Text를 기반으로 새로운 Speech 생성 |
| **Voice Conversion (VC)** | Source Speech | 발화 내용은 유지하면서 화자의 음색 및 특성 변환 |

최근 TTS는 단순한 Text-to-Speech를 넘어, 짧은 **Reference Audio**만으로 대상 화자의 음색과 발화 특성을 재현하는 **Voice Cloning**으로 확장되고 있습니다.

### 1.2 TTS Framework

일반적인 TTS 시스템은 **Text Analysis → Acoustic Model → Vocoder**의 흐름으로 구성됩니다.

<p align="center">
  <img src="tts_framework.png" width="1000" alt="TTS Framework" />
</p>

- **Text Analysis**: Text Normalization, Word Segmentation, POS Tagging, Prosody Prediction, G2P 등
- **Acoustic Model**: Linguistic Feature를 Acoustic Feature로 변환
- **Vocoder**: Acoustic Feature를 최종 Speech Waveform으로 변환

본 연구에서는 한국어 발음 오류를 분석하면서 특히 **Text Normalization → Language Routing → G2P → IPA Tokenization** 과정에 집중했습니다.

### 1.3 Voice Cloning

Voice Cloning은 TTS가 Text의 내용뿐 아니라 특정 화자의 **Speaker Identity**까지 재현하도록 확장된 기술입니다.

| Method | Description |
|---|---|
| **Speaker Adaptation** | 대상 화자 데이터로 모델을 Fine-tuning |
| **Few-shot Voice Cloning** | 소량의 대상 화자 음성으로 Adaptation |
| **Zero-shot Voice Cloning** | 짧은 Reference Audio만으로 별도 화자 학습 없이 생성 |

본 연구에서 사용한 [X-Voice](https://github.com/sunnyxrxrx/X-Voice)는 **Zero-shot Voice Cloning** 방식의 모델이며, 다국어 환경에서 Reference Audio와 다른 언어의 음성을 생성하는 **Cross-lingual Voice Cloning**을 지원합니다.

### 1.4 Evaluation Metrics

| Evaluation Target | Metric | Description | Better |
|---|---|---|:---:|
| Pronunciation / Content | **WER / CER** | Target Text와 ASR Transcript의 오류율 | ↓ |
| Speaker Similarity | **SIM-o / SECS** | Reference와 Generated Speech의 Speaker Embedding Similarity | ↑ |
| Naturalness | **MOS** | Human Listener가 평가한 자연스러움 | ↑ |
| Speech Quality | **DNSMOS** | 비침습 Speech Quality Prediction | ↑ |
| Prosody | **FFE** | Pitch / Voicing Error | ↓ |
| Generation Speed | **RTF** | Generation Time / Audio Duration | ↓ |

최종 정량 평가는 X-Voice Benchmark와 동일하게 **WER**과 **SIM-o**를 중심으로 수행했습니다.

---

## 2. X-Voice

본 연구의 Baseline Model로 [X-Voice](https://github.com/sunnyxrxrx/X-Voice)를 사용했습니다.

X-Voice는 약 **420K hours**의 다국어 음성으로 학습된 약 **0.4B** 규모의 Non-Autoregressive Voice Cloning 모델로, 하나의 모델에서 **30개 언어**의 Zero-shot Cross-lingual Voice Cloning을 지원합니다.

주요 특징은 다음과 같습니다.

- **Conditional Flow Matching** 기반 Speech Generation
- **Unified Multilingual Representation**
- **Dual-Level Language Injection**
- **Decoupled & Scheduled CFG**
- **Two-stage Training**

### 2.1 Two-stage Training

- **Stage 1**: F5-TTS-v1-Base를 초기화 모델로 사용하여 대규모 Multilingual Data로 600K updates 학습
- **Stage 2**: Stage 1 Checkpoint를 기반으로 Synthetic Reference를 활용한 Transcript-Free Voice Cloning 학습

본 연구에서는 한국어 데이터를 직접 추가 학습하고 성능 변화를 분석하기 위해 **Stage 1을 기준으로 실험**했습니다.

### 2.2 Baseline Inference

공개 Stage 1 600K Checkpoint를 한국어·영어·중국어에 적용하여 Baseline Inference를 수행했습니다.

- 한국어 생성 음성에서 **부자연스러운 발음**
- 일부 발화가 비정상적으로 **길게 늘어지는 현상**
- 한·영 Code-switching에서 **발음 및 자연스러움 저하**
- 공개 Evaluation Code의 한국어 WER이 논문 보고값과 크게 다른 현상

이 결과를 바탕으로 **한국어 데이터 Fine-tuning**과 **Korean Frontend 원인 분석**을 진행했습니다.

---

## 3. Korean Speech Dataset Survey

한국어 Fine-tuning 및 후속 실험에 적합한 데이터를 선정하기 위해 **11개 한국어 음성 데이터셋의 스크립트 구성과 특징**을 조사했습니다.

### 3.1 Dataset Comparison

| Dataset | Script Type | Main Characteristics |
|---|---|---|
| [AI Hub 다화자 음성합성](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=542&topMenu=100) | 단문 낭독 | 많은 일반인 화자의 깨끗한 TTS 발화 |
| [AI Hub 감성 및 발화 스타일별 음성합성](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=466&topMenu=100) | 감성·스타일 낭독 | 감정에 따른 억양·운율·말투 |
| [KSS](https://huggingface.co/datasets/Bingsu/KSS_Dataset) | 단일 화자 문장 낭독 | 원문·정규화문·자모 분해문 제공 |
| [Deeply Korean Read Speech](https://www.openslr.org/97/) | 2인 화자 낭독 | Text Sentiment × Voice Sentiment + 녹음 환경 변화 |
| [Zeroth Korean](https://www.openslr.org/40/) | 다화자 문장 낭독 | 비교적 긴 정형 문장 + Audio/Text 1:1 |
| [KsponSpeech](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=123) | 2인 자유대화 | 머뭇거림·반복·말 고침 등 실제 구어 |
| [AI Hub 숫자가 포함된 패턴 발화](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=484) | 패턴 문장 낭독 | 숫자의 문맥별 다양한 읽기 방식 |
| [AI Hub 한국인 외래어 발화](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=131) | 단어·짧은 문장 | 한국인의 외래어·고유명사 발음 |
| [AI Hub 한영 혼합 인식](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71260) | 2인 대화 | 한국어 문장 속 영어계 표현 |
| [AI Hub 중·노년층 한국어 방언](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71517) | 낭독 + 자유발화 + 2인 대화 | 강원·경상 방언 및 중·노년층 말투 |
| [Seoul Corpus](https://www.openslr.org/113/) | 인터뷰형 자연발화 | 서울말 자연발화 + 음소 수준 정밀 Label |

### 3.2 Dataset Script Examples

#### 1) AI Hub 다화자 음성합성 데이터

여러 일반인 화자가 미리 정해진 짧고 정제된 한국어 문장을 읽는 **Read Speech** 데이터입니다. AI 비서·스마트홈·날씨·의료·스포츠·생활정보 등 다양한 주제의 질의형·명령형 문장이 포함되어 있습니다.

**Script Examples**
> “자려고 하니까 불 꺼.”
> “내일 날씨는 비교적 어때?”
> “오늘 축구 경기 언제부터야?”
> “집 근처 운전면허 학원 알려 줘.”

→ 다양한 화자의 음색과 발음 특성을 학습하기에 적합한 다화자 음성합성 데이터

#### 2) AI Hub 감성 및 발화 스타일별 음성합성 데이터

전문 성우가 정해진 문장을 특정 **감정 및 발화 스타일**에 맞춰 읽는 데이터입니다. 기쁨·놀람·슬픔·분노·두려움·혐오·중립 등의 감정과 낭독체·대화체·뉴스체·중계체·구연체 등의 발화 유형을 포함합니다.

**Script Examples**
> “은행가가 농부에게 말했다.”
> “한 번 그 증거를 보여 주시면 꼭 믿겠습니다.”
> “궁금증이 가는 것이다.”
> “행복한 신부가 스스로 물에 몸을 던졌을 리는 만무해.”

→ 감정에 따라 변화하는 억양·강세·속도·운율·말투를 학습할 수 있는 데이터

#### 3) KSS (Korean Single Speaker Speech) Dataset

여성 화자 1명이 정해진 한국어 문장을 읽은 단일 화자 낭독 데이터이며, `original_script`, `expanded_script`, `decomposed_script`를 함께 제공합니다.

**Script Examples**
> “용돈을 아껴 써라.”
> “그 애 전화번호 알아?”
> “거기 도착하면 나한테 알려 줘.”
> `114에 전화를 해서 ...` → `일일사에 전화를 해서 ...`

→ 원문·정규화문·자모 분해문을 함께 제공하여 Text Normalization 및 입력 표현 비교에 활용 가능

#### 4) Deeply Korean Read Speech Corpus

2인 1조의 화자가 정해진 Script를 읽는 한국어 다화자 낭독 데이터입니다. Text Sentiment와 Voice Sentiment가 별도로 제공되며 녹음 장소·거리·기기 변화도 포함합니다.

**Script Example**
> “저 식당 음식이 정말 맛있나 봐요.”

→ 동일한 낭독 문장을 여러 감정·화자·녹음 환경에서 수집한 데이터

#### 5) Zeroth Korean

한국어 ASR 연구를 위해 구축된 다화자 문장 낭독 데이터로, 짧은 일상 회화보다 뉴스·시사·사회·정보 전달형의 비교적 긴 문장이 많이 포함됩니다.

**Script Examples**
> “그밖의 자세한 사항은 중앙선거여론조사심의위원회 홈페이지를 참조하면 된다.”
> “가방을 비롯한 수하물들을 컨베이어 벨트 위에 가능한 한 납작하게 올려놓으십시오.”

→ 비교적 길고 정형화된 한국어 문장과 정확한 전사문이 1:1로 대응

#### 6) KsponSpeech

약 2,000명의 한국인이 두 명씩 짝을 이루어 다양한 주제로 대화한 **Spontaneous Speech** 데이터입니다. 간투사·머뭇거림·반복·말 고침·문장 중단 등 실제 구어 특성을 포함합니다.

**Script Examples**
> “아/ 몬 소리야, 그건 또.”
> “아/ 내일 나 알바하구나.”
> “아/ 근데 강남 너무 비싸.”
> “목포 너무 비싸, 어, 시외버스도 비싸고, 한 (3만원)/(삼만 원)인가 그럴걸.”

→ 실제 한국어 대화의 발화 속도·호흡·억양·축약 및 비정형 문장 구조를 포함

#### 7) AI Hub 숫자가 포함된 패턴 발화 데이터

비밀번호·이메일·우편번호·스포츠 기록·날짜·시간·금액·수량 등 **숫자가 실제 사용되는 문맥**을 포함합니다. 숫자가 표기된 `scriptITN`과 실제 읽는 형태의 `scriptTN`을 함께 제공합니다.

**Script Examples**
> 비밀번호 `1111` → “일 일 일 일”
> 이메일 ID `PJS0023` → “피 제이 에스 공 공 이 삼”
> `29승 6패, 승률 83%` → “이십 구 승 육 패, 팔십 삼 퍼센트”
> 타율 `0.143` → “일할 사푼 삼리”
> 우편번호 `61342` → “육 일 삼 사 이”

→ 동일한 숫자라도 문맥에 따라 읽는 방식이 달라지는 한국어 Number Normalization 분석에 유용

#### 8) AI Hub 한국인 외래어 발화 데이터

한국인이 실제 사용하는 외래어와 외국 고유명사의 한국어 발음을 수집한 데이터입니다. 목표 외래어가 포함된 문장과 단독 단어 발화가 함께 존재합니다.

**Script Examples**
> “스토리텔링을 통해서 자신의 생각과 의견을 전달하는 방법을 배우게 돼요.”
> “혹시 트레인 언제 하는지 알아.”
> “둠.”

→ 외래어·외국 고유명사가 한국인 화자에게서 실제로 어떻게 발음되는지 분석 가능

#### 9) AI Hub 한영 혼합 인식 데이터

일상 및 전문 분야의 다양한 주제로 두 화자가 대화하는 **Korean-English Mixed Speech** 데이터입니다. 한국어 문장 속 영어계 표현이 한국어 발음 형태로 전사되며 영어 원형은 `originalForm`에 별도로 기록됩니다.

**Script Examples**
> “여기 런치 메뉴 중에선 리조또랑 로제파스타가 젤 나음.”
> “매니저들 패션감각 넘치는거 봤어? 화이트셔츠에 데님인데 멋져.”
> “그렇지? 너도 그렇게 씽킹했구나 나도 그랬는데.”

→ 한국어 문장 안에 영어계 단어와 표현이 포함된 한영 혼용 발화 분석에 적합

#### 10) AI Hub 중·노년층 한국어 방언 데이터

강원도·경상도 중·노년층 화자의 실제 방언을 수집한 데이터로 **낭독형, 질문 기반 자유발화형, 2인 대화형**을 모두 포함합니다.

**Script Examples**
> 강원도 낭독: “그 집 메누리는 아츰지냑으로 집안 으른덜께 문안 인사를 디레고 …”
> 경상도 낭독: “게얼에 먹을 채소나 가일 같은 것은 오데 보관을 했습니껴?”
> 2인 대화 주제: “1. 산이 더 좋다. 2. 바다가 더 좋다.” → “나는 산이 좋더라.” / “나는 바닷가 좋아.”

→ 지역 방언의 발음·억양·어휘와 중·노년층의 자연스러운 말투를 폭넓게 포함

#### 11) Seoul Corpus

서울말 원어민의 실제 발화를 수집한 인터뷰형 자연발화 데이터입니다. 철자 형태와 실제 발음 형태를 별도 Tier로 전사하고, 어절 경계와 음소 단위까지 Labeling합니다.

**Script Example**
> “자기 전에두 하고 와.”

→ 문어체와 다른 실제 서울말의 발음 변화·연음·축약·발화 속도·억양을 분석할 수 있는 음성학 중심 데이터

### 3.3 Script Type Summary

- **정제된 낭독 중심**: 다화자 음성합성 / 감성 및 발화 스타일 / KSS / Deeply Korean Read Speech / Zeroth Korean / 숫자 패턴 발화
- **자연발화·대화 중심**: KsponSpeech / Seoul Corpus
- **특정 발음 특화**: 한국인 외래어 발화 / 숫자 패턴 발화
- **언어·지역적 다양성 특화**: 한영 혼합 인식 / 중·노년층 한국어 방언

### 3.4 Selected Training Dataset

실제 Fine-tuning에는 **AI Hub 다화자 음성합성 데이터**를 사용했습니다.

- 총 **10,152 hours**
- 총 **3,495 speakers**
- 10대–60대 이상의 다양한 연령 및 성별
- 화자별 약 2,000–2,400개 발화
- AI 비서·스마트홈·날씨·의료·스포츠·생활정보 등 다양한 단문
- WAV + JSON Transcript Pair

다수의 일반인 화자가 정해진 문장을 명확하게 읽는 형태로 구성되어 있어 **다양한 Speaker Identity를 유지하면서 한국어 발음과 음색을 학습하기에 적합**하다고 판단했습니다.

---

## 4. Dataset Preprocessing

AI Hub 다화자 음성합성 데이터를 X-Voice 학습 형식으로 변환하고 품질을 정제했습니다.

1. WAV–JSON Pair 구성 및 Duration 계산
2. **0.5–30 sec** 발화만 유지
3. **DNSMOS < 1.5** 저품질 음성 제거
4. 동일 Transcript가 **20회 초과** 반복되는 Sample 제거
5. Speaking Rate 계산 후 **IQR Outlier** 제거
6. Transcript를 X-Voice Pronunciation Token으로 변환
7. `raw.arrow`, `vocab.txt`, `vocab_stats.txt`, `duration.json` 생성


---

## 5. Fine-tuning

### 5.1 Korean-only Fine-tuning

먼저 Stage 1 600K Checkpoint를 기반으로 **AI Hub 한국어 500 h**를 사용하여 Fine-tuning을 수행했습니다.

한국어 생성 성능은 향상되었지만, 한국어 데이터만 반복적으로 학습하면서 기존 Multilingual Representation이 손상되는 **Catastrophic Forgetting**이 관찰되었습니다.

### 5.2 Multilingual Data Replay

한국어 Adaptation과 기존 Multilingual 성능을 함께 유지하기 위해 **Multilingual Data Replay**를 적용했습니다.

| Data | Hours |
|---|---:|
| AI Hub Korean | 500 h |
| English | 100 h |
| Chinese | 50 h |
| Japanese | 50 h |
| Original Korean | 50 h |
| German / French / Spanish / Italian / Portuguese / Russian | 25 h each |
| Remaining Languages | 10 h each |

---

## 6. Korean Frontend Analysis

Fine-tuning 결과를 평가하는 과정에서 공개 Stage 1 600K Checkpoint의 한국어 WER이 논문의 결과보다 비정상적으로 높게 나타났습니다.

이에 Model Weight뿐 아니라 Text가 X-Voice 입력 Token으로 변환되는 전체 과정을 추적했습니다.

**Text → Language Routing → Text Normalization → g2pK → eSpeak → IPA → Tokenizer → X-Voice**


### 6.1 `g2pK to_syl`

한국어 G2P에 사용되는 `g2pK`의 `to_syl` 옵션을 분석했습니다.

- `to_syl=True` → 완성형 한글 음절 유지
- `to_syl=False` → Hangul Jamo 단위로 분해

해당 옵션은 다음 Tokenizer 코드에서 설정됩니다.

- 코드 경로: [`src/x_voice/train/datasets/ipa_v6_tokenizer.py`](src/x_voice/train/datasets/ipa_v6_tokenizer.py)
- 적용 위치: `PhonemizeTextTokenizer.__init__()`의 한국어 `G2pk` 초기화 부분

```python
if language == "ko":
    self.g2p = G2pk(no_space=False, to_syl=True)
```

공개 X-Voice 코드에서는 `to_syl=False`가 사용되고 있었으며, 분해된 Jamo가 eSpeak로 전달되면서 **비정상적인 IPA Sequence**가 생성되는 것을 확인했습니다.

Model Weight와 Inference Setting을 고정하고 `to_syl`만 `True`로 변경한 결과,

| Model / Setting | WER ↓ |
|---|---:|
| Paper X-Voice Stage 1 | 2.42 |
| Base 600K (Reproduced) | 12.131 |
| `to_syl=True` | **3.016** |

즉, 높은 한국어 WER의 주요 원인 중 하나가 Acoustic Model 자체가 아니라 **Korean Frontend의 Phonetic Representation**임을 확인했습니다.

### 6.2 Number & English Unit Analysis

`to_syl` 수정 후 일반적인 한국어 발음은 크게 개선되었지만 숫자·조수사·영문 단위가 포함된 문장에서 추가 오류가 나타났습니다.

| Input | Generated Pronunciation Problem |
|---|---|
| `3명이 회의에 참석했습니다.` | `삼 명이` |
| `5km를 이동했습니다.` | `파이브 킬로미터` |
| `오늘 5km를 이동했습니다.` | `오 케이엠` |

X-Voice의 Language Routing에서 한글은 Korean, Latin Alphabet은 English로 분류하지만 숫자는 **Neutral**로 처리됩니다. 따라서 같은 `5km`도 주변 Span에 따라 서로 다른 Language Context에 결합될 수 있음을 확인했습니다.

이 문제를 통해 한국어 생성 성능에는 Acoustic Model뿐 아니라 **Text Normalization과 Language Routing** 역시 중요한 영향을 준다는 것을 확인했습니다.

### 6.3 Modified Code

본 연구에서 분석 및 개선한 내용은 다음 코드에 반영되어 있습니다. `*_original.py`는 공개 구현을 기준으로 보존한 파일이며, `*_improved.py`는 한국어 처리 개선을 적용한 비교용 구현입니다.

| Improvement | Code Path | Description |
|---|---|---|
| Korean Syllable Preservation | [`src/x_voice/train/datasets/ipa_v6_tokenizer.py`](src/x_voice/train/datasets/ipa_v6_tokenizer.py) | `G2pk(no_space=False, to_syl=True)`를 적용하여 Hangul Jamo 분해로 인한 비정상 IPA 변환 방지 |
| Korean Text Normalization | [`src/x_voice/infer/korean_tn.py`](src/x_voice/infer/korean_tn.py) | N2gkPlus 기반 숫자·조수사 처리와 `km`, `GB`, `GHz`, `MB/s` 등의 숫자+영문 단위 확장 |
| Number & Unit Language Routing | [`src/x_voice/infer/utils_infer_improved.py`](src/x_voice/infer/utils_infer_improved.py) | 숫자+Latin 표현을 하나의 Span으로 보호하고 문장의 Dominant Script에 따라 언어를 할당 |
| Original / Improved Inference | [`infer_cli_stage1_original.py`](src/x_voice/infer/infer_cli_stage1_original.py), [`infer_cli_stage1_improved.py`](src/x_voice/infer/infer_cli_stage1_improved.py) | 동일 조건에서 공개 구현과 개선 구현을 각각 실행하기 위한 Stage 1 CLI |
| Original / Improved Inference Utilities | [`utils_infer_original.py`](src/x_voice/infer/utils_infer_original.py), [`utils_infer_improved.py`](src/x_voice/infer/utils_infer_improved.py) | Text Normalization, Language Routing, IPA Tokenization 전후 비교 |
| Korean Evaluation Normalization | [`src/x_voice/eval/text_normalizer_improved.py`](src/x_voice/eval/text_normalizer_improved.py) | WER 평가 전 한국어 문장에 N2gkPlus Normalization 적용 |
| Batch Inference Evaluation | [`eval_infer_batch_original.py`](src/x_voice/eval/eval_infer_batch_original.py), [`eval_infer_batch_improved.py`](src/x_voice/eval/eval_infer_batch_improved.py) | Base 600K와 Ours 15.5k의 Benchmark 음성을 동일 조건으로 생성 |
| WER Evaluation | [`src/x_voice/eval/utils/run_wer.py`](src/x_voice/eval/utils/run_wer.py) | ASR Transcript 생성, 언어별 Text Normalization 및 WER 계산 |
| Speaker Similarity Evaluation | [`src/x_voice/eval/eval_similarity.py`](src/x_voice/eval/eval_similarity.py), [`ecapa_tdnn.py`](src/x_voice/eval/ecapa_tdnn.py) | ECAPA-TDNN Speaker Embedding 기반 Reference–Generated Speech 유사도 계산 |
| Checkpoint Loading | [`src/x_voice/model/trainer.py`](src/x_voice/model/trainer.py) | Vocabulary 확장 시 기존 Text Embedding을 보존하여 부분 로드하고 Fine-tuning Checkpoint Resume 처리 개선 |
| Training Configurations | [`XVoice_KO_Replay_Stage1.yaml`](src/x_voice/configs/XVoice_KO_Replay_Stage1.yaml), [`XVoice_KO_Replay_SylFix_Stage1.yaml`](src/x_voice/configs/XVoice_KO_Replay_SylFix_Stage1.yaml), [`XVoice_KOEN_CS_249h_Stage1.yaml`](src/x_voice/configs/XVoice_KOEN_CS_249h_Stage1.yaml) | Multilingual Replay, Syllable Fix, Korean–English Code-switching 실험 설정 |

---

## 7. Inference

Stage 1 inference는 [`basic_stage1.toml`](src/x_voice/infer/examples/basic/basic_stage1.toml)을 읽어 실행합니다. 기존 Base inference 설정에서 `ckpt_file`을 Ours 15.5k Checkpoint 경로로 바꾸고, 함께 제공된 Ours `vocab.txt` 경로를 지정한 뒤 동일한 명령을 실행합니다.

### 7.1 TOML에서 수정할 항목

| Field | Description | Example |
|---|---|---|
| `ckpt_file` | 사용할 Stage 1 Checkpoint | `ckpts/XVoice_KO_Replay_1100h_SylFix_v2/model_15500.pt` |
| `vocab_file` | Checkpoint와 함께 사용하는 Vocabulary | `ckpts/XVoice_KO_Replay_1100h_SylFix_v2/vocab.txt` |
| `ref_audio` | 복제할 화자의 Reference Audio | `reference_audio_ko.wav` |
| `ref_text` | Reference Audio의 정확한 Transcript | 아래 Reference Script |
| `gen_text` | 생성할 Target Text | `3명이 회의에 참석했습니다.` |
| `ref_lang`, `gen_lang` | Reference/Target 언어 | `ko` |
| `output_dir` | 결과 저장 폴더 | `inference_outputs` |
| `output_file` | 결과 WAV 파일명 | `result2.wav` |

Ours 15.5k를 실행하려면 TOML의 주요 부분을 다음처럼 수정합니다.

> **Checkpoint 안내:** `vocab.txt`는 이 저장소에 포함되어 있지만, 약 5.1GB인 `model_15500.pt`는 GitHub에 포함되어 있지 않습니다. [Notion Checkpoint 페이지](https://app.notion.com/p/ckpt-3cd4e0f5f31480be9d75fbd37a09dcbc?source=copy_link)에서 파일을 다운로드한 뒤 아래 경로에 배치합니다.

```bash
mkdir -p ckpts/XVoice_KO_Replay_1100h_SylFix_v2

# 브라우저에서 받은 파일의 실제 위치에 맞게 첫 번째 경로를 수정하세요.
mv ~/Downloads/model_15500.pt \
  ckpts/XVoice_KO_Replay_1100h_SylFix_v2/model_15500.pt

# 파일 배치 확인
test -f ckpts/XVoice_KO_Replay_1100h_SylFix_v2/model_15500.pt && \
  echo "Ours 15.5k checkpoint is ready."
```

```toml
model = "XVoice_Base_Stage1"
model_cfg = "src/x_voice/configs/XVoice_Base_Stage1.yaml"
ckpt_file = "ckpts/XVoice_KO_Replay_1100h_SylFix_v2/model_15500.pt"
vocab_file = "ckpts/XVoice_KO_Replay_1100h_SylFix_v2/vocab.txt"

ref_audio = "reference_audio_ko.wav"
ref_text = "저는 지금 음성 생성 모델을 테스트하고 있습니다. 제 목소리가 얼마나 자연스럽게 생성되는지 확인해보겠습니다."
gen_text = "3명이 회의에 참석했습니다."

ref_lang = "ko"
gen_lang = "ko"
auto_detect_lang = false
normalize_text = true
sp_type = "syllable"

output_dir = "inference_outputs"
output_file = "result2.wav"
```

TOML을 저장한 뒤 다음 명령을 실행합니다.

```bash
python -m x_voice.infer.infer_cli_stage1_improved \
  -c src/x_voice/infer/examples/basic/basic_stage1.toml
```

---
## 8. Evaluation

X-Voice Multilingual Benchmark를 이용하여 **Intra-lingual / Cross-lingual** 조건에서 Base 600K와 최종 **Ours 15.5k** 모델을 비교했습니다.

평가 지표는 다음과 같습니다.

- **WER ↓**: 발음 및 내용 정확도
- **SIM-o ↑**: Reference와 Generated Speech 사이의 Speaker Similarity

### 8.1 Intra-lingual Results

30개 언어에 대해 동일 언어 Reference → 동일 언어 Target 조건으로 평가했습니다.

<p align="center">
  <img src="intra-lingual%20result.png" width="800" alt="Intra-lingual Evaluation Results" />
</p>

주요 결과는 다음과 같습니다.

- **Korean WER:** `2.979 → 2.680`
- **Korean SIM-o:** `0.7215 → 0.7265`
- **English WER:** `2.381 → 2.152`
- **English SIM-o:** `0.5852 → 0.5939`

한국어뿐 아니라 다수 언어에서 Base 600K 대비 WER 또는 SIM-o가 개선되어, 한국어 Adaptation 과정에서도 기존 Multilingual 능력이 상당 부분 유지되었음을 확인했습니다.

### 8.2 Cross-lingual Results

한국어와 영어 사이의 Cross-lingual Voice Cloning 성능을 비교했습니다.

<p align="center">
  <img src="cross-lingual%20result.png" width="800" alt="Cross-lingual Evaluation Results" />
</p>

| Language Pair | Base WER ↓ | Ours WER ↓ | Base SIM-o ↑ | Ours SIM-o ↑ |
|---|---:|---:|---:|---:|
| KO→EN | **2.921** | 3.360 | **0.4497** | 0.4488 |
| EN→KO | 4.464 | **3.278** | 0.4727 | **0.4781** |

**EN→KO**에서는 발음 정확도와 Speaker Similarity가 모두 향상된 반면, **KO→EN**에서는 일부 성능 Trade-off가 나타났습니다.

### 8.3 Inference Audio Results

#### Reference Audio

- Audio: [▶ `reference_audio_ko.wav`](reference_audio_ko.wav)
- Script: “저는 지금 음성 생성 모델을 테스트하고 있습니다. 제 목소리가 얼마나 자연스럽게 생성되는지 확인해보겠습니다.”

Base 600K와 Ours 15.5k는 동일한 Reference Audio와 Target Text를 사용했습니다. 차이는 inference에 사용한 Checkpoint와 해당 Vocabulary입니다.

| No. | Target Text | Base 600K | Ours 15.5k |
|---:|---|---|---|
| 1 | 퇴근길에 마트에 들러 우유와 과일을 사고 집으로 돌아왔습니다. | [▶ `base_result1.wav`](base_result1.wav) | [▶ `result1.wav`](result1.wav) |
| 2 | 3명이 회의에 참석했습니다. | [▶ `base_result2.wav`](base_result2.wav) | [▶ `result2.wav`](result2.wav) |
| 3 | 오늘 5km를 이동했습니다. | [▶ `base_result3.wav`](base_result3.wav) | [▶ `result3.wav`](result3.wav) |

---

## 9. Conclusion

본 연구에서는 [X-Voice](https://github.com/sunnyxrxrx/X-Voice)를 기반으로 한국어 Zero-shot Cross-lingual Voice Cloning 성능을 개선하기 위해 **Dataset → Fine-tuning → Evaluation → Text Frontend**를 단계적으로 분석했습니다.

- 한국어 음성 데이터셋 **11종의 Script 구조 및 특성 조사**
- AI Hub 다화자 음성합성 데이터 기반 Korean Fine-tuning
- Korean-only Fine-tuning에서 **Catastrophic Forgetting** 확인
- **Multilingual Data Replay**를 통한 기존 다국어 성능 보존
- `g2pK to_syl=False`에서 발생하는 비정상 IPA 문제 발견
- `to_syl=True` 적용 시 한국어 WER **12.131 → 3.016**
- 숫자·조수사·영문 단위 오류를 **Text Normalization / Language Routing** 관점에서 분석
- 최종 모델에서 KO→KO WER **2.979 → 2.680**, SIM-o **0.7215 → 0.7265**
- EN→KO Cross-lingual WER **4.464 → 3.278**, SIM-o **0.4727 → 0.4781**

### Future Work

- KO→EN Cross-lingual 성능 저하 원인 분석
- 숫자·단위·외래어를 고려한 Korean Text Normalization 고도화
- AI Hub 숫자 패턴 / 외래어 / 한영 혼합 데이터의 추가 활용
- 한·영 Code-switching 전용 Test Set 구축 및 일반화 성능 평가
- WER / SIM-o뿐 아니라 MOS 및 Prosody 관련 지표를 이용한 다각도 평가

---

## References

- [X-Voice: Official Repository](https://github.com/sunnyxrxrx/X-Voice)
- [X-Voice Paper](https://arxiv.org/abs/2605.05611)
- [X-Voice Model](https://huggingface.co/XRXRX/X-Voice)
- [X-Voice Benchmark](https://huggingface.co/datasets/XRXRX/X-Voice-Testset)

```bibtex
@article{xu2026xvoiceenablingspeak30,
  title={X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning},
  author={Rixi Xu and Qingyu Liu and Haitao Li and Yushen Chen and Zhikang Niu and Yunting Yang and Jian Zhao and Ke Li and Berrak Sisman and Qinyuan Cheng and Xipeng Qiu and Kai Yu and Xie Chen},
  journal={arXiv preprint arXiv:2605.05611},
  year={2026}
}
```
