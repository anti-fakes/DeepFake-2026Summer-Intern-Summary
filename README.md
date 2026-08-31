# DeepFake-2026Summer-Intern-Summary (LEE Minseo)

> **2026 하계 연구연수 최종 정리**
> **Research Period:** 2026.07.01 – 2026.08.31

본 연구는 [X-Voice](https://github.com/sunnyxrxrx/X-Voice)를 기반으로 진행하였으며, **한국어 Zero-shot Cross-lingual Voice Cloning 성능 향상**을 목표로 합니다.

Audio Deepfake 및 Voice Cloning 기술을 조사하고, X-Voice의 공개 Stage 1 모델을 기반으로 한국어 음성 데이터 Fine-tuning과 Multilingual Data Replay를 수행했습니다. 또한 한국어 생성 과정에서 발생하는 발음 오류의 원인을 Text Frontend 관점에서 분석하고 개선했습니다.

---

## 1. Audio Deepfake & Voice Cloning Survey

### 1.1 Audio Deepfake

**Audio Deepfake**는 AI 기반 음성 생성 및 변환 기술을 이용하여 실제 사람이 발화하지 않은 음성을 생성하거나, 기존 음성을 다른 화자의 음성처럼 변조하는 기술입니다.

Audio Deepfake 생성 방식은 크게 **Speech Synthesis (TTS)**와 **Voice Conversion (VC)**으로 구분할 수 있습니다.

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

| Dataset | Script Type | Main Characteristics | Research Use |
|---|---|---|---|
| [AI Hub 다화자 음성합성](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=542&topMenu=100) | 단문 낭독 | 다양한 일반인 화자의 깨끗한 TTS 발화 | **한국어 Fine-tuning 주 데이터** |
| [AI Hub 감성 및 발화 스타일별 음성합성](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=466&topMenu=100) | 감성·스타일 낭독 | 감정에 따른 억양·강세·속도·운율·말투 | Prosody / Emotion |
| [KSS](https://huggingface.co/datasets/Bingsu/KSS_Dataset) | 단일 화자 문장 낭독 | Original / Expanded / Decomposed Script 제공 | Text Normalization 비교 |
| [Deeply Korean Read Speech](https://www.openslr.org/97/) | 2인 화자 낭독 | Text Sentiment × Voice Sentiment + 녹음 환경 변화 | 환경·감성 다양성 |
| [Zeroth Korean](https://www.openslr.org/40/) | 다화자 문장 낭독 | 비교적 긴 정형 문장, Audio/Text 1:1 | 장문·ASR 평가 |
| [KsponSpeech](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=123) | 2인 자유대화 | 머뭇거림·반복·말 고침 등 실제 구어 | 자연발화 |
| [AI Hub 숫자가 포함된 패턴 발화](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=484) | 패턴 문장 낭독 | ScriptITN / ScriptTN, 문맥별 숫자 읽기 | **숫자·조수사 TN** |
| [AI Hub 한국인 외래어 발화](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=131) | 단어·짧은 문장 | 한국인의 외래어·외국 고유명사 발음 | 외래어 발음 |
| [AI Hub 한영 혼합 인식](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71260) | 2인 대화 | 한국어 문장 속 영어계 표현, `originalForm` 제공 | **Code-switching** |
| [AI Hub 중·노년층 한국어 방언](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71517) | 낭독 + 자유발화 + 2인 대화 | 강원·경상 방언 및 중·노년층 말투 | 지역·연령 다양성 |
| [Seoul Corpus](https://www.openslr.org/113/) | 인터뷰형 자연발화 | 서울말 자연발화 + 음소 수준 TextGrid Label | 음성학 분석 |

### 3.2 Script Characteristics

조사한 데이터셋은 발화 형태에 따라 다음과 같이 구분할 수 있습니다.

- **정제된 낭독 중심**: 다화자 음성합성 / 감성 및 발화 스타일 / KSS / Deeply Korean Read Speech / Zeroth Korean / 숫자 패턴 발화
- **자연발화·대화 중심**: KsponSpeech / Seoul Corpus
- **특정 발음 특화**: 한국인 외래어 발화 / 숫자 패턴 발화
- **언어·지역적 다양성 특화**: 한영 혼합 인식 / 중·노년층 한국어 방언

특히 숫자 패턴 데이터는 동일한 숫자라도 **비밀번호·주소·날짜·금액·비율·스포츠 기록 등 문맥에 따라 읽는 방식이 달라지는 한국어 특성**을 포함하고 있으며, 한영 혼합 인식 데이터는 한국어 대화 속 영어계 표현과 영어 원형을 함께 제공하여 Code-switching 분석에 활용할 수 있습니다.

### 3.3 Selected Training Dataset

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

공개 X-Voice 코드에서는 `to_syl=False`가 사용되고 있었으며, 분해된 Jamo가 eSpeak로 전달되면서 **비정상적인 IPA Sequence**가 생성되는 것을 확인했습니다.


Model Weight와 Inference Setting을 고정하고 `to_syl`만 `True`로 변경한 결과,

| Model / Setting | WER ↓ | SIM-o ↑ |
|---|---:|---:|
| Paper X-Voice Stage 1 | 2.42 | 0.723 |
| Base 600K (Reproduced) | 12.131 | 0.7197 |
| `to_syl=True` | **3.016** | - |

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

---

## 7. Evaluation

X-Voice Multilingual Benchmark를 이용하여 **Intra-lingual / Cross-lingual** 조건에서 Base 600K와 최종 **Replay + SylFix 15.5K** 모델을 비교했습니다.

평가 지표는 다음과 같습니다.

- **WER ↓**: 발음 및 내용 정확도
- **SIM-o ↑**: Reference와 Generated Speech 사이의 Speaker Similarity

### 7.1 Intra-lingual Results

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

### 7.2 Cross-lingual Results

한국어와 영어 사이의 Cross-lingual Voice Cloning 성능을 비교했습니다.

<p align="center">
  <img src="cross-lingual%20result.png" width="800" alt="Cross-lingual Evaluation Results" />
</p>

| Language Pair | Base WER ↓ | Ours WER ↓ | Base SIM-o ↑ | Ours SIM-o ↑ |
|---|---:|---:|---:|---:|
| KO→EN | **2.921** | 3.360 | **0.4497** | 0.4488 |
| EN→KO | 4.464 | **3.278** | 0.4727 | **0.4781** |

**EN→KO**에서는 발음 정확도와 Speaker Similarity가 모두 향상된 반면, **KO→EN**에서는 일부 성능 Trade-off가 나타났습니다.

---

## 8. Conclusion

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
