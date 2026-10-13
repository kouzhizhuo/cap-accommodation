# CORPORA survey (P3 accommodation) -- final, 2026-09-29

Legend: Use = (a) authentic prompts, (b) held-out detector validation, (c) code to reuse. Verified = Y only if the page/card/repo file was opened this session; otherwise N/UNVERIFIED.
Project rule: no gated models, no HF logins. Every item needing an account, form, licence request or DUA is repeated under "NEEDS LOGIN / USER ACTION".
Caveats on method: GitHub REST API (unauthenticated) was rate-limited, so licences were read from raw LICENSE files / repo pages; Semantic Scholar 429; DuckDuckGo CAPTCHA; english-corpora.org 403; lingtools.uoregon.edu connection refused from this machine. The CC licence shown on an arXiv abstract page is the licence of the PAPER, not of any dataset -- not used as a data licence below.

## 1. African American English (AAE / AAVE)

| Resource | Variety | URL | Licence | Access | Size | Use | Verified |
|---|---|---|---|---|---|---|---|
| CORAAL v2023.06 (Kendall & Farrington) | AAE speech transcripts (sociolinguistic interviews; components DCA/DCB/PRV/ROC/VLD/ATL/LES/DTA, component list not opened) | https://oraal.github.io/coraal (old URL https://oraal.uoregon.edu/coraal 301s here); downloads http://lingtools.uoregon.edu/coraal | CC BY-NC-SA 4.0 (stated on project page) | Open, no registration stated; per-component zips; download host refused connection from this machine (retry from another network) | >220 interviews, speakers born 1888-2005; hours/words not on main page; /components page 404 | (a) speech-style AAE turns, not chat-style; (b) good for copula deletion/habitual be/negative concord detectors | licence+description Y; links/size N |
| TwitterAAE-full-v1 (Blodgett, Green, O'Connor 2016) | AAE / Hispanic / Asian / White-aligned tweets w/ demographic posteriors | http://slanglab.cs.umass.edu/TwitterAAE/ | "research purposes only" + citation (no SPDX) | Open zip, full tweet text (not IDs) | 5.5 GB | (a) authentic AAE tweets (filter AAE proportion high, e.g. >0.8); (b) yes | Y |
| UD-TwitterAAE (Blodgett et al. ACL 2018) | 250 AAE + 250 mainstream-American tweets, UD 2.0 | same page (TwitterAAE-UD-v1.zip) | research only + citation per page; Fleisig et al. App. D say the AAE data used was under MIT (not confirmed in zip) | Open | 500 tweets | (b) small gold set; (a) tiny | page Y; licence partly N |
| slanglab/twitteraae (demographic dialect model) | AAE proportion scorer | https://github.com/slanglab/twitteraae | MIT (repo page) | Open; Python 2.7 only (port needed) | model/vocab + count tables | (c) reusable AAE-ness scorer / detector baseline | Y |
| Groenwold et al. 2020 AAVE/SAE pairs | 2,019 intent-equivalent AAVE (from TwitterAAE) / SAE pairs, MTurk-translated | https://github.com/sophiegroenwold/AAVE_SAE_dataset (only sae_samples.txt public "to comply with Twitter ToS"); full pairs = optional supplementary zip at https://aclanthology.org/2020.emnlp-main.473/ | repo: no licence stated | supplement zip is on the ACL Anthology page (open, no login) | 2,019 pairs | (a)/(b) AAE tweets with SAE glosses; the supplement is what Hofmann et al. consumed | Y (repo, anthology page); supplement contents N |
| furquan/dialectic-preferences-bias-aae-sae-parallel (HF, third party) | AAE<->SAE tweet pairs (provenance not stated; likely Groenwold-derived) | https://huggingface.co/datasets/furquan/dialectic-preferences-bias-aae-sae-parallel | MIT (card metadata) | Public, no login | 4,992 pairs, ~1 MB parquet | (a)/(b) with provenance caveat | Y (API metadata) |
| ReDial (Lin et al., ACL 2025 long 317; arXiv:2410.11005) | AAVE rewrites (human) of HumanEval/GSM8K/etc. | HF https://huggingface.co/datasets/fangrulin/redial ; https://huggingface.co/datasets/fangrulin/asynchow_aave | MIT (both cards) | Public, ungated, no login | redial: math 300, comprehensive 240, logic 362, algorithm 314 pairs (original+AAVE, vanilla+CoT splits) ; asynchow_aave 240 rows | (a) task-style AAVE queries (instruction-like, not chat); (b) AAVE/SAE minimal pairs for detector validation | Y (API + anthology + arXiv abs) |
| SALT-NLP/CoQA_AAVE (HF) | AAVE version of CoQA (VALUE) | https://huggingface.co/datasets/SALT-NLP/CoQA_AAVE | none on card metadata | Public, ungated | coqa-train/dev json | (a)/(b) synthetic (rule-based) AAVE -- not authentic | Y (file list); licence N |
| VALUE (Ziems et al. 2022) | AAE (rule-based transforms of GLUE) | https://github.com/SALT-NLP/value | no LICENSE file at /main/LICENSE (404); UNVERIFIED | data behind a "Data Use Agreement" Google Form (see NEEDS LOGIN) | 10 commits, 9 stars | (c) transformation code; data gated | Y (repo page) |
| Blodgett AAE-to-SAE conversion resources | -- | none found on TwitterAAE page | -- | -- | -- | -- | Y (negative) |

## 2. Singlish / Colloquial Singapore English

| Resource | Variety | URL | Licence | Access | Size | Use | Verified |
|---|---|---|---|---|---|---|---|
| CoSEM (Gonzales et al., World Englishes 2023, doi:10.1111/weng.12534) | Singlish text messages 2016-2022, with age/gender/race/year metadata | https://github.com/wdwgonzales/CoSEM | CC BY-NC-SA 4.0 (repo README) | Open GitHub zip (hierarchical text, AntConc-style), no login | ~900k lines, ~6.9M tokens (Nov 2024); PII scrubbing imperfect | (a) BEST authentic chat-style Singlish prompts; (b) detectors (particles lah/lor/leh, copula drop) | Y |
| Colloquial Singaporean English Style Transfer (Liang et al., ACL 2025 long 1309, "Singlish 140K") | formal<->Singlish sentence pairs w/ 6-aspect explanations (syntax, lexical borrowing, pragmatics, prosody, emoticons, code-switching); scraped from forums (hardwarezone etc.) | paper https://aclanthology.org/2025.acl-long.1309/ ; code+data claim https://github.com/dungxibo123/colloquial_tst | none stated (README is one line) | GitHub; repo shows test_data.csv, scripts; whether full 140K is released is UNVERIFIED | 140K claimed; repo has only test_data.csv visible | (a)/(b) if full data present; (c) MACoE-Style multi-agent code | paper Y; repo partial |
| Ng & Chan 2024 "What talking you?" | Singlish messaging texts -> English | arXiv https://arxiv.org/abs/2411.05253 (dataset "on GitHub", link inside PDF, not extracted) | dataset licence UNVERIFIED (the CC BY-NC-ND on the abs page is the paper's) | GitHub | ? | (a)/(b) | abs Y; data N |
| Pham, Pham, Meyers 2024 "Diverse Tweet Corpus of English Varieties" | Indian English, Singlish, AAE tweets | arXiv https://arxiv.org/abs/2401.11487 ; https://github.com/phamleyennhi/diverse_english_corpus (dirs: analysis, diverse_corpus, english_words_corpora; README says some files removed for anonymity) | dataset licence UNVERIFIED (CC BY-NC-SA on abs is the paper's) | GitHub | 170,800 tweets, 7 countries (paper) | (a) if diverse_corpus is populated | abs+tree Y; contents N |
| MNSC / IMDA National Speech Corpus (Wang et al. arXiv:2501.01034) | spoken Singlish (transcripts) | https://github.com/AudioLLMs/Singlish (README says HF "MNSC-V1"); HF mirror https://huggingface.co/datasets/knoveleng/nsc-singlish (third party; empty card, licence not given) | NSC licence NOT confirmed (imda.gov.sg page content not retrievable) | mirror is ungated but licence unclear; original NSC via IMDA | source NSC ~10,000 h; mirror 105,997 clips, 17.8 GB | (b) audio+transcripts; (a) speech transcripts only. Not a priority | partial |
| gabrielchua/singlish-to-english-synthetic (HF) | GPT-4-synthetic Singlish | https://huggingface.co/datasets/gabrielchua/singlish-to-english-synthetic | CC BY-NC-SA 4.0 (card) | public | <1K rows | NOT authentic; skip for (a) | Y (API) |
| NUS SMS Corpus (SEACrowd loader) | SMS incl. Singapore English | https://huggingface.co/datasets/SEACrowd/nus_sms_corpus (loader stub only: 2.32 kB, 1 visible row, no card/licence; original NUS host ctc.nus.edu.sg does not resolve) | UNVERIFIED | stub; data must come from original NUS source (not located) | original ~67k msgs (memory, UNVERIFIED) | do not rely on; use CoSEM instead | stub Y; corpus N |
| UD Singlish (Wang et al. 2017, arXiv:1705.06463) | Singlish treebank | arXiv https://arxiv.org/abs/1705.06463 (data repo not opened) | UNVERIFIED | -- | -- | (b) | N |

## 3. ICE, GLoWbE, SCOTS and other regional corpora

| Resource | Variety | URL | Licence | Access | Size | Use | Verified |
|---|---|---|---|---|---|---|---|
| ICE components via Univ. Zurich | Canada, Jamaica, Hong Kong, East Africa (Kenya+Tanzania), India, Singapore, Philippines, USA (~400k words written), Ireland, SPICE-Ireland | https://www.ice-corpora.uzh.ch/en/access.html | academic licence, .edu email required, form emailed to <email>, credentials in ~10 working days; redistribution/LLM-use terms not stated on page | REGISTRATION / licence form | ~1M words each (500 texts x ~2,000 words; 300 spoken, 200 written) | (a) social letters section (as Fleisig et al. did) for Indian, Jamaican, Nigerian, Kenyan, Irish; (b) yes | Y |
| ICE-GB / NZ / Sri Lanka / Uganda / Gibraltar | -- | same page: GB via UCL, NZ via Victoria Univ. Wellington, SL via Giessen, Uganda via Bochum, Gibraltar via Vigo | restricted, per-compiler | contact compilers | ~1M | not needed; ICE-GB is Standard British control only | Y (page listing) |
| ICE-Nigeria | Nigerian English (educated, written+spoken) | https://sourceforge.net/projects/ice-nigeria/ (files: https://sourceforge.net/projects/ice-nigeria/files/) | SourceForge lists "Other License"; readme.txt (1.1 kB) not read -- terms UNVERIFIED | OPEN direct download, no login: ICE-Nigeria-txt-and-xml-files_version_Nov_03_2015.zip (12.2 MB); also POS-tagged written zip (4.8 MB), raw files, ELAN spoken | ~1M words, 500 texts | (a) social letters section = best open Nigerian English prompts (matches Fleisig recipe); (b) yes. Read readme.txt before redistributing | Y |
| GLoWbE | 20 countries web English (US, GB, AU, IN, SG, NG, JM ...) | https://www.english-corpora.org/glowbe/ (403 from here); https://www.corpusdata.org/ | commercial-ish licence for full text; price/terms not verified | web interface free w/ registration; downloadable full-text is PAID | 1.8B words, 1.8M pages, ~60% blogs | (a) blogs per-country (noisy, includes non-native variance); web search UI only w/o purchase | partial (size only) |
| SCOTS Corpus | Scots / Scottish English | https://www.scottishcorpus.ac.uk/ (Terms page URL guesses gave 404) | Fleisig et al.: "licence permits use of the data for research purposes"; page terms not read | Open web, no registration mentioned; bulk download not confirmed | ~4.6M words (+ CMSW 5.4M written) | (a) Scots correspondence/letters (used by Fleisig); (b) yes | Y (size); licence secondary |
| Scots Wikipedia dump | "Scots" -- WARNING quality: widely known to be largely non-native pseudo-Scots (2020 controversy; from memory, UNVERIFIED) | https://dumps.wikimedia.org/scowiki/latest/ | Wikipedia default CC BY-SA (from memory; not read this session) | Open | scowiki pages-articles 52.2 MB bz2 (dump 2026-09-01) | NOT recommended as authentic; only as a synthetic-Scots contrast | dump Y; licence N |
| britllm/*_scots (arc_scots, truthfulqa_scots, piqa_scots) | Scots translations of benchmarks | https://huggingface.co/datasets/britllm/arc_scots | none in card metadata | public | ARC-E/C train/val/test | (a) task prompts, translated (not authentic) | Y (file list) |

## 4. Nigerian Pidgin / Jamaican / other

| Resource | Variety | URL | Licence | Access | Size | Use | Verified |
|---|---|---|---|---|---|---|---|
| UD_Naija-NSC | Nigerian Pidgin treebank (spoken) | https://github.com/UniversalDependencies/UD_Naija-NSC | CC BY-SA 4.0 | open | 9,242 sents / 140,729 tokens | (b) Pidgin detector validation | Y |
| NaijaSenti-Twitter | Nigerian Pidgin (plus Hausa, Igbo, Yoruba) tweets | https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter | CC BY-NC-SA 4.0 (card metadata); mteb/NaijaSenti mirror tagged cc-by-4.0 (disagree -- use the original) | public (loader script; files hosted via GitHub) | ~30k tweets per language | (a) authentic Pidgin tweets | Y (API) |
| PidginUNMT (Ogueji & Ahia) | English-Pidgin unsupervised MT | https://github.com/keleog/PidginUNMT | LICENSE file exists, type UNVERIFIED | open | eval 2,101 pairs (JW300); corpus/ folder | (c) minor | Y (README) |
| asr-nigerian-pidgin/nigerian-pidgin-1.0 | Nigerian Pidgin speech+text | https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0 | cc-by-4.0 (tag) | public | ? | (b) | tag only |
| michsethowusu/english-nigerian-pidgin_sentence-pairs_mt560 | En-Pidgin pairs | HF | cc-by-4.0 (tag) | public | ? | translated/parallel | tag only |
| eWAVE 3.0 (Kortmann, Lunkenheimer, Ehret 2020) | 77 English varieties + 26 pidgins/creoles, 235 features | https://ewave-atlas.org/ ; https://github.com/clld/ewave | CC BY 3.0 Unported (site) | open, CLDF | 235 features x 103 varieties | (b)/(c) feature inventory; per-variety feature lists to build detectors (Jamaican, Nigerian, Indian, Singapore, Chicano, Appalachian ...) | Y (site); DOI in tool output N |
| Chicano English, Appalachian English, Jamaican | -- | no open authentic-text corpus located this session. Candidates NOT verified: ICE-Jamaica (Zurich licence above), Linguistic Atlas Project, SLAAP (NCSU) | -- | -- | -- | use Multi-VALUE rule output and eWAVE features as proxies, flag as synthetic | N |

## 5. Benchmarks and multi-dialect toolkits

| Resource | Variety | URL | Licence | Access | Size | Use | Verified |
|---|---|---|---|---|---|---|---|
| Multi-VALUE (Ziems et al. ACL 2023, arXiv:2212.08011) | 50 dialects / 189 features; code classes include aave, appalachian, chicano, indian, singapore (Southeast American Enclave, ...) | https://github.com/SALT-NLP/multi-value ; pip `value-nlp`; http://value-nlp.org | Apache-2.0 (raw LICENSE read) | open | Python 3.10; spaCy, NLTK WordNet | (c) rule-based dialect transformer to produce synthetic dialect stimuli for ALL target varieties (a fallback where no authentic corpus exists); README mentions gold CoQA in Chicano+Indian English but hosting location NOT stated in README | Y; gold-CoQA location N |
| EnDive (Gupta et al., arXiv:2504.07100) | AAVE, ChcE, CollSgE, IndE, JamE; LLM-translated (few-shot) benchmarks | https://huggingface.co/datasets/abhaygupta1266/EnDive_All_Datasets (viewer broken; CSVs with BLEU scores); ren1034/EnDive is an empty stub | none stated | public, no login | per-task/per-dialect CSVs (BoolQ, COPA, FOLIO, GSM8K, HumanEval, Logic Bench, MBPP ...) | (a) SYNTHETIC only; venue claim "EMNLP Findings 2025" NOT confirmed (arXiv abs lists none; my guessed anthology ID was wrong) | Y (file list) |
| DIALECT-BENCH (Faisal et al. 2024, arXiv:2403.11009) | 281 varieties, 10 text tasks; English side is only UD_English-EWT (standard) as baseline | https://arxiv.org/abs/2403.11009 ; https://github.com/ffaisal93/DialectBench ; site repo ffaisal93/DialectBench.io | Apache-2.0 (repo page) | open; `bash download_data.sh --task all` | multi-task | low value for this project (no English dialect prompts) | Y |
| SALT-NLP/CoQA_AppE, CoQA_ChcE, CoQA_CollSgE, CoQA_IndE, CoQA_Multi, CoQA_AAVE | Appalachian, Chicano, Colloquial Singapore, Indian, multi-dialect, AAVE versions of CoQA | https://huggingface.co/datasets/SALT-NLP/CoQA_ChcE (same pattern for the others) | none on cards (upstream CoQA licence is mixed per domain; check before redistribution) | public, ungated, no login | each: coqa-dev-v1.0.json ~7.7 MB, coqa-train-v1.0.json ~58 MB | (a)/(b) only as controlled dialect-transformed stories/questions; README says "gold" Chicano/Indian variants were made with native speakers -- UNVERIFIED which of these files are gold vs rule-based | Y (file listing via HF API); provenance N |
| SALT-NLP/value (HF) | -- | https://huggingface.co/datasets/SALT-NLP/value | -- | ungated but EMPTY (only .gitattributes); VALUE data is behind the DUA form | -- | -- | Y |

## 6. Hofmann et al. 2024 (Nature; arXiv:2403.00742) -- dialect prejudice

| Item | Detail |
|---|---|
| Repo | https://github.com/valentinhofmann/dialect-prejudice (70 commits, 39 stars) |
| Licence | MIT, Copyright (c) 2024 Valentin Hofmann (raw LICENSE read) |
| Top level | data/ demo/ notebooks/ perplexity/ probing/ scripts/ LICENSE README.md requirements.txt |
| data/pairs | ONLY habitual.txt (1,200 tab-separated lines, e.g. "he be cracking<TAB>he's usually cracking"). The tweet pairs of Blodgett 2016 / Groenwold 2020 are NOT shipped; README: place them as tab-separated files in data/pairs (Groenwold supplement, see section 1) |
| data/attributes | guilt.txt, katz.txt, occupations.txt, penalty.txt |
| probing/ | helpers.py, mgp.py, mgp_gpt3.py, mgp_gpt4.py, prompting.py (prompt templates live in prompting.py) |
| Run | Python 3.10; `python3.10 mgp.py --model $model --variable $variable --attribute $attribute --device $device`; OpenAI scripts need an API key in `.env` (python-decouple) |
| Pinned deps | adjustText 0.8, ipykernel 6.29.2, matplotlib 3.6.2, numpy 1.22.4, openai 0.28.1, pandas 1.5.2, python-decouple 3.8, scipy 1.7.3, seaborn 0.12.1, sentencepiece 0.2.0, statsmodels 0.13.2, tiktoken 0.5.2, torch 2.1.2, tqdm 4.64.1, transformers 4.36.2 |
| Use | (c) matched-guise probing scaffold + attribute lists; HF models it loads via transformers must be ungated ones (choose accordingly). Verified: Y |

## 7. Fleisig et al. 2024 (arXiv:2406.08818v3), "Linguistic Bias in ChatGPT"

- No released code or data found: full PDF text has no GitHub/HF link; GitHub repo search for the title returned 0. Status: not released, as far as could be checked (authors' own pages not checked).
- Reusable recipe (App. D of the PDF): Nigerian, Jamaican, Indian, Irish, Kenyan = ICE social letters; SAE/SBE = Reddit US/UK city subreddits (Zhang 2023); AAE = Blodgett et al. 2018 (UD-TwitterAAE); Scots = SCOTS correspondence/letters; Singlish = CoSEM text messages. They report an ICE licence for non-profit academic research, SCOTS research-use licence, SAE/SBE/AAE MIT, CoSEM CC BY-NC-SA 4.0. Ten varieties; models GPT-3.5/GPT-4; native-speaker ratings via Prolific.

## 8. LLM accommodation / linguistic style matching code

No released LLM-vs-dialect accommodation / LSM code found. Searches done: GitHub repo search "language style matching LSM function words" (0), "Jurgens-Lab style matching reddit" (0), "linguistic style matching LLM" (only ZhanxyR/TTM, a role-play repo, unrelated); HF dataset search "style matching"/"accommodation" (nothing relevant); arXiv title search. Relevant papers only (no code link found on abs pages): Ananthasubramaniam et al. 2023 "Exploring Linguistic Style Matching in Online Communities" (arXiv:2307.02758; LSM = function-word/formality mirroring on Reddit threads) and Niederhoffer & Pennebaker/Ireland-style LSM (LIWC-based, from memory). Recommendation: implement LSM ourselves (per-category function-word rate, LSM = 1 - |a-b|/(a+b+eps), averaged) with an open function-word list rather than the proprietary LIWC dictionary; verify no LIWC dependency is introduced. Verified: negative search results only.

## NEEDS LOGIN / USER ACTION (running list)

1. ICE (Zurich) -- Indian, Jamaican, Singapore, Kenya/Tanzania, Irish, Hong Kong, Philippines, Canada: fill the licence form and email <email> from a .edu address; ~10 working days. Why: only route to authentic Indian/Jamaican English social letters (the Fleisig recipe). ICE-GB (UCL) is a separate request (Standard British control only; can use Reddit UK instead).
2. VALUE (AAE GLUE) data: Data Use Agreement Google Form linked from https://github.com/SALT-NLP/value. Optional; Multi-VALUE code and the SALT-NLP/CoQA_* HF files need nothing.
3. GLoWbE full text: paid licence via english-corpora.org; the site also returned 403 to us and the web UI requires an account. Not recommended.
4. CORAAL: no login required, but http://lingtools.uoregon.edu/coraal refused connections from this machine; user should download from another network (or the GPU box) and place under data_sources/. Confirm component list/sizes then.
5. IMDA National Speech Corpus: licence terms unconfirmed (IMDA page content not retrievable); user should read the IMDA NSC page before we touch the HF mirror. Low priority.
6. Groenwold et al. 2020 AAVE side: obtain the supplementary zip from https://aclanthology.org/2020.emnlp-main.473/ (no login, but a manual download; needed to run Hofmann's matched-guise pairs on tweets).
7. Twitter/TwitterAAE terms: full-text zip is "research purposes only"; user must be comfortable with that before we train/publish on it.
8. HF: nothing needs an HF login or a gated model. Verified gated=False via the API for redial, furquan/*, EnDive_All_Datasets, britllm/arc_scots, nigerian-pidgin-1.0, mt560 pairs, mteb/NaijaSenti, SALT-NLP/CoQA_*. HausaNLP/NaijaSenti-Twitter and knoveleng/nsc-singlish were also gated=False.
9. OpenAI/Anthropic API keys are needed only if we run Hofmann's mgp_gpt3/gpt4 scripts; use open ungated local models instead.

## RECOMMENDATIONS (per variety)

| Variety | (a) authentic prompts | (b) detector validation |
|---|---|---|
| AAE | TwitterAAE-full (filter AAE proportion), UD-TwitterAAE (500), Groenwold pairs; ReDial (task-style) | UD-TwitterAAE + Groenwold/ReDial minimal pairs + twitteraae scorer; CORAAL once downloadable |
| Singlish | CoSEM (CC BY-NC-SA, open); Liang 2025 140K only if colloquial_tst really ships it | CoSEM particles (lah/lor/leh); ACL-1309 explanations if released |
| Nigerian English | ICE-Nigeria (open on SourceForge) | same, plus eWAVE features |
| Nigerian Pidgin (optional) | NaijaSenti-Twitter (pcm) | UD_Naija-NSC (CC BY-SA 4.0) |
| Indian, Jamaican | ICE-India/ICE-Jamaica (licence form); fall back to Multi-VALUE / SALT-NLP CoQA_IndE (synthetic) | eWAVE features; ICE after licence |
| Scots/Scottish | SCOTS correspondence/letters (research-use terms; check bulk download) | SCOTS; Scots Wikipedia NOT authentic |
| Chicano, Appalachian | no authentic open corpus found: use Multi-VALUE (chicano, appalachian) + SALT-NLP CoQA_ChcE/AppE, flagged synthetic | eWAVE features only |
| SAE / SBE controls | Reddit US/UK city subreddits (Zhang 2023; not retrieved) or UD-TwitterAAE mainstream half | -- |
