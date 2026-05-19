# State-of-the-Art Methods for Monocular RGB Hand-Contact Detection and Zero-Shot Cross-Scenario Recognition (2023–2026)

## TL;DR
- The closest published match to a JCST-style 2D-distance + appearance + online-baseline contact detector is **Palmpad (He et al., CHI 2025)**, which uses MediaPipe landmark-driven crops of the index fingertip and opposite palm plus per-frame optical flow into a ResNet to reach 97.0% accuracy on index-to-palm touch from a single RGB camera; it should be the primary baseline.
- For zero-shot cross-scenario generalization (writing vs tapping vs gesturing), the strongest building blocks are **EgoTouch (Mollyn & Harrison, UIST 2024)** with calibration-free DIP-to-fingertip scale normalization, **PressureVision++ (Grady et al., WACV 2024)** with weak-label prompt supervision that transfers across surfaces, and **EgoChoir (Yang et al., NeurIPS 2024)** which uses gradient-modulated cross-attention to adaptively weight motion vs. appearance cues across scenarios.
- An online-baseline / anomaly-style formulation is well supported in adjacent areas: TriPad (CHI 2024) uses initial dwell frames to define a touch plane; "Future Frame Prediction" (Liu et al., CVPR 2018) and "Look at Adjacent Frames" (Ouyang, Shen & Sanchez, ECCV-W 2022) provide the contrastive deviation-from-baseline template; combining these with **relative** (rather than absolute) 2D fingertip–palm distances yields a clean blueprint for a domain-invariant JCST submission.

## Key Findings

### 1. Monocular RGB hand-contact detection — current SOTA

| System | Venue / Year | Modality | Key idea | Reported accuracy |
|---|---|---|---|---|
| **Palmpad** (He, Wang, Shi, Hsia, Liang, Yu) | CHI 2025 | Single RGB camera | MediaPipe-cropped index-fingertip & palm patches + optical flow into ResNet | 97.0% acc, F1 ≈ 96.1% |
| **EgoTouch** (Mollyn, Harrison) | UIST 2024 | RGB passthrough HMD | MediaPipe 21-keypoint hand, DIP–tip distance scale normalisation, CNN on cropped fingertip image | 94.9% frame-wise; TP 96.4%, FP 5.6% |
| **TouchInsight** (Streli, Richardson, Botros, Ma, Wang, Holz) | UIST 2024 | Egocentric HMD vision | Uncertainty-aware Gaussian touch model fused with language priors | F1 = 0.99 touch, F1 = 0.96 finger ID, mean 2D error 6.3 mm, < 70 ms latency |
| **StegoType** (Richardson et al., Meta) | UIST 2024 | Egocentric vision | Surface-typing decoder over hand-tracking pose only | Ten-finger surface typing |
| **PressureVision++** (Grady, Collins, Tang, Twigg, Aneja, Hays, Kemp) | WACV 2024 | Single RGB | Weak "press-prompt" labels; pressure regression across 51 participants / diverse surfaces | 89.3% contact accuracy on diverse surfaces |
| **EgoPressure** (Zhao, Kwon, Streli, Pollefeys, Holz) | CVPR 2025 | Egocentric RGB(-D) | 5-h dataset (21 participants) + baseline pressure & pose joint estimation | Establishes egocentric pressure benchmark |
| **EclipseTouch** (Mollyn, DeVrio, Harrison) | UIST 2025 | RGB + worn IR emitter | Structured IR shadow → hover-distance mean error 6.9 mm; touch acc 98.0% | Not strictly passive RGB |
| **ShadowTouch** (Liang, Wang, Li, Hsia, Fan, Yu, Shi) | UIST 2023 | RGB + wrist illuminant | Forward-facing wrist LED casts shadows on surface; two-stage CNN | 99.1% acc, 96.8% F1, ~2 mm sensitivity |

**Soft-tissue deformation handling.** EgoTouch is explicit that finger-press appearance cues — fingertip color loss, hyperextension of the DIP joint, fingerpad widening, change in skin texture, and local shading near the contact patch — are the principal monocular cues. These are the same cues exploited by PressureVision/PressureVision++. None of these systems use monocular depth estimation; they exploit local appearance around the contact region instead. EgoTouch additionally hand-crafts a **DIP-to-fingertip-length normalization** that effectively cancels camera-distance variation, a critical step because soft-tissue cues scale with image resolution.

**Real-time properties.** Palmpad samples at 30 Hz (≈ 33 ms/frame) on a ResNet backbone. TouchInsight reports < 70 ms mean detection latency for HMD-grade inference. ShadowTouch's two-stage CNN runs at headset frame rates. Strictly < 10 ms/frame is only achievable today with very small CNNs on hand-cropped patches (Palmpad's ablation suggests the ResNet head, not MediaPipe, dominates latency) or with shallow MLPs over distance features alone. The 2D-distance + appearance fusion proposed in the brief is well aligned with this latency target.

**Avoiding monocular depth.** None of the listed state-of-the-art monocular RGB systems rely on per-pixel monocular depth estimation. They use (a) hand keypoints/MANO meshes for geometry (Palmpad, EgoTouch, TouchInsight, TriPad), (b) appearance crops around the fingertip (EgoTouch, PressureVision++), or (c) external optical cues (ShadowTouch, EclipseTouch). This corroborates the design choice to skip monocular depth.

### 2. Zero-shot / domain-invariant contact & gesture recognition

- **EgoChoir** (Yang et al., NeurIPS 2024 — *Capturing 3D Human-Object Interaction Regions from Egocentric Views*) explicitly addresses cross-scenario contact estimation. Its **gradient-modulation tokens** dynamically reweight appearance, head-motion, and 3D-object branches per scenario, so the network adopts the most informative cue for hand vs. body interactions. Vertex-wise contact F1 = 0.76, affordance AUC = 78%. This is the most direct precedent for **adaptive multi-cue weighting** in a hand-contact pipeline.
- **PressureVision++** (Grady et al., WACV 2024) demonstrates near zero-shot transfer across surfaces using a weak-supervision "prompt people to apply pressure" paradigm; one of the first vision-based contact systems to test outside its training distribution.
- **Practical Single-Domain Generalization via Training-time and Test-time Learning** (Yang et al., ACM SIGKDD 2024, DOI 10.1145/3637528.3671806) and **Improved Test-Time Adaptation for Domain Generalization** (Chen, Zhang, Song, Shan, Liu — CVPR 2023) provide the canonical training-/test-time adaptation toolkit (entropy minimization, batch-norm statistics on-the-fly) — directly portable to a contact classifier.
- **GestureGPT: Toward Zero-Shot Free-Form Hand Gesture Understanding with Large Language Model Agents** (Zeng, Wang, Zhang, Yu, Zhao, Chen — Best Paper, ACM ISS 2024; *Proc. ACM Hum.-Comput. Interact.* vol. 8, DOI 10.1145/3698145) is the leading LLM-based zero-shot free-form gesture *understanding* system; it does not solve low-level contact, but provides a complementary semantic layer for action recognition built on top of a contact event stream.
- For **online normalization** specifically, the most relevant published HCI move is EgoTouch's per-frame DIP-to-fingertip rescaling. Time-windowed running normalization of fingertip–palm distance has not been published in a CVPR/UIST/CHI venue we could find — this is precisely the gap a JCST paper can fill.

### 3. Online adaptive baselines / hover-as-reference / anomaly formulations

- **TriPad** (Dupré, Appert, Rey, Saidi, Pietriga — CHI 2024) is the cleanest published example of "use initial idle/hover frames as reference": users dwell with thumb-middle-pinky on a surface to fit a plane, and subsequent touches are defined as fingertip proximity to that plane. Their evaluation reports 99.0% click accuracy and zero machine-learning, zero per-scene scanning. The model is precisely the **hover-as-baseline** formulation requested in the brief.
- **Future Frame Prediction for Anomaly Detection — A New Baseline** (Liu, Luo, Lian, Gao — CVPR 2018) is the canonical contrastive-deviation backbone: train a predictor on "normal" frames, flag drift between prediction and ground truth as anomaly. Directly transferable to "predict next-frame fingertip–palm distance under hover; deviation = contact".
- **Look at Adjacent Frames: Video Anomaly Detection without Offline Training** (Ouyang, Shen & Sanchez — University of Warwick, ECCV 2022 Workshops, LNCS vol. 13805 / arXiv 2207.13798) builds an incremental learner whose adaptation rate distinguishes gradual (normal) from drastic (anomalous) frame-to-frame shifts. The same incremental-learner-difficulty signal can be wired into a contact detector to separate hover drift from a contact transient.
- **Breakpoint based online anomaly detection** (Etienne Krönert, Dalila Hattab — Worldline FS Lab; Alain Celisse — Paris 1 Panthéon-Sorbonne University; arXiv 2402.03565, Feb/Jul 2024) gives a principled change-point statistics framework with controlled FDR — useful for time-stamping the touch-down event precisely.

### 4. Palm-on-hand / skin-surface writing systems

- **Palmpad** (He et al., CHI 2025) — the most direct match. RGB only; index-fingertip-to-palm tap and swipe; MediaPipe crop + optical flow + ResNet; 97.0% accuracy, F1 ≈ 96.1%; outperforms Meta Quest native hand tracking; 211-min dataset (16 participants) released on Hugging Face. Authors note their model "surpasses similar work" — concretely, they reproduced an unnamed prior method (their reference [62], whose dataset was not released) on the Palmpad data and obtained only 89.1% accuracy / 85.3% F1, illustrating how brittle prior approaches are without scenario-matched training data.
- **HandPad: Make Your Hand an On-the-go Writing Pad via Human Capacitance** (Lu, Ding, Pan, Li, Zhou, Fu, Zhang, Chen, Xue — UIST 2024) turns the back of the hand into a writing pad via **human-capacitance** sensing (not RGB), feeds bi-LSTM+ResNet for letters/digits/Chinese characters with 99.1%/97.6%/97.9% accuracy. Vital baseline because it solves the *handwriting recognition* portion of the problem with miniature electrodes; vision-based replacement is an open problem.
- **Handwriting for Text Input and the Impact of XR Displays, Surface Alignments, and Sentence Complexities** (Kern, Tschanter, Latoschik — IEEE TVCG 2024; PubMed 38442066) is the canonical recent XR handwriting study; compares physically-aligned vs mid-air surfaces in VR and VST-AR with 72 participants. It establishes user-experience benchmarks that an on-palm system should beat.
- **Finger-to-Palm: Single-Handed Touch Gestures Using the Palm** (Hsu & Chan — CHI EA 2025) is the recent gesture-taxonomy reference for finger-on-palm interaction (uses pressure sensors).
- **PalmGesture** (Wang et al., MobileHCI 2015), **PalmType** (MobileHCI 2015) and **PalmBoard** (Yi et al., CHI 2020) remain the standard historical baselines.

### 5. Multi-cue fusion for contact detection (geometric + appearance + kinematic)

- **EgoChoir** (NeurIPS 2024) — parallel cross-attention with gradient-modulation tokens that adaptively reweight appearance, head motion, and 3D object cues per scenario. The most relevant template for an "adaptive weighting" contact detector.
- **TouchInsight** (UIST 2024) fuses geometric (hand-pose-derived 2D touch location) and kinematic (sequence model over hand pose) signals under a Bayesian uncertainty model and combines with language priors at decoding time.
- **PressureVision++** (WACV 2024) implicitly fuses appearance and pose — its model uses the global image and learns to localize the contact via the hand's appearance.
- **LI-TFMNet** (Z. Li & D. Shou, *Information Fusion* vol. 115, March 2025) introduces an **adaptive-weight residual multi-head cross-attention (ARMHCA)** fusion strategy that refines within-modal features and exploits inter-modal correlations across temporal modalities — methodologically a clean recipe for adaptive geometric/appearance/kinematic fusion.
- **Domain Adaptation with Contrastive Simultaneous Multi-Loss Training for Hand Gesture Recognition** (Sensors 2023) — two-headed cross-entropy + contrastive loss network for cross-user gesture generalisation; useful as a contrastive backbone for cross-scenario contact training.

## Details

### 1.A. Palmpad in depth (the primary precedent)
He, Wang, Shi, Hsia, Liang, Yu, *Palmpad: Enabling Real-Time Index-to-Palm Touch Interaction with a Single RGB Camera*, CHI 2025 (Article 551, DOI 10.1145/3706598.3714130; dataset at huggingface.co/datasets/Teburile/Palmpad_Dataset). Pipeline: (i) MediaPipe Hand on every frame; (ii) cropped right-index-fingertip region and left-palm region; (iii) per-frame optical flow over consecutive fingertip crops; (iv) ResNet (ImageNet-initialised) + FC classifier → touch / no-touch. Ground-truth labelling used an AC circuit through the user's body. Reported 97.0% accuracy (SD = 2.0%), F1 ≈ 96.1%; in usability evaluation 95.3% accuracy in unconstrained MR use. The authors reproduced the best prior method they could find ([62] in their references, whose dataset was unreleased) on their own data and obtained 89.1% accuracy / 85.3% F1 — quantifying the margin a new method must clear. 25% of remaining error stems from MediaPipe tracking inaccuracies; that is the bottleneck most amenable to **2D geometric-distance smoothing and online baseline normalization**.

### 1.B. EgoTouch in depth (the RGB-only on-skin reference)
Mollyn & Harrison, *EgoTouch: On-Body Touch Input Using AR/VR Headset Cameras*, UIST 2024 (DOI 10.1145/3654777.3676455; arXiv 2509.01786). The system uses 21 MediaPipe keypoints, gates input on a "fingertip within ~3 palm-lengths of the opposing palm" condition, then rotates and scales the input image based on the **DIP-to-fingertip distance** before passing a cropped patch to a CNN. Reports mean frame-wise accuracy 94.9% (SD = 3.5%), TP = 96.4%, FP = 5.6%; calibration-free across skin tones, lighting, and body motion (including walking). The DIP-tip normalisation is precisely the "online normalization" idea the JCST design is converging on — adopt this and extend it to a **temporally-running** version that estimates the baseline distance over a short hover window.

### 1.C. TouchInsight (uncertainty-aware geometric fusion)
Streli, Richardson, Botros, Ma, Wang, Holz, *TouchInsight*, UIST 2024 (DOI 10.1145/3654777.3676330). Predicts a 2D Gaussian over touch location with explicit uncertainty, sums log-probabilities with character LM priors at decoding time. Reports touch-event F1 = 0.99, finger-ID F1 = 0.96, mean 2D location error 6.3 mm, < 70 ms latency. The Bayesian uncertainty representation is the right template for a robust cross-scenario contact classifier that needs to "know when it does not know".

### 2.A. EgoChoir's adaptive cue weighting
Yang et al., *EgoChoir: Capturing 3D Human-Object Interaction Regions from Egocentric Views*, NeurIPS 2024 (arXiv 2405.13659). Modality-wise features (appearance, head-motion, object geometry) are mapped into interaction clues that query each other through parallel cross-attention. **Gradient-modulation tokens** adjust the gradients of specific layers so the network selects appropriate clues per scenario (hand vs. body interaction). Vertex-wise contact F1 = 0.76, affordance AUC = 78%. The gradient-modulation idea is directly transferable to a 2D-distance + appearance contact classifier that must auto-route to whichever cue is reliable in the current scenario.

### 2.B. Test-time adaptation for domain generalization
Chen, Zhang, Song, Shan, Liu, *Improved Test-Time Adaptation for Domain Generalization*, CVPR 2023; Yang et al., *Practical Single-Domain Generalization via Training-time and Test-time Learning*, ACM SIGKDD 2024 (DOI 10.1145/3637528.3671806). Both update batch-norm statistics or auxiliary heads at deployment using the unlabelled test stream. The simplest port to a contact classifier is: at deployment, use the first hover-only frames to **refit the BN statistics** of the appearance branch, providing scenario-specific calibration with zero labels.

### 3.A. TriPad as the touchscreen-via-dwell baseline
Dupré, Appert, Rey, Saidi, Pietriga, *TriPad*, CHI 2024 (DOI 10.1145/3613904.3642323). Users dwell with thumb-middle-pinky on a target surface; the system fits a plane to these three fingertips, then defines touch as fingertip proximity to that plane. No machine learning, no scene understanding, 99.0% click accuracy. The data point that an entirely hand-tracking-only baseline can hit 99% accuracy on flat *rigid* surfaces sets the bar — for the deformable, non-planar palm surface, additional appearance cues are required.

### 3.B. Anomaly-detection templates
- Liu, Luo, Lian, Gao, *Future Frame Prediction for Anomaly Detection — A New Baseline*, CVPR 2018: predict next frame from prior frames using appearance + optical-flow constraints; deviation = anomaly. Translation: predict expected fingertip–palm distance under hover; the actual sudden drop = contact event.
- Ouyang, Shen, Sanchez, *Look at Adjacent Frames: Video Anomaly Detection without Offline Training*, ECCV 2022 Workshops / LNCS vol. 13805 (arXiv 2207.13798): an incremental learner adapts to gradual shifts but lags drastic ones, the lag itself being the anomaly signal. Maps cleanly onto "fingertip drift while hovering ≠ touch-down transient".
- Krönert, Hattab & Celisse, *Breakpoint based online anomaly detection*, arXiv 2402.03565, 2024: principled change-point inference with controlled FDR — apply directly on the time series of 2D fingertip-to-palm distance.

### 4.A. HandPad: the writing-side counterpart
Lu, Ding, Pan, Li, Zhou, Fu, Zhang, Chen, Xue, *HandPad: Make Your Hand an On-the-go Writing Pad via Human Capacitance*, UIST 2024 (DOI 10.1145/3654777.3676328). Capacitive electrodes on the writing finger; bi-LSTM + ResNet recognizes English letters (99.1%), numbers (97.6%), and Chinese characters (97.9%) on the back-of-hand surface. Strong evidence that, *given* a reliable per-frame contact + trajectory, downstream handwriting recognition is essentially a solved problem. The JCST contribution should focus on the *vision-based* contact and trajectory extraction.

### 4.B. XR handwriting baselines
Kern, Tschanter, Latoschik, *Handwriting for Text Input and the Impact of XR Displays, Surface Alignments, and Sentence Complexities*, IEEE TVCG 2024 (PubMed 38442066). 72-participant 2×2×2 study (VR vs VST-AR, physically-aligned vs mid-air, simple vs complex sentences). Key result: physically-aligned surfaces score significantly higher on learnability and lower on physical demand; mid-air scores higher novelty but those gains may diminish with experience. The user-study template is what a JCST submission should adopt for its evaluation chapter.

### 5.A. Adaptive multi-cue fusion templates
Li & Shou, *LI-TFMNet*, *Information Fusion* vol. 115, March 2025: cross-modal branch (learnable temporal 2D-isation) + cross-domain branch (time-frequency fusion), combined by **adaptive-weight residual multi-head cross-attention (ARMHCA)** with multi-scale depthwise convolutions. SOTA on UCI-HAR. The ARMHCA module is a drop-in for fusing 2D distance, appearance crop, and kinematic-pose streams in a contact classifier, and explicitly weights features by reliability.

## Recommendations

**Staged plan for a JCST-level "zero-shot cross-scenario contact detection with 2D-distance + appearance + online baseline normalization" paper.**

1. **Adopt Palmpad's input pipeline as the baseline (Stage 1).**
   - Use MediaPipe Hand for landmarks on both hands.
   - Compute **2D Euclidean distances** between right-index-fingertip and a set of left-palm landmarks (palm center, MCPs of each finger).
   - Compute **per-frame DIP-to-fingertip length** for scale normalization (as in EgoTouch).
   - Baseline benchmark: replicate Palmpad's 97.0% on the released Hugging Face dataset (Teburile/Palmpad_Dataset).
   - **Threshold to escalate to Stage 2:** if relative-distance + appearance fusion does not exceed 97.5% in-domain on Palmpad data, the geometric-feature design is under-specified; revisit landmark selection.

2. **Add online baseline normalization (Stage 2 — the core JCST contribution).**
   - Maintain a 1–2 s sliding window of hover-frame distances; treat the *median* (robust) and IQR as the running baseline `(μ_t, σ_t)`.
   - Feed **relative features** `z_t = (d_t − μ_t) / σ_t` to the classifier rather than raw `d_t`.
   - Add a learned gating module (à la EgoChoir's gradient modulation) that down-weights the geometric branch when its variance is high (e.g., fast hand motion) and up-weights the appearance branch instead.
   - **Threshold to escalate to Stage 3:** if leave-one-scenario-out F1 (writing / tapping / gesturing) does not exceed 90%, augment with test-time BN adaptation (Chen et al., CVPR 2023).

3. **Add a contrastive anomaly head (Stage 3).**
   - Treat the first N hover frames per session as the "normal" class; train a one-class or contrastive head to flag deviations.
   - Combine with the supervised classifier via late fusion of log-likelihoods (TouchInsight pattern).
   - Use the Krönert–Hattab–Celisse breakpoint-based online anomaly framework (arXiv 2402.03565) to time-stamp touch-down events precisely with controlled FDR.

4. **Multi-cue fusion at < 10 ms/frame (Stage 4).**
   - Two streams: (a) a small MLP over 12–20 relative geometric features per frame, (b) a MobileNet-tiny over the 64×64 fingertip patch.
   - Combine with ARMHCA-style attention (Li & Shou, *Information Fusion* 2025) producing scenario-adaptive weights.
   - **Threshold to escalate to Stage 5:** if total inference exceeds 8 ms on the target device, quantize the appearance branch to INT8 or drop frames adaptively (use the 2D-distance head as a gating predictor).

5. **Evaluation protocol (Stage 5).**
   - Datasets: Palmpad (CHI 2025), EgoPressure (CVPR 2025), EgoTouch's evaluation set if released, plus a new in-the-wild palm-writing set following Kern et al.'s TVCG 2024 study design.
   - Compute leave-one-scenario-out and leave-one-user-out accuracy/F1; report **zero-shot transfer** numbers explicitly.
   - User study: 16–20 participants writing single letters/digits on the palm, comparing against TouchInsight-style mid-air and TriPad-style physical-surface conditions.

6. **What would change these recommendations.**
   - If the target device has < 50 mW power budget (smart-glasses), replace the appearance branch with the Helios event-camera approach (arXiv 2407.05206).
   - If precise pressure regression is required, swap PressureVision++ in as the appearance branch.
   - If text-entry throughput is the metric, integrate HandPad's bi-LSTM character recogniser on the trajectory stream.

## Caveats

- **Speculative dates.** Some 2026 references in the literature (e.g., a SurfaceXR arXiv preprint dated 2026, FlickPose) are pre-publication / not yet peer-reviewed; treat their numbers as provisional.
- **JCST coverage gap.** Our searches did not surface any JCST (*J. Comp. Sci. & Tech.*) papers from 2024–2026 on monocular RGB hand-on-hand contact detection specifically. Either (a) such a paper exists very recently and is not yet indexed, or (b) this is a genuine gap, which actually strengthens the case for a JCST submission on exactly this topic.
- **Strict < 10 ms/frame budget is aspirational.** Palmpad runs at ~33 ms/frame; TouchInsight reports < 70 ms latency. Hitting < 10 ms end-to-end will require careful engineering (INT8 quantization, custom MobileNet, possibly dropping the optical-flow branch). The plan above shows where to compromise.
- **MediaPipe is the geometric bottleneck.** Palmpad explicitly attributes 25% of its remaining errors to MediaPipe inaccuracies. Online baseline normalization will *partially* compensate, but a more accurate hand-tracking front-end (e.g., HaMeR, CVPR 2024) would help more on the other 75%.
- **Soft-tissue deformation cues are subtle in passive RGB.** EgoTouch achieves 94.9% accuracy, not the 99%+ that ShadowTouch (with active wrist illumination) or EclipseTouch (with worn IR) achieve. A purely passive RGB system on a non-planar deformable palm surface will likely top out near EgoTouch's range; if higher accuracy is needed, consider EclipseTouch-style worn IR as an optional sensor.
- **Cross-scenario evaluation is rare in the literature.** Even the strongest published systems (Palmpad, EgoTouch) report mostly in-domain numbers; zero-shot cross-scenario performance is essentially an open evaluation gap, and is itself a reason a JCST paper centred on this evaluation question would be timely.