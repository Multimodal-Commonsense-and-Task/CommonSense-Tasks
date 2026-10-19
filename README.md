# CommonSense-Tasks

#### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially CommonSense task.
#### This repository summarizes following researches.

## Research list
* Context-Aware Planning and Environment-Aware Memory for Instruction Following Embodied Agents (CAPEAM) (ICCV 2023) - Byeonghwi Kim, Jinyeon Kim, Yuyeong Kim, Cheolhong Min, Jonghyun Choi.

  * CAPEAM (Context-Aware Planning and Environment-Aware Memory) is an embodied instruction following agent that incorporates semantic context (e.g., appropriate objects to interact with) in a sequence of actions and maintains the changed spatial arrangement and states of interacted objects (e.g., location that the object has been moved to) in inferring the subsequent actions.

* Story Visualization by Online Text Augmentation with Context Memory (CMOTA) (ICCV 2023) - Daechul Ahn, Daneul Kim, Gwangmo Song, SeungHwan Kim, Dongyeop Kang, and Jonghyun Choi.

  * The proposed Context Memory and Online Text Augmentation (CMOTA), is a novel memory architecture for Bi-directional Transformers along with online text augmentation. This augmentation generates multiple pseudo-descriptions to provide additional supervision during training, enhancing the model's ability to generalize to language variations during inference.

* Online Continual Learning on Hierarchical Label Expansion (Hierarchical CL) (ICCV 2023) - Byung Hyun Lee, Okchul Jung, Jonghyun Choi, Se Young Chun.

  * Continual learning enables models to adapt to new tasks while retaining previously learned knowledge. We propose a novel approach called hierarchical label expansion (HLE) that allows networks to first learn coarse-grained classes before gradually expanding to more fine-grained classes at various hierarchical depths. Our experiments show that our proposed method, which uses hierarchy-aware pseudo-labeling and selective memory management, significantly outperforms existing approaches across all hierarchical levels and conventional continual learning setups.

* Multi-level Compositional Reasoning for Interactive Instruction Following (MOCHA) (AAAI 2023) - Suvaansh Bhambri*, Byeonghwi Kim*, Jonghyun Choi

  * This work introduce MCR-Agent, a robotic system that breaks down complex domestic tasks into manageable subgoals using a three-level action policy (high-level task planning, mid-level navigation/interaction control, and low-level object manipulation). This approach achieves better performance than existing methods while maintaining human-interpretable actions.

* ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments (ReALFREAD) (ECCV 2024) - Taewoong Kim*, Cheolhong Min*, Byeonghwi Kim, Jinyeon kim, Wonje Jeung and Jonghyun Choi

  * ReALFRED is a new benchmark that uses real-world scenes, objects, and room layouts to train robotic agents for household tasks, addressing the limitations of existing simulated environments that lack real-world applicability. The benchmark extends ALFRED with larger, more realistic 3D-captured spaces, and testing shows that existing methods perform significantly worse in these more realistic environments, highlighting the need for better solutions.

* Online Boundary-Free Continual Learning by Scheduled Data Prior (SDP) (ICLR 2023) - Hyunseo Koh, Minhyuk Seo, Jihwan Bang, Hwanjun Song, Deokki Hong, Seulki Park, Jung-Woo Ha and Jonghyun Choi

  * This work introduce a more realistic "boundary-free" approach to continual learning that handles continuously changing data distributions without explicit task boundaries, unlike traditional methods that rely on discrete task splits. Our solution includes scheduled knowledge transfer and data-driven balancing between past and present learning, outperforming existing methods across multiple benchmark datasets while using a novel information theory-based forgetting measure.

* Tuning Large Multimodal Models for Videos using Reinforcement Learning from AI Feedback (VLM-RLAIF) (ACL 2024) - Daechul Ahn, Yura Choi, Youngjae Yu, Dongyeop Kang and Jonghyun Choi

  * A new approach called VLM-RLAIF uses Reinforcement Learning from AI Feedback to improve video-text alignment in video large multimodal models (VLMMs), addressing the limitations of existing methods that struggle due to insufficient multimodal training data. The system incorporates detailed video descriptions during preference feedback generation (context-aware reward modeling) and outperforms existing approaches on various video benchmarks.

### 2025

* Multi-Modal Grounded Planning and Efficient Replanning For Learning Embodied Agents with A Few Examples (FLARE) (AAAI 2025) - Taewoong Kim, Byeonghwi Kim and Jonghyun Choi.

  * FLARE grounds few-shot language planning in visual observations and uses visual cues to correct ambiguous instructions and efficiently replan embodied tasks.

* ISR-DPO: Aligning Large Multimodal Models for Videos by Iterative Self-Retrospective DPO (ISR-DPO) (AAAI 2025) - Daechul Ahn, Yura Choi, San Kim, Youngjae Yu, Dongyeop Kang and Jonghyun Choi.

  * ISR-DPO uses iterative self-retrospection to produce more visually grounded video preferences while reducing hallucination and length bias during preference optimization.

* Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling (Budgeted-CL) (ICLR 2025) - Minhyuk Seo, Hyunseo Koh and Jonghyun Choi.

  * Budgeted-CL compares continual-learning methods under explicit compute and memory budgets, then saves resources through adaptive layer freezing and frequency-based replay sampling.

* Moment-Aware Video Retrieval for Video Corpus Moment Retrieval (IEEE Access 2025) - Yura Choi, Daechul Ahn and Jonghyun Choi.

  * This work jointly connects video retrieval and moment retrieval and uses curriculum-based negative sampling for finer temporal query-video alignment.

* Representation Bending for Large Language Model Safety (RepBend) (ACL 2025) - Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu and Jonghyun Choi.

  * RepBend turns activation-steering ideas into a loss-based fine-tuning method that disrupts harmful internal representations while preserving model utility.

* Large Language Models Still Exhibit Bias in Long Text (LTF-TEST) (ACL Findings 2025) - Wonje Jeung, Dongjae Jeon, Ashkan Yousefpour and Jonghyun Choi.

  * LTF-TEST evaluates subtle demographic bias in long-form generation across many topics and demographic axes, together with a fine-tuning approach for mitigation.

* Society of Mind Meets Real-Time Strategy: A Hierarchical Multi-Agent Framework for Strategic Reasoning (HIMA) (COLM 2025) - Daechul Ahn, San Kim and Jonghyun Choi.

  * HIMA combines specialized imitation-learning agents with a strategic planner to produce adaptive long-horizon plans in real-time strategy environments.

* Online Generic Event Boundary Detection (ESTimator) (ICCV 2025) - Hyungrok Jung, Daneul Kim, Seunggyun Lim, Jeany Son and Jonghyun Choi.

  * ESTimator detects generic event boundaries in streaming video by anticipating future frames and identifying statistically meaningful prediction errors online.

* GenOL: Generating Diverse Examples for Name-only Online Learning (GenOL) (TMLR 2025) - Minhyuk Seo, Seongwon Cho, Minjae Lee, Diganta Misra, Hyeonbeom Choi, Seon Joo Kim and Jonghyun Choi.

  * GenOL replaces manually labeled or web-scraped supervision with diverse generated examples selected across prompts and generative models for name-only online learning.

### 2026

* An Information Theoretic Evaluation Metric for Strong Unlearning (IDI) (AAAI 2026) - Dongjae Jeon, Wonje Jeung, Taeheon Kim, Albert No and Jonghyun Choi.

  * The Information Difference Index evaluates strong machine unlearning by measuring how much label information remains in a model's intermediate representations.

* What Happens When: Learning Temporal Order of Events in Videos (VECTOR) (WACV 2026) - Daechul Ahn, Yura Choi, Hyeonbeom Choi, Seongwon Cho, San Kim and Jonghyun Choi.

  * VECTOR benchmarks whether video-language models truly understand event order, while MECoT improves temporal reasoning with multi-event instruction tuning and chain-of-thought prompting.

* Becoming Experienced Judges: Selective Test-Time Learning for Evaluators (LWE) (EACL 2026) - Seongyeon Jwa, Daechul Ahn, Reokyoung Kim, Dongyeop Kang and Jonghyun Choi.

  * LWE lets language-model evaluators accumulate reusable evaluation insights at test time and selectively update them on self-inconsistent cases.

* Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients (Co-LoRA) (ICLR 2026) - Minhyuk Seo, Taeheon Kim, Hankook Lee, Jonghyun Choi and Tinne Tuytelaars.

  * Co-LoRA enables personalized federated learning across clients with different tasks and model architectures through relevance-aware aggregation and dimension-invariant adapters.

* [BINDER: Instantly Adaptive Mobile Manipulation with Open-Vocabulary Commands (BINDER)](./BINDER) (ICRA 2026) - Seongwon Cho, Daechul Ahn, Donghyun Shin, Hyeonbeom Choi, San Kim and Jonghyun Choi.

  * BINDER couples deliberative task planning with continuous video-based monitoring so mobile manipulators can react and replan as environments change.

* DBMSolver: A Training-free Diffusion Bridge Sampler for High-Quality Image-to-Image Translation (DBMSolver) (CVPR 2026) - Sankarshana Venugopal, Mohammad Mostafavi and Jonghyun Choi.

  * DBMSolver applies exponential integrators to diffusion bridge dynamics for faster high-quality image-to-image translation without additional training.

* FALCON: Fast Adaptive Lightweight Computation of Intensities and Events for Depth Estimation (FALCON) (CVPR Findings 2026) - Sankarshana Venugopal, Mohammad Mostafavi and Jonghyun Choi.

  * FALCON adaptively computes image intensities and events to make event-assisted depth estimation more efficient.

* OASIS: Online Sample Selection for Continual Instruction Tuning (OASIS) (ACL 2026) - Minjae Lee, Minhyuk Seo, Tingyu Qu, Tinne Tuytelaars and Jonghyun Choi.

  * OASIS adaptively selects informative, non-redundant samples from a streaming instruction-tuning workload without relying on a pre-trained reference model.

* Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts (DART) (ECCV 2026) - Taewook Kang*, Taeheon Kim*, Donghyun Shin and Jonghyun Choi.

  * DART isolates and transfers domain-specific weight components so a vision-language-action model can adapt to visual or robot embodiment shifts from a single demonstration.

* Risk-adaptive Activation Steering for Safe Multimodal Large Language Models (MoRAS) (ECCV 2026) - Jonghyun Park, Minhyuk Seo, Chaewon Yeo and Jonghyun Choi.

  * MoRAS detects safety-critical visual risk and adaptively steers multimodal model activations to mitigate image-based jailbreaks while preserving utility and reducing inference overhead.

* SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models (SCALE) (ICML 2026) - Hyunbeom Choi, Daechul Ahn, Youhan Lee, Taewook Kang, Seongwon Cho and Jonghyun Choi.

  * SCALE uses a vision-language-action model's own uncertainty to adapt both visual perception and action decoding in a single inference pass.
