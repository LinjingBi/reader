
==========================================================================================
# month=2025-01 BEST CLUSTERING (mode=B, k=5)

------------------------------------------------------------------------------------------
Cluster 2 | size=24

[2501.12599] Kimi k1.5: Scaling Reinforcement Learning with LLMs
Summary: Language model pretraining with next token prediction has proved effective
for scaling compute but is limited to the amount of available training data.
Scaling reinforcement learning (RL) unlocks a new axis for the continued
improvement of artificial intelligence, with the promise that large language
models (LLMs) can scale their training data by l…

[2501.11425] Agent-R: Training Language Model Agents to Reflect via Iterative
  Self-Training
Summary: Large Language Models (LLMs) agents are increasingly pivotal for addressing
complex tasks in interactive environments. Existing work mainly focuses on
enhancing performance through behavior cloning from stronger experts, yet such
approaches often falter in real-world applications, mainly due to the inability
to recover from errors. However, step-le…

[2501.12948] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning
Summary: We introduce our first-generation reasoning models, DeepSeek-R1-Zero and
DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement
learning (RL) without supervised fine-tuning (SFT) as a preliminary step,
demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero
naturally emerges with numerous powerful and intr…

[2501.12895] Test-Time Preference Optimization: On-the-Fly Alignment via Iterative
  Textual Feedback
Summary: Large language models (LLMs) demonstrate impressive performance but lack the
flexibility to adapt to human preferences quickly without retraining. In this
work, we introduce Test-time Preference Optimization (TPO), a framework that
aligns LLM outputs with human preferences during inference, removing the need
to update model parameters. Rather than …

[2501.15383] Qwen2.5-1M Technical Report
URL: https://qwenlm.github.io/blog/qwen2.5-1m/
Summary: We introduce Qwen2.5-1M, a series of models that extend the context length to
1 million tokens. Compared to the previous 128K version, the Qwen2.5-1M series
have significantly enhanced long-context capabilities through long-context
pre-training and post-training. Key techniques such as long data synthesis,
progressive pre-training, and multi-stage …

[2501.17161] SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model
  Post-training
URL: https://tianzhechu.com/SFTvsRL/
Summary: Supervised fine-tuning (SFT) and reinforcement learning (RL) are widely used
post-training techniques for foundation models. However, their roles in
enhancing model generalization capabilities remain unclear. This paper studies
the difference between SFT and RL on generalization and memorization, focusing
on text-based rule variants and visual vari…

[2501.05032] Enhancing Human-Like Responses in Large Language Models
Summary: This paper explores the advancements in making large language models (LLMs)
more human-like. We focus on techniques that enhance natural language
understanding, conversational coherence, and emotional intelligence in AI
systems. The study evaluates various approaches, including fine-tuning with
diverse datasets, incorporating psychological principl…

[2501.08313] MiniMax-01: Scaling Foundation Models with Lightning Attention
Summary: We introduce MiniMax-01 series, including MiniMax-Text-01 and MiniMax-VL-01,
which are comparable to top-tier models while offering superior capabilities in
processing longer contexts. The core lies in lightning attention and its
efficient scaling. To maximize computational capacity, we integrate it with
Mixture of Experts (MoE), creating a model w…

[2501.09891] Evolving Deeper LLM Thinking
Summary: We explore an evolutionary search strategy for scaling inference time compute
in Large Language Models. The proposed approach, Mind Evolution, uses a
language model to generate, recombine and refine candidate responses. The
proposed approach avoids the need to formalize the underlying inference problem
whenever a solution evaluator is available. Co…

[2501.06252] Transformer^2: Self-adaptive LLMs
Summary: Self-adaptive large language models (LLMs) aim to solve the challenges posed
by traditional fine-tuning methods, which are often computationally intensive
and static in their ability to handle diverse tasks. We introduce \implname, a
novel self-adaptation framework that adapts LLMs for unseen tasks in real-time
by selectively adjusting only the sin…

[2501.07301] The Lessons of Developing Process Reward Models in Mathematical
  Reasoning
Summary: Process Reward Models (PRMs) emerge as a promising approach for process
supervision in mathematical reasoning of Large Language Models (LLMs), which
aim to identify and mitigate intermediate errors in the reasoning processes.
However, the development of effective PRMs faces significant challenges,
particularly in data annotation and evaluation meth…

[2501.10120] PaSa: An LLM Agent for Comprehensive Academic Paper Search
Summary: We introduce PaSa, an advanced Paper Search agent powered by large language
models. PaSa can autonomously make a series of decisions, including invoking
search tools, reading papers, and selecting relevant references, to ultimately
obtain comprehensive and accurate results for complex scholarly queries. We
optimize PaSa using reinforcement learning…

[2501.04682] Towards System 2 Reasoning in LLMs: Learning How to Think With Meta
  Chain-of-Though
Summary: We propose a novel framework, Meta Chain-of-Thought (Meta-CoT), which extends
traditional Chain-of-Thought (CoT) by explicitly modeling the underlying
reasoning required to arrive at a particular CoT. We present empirical evidence
from state-of-the-art models exhibiting behaviors consistent with in-context
search, and explore methods for producing …

[2501.03262] REINFORCE++: A Simple and Efficient Approach for Aligning Large Language
  Models
Summary: Reinforcement Learning from Human Feedback (RLHF) has emerged as a critical
approach for aligning large language models with human preferences, witnessing
rapid algorithmic evolution through methods such as Proximal Policy
Optimization (PPO), Direct Preference Optimization (DPO), REINFORCE Leave
One-Out (RLOO), ReMax, and Group Relative Policy Opti…

[2501.05366] Search-o1: Agentic Search-Enhanced Large Reasoning Models
Summary: Large reasoning models (LRMs) like OpenAI-o1 have demonstrated impressive
long stepwise reasoning capabilities through large-scale reinforcement
learning. However, their extended reasoning processes often suffer from
knowledge insufficiency, leading to frequent uncertainties and potential
errors. To address this limitation, we introduce Search-o1, …

[2501.04519] rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep
  Thinking
Summary: We present rStar-Math to demonstrate that small language models (SLMs) can
rival or even surpass the math reasoning capability of OpenAI o1, without
distillation from superior models. rStar-Math achieves this by exercising "deep
thinking" through Monte Carlo Tree Search (MCTS), where a math policy SLM
performs test-time search guided by an SLM-base…

[2501.11873] Demons in the Detail: On Implementing Load Balancing Loss for Training
  Specialized Mixture-of-Expert Models
Summary: This paper revisits the implementation of
Load-balancing Loss (LBL) when training
Mixture-of-Experts (MoEs) models. Specifically, LBL for MoEs is defined as N_E
sum_{i=1}^{N_E} f_i p_i, where N_E is the total number of experts, f_i
represents the frequency of expert i being selected, and p_i denotes the
average gating score of the expert i. Existin…

[2501.13200] SRMT: Shared Memory for Multi-agent Lifelong Pathfinding
Summary: Multi-agent reinforcement learning (MARL) demonstrates significant progress
in solving cooperative and competitive multi-agent problems in various
environments. One of the principal challenges in MARL is the need for explicit
prediction of the agents' behavior to achieve cooperation. To resolve this
issue, we propose the Shared Recurrent Memory Tra…

[2501.06282] MinMo: A Multimodal Large Language Model for Seamless Voice Interaction
URL: https://funaudiollm.github.io/minmo
Summary: Recent advancements in large language models (LLMs) and multimodal
speech-text models have laid the groundwork for seamless voice interactions,
enabling real-time, natural, and human-like conversations. Previous models for
voice interactions are categorized as native and aligned. Native models
integrate speech and text processing in one framework b…

[2501.09732] Inference-Time Scaling for Diffusion Models beyond Scaling Denoising
  Steps
Summary: Generative models have made significant impacts across various domains,
largely due to their ability to scale during training by increasing data,
computational resources, and model size, a phenomenon characterized by the
scaling laws. Recent research has begun to explore inference-time scaling
behavior in Large Language Models (LLMs), revealing how…

[2501.06425] Tensor Product Attention Is All You Need
URL: https://tensorgi.github.io/TPA/
Summary: Scaling language models to handle longer input sequences typically
necessitates large key-value (KV) caches, resulting in substantial memory
overhead during inference. In this paper, we propose Tensor Product Attention
(TPA), a novel attention mechanism that uses tensor decompositions to represent
queries, keys, and values compactly, significantly …

[2501.18492] GuardReasoner: Towards Reasoning-based LLM Safeguards
Summary: As LLMs increasingly impact safety-critical applications, ensuring their
safety using guardrails remains a key challenge. This paper proposes
GuardReasoner, a new safeguard for LLMs, by guiding the guard model to learn to
reason. Concretely, we first create the GuardReasonerTrain dataset, which
consists of 127K samples with 460K detailed reasoning …

[2501.08365] Towards Best Practices for Open Datasets for LLM Training
Summary: Many AI companies are training their large language models (LLMs) on data
without the permission of the copyright owners. The permissibility of doing so
varies by jurisdiction: in countries like the EU and Japan, this is allowed
under certain restrictions, while in the United States, the legal landscape is
more ambiguous. Regardless of the legal st…

[2501.03575] Cosmos World Foundation Model Platform for Physical AI
URL: https://research.nvidia.com/labs/dir/cosmos1/
Summary: Physical AI needs to be trained digitally first. It needs a digital twin of
itself, the policy model, and a digital twin of the world, the world model. In
this paper, we present the Cosmos World Foundation Model Platform to help
developers build customized world models for their Physical AI setups. We
position a world foundation model as a general-…

------------------------------------------------------------------------------------------
Cluster 0 | size=12

[2501.00958] 2.5 Years in Class: A Multimodal Textbook for Vision-Language
  Pretraining
Summary: Compared to image-text pair data, interleaved corpora enable Vision-Language
Models (VLMs) to understand the world more naturally like humans. However, such
existing datasets are crawled from webpage, facing challenges like low
knowledge density, loose image-text relations, and poor logical coherence
between images. On the other hand, the internet …

[2501.13106] VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video
  Understanding
URL: https://github.com/DAMO-NLP-SG/VideoLLaMA3
Summary: In this paper, we propose VideoLLaMA3, a more advanced multimodal foundation
model for image and video understanding. The core design philosophy of
VideoLLaMA3 is vision-centric. The meaning of "vision-centric" is two-fold: the
vision-centric training paradigm and vision-centric framework design. The key
insight of our vision-centric training parad…

[2501.06186] LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs
Summary: Reasoning is a fundamental capability for solving complex multi-step
problems, particularly in visual contexts where sequential step-wise
understanding is essential. Existing approaches lack a comprehensive framework
for evaluating visual reasoning and do not emphasize step-wise problem-solving.
To this end, we propose a comprehensive framework for…

[2501.03895] LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One
  Vision Token
Summary: The advent of real-time large multimodal models (LMMs) like GPT-4o has
sparked considerable interest in efficient LMMs. LMM frameworks typically
encode visual inputs into vision tokens (continuous representations) and
integrate them and textual instructions into the context of large language
models (LLMs), where large-scale parameters and numerous …

[2501.12380] MMVU: Measuring Expert-Level Multi-Discipline Video Understanding
URL: https://mmvu-benchmark.github.io/
Summary: We introduce MMVU, a comprehensive expert-level, multi-discipline benchmark
for evaluating foundation models in video understanding. MMVU includes 3,000
expert-annotated questions spanning 27 subjects across four core disciplines:
Science, Healthcare, Humanities & Social Sciences, and Engineering. Compared to
prior benchmarks, MMVU features three k…

[2501.05874] VideoRAG: Retrieval-Augmented Generation over Video Corpus
Summary: Retrieval-Augmented Generation (RAG) is a powerful strategy to address the
issue of generating factually incorrect outputs in foundation models by
retrieving external knowledge relevant to queries and incorporating it into
their generation process. However, existing RAG approaches have primarily
focused on textual information, with some recent adva…

[2501.07171] BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and
  Vision-Language Models Derived from Scientific Literature
URL: https://minwoosun.github.io/biomedica-website/
Summary: The development of vision-language models (VLMs) is driven by large-scale and
diverse multimodal datasets. However, progress toward generalist biomedical
VLMs is limited by the lack of annotated, publicly accessible datasets across
biology and medicine. Existing efforts are restricted to narrow domains,
missing the full diversity of biomedical know…

[2501.03841] OmniManip: Towards General Robotic Manipulation via Object-Centric
  Interaction Primitives as Spatial Constraints
URL: https://omnimanip.github.io/
Summary: The development of general robotic systems capable of manipulating in
unstructured environments is a significant challenge. While Vision-Language
Models(VLM) excel in high-level commonsense reasoning, they lack the
fine-grained 3D spatial understanding required for precise manipulation tasks.
Fine-tuning VLM on robotic datasets to create Vision-Lan…

[2501.12909] FilmAgent: A Multi-Agent Framework for End-to-End Film Automation in
  Virtual 3D Spaces
URL: https://filmagent.github.io/
Summary: Virtual film production requires intricate decision-making processes,
including scriptwriting, virtual cinematography, and precise actor positioning
and actions. Motivated by recent advances in automated decision-making with
language agent-based societies, this paper introduces FilmAgent, a novel
LLM-based multi-agent collaborative framework for en…

[2501.01427] VideoAnydoor: High-fidelity Video Object Insertion with Precise Motion
  Control
URL: https://videoanydoor.github.io/
Summary: Despite significant advancements in video generation, inserting a given
object into videos remains a challenging task. The difficulty lies in
preserving the appearance details of the reference object and accurately
modeling coherent motions at the same time. In this paper, we propose
VideoAnydoor, a zero-shot video object insertion framework with h…

[2501.12326] UI-TARS: Pioneering Automated GUI Interaction with Native Agents
Summary: This paper introduces UI-TARS, a native GUI agent model that solely perceives
the screenshots as input and performs human-like interactions (e.g., keyboard
and mouse operations). Unlike prevailing agent frameworks that depend on
heavily wrapped commercial models (e.g., GPT-4o) with expert-crafted prompts
and workflows, UI-TARS is an end-to-end mode…

[2501.15368] Baichuan-Omni-1.5 Technical Report
Summary: We introduce Baichuan-Omni-1.5, an omni-modal model that not only has
omni-modal understanding capabilities but also provides end-to-end audio
generation capabilities. To achieve fluent and high-quality interaction across
modalities without compromising the capabilities of any modality, we
prioritized optimizing three key aspects. First, we establi…

------------------------------------------------------------------------------------------
Cluster 4 | size=6

[2501.02976] STAR: Spatial-Temporal Augmentation with Text-to-Video Models for
  Real-World Video Super-Resolution
Summary: Image diffusion models have been adapted for real-world video
super-resolution to tackle over-smoothing issues in GAN-based methods. However,
these models struggle to maintain temporal consistency, as they are trained on
static images, limiting their ability to capture temporal dynamics effectively.
Integrating text-to-video (T2V) models into video…

[2501.00103] LTX-Video: Realtime Video Latent Diffusion
Summary: We introduce LTX-Video, a transformer-based latent diffusion model that
adopts a holistic approach to video generation by seamlessly integrating the
responsibilities of the Video-VAE and the denoising transformer. Unlike
existing methods, which treat these components as independent, LTX-Video aims
to optimize their interaction for improved efficien…

[2501.12202] Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D
  Assets Generation
Summary: We present Hunyuan3D 2.0, an advanced large-scale 3D synthesis system for
generating high-resolution textured 3D assets. This system includes two
foundation components: a large-scale shape generation model -- Hunyuan3D-DiT,
and a large-scale texture synthesis model -- Hunyuan3D-Paint. The shape
generative model, built on a scalable flow-based diffu…

[2501.05441] The GAN is dead; long live the GAN! A Modern GAN Baseline
Summary: There is a widely-spread claim that GANs are difficult to train, and GAN
architectures in the literature are littered with empirical tricks. We provide
evidence against this claim and build a modern GAN baseline in a more
principled manner. First, we derive a well-behaved regularized relativistic GAN
loss that addresses issues of mode dropping and …

[2501.01895] EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation
URL: https://sites.google.com/view/enerverse
Summary: We introduce EnerVerse, a comprehensive framework for embodied future space
generation specifically designed for robotic manipulation tasks. EnerVerse
seamlessly integrates convolutional and bidirectional attention mechanisms for
inner-chunk space modeling, ensuring low-level consistency and continuity.
Recognizing the inherent redundancy in video …

[2501.08332] MangaNinja: Line Art Colorization with Precise Reference Following
URL: https://johanan528.github.io/MangaNinjia/
Summary: Derived from diffusion models, MangaNinjia specializes in the task of
reference-guided line art colorization. We incorporate two thoughtful designs
to ensure precise character detail transcription, including a patch shuffling
module to facilitate correspondence learning between the reference color image
and the target line art, and a point-driven c…

------------------------------------------------------------------------------------------
Cluster 1 | size=5

[2501.01257] CodeElo: Benchmarking Competition-level Code Generation of LLMs with
  Human-comparable Elo Ratings
Summary: With the increasing code reasoning capabilities of existing large language
models (LLMs) and breakthroughs in reasoning models like OpenAI o1 and o3,
there is a growing need to develop more challenging and comprehensive
benchmarks that effectively test their sophisticated competition-level coding
abilities. Existing benchmarks, like LiveCodeBench a…

[2501.14249] Humanity's Last Exam
URL: https://lastexam.ai/
Summary: Benchmarks are important tools for tracking the rapid advancements in large
language model (LLM) capabilities. However, benchmarks are not keeping pace in
difficulty: LLMs now achieve over 90\% accuracy on popular benchmarks like
MMLU, limiting informed measurement of state-of-the-art LLM capabilities. In
response, we introduce Humanity's Last Exam…

[2501.04227] Agent Laboratory: Using LLM Agents as Research Assistants
Summary: Historically, scientific discovery has been a lengthy and costly process,
demanding substantial time and resources from initial conception to final
results. To accelerate scientific discovery, reduce research costs, and improve
research quality, we introduce Agent Laboratory, an autonomous LLM-based
framework capable of completing the entire resear…

[2501.14342] Chain-of-Retrieval Augmented Generation
Summary: This paper introduces an approach for training o1-like RAG models that
retrieve and reason over relevant information step by step before generating
the final answer. Conventional RAG methods usually perform a single retrieval
step before the generation process, which limits their effectiveness in
addressing complex queries due to imperfect retrieva…

[2412.19723] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse
  Task Synthesis
URL: https://qiushisun.github.io/OS-Genesis-Home/
Summary: Graphical User Interface (GUI) agents powered by Vision-Language Models
(VLMs) have demonstrated human-like computer control capability. Despite their
utility in advancing digital automation, a critical bottleneck persists:
collecting high-quality trajectory data for training. Common practices for
collecting such data rely on human supervision or s…

------------------------------------------------------------------------------------------
Cluster 3 | size=3

[2501.05727] Enabling Scalable Oversight via Self-Evolving Critic
Summary: Despite their remarkable performance, the development of Large Language
Models (LLMs) faces a critical challenge in scalable oversight: providing
effective feedback for tasks where human evaluation is difficult or where LLMs
outperform humans. While there is growing interest in using LLMs for critique,
current approaches still rely on human annotat…

[2501.17703] Critique Fine-Tuning: Learning to Critique is More Effective than
  Learning to Imitate
Summary: Supervised Fine-Tuning (SFT) is commonly used to train language models to
imitate annotated responses for given instructions. In this paper, we challenge
this paradigm and propose Critique Fine-Tuning (CFT), a strategy where models
learn to critique noisy responses rather than simply imitate correct ones.
Inspired by human learning processes that e…

[2501.18585] Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs
Summary: Large language models (LLMs) such as OpenAI's o1 have demonstrated remarkable
abilities in complex reasoning tasks by scaling test-time compute and
exhibiting human-like deep thinking. However, we identify a phenomenon we term
underthinking, where o1-like LLMs frequently switch between different reasoning
thoughts without sufficiently exploring pro…

==========================================================================================
# month=2025-02 BEST CLUSTERING (mode=B, k=4)

------------------------------------------------------------------------------------------
Cluster 2 | size=19

[2502.03373] Demystifying Long Chain-of-Thought Reasoning in LLMs
Summary: Scaling inference compute enhances reasoning in large language models (LLMs),
with long chains-of-thought (CoTs) enabling strategies like backtracking and
error correction. Reinforcement learning (RL) has emerged as a crucial method
for developing these capabilities, yet the conditions under which long CoTs
emerge remain unclear, and RL training re…

[2502.06703] Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time
  Scaling
URL: https://ryanliu112.github.io/compute-optimal-tts
Summary: Test-Time Scaling (TTS) is an important method for improving the performance
of Large Language Models (LLMs) by using additional computation during the
inference phase. However, current studies do not systematically analyze how
policy models, Process Reward Models (PRMs), and problem difficulty influence
TTS. This lack of analysis limits the unders…

[2502.05171] Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth
  Approach
Summary: We study a novel language model architecture that is capable of scaling
test-time computation by implicitly reasoning in latent space. Our model works
by iterating a recurrent block, thereby unrolling to arbitrary depth at
test-time. This stands in contrast to mainstream reasoning models that scale up
compute by producing more tokens. Unlike approa…

[2502.06807] Competitive Programming with Large Reasoning Models
Summary: We show that reinforcement learning applied to large language models (LLMs)
significantly boosts performance on complex coding and reasoning tasks.
Additionally, we compare two general-purpose reasoning models - OpenAI o1 and
an early checkpoint of o3 - with a domain-specific system, o1-ioi, which uses
hand-engineered inference strategies designed …

[2502.19613] Self-rewarding correction for mathematical reasoning
Summary: We study self-rewarding reasoning large language models (LLMs), which can
simultaneously generate step-by-step reasoning and evaluate the correctness of
their outputs during the inference time-without external feedback. This
integrated approach allows a single model to independently guide its reasoning
process, offering computational advantages for…

[2501.19393] s1: Simple test-time scaling
Summary: Test-time scaling is a promising new approach to language modeling that uses
extra test-time compute to improve performance. Recently, OpenAI's o1 model
showed this capability but did not publicly share its methodology, leading to
many replication efforts. We seek the simplest approach to achieve test-time
scaling and strong reasoning performance. …

[2502.14739] SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines
URL: https://supergpqa.github.io/
Summary: Large language models (LLMs) have demonstrated remarkable proficiency in
mainstream academic disciplines such as mathematics, physics, and computer
science. However, human knowledge encompasses over 200 specialized disciplines,
far exceeding the scope of existing benchmarks. The capabilities of LLMs in
many of these specialized fields-particularly …

[2502.18449] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open
  Software Evolution
Summary: The recent DeepSeek-R1 release has demonstrated the immense potential of
reinforcement learning (RL) in enhancing the general reasoning capabilities of
large language models (LLMs). While DeepSeek-R1 and other follow-up work
primarily focus on applying RL to competitive coding and math problems, this
paper introduces SWE-RL, the first approach to s…

[2502.14499] MLGym: A New Framework and Benchmark for Advancing AI Research Agents
Summary: We introduce Meta MLGym and MLGym-Bench, a new framework and benchmark for
evaluating and developing LLM agents on AI research tasks. This is the first
Gym environment for machine learning (ML) tasks, enabling research on
reinforcement learning (RL) algorithms for training such agents. MLGym-bench
consists of 13 diverse and open-ended AI research t…

[2502.06781] Exploring the Limit of Outcome Reward for Learning Mathematical
  Reasoning
Summary: Reasoning abilities, especially those for solving complex math problems, are
crucial components of general intelligence. Recent advances by proprietary
companies, such as o-series models of OpenAI, have made remarkable progress on
reasoning tasks. However, the complete technical details remain unrevealed, and
the techniques that are believed certai…

[2502.01456] Process Reinforcement through Implicit Rewards
Summary: Dense process rewards have proven a more effective alternative to the sparse
outcome-level rewards in the inference-time scaling of large language models
(LLMs), particularly in tasks requiring complex multi-step reasoning. While
dense rewards also offer an appealing choice for the reinforcement learning
(RL) of LLMs since their fine-grained reward…

[2502.08127] Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance
Summary: Recent advancements in large language models (LLMs) have shown strong general
reasoning abilities, yet their effectiveness in financial reasoning remains
underexplored. In this study, we comprehensively evaluate 16 powerful reasoning
and general LLMs on three complex financial tasks involving financial text,
tabular data, and equations, assessing n…

[2502.03387] LIMO: Less is More for Reasoning
Summary: We present a fundamental discovery that challenges our understanding of how
complex reasoning emerges in large language models. While conventional wisdom
suggests that sophisticated reasoning tasks demand extensive training data
(>100,000 examples), we demonstrate that complex mathematical reasoning
abilities can be effectively elicited with surpri…

[2502.14382] S*: Test Time Scaling for Code Generation
URL: https://novasky-ai.github.io/posts/S*/
Summary: Increasing test-time compute for LLMs shows promise across domains but
remains underexplored in code generation, despite extensive study in math. In
this paper, we propose S*, the first hybrid test-time scaling framework that
substantially improves the coverage and selection accuracy of generated code.
S* extends the existing parallel scaling parad…

[2502.08946] The Stochastic Parrot on LLM's Shoulder: A Summative Assessment of
  Physical Concept Understanding
URL: https://physico-benchmark.github.io
Summary: In a systematic way, we investigate a widely asked question: Do LLMs really
understand what they say?, which relates to the more familiar term Stochastic
Parrot. To this end, we propose a summative assessment over a carefully
designed physical concept understanding task, PhysiCo. Our task alleviates the
memorization issue via the usage of grid-form…

[2502.01237] The Differences Between Direct Alignment Algorithms are a Blur
Summary: Direct Alignment Algorithms (DAAs) simplify language model alignment by
replacing reinforcement learning (RL) and reward modeling (RM) in Reinforcement
Learning from Human Feedback (RLHF) with direct policy optimization. DAAs can
be classified by their ranking losses (pairwise vs. pointwise), by the rewards
used in those losses (e.g., likelihood ra…

[2502.19634] MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language
  Models (VLMs) via Reinforcement Learning
Summary: Reasoning is a critical frontier for advancing medical image analysis, where
transparency and trustworthiness play a central role in both clinician trust
and regulatory approval. Although Medical Visual Language Models (VLMs) show
promise for radiological tasks, most existing VLMs merely produce final answers
without revealing the underlying reason…

[2502.13130] Magma: A Foundation Model for Multimodal AI Agents
Summary: We present Magma, a foundation model that serves multimodal AI agentic tasks
in both the digital and physical worlds. Magma is a significant extension of
vision-language (VL) models in that it not only retains the VL understanding
ability (verbal intelligence) of the latter, but is also equipped with the
ability to plan and act in the visual-spatia…

[2502.18864] Towards an AI co-scientist
Summary: Scientific discovery relies on scientists generating novel hypotheses that
undergo rigorous experimental validation. To augment this process, we introduce
an AI co-scientist, a multi-agent system built on Gemini 2.0. The AI
co-scientist is intended to help uncover new, original knowledge and to
formulate demonstrably novel research hypotheses and p…

------------------------------------------------------------------------------------------
Cluster 0 | size=18

[2502.07346] BenchMAX: A Comprehensive Multilingual Evaluation Suite for Large
  Language Models
Summary: Previous multilingual benchmarks focus primarily on simple understanding
tasks, but for large language models(LLMs), we emphasize proficiency in
instruction following, reasoning, long context understanding, code generation,
and so on. However, measuring these advanced capabilities across languages is
underexplored. To address the disparity, we intr…

[2502.17129] Thus Spake Long-Context Large Language Model
Summary: Long context is an important topic in Natural Language Processing (NLP),
running through the development of NLP architectures, and offers immense
opportunities for Large Language Models (LLMs) giving LLMs the lifelong
learning potential akin to humans. Unfortunately, the pursuit of a long context
is accompanied by numerous obstacles. Nevertheless, …

[2502.15007] LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context
  Memory of Transformers
Summary: We introduce methods to quantify how Large Language Models (LLMs) encode and
store contextual information, revealing that tokens often seen as minor (e.g.,
determiners, punctuation) carry surprisingly high context. Notably, removing
these tokens -- especially stopwords, articles, and commas -- consistently
degrades performance on MMLU and BABILong-…

[2502.09992] Large Language Diffusion Models
URL: https://ml-gsai.github.io/LLaDA-demo/
Summary: Autoregressive models (ARMs) are widely regarded as the cornerstone of large
language models (LLMs). We challenge this notion by introducing LLaDA, a
diffusion model trained from scratch under the pre-training and supervised
fine-tuning (SFT) paradigm. LLaDA models distributions through a forward data
masking process and a reverse process, paramete…

[2502.02737] SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model
URL: https://github.com/huggingface/smollm
Summary: While large language models have facilitated breakthroughs in many applications of artificial intelligence, their inherent largeness makes them computationally expensive and challenging to deploy in resource-constrained settings. In this paper, we document the development of SmolLM2, a state-of-the-art "small" (1.7 billion parameter) language model…

[2502.08910] InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on
  a Single GPU
Summary: In modern large language models (LLMs), handling very long context lengths
presents significant challenges as it causes slower inference speeds and
increased memory costs. Additionally, most existing pre-trained LLMs fail to
generalize beyond their original training sequence lengths. To enable efficient
and practical long-context utilization, we in…

[2502.07864] TransMLA: Multi-head Latent Attention Is All You Need
Summary: Modern large language models (LLMs) often encounter communication bottlenecks
on current hardware, rather than purely computational constraints. Multi-head
Latent Attention (MLA) tackles this challenge by using low-rank matrices in the
key-value (KV) layers, thereby allowing compressed latent KV states to be
cached. This approach significantly redu…

[2502.14502] How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?
Summary: The performance of Large Language Models (LLMs) on many tasks is greatly
limited by the knowledge learned during pre-training and stored in the model's
parameters. Low-rank adaptation (LoRA) is a popular and efficient training
technique for updating or domain-specific adaptation of LLMs. In this study, we
investigate how new facts can be incorporat…

[2502.11089] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
  Attention
Summary: Long-context modeling is crucial for next-generation language models, yet the
high computational cost of standard attention mechanisms poses significant
computational challenges. Sparse attention offers a promising direction for
improving efficiency while maintaining model capabilities. We present NSA, a
Natively trainable Sparse Attention mechanis…

[2502.03032] Analyze Feature Flow to Enhance Interpretation and Steering in Language
  Models
Summary: We introduce a new approach to systematically map features discovered by
sparse autoencoder across consecutive layers of large language models,
extending earlier work that examined inter-layer feature links. By using a
data-free cosine similarity technique, we trace how specific features persist,
transform, or first appear at each stage. This metho…

[2502.13063] Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the
  Limits of Embedding Space Capacity
Summary: A range of recent works addresses the problem of compression of sequence of
tokens into a shorter sequence of real-valued vectors to be used as inputs
instead of token embeddings or key-value cache. These approaches allow to
reduce the amount of compute in existing language models. Despite relying on
powerful models as encoders, the maximum attaina…

[2502.12900] Soundwave: Less is More for Speech-Text Alignment in LLMs
Summary: Existing end-to-end speech large language models (LLMs) usually rely on
large-scale annotated data for training, while data-efficient training has not
been discussed in depth. We focus on two fundamental problems between speech
and text: the representation space gap and sequence length inconsistency. We
propose Soundwave, which utilizes an efficien…

[2502.18934] Kanana: Compute-efficient Bilingual Language Models
URL: https://huggingface.co/kakaocorp
Summary: We introduce Kanana, a series of bilingual language models that demonstrate
exceeding performance in Korean and competitive performance in English. The
computational cost of Kanana is significantly lower than that of
state-of-the-art models of similar size. The report details the techniques
employed during pre-training to achieve compute-efficient …

[2502.15814] Slamming: Training a Speech Language Model on One GPU in a Day
URL: https://pages.cs.huji.ac.il/adiyoss-lab/slamming/
Summary: We introduce Slam, a recipe for training high-quality Speech Language Models
(SLMs) on a single academic GPU in 24 hours. We do so through empirical
analysis of model initialisation and architecture, synthetic training data,
preference optimisation with synthetic data and tweaking all other components.
We empirically demonstrate that this training …

[2502.18411] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference
Summary: Recent advancements in open-source multi-modal large language models (MLLMs)
have primarily focused on enhancing foundational capabilities, leaving a
significant gap in human preference alignment. This paper introduces
OmniAlign-V, a comprehensive dataset of 200K high-quality training samples
featuring diverse images, complex questions, and varied …

[2502.06329] Expect the Unexpected: FailSafe Long Context QA for Finance
URL: https://writer.com/research/
Summary: We propose a new long-context financial benchmark, FailSafeQA, designed to
test the robustness and context-awareness of LLMs against six variations in
human-interface interactions in LLM-based query-answer systems within finance.
We concentrate on two case studies: Query Failure and Context Failure. In the
Query Failure scenario, we perturb the ori…

[2502.14776] SurveyX: Academic Survey Automation via Large Language Models
URL: http://www.surveyx.cn
Summary: Large Language Models (LLMs) have demonstrated exceptional comprehension
capabilities and a vast knowledge base, suggesting that LLMs can serve as
efficient tools for automated survey generation. However, recent research
related to automated survey generation remains constrained by some critical
limitations like finite context window, lack of in-de…

[2502.06394] SynthDetoxM: Modern LLMs are Few-Shot Parallel Detoxification Data
  Annotators
URL: https://s-nlp.github.io/synthdetoxm/
Summary: Existing approaches to multilingual text detoxification are hampered by the
scarcity of parallel multilingual datasets. In this work, we introduce a
pipeline for the generation of multilingual parallel detoxification data. We
also introduce SynthDetoxM, a manually collected and synthetically generated
multilingual parallel text detoxification datas…

------------------------------------------------------------------------------------------
Cluster 3 | size=10

[2502.02492] VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion
  Generation in Video Models
Summary: Despite tremendous recent progress, generative video models still struggle to
capture real-world motion, dynamics, and physics. We show that this limitation
arises from the conventional pixel reconstruction objective, which biases
models toward appearance fidelity at the expense of motion coherence. To
address this, we introduce VideoJAM, a novel f…

[2502.10248] Step-Video-T2V Technical Report: The Practice, Challenges, and Future of
  Video Foundation Model
URL: https://yuewen.cn/videos
Summary: We present Step-Video-T2V, a state-of-the-art text-to-video pre-trained model
with 30B parameters and the ability to generate videos up to 204 frames in
length. A deep compression Variational Autoencoder, Video-VAE, is designed for
video generation tasks, achieving 16x16 spatial and 8x temporal compression
ratios, while maintaining exceptional vide…

[2502.17258] VideoGrain: Modulating Space-Time Attention for Multi-grained Video
  Editing
URL: https://knightyxp.github.io/VideoGrain_project_page/
Summary: Recent advancements in diffusion models have significantly improved video
generation and editing capabilities. However, multi-grained video editing,
which encompasses class-level, instance-level, and part-level modifications,
remains a formidable challenge. The major difficulties in multi-grained editing
include semantic misalignment of text-to-reg…

[2502.04896] Goku: Flow Based Video Generative Foundation Models
Summary: This paper introduces Goku, a state-of-the-art family of joint
image-and-video generation models leveraging rectified flow Transformers to
achieve industry-leading performance. We detail the foundational elements
enabling high-quality visual generation, including the data curation pipeline,
model architecture design, flow formulation, and advanced …

[2502.11079] Phantom: Subject-consistent video generation via cross-modal alignment
URL: https://phantom-video.github.io/Phantom/
Summary: The continuous development of foundational models for video generation is
evolving into various applications, with subject-consistent video generation
still in the exploratory stage. We refer to this as Subject-to-Video, which
extracts subject elements from reference images and generates
subject-consistent video through textual instructions. We bel…

[2502.01061] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human
  Animation Models
URL: https://omnihuman-lab.github.io/
Summary: End-to-end human animation, such as audio-driven talking human generation,
has undergone notable advancements in the recent few years. However, existing
methods still struggle to scale up as large general video generation models,
limiting their potential in real applications. In this paper, we propose
OmniHuman, a Diffusion Transformer-based framew…

[2502.05173] VideoRoPE: What Makes for Good Video Rotary Position Embedding?
URL: https://wiselnn570.github.io/VideoRoPE/
Summary: While Rotary Position Embedding (RoPE) and its variants are widely adopted
for their long-context capabilities, the extension of the 1D RoPE to video,
with its complex spatio-temporal structure, remains an open challenge. This
work first introduces a comprehensive analysis that identifies four key
characteristics essential for the effective adaptat…

[2502.14786] SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic
  Understanding, Localization, and Dense Features
URL: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md
Summary: We introduce SigLIP 2, a family of new multilingual vision-language encoders
that build on the success of the original SigLIP. In this second iteration, we
extend the original image-text training objective with several prior,
independently developed techniques into a unified recipe -- this includes
captioning-based pretraining, self-supervised loss…

[2502.13923] Qwen2.5-VL Technical Report
URL: https://chat.qwenlm.ai
Summary: We introduce Qwen2.5-VL, the latest flagship model of Qwen vision-language
series, which demonstrates significant advancements in both foundational
capabilities and innovative functionalities. Qwen2.5-VL achieves a major leap
forward in understanding and interacting with the world through enhanced visual
recognition, precise object localization, ro…

[2502.18417] GHOST 2.0: generative high-fidelity one shot transfer of heads
Summary: While the task of face swapping has recently gained attention in the research
community, a related problem of head swapping remains largely unexplored. In
addition to skin color transfer, head swap poses extra challenges, such as the
need to preserve structural information of the whole head during synthesis and
inpaint gaps between swapped head and…

------------------------------------------------------------------------------------------
Cluster 1 | size=3

[2502.10389] Region-Adaptive Sampling for Diffusion Transformers
URL: https://aka.ms/ras-dit
Summary: Diffusion models (DMs) have become the leading choice for generative tasks
across diverse domains. However, their reliance on multiple sequential forward
passes significantly limits real-time performance. Previous acceleration
methods have primarily focused on reducing the number of sampling steps or
reusing intermediate results, failing to leverag…

[2502.11564] Continuous Diffusion Model for Language Modeling
Summary: Diffusion models have emerged as a promising alternative to autoregressive
models in modeling discrete categorical data. Yet diffusion models that
directly work on discrete data space do not fully exploit the power of
iterative refinement, as the signals are lost during the transition between
discrete states. Existing continuous diffusion models fo…

[2502.18137] SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference
URL: https://github.com/thu-ml/SageAttention
Summary: An efficient attention implementation is essential for large models due to
its quadratic time complexity. Fortunately, attention commonly exhibits
sparsity, i.e., many values in the attention map are near zero, allowing for
the omission of corresponding computations. Many studies have utilized the
sparse pattern to accelerate attention. However, mo…

==========================================================================================
# month=2025-03 BEST CLUSTERING (mode=C, k=5)

------------------------------------------------------------------------------------------
Cluster 1 | size=17

[2503.16419] Stop Overthinking: A Survey on Efficient Reasoning for Large Language
  Models
Summary: Large Language Models (LLMs) have demonstrated remarkable capabilities in
complex tasks. Recent advancements in Large Reasoning Models (LRMs), such as
OpenAI o1 and DeepSeek-R1, have further improved performance in System-2
reasoning domains like mathematics and programming by harnessing supervised
fine-tuning (SFT) and reinforcement learning (RL) …

[2503.18878] I Have Covered All the Bases Here: Interpreting Reasoning Features in
  Large Language Models via Sparse Autoencoders
Summary: Large Language Models (LLMs) have achieved remarkable success in natural
language processing. Recent advances have led to the developing of a new class
of reasoning LLMs; for example, open-source DeepSeek-R1 has achieved
state-of-the-art performance by integrating deep thinking and complex
reasoning. Despite these impressive capabilities, the inter…

[2503.16219] Reinforcement Learning for Reasoning in Small LLMs: What Works and What
  Doesn't
Summary: Enhancing the reasoning capabilities of large language models (LLMs)
typically relies on massive computational resources and extensive datasets,
limiting accessibility for resource-constrained settings. Our study
investigates the potential of reinforcement learning (RL) to improve reasoning
in small LLMs, focusing on a 1.5-billion-parameter model,
…

[2503.00865] Babel: Open Multilingual Large Language Models Serving Over 90% of
  Global Speakers
URL: https://babel-llm.github.io/babel-llm/
Summary: Large language models (LLMs) have revolutionized natural language processing
(NLP), yet open-source multilingual LLMs remain scarce, with existing models
often limited in language coverage. Such models typically prioritize
well-resourced languages, while widely spoken but under-resourced languages are
often overlooked. To address this disparity, we…

[2503.19693] AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through
  Lightweight Vocabulary Adaptation
URL: https://itay-nakash.github.io/AdaptiVocab/
Summary: Large Language Models (LLMs) have shown impressive versatility as general
purpose models. However, their broad applicability comes at a high-cost
computational overhead, particularly in auto-regressive decoding where each
step requires a forward pass. In domain-specific settings, general-purpose
capabilities are unnecessary and can be exchanged for…

[2503.15299] Inside-Out: Hidden Factual Knowledge in LLMs
Summary: This work presents a framework for assessing whether large language models
(LLMs) encode more factual knowledge in their parameters than what they express
in their outputs. While a few studies hint at this possibility, none has
clearly defined or demonstrated this phenomenon. We first propose a formal
definition of knowledge, quantifying it for a g…

[2503.14456] RWKV-7 "Goose" with Expressive Dynamic State Evolution
URL: https://rwkv.cn
Summary: We present RWKV-7 "Goose", a new sequence modeling architecture, along with
pre-trained language models that establish a new state-of-the-art in downstream
performance at the 3 billion parameter scale on multilingual tasks, and match
current SoTA English language performance despite being trained on dramatically
fewer tokens than other top 3B model…

[2503.01743] Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language
  Models via Mixture-of-LoRAs
URL: https://huggingface.co/microsoft/Phi-4-multimodal-instruct
Summary: We introduce Phi-4-Mini and Phi-4-Multimodal, compact yet highly capable
language and multimodal models. Phi-4-Mini is a 3.8-billion-parameter language
model trained on high-quality web and synthetic data, significantly
outperforming recent open-source models of similar size and matching the
performance of models twice its size on math and coding t…

[2503.21460] Large Language Model Agent: A Survey on Methodology, Applications and
  Challenges
URL: https://huggingface.co/spaces/luojunyu/Agent-Papers
Summary: The era of intelligent agents is upon us, driven by revolutionary
advancements in large language models. Large Language Model (LLM) agents, with
goal-driven behaviors and dynamic adaptation capabilities, potentially
represent a critical pathway toward artificial general intelligence. This
survey systematically deconstructs LLM agent systems through…

[2503.07605] SEAP: Training-free Sparse Expert Activation Pruning Unlock the
  Brainpower of Large Language Models
Summary: Large Language Models have achieved remarkable success across various natural
language processing tasks, yet their high computational cost during inference
remains a major bottleneck. This paper introduces Sparse Expert Activation
Pruning (SEAP), a training-free pruning method that selectively retains
task-relevant parameters to reduce inference ov…

[2503.14476] DAPO: An Open-Source LLM Reinforcement Learning System at Scale
URL: https://dapo-sia.github.io/
Summary: Inference scaling empowers LLMs with unprecedented reasoning ability, with
reinforcement learning as the core technique to elicit complex reasoning.
However, key technical details of state-of-the-art reasoning LLMs are concealed
(such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the
community still struggles to reproduce their RL tr…

[2503.05500] EuroBERT: Scaling Multilingual Encoders for European Languages
URL: https://huggingface.co/EuroBERT
Summary: General-purpose multilingual vector representations, used in retrieval,
regression and classification, are traditionally obtained from bidirectional
encoder models. Despite their wide applicability, encoders have been recently
overshadowed by advances in generative decoder-only models. However, many
innovations driving this progress are not inheren…

[2503.04625] START: Self-taught Reasoner with Tools
Summary: Large reasoning models (LRMs) like OpenAI-o1 and DeepSeek-R1 have
demonstrated remarkable capabilities in complex reasoning tasks through the
utilization of long Chain-of-thought (CoT). However, these models often suffer
from hallucinations and inefficiencies due to their reliance solely on internal
reasoning processes. In this paper, we introduce …

[2503.00808] Predictive Data Selection: The Data That Predicts Is the Data That
  Teaches
Summary: Language model pretraining involves training on extensive corpora, where data
quality plays a pivotal role. In this work, we aim to directly estimate the
contribution of data during pretraining and select pretraining data in an
efficient manner. Specifically, we draw inspiration from recent findings
showing that compression efficiency (i.e., the no…

[2503.03601] Feature-Level Insights into Artificial Text Detection with Sparse
  Autoencoders
URL: https://mgtsaevis.github.io/mgt-sae-visualization/
Summary: Artificial Text Detection (ATD) is becoming increasingly important with the
rise of advanced Large Language Models (LLMs). Despite numerous efforts, no
single algorithm performs consistently well across different types of unseen
text or guarantees effective generalization to new LLMs. Interpretability plays
a crucial role in achieving this goal. In…

[2503.16416] Survey on Evaluation of LLM-based Agents
Summary: The emergence of LLM-based agents represents a paradigm shift in AI, enabling
autonomous systems to plan, reason, use tools, and maintain memory while
interacting with dynamic environments. This paper provides the first
comprehensive survey of evaluation methodologies for these increasingly capable
agents. We systematically analyze evaluation bench…

[2502.21263] RuCCoD: Towards Automated ICD Coding in Russian
Summary: This study investigates the feasibility of automating clinical coding in
Russian, a language with limited biomedical resources. We present a new dataset
for ICD coding, which includes diagnosis fields from electronic health records
(EHRs) annotated with over 10,000 entities and more than 1,500 unique ICD
codes. This dataset serves as a benchmark fo…

------------------------------------------------------------------------------------------
Cluster 4 | size=15

[2503.07365] MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale
  Reinforcement Learning
Summary: We present MM-Eureka, a multimodal reasoning model that successfully extends
large-scale rule-based reinforcement learning (RL) to multimodal reasoning.
While rule-based RL has shown remarkable success in improving LLMs' reasoning
abilities in text domains, its application to multimodal settings has remained
challenging. Our work reproduces key cha…

[2503.05132] R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model
Summary: Recently DeepSeek R1 demonstrated how reinforcement learning with simple
rule-based incentives can enable autonomous development of complex reasoning in
large language models, characterized by the "aha moment", in which the model
manifest self-reflection and increased response length during training.
However, attempts to extend this success to mult…

[2503.21776] Video-R1: Reinforcing Video Reasoning in MLLMs
Summary: Inspired by DeepSeek-R1's success in eliciting reasoning abilities through
rule-based reinforcement learning (RL), we introduce Video-R1 as the first
attempt to systematically explore the R1 paradigm for eliciting video reasoning
within multimodal large language models (MLLMs). However, directly applying RL
training with the GRPO algorithm to video…

[2503.10639] GoT: Unleashing Reasoning Capability of Multimodal Large Language Model
  for Visual Generation and Editing
Summary: Current image generation and editing methods primarily process textual
prompts as direct inputs without reasoning about visual composition and
explicit operations. We present Generation Chain-of-Thought (GoT), a novel
paradigm that enables generation and editing through an explicit language
reasoning process before outputting images. This approach …

[2503.07536] LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through
  Two-Stage Rule-Based RL
URL: https://forjadeforest.github.io/LMM-R1-ProjectPage
Summary: Enhancing reasoning in Large Multimodal Models (LMMs) faces unique challenges
from the complex interplay between visual perception and logical reasoning,
particularly in compact 3B-parameter architectures where architectural
constraints limit reasoning capacity and modality alignment.
  While rule-based reinforcement learning (RL) excels in text-on…

[2503.21620] UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement
  Learning
URL: https://yxchai.com/UI-R1/
Summary: The recent DeepSeek-R1 has showcased the emergence of reasoning capabilities
in LLMs through reinforcement learning (RL) with rule-based rewards. Building
on this idea, we are the first to explore how rule-based RL can enhance the
reasoning capabilities of multimodal large language models (MLLMs) for graphic
user interface (GUI) action prediction t…

[2503.10480] World Modeling Makes a Better Planner: Dual Preference Optimization for
  Embodied Task Planning
Summary: Recent advances in large vision-language models (LVLMs) have shown promise
for embodied task planning, yet they struggle with fundamental challenges like
dependency constraints and efficiency. Existing approaches either solely
optimize action selection or leverage world models during inference,
overlooking the benefits of learning to model the worl…

[2503.01785] Visual-RFT: Visual Reinforcement Fine-Tuning
URL: https://github.com/Liuziyu77/Visual-RFT
Summary: Reinforcement Fine-Tuning (RFT) in Large Reasoning Models like OpenAI o1
learns from feedback on its answers, which is especially useful in applications
when fine-tuning data is scarce. Recent open-source work like DeepSeek-R1
demonstrates that reinforcement learning with verifiable reward is one key
direction in reproducing o1. While the R1-style …

[2503.05236] Unified Reward Model for Multimodal Understanding and Generation
URL: https://codegoat24.github.io/UnifiedReward/
Summary: Recent advances in human preference alignment have significantly enhanced
multimodal generation and understanding. A key approach is training reward
models to guide preference optimization. However, existing models are often
task-specific, limiting their adaptability across diverse visual applications.
We also argue that jointly learning to assess …

[2503.10613] CoSTAast: Cost-Sensitive Toolpath Agent for Multi-turn Image Editing
URL: https://huggingface.co/datasets/umd-zhou-lab/CoSTAR
Summary: Text-to-image models like stable diffusion and DALLE-3 still struggle with
multi-turn image editing. We decompose such a task as an agentic workflow
(path) of tool use that addresses a sequence of subtasks by AI tools of varying
costs. Conventional search algorithms require expensive exploration to find
tool paths. While large language models (LLMs…

[2503.07920] Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural
  Vision-Language Dataset for Southeast Asia
Summary: Southeast Asia (SEA) is a region of extraordinary linguistic and cultural
diversity, yet it remains significantly underrepresented in vision-language
(VL) research. This often results in artificial intelligence (AI) models that
fail to capture SEA cultural nuances. To fill this gap, we present SEA-VL, an
open-source initiative dedicated to developi…

[2503.12533] Being-0: A Humanoid Robotic Agent with Vision-Language Models and
  Modular Skills
URL: https://beingbeyond.github.io/Being-0/
Summary: Building autonomous robotic agents capable of achieving human-level
performance in real-world embodied tasks is an ultimate goal in humanoid robot
research. Recent advances have made significant progress in high-level
cognition with Foundation Models (FMs) and low-level skill development for
humanoid robots. However, directly combining these compon…

[2503.23307] MoCha: Towards Movie-Grade Talking Character Synthesis
URL: https://congwei1230.github.io/MoCha/
Summary: Recent advancements in video generation have achieved impressive motion
realism, yet they often overlook character-driven storytelling, a crucial task
for automated film, animation generation. We introduce Talking Characters, a
more realistic task to generate talking character animations directly from
speech and text. Unlike talking head, Talking C…

[2503.16905] MAPS: A Multi-Agent Framework Based on Big Seven Personality and
  Socratic Guidance for Multimodal Scientific Problem Solving
Summary: Multimodal scientific problems (MSPs) involve complex issues that require the
integration of multiple modalities, such as text and diagrams, presenting a
significant challenge in artificial intelligence. While progress has been made
in addressing traditional scientific problems, MSPs still face two primary
issues: the challenge of multi-modal compr…

[2503.11576] SmolDocling: An ultra-compact vision-language model for end-to-end
  multi-modal document conversion
URL: https://huggingface.co/ds4sd/SmolDocling-256M-preview
Summary: We introduce SmolDocling, an ultra-compact vision-language model targeting
end-to-end document conversion. Our model comprehensively processes entire
pages by generating DocTags, a new universal markup format that captures all
page elements in their full context with location. Unlike existing approaches
that rely on large foundational models, or en…

------------------------------------------------------------------------------------------
Cluster 3 | size=9

[2503.19325] Long-Context Autoregressive Video Modeling with Next-Frame Prediction
URL: https://farlongctx.github.io/
Summary: Long-context autoregressive modeling has significantly advanced language
generation, but video generation still struggles to fully utilize extended
temporal contexts. To investigate long-context video modeling, we introduce
Frame AutoRegressive (FAR), a strong baseline for video autoregressive
modeling. Just as language models learn causal dependen…

[2503.04130] Token-Efficient Long Video Understanding for Multimodal LLMs
URL: https://research.nvidia.com/labs/lpr/storm/
Summary: Recent advances in video-based multimodal large language models (Video-LLMs)
have significantly improved video understanding by processing videos as
sequences of image frames. However, many existing methods treat frames
independently in the vision backbone, lacking explicit temporal modeling, which
limits their ability to capture dynamic patterns a…

[2503.06053] DropletVideo: A Dataset and Approach to Explore Integral Spatio-Temporal
  Consistent Video Generation
Summary: Spatio-temporal consistency is a critical research topic in video generation.
A qualified generated video segment must ensure plot plausibility and coherence
while maintaining visual consistency of objects and scenes across varying
viewpoints. Prior research, especially in open-source projects, primarily
focuses on either temporal or spatial consis…

[2503.11647] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video
URL: https://jianhongbai.github.io/ReCamMaster/
Summary: Camera control has been actively studied in text or image conditioned video
generation tasks. However, altering camera trajectories of a given video
remains under-explored, despite its importance in the field of video creation.
It is non-trivial due to the extra constraints of maintaining multiple-frame
appearance and dynamic synchronization. To ad…

[2503.13358] One-Step Residual Shifting Diffusion for Image Super-Resolution via
  Distillation
Summary: Diffusion models for super-resolution (SR) produce high-quality visual
results but require expensive computational costs. Despite the development of
several methods to accelerate diffusion-based SR models, some (e.g., SinSR)
fail to produce realistic perceptual details, while others (e.g., OSEDiff) may
hallucinate non-existent structures. To overco…

[2503.16660] When Less is Enough: Adaptive Token Reduction for Efficient Image
  Representation
Summary: Vision encoders typically generate a large number of visual tokens, providing
information-rich representations but significantly increasing computational
demands. This raises the question of whether all generated tokens are equally
valuable or if some of them can be discarded to reduce computational costs
without compromising quality. In this paper…

[2503.09573] Block Diffusion: Interpolating Between Autoregressive and Diffusion
  Language Models
URL: https://m-arriola.com/bd3lms/
Summary: Diffusion language models offer unique benefits over autoregressive models
due to their potential for parallelized generation and controllability, yet
they lag in likelihood modeling and are limited to fixed-length generation. In
this work, we introduce a class of block diffusion language models that
interpolate between discrete denoising diffusion…

[2503.07677] PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference
  Time by Leveraging Sparsity
URL: https://cubeyoung.github.io/pladis-proejct/
Summary: Diffusion models have shown impressive results in generating high-quality
conditional samples using guidance techniques such as Classifier-Free Guidance
(CFG). However, existing methods often require additional training or neural
function evaluations (NFEs), making them incompatible with guidance-distilled
models. Also, they rely on heuristic appro…

[2503.10633] Charting and Navigating Hugging Face's Model Atlas
URL: https://horwitz.ai/model-atlas
Summary: As there are now millions of publicly available neural networks, searching
and analyzing large model repositories becomes increasingly important.
Navigating so many models requires an atlas, but as most models are poorly
documented charting such an atlas is challenging. To explore the hidden
potential of model repositories, we chart a preliminary a…

------------------------------------------------------------------------------------------
Cluster 2 | size=5

[2503.20215] Qwen2.5-Omni Technical Report
URL: https://qwenlm.github.io/blog/qwen2.5-omni/
Summary: In this report, we present Qwen2.5-Omni, an end-to-end multimodal model
designed to perceive diverse modalities, including text, images, audio, and
video, while simultaneously generating text and natural speech responses in a
streaming manner. To enable the streaming of multimodal information inputs,
both audio and visual encoders utilize a block-w…

[2503.04724] LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM
URL: https://mbzuai-oryx.github.io/LLMVoX/
Summary: Recent advancements in speech-to-speech dialogue systems leverage LLMs for
multimodal interactions, yet they remain hindered by fine-tuning requirements,
high computational overhead, and text-speech misalignment. Existing
speech-enabled LLMs often degrade conversational quality by modifying the LLM,
thereby compromising its linguistic capabilities.…

[2503.19786] Gemma 3 Technical Report
Summary: We introduce Gemma 3, a multimodal addition to the Gemma family of
lightweight open models, ranging in scale from 1 to 27 billion parameters. This
version introduces vision understanding abilities, a wider coverage of
languages and longer context - at least 128K tokens. We also change the
architecture of the model to reduce the KV-cache memory that…

[2503.08638] YuE: Scaling Open Foundation Models for Long-Form Music Generation
URL: https://map-yue.github.io/
Summary: We tackle the task of long-form music generation--particularly the
challenging lyrics-to-song problem--by introducing YuE, a family of
open foundation models based on the LLaMA2 architecture. Specifically, YuE
scales to trillions of tokens and generates up to five minutes of music while
maintaining lyrical alignment, coherent musical structure, and…

[2503.10622] Transformers without Normalization
Summary: Normalization layers are ubiquitous in modern neural networks and have long
been considered essential. This work demonstrates that Transformers without
normalization can achieve the same or better performance using a remarkably
simple technique. We introduce Dynamic Tanh (DyT), an element-wise operation
DyT(x) = tanh(alpha x), as a drop-in replacem…

------------------------------------------------------------------------------------------
Cluster 0 | size=4

[2503.20314] Wan: Open and Advanced Large-Scale Video Generative Models
Summary: This report presents Wan, a comprehensive and open suite of video foundation
models designed to push the boundaries of video generation. Built upon the
mainstream diffusion transformer paradigm, Wan achieves significant
advancements in generative capabilities through a series of innovations,
including our novel VAE, scalable pre-training strategies…

[2503.18942] Video-T1: Test-Time Scaling for Video Generation
URL: https://liuff19.github.io/Video-T1/
Summary: With the scale capability of increasing training data, model size, and
computational cost, video generation has achieved impressive results in digital
creation, enabling users to express creativity across various domains.
Recently, researchers in Large Language Models (LLMs) have expanded the scaling
to test-time, which can significantly improve LL…

[2503.07598] VACE: All-in-One Video Creation and Editing
URL: https://ali-vilab.github.io/VACE-Page/
Summary: Diffusion Transformer has demonstrated powerful capability and scalability in
generating high-quality images and videos. Further pursuing the unification of
generation and editing tasks has yielded significant progress in the domain of
image content creation. However, due to the intrinsic demands for consistency
across both temporal and spatial dyn…

[2503.14378] Impossible Videos
URL: https://showlab.github.io/Impossible-Videos/
Summary: Synthetic videos nowadays is widely used to complement data scarcity and
diversity of real-world videos. Current synthetic datasets primarily replicate
real-world scenarios, leaving impossible, counterfactual and anti-reality video
concepts underexplored. This work aims to answer two questions: 1) Can today's
video generation models effectively fol…

==========================================================================================
# month=2025-04 BEST CLUSTERING (mode=C, k=5)

------------------------------------------------------------------------------------------
Cluster 1 | size=22

[2504.05299] SmolVLM: Redefining small and efficient multimodal models
URL: https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7
Summary: Large Vision-Language Models (VLMs) deliver exceptional performance but
require significant computational resources, limiting their deployment on
mobile and edge devices. Smaller VLMs typically mirror design choices of larger
models, such as extensive image tokenization, leading to inefficient GPU memory
usage and constrained practicality for on-de…

[2504.15271] Eagle 2.5: Boosting Long-Context Post-Training for Frontier
  Vision-Language Models
URL: https://nvlabs.github.io/EAGLE/
Summary: We introduce Eagle 2.5, a family of frontier vision-language models (VLMs)
for long-context multimodal learning. Our work addresses the challenges in long
video comprehension and high-resolution image understanding, introducing a
generalist framework for both tasks. The proposed training framework
incorporates Automatic Degrade Sampling and Image A…

[2504.15279] VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal
  Large Language Models
URL: https://visulogic-benchmark.github.io/VisuLogic/
Summary: Visual reasoning is a core component of human intelligence and a critical
capability for advanced multimodal models. Yet current reasoning evaluations of
multimodal large language models (MLLMs) often rely on text descriptions and
allow language-based reasoning shortcuts, failing to measure genuine
vision-centric reasoning. To address this, we intr…

[2504.05599] Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought
Summary: We introduce Skywork R1V, a multimodal reasoning model extending the an
R1-series Large language models (LLM) to visual modalities via an efficient
multimodal transfer method. Leveraging a lightweight visual projector, Skywork
R1V facilitates seamless multimodal adaptation without necessitating retraining
of either the foundational language model o…

[2504.10479] InternVL3: Exploring Advanced Training and Test-Time Recipes for
  Open-Source Multimodal Models
URL: https://internvl.github.io/blog/2025-04-11-InternVL-3.0/
Summary: We introduce InternVL3, a significant advancement in the InternVL series
featuring a native multimodal pre-training paradigm. Rather than adapting a
text-only large language model (LLM) into a multimodal large language model
(MLLM) that supports visual inputs, InternVL3 jointly acquires multimodal and
linguistic capabilities from both diverse multi…

[2504.00883] Improved Visual-Spatial Reasoning via R1-Zero-Like Training
Summary: Increasing attention has been placed on improving the reasoning capacities of
multi-modal large language models (MLLMs). As the cornerstone for AI agents
that function in the physical realm, video-based visual-spatial intelligence
(VSI) emerges as one of the most pivotal reasoning capabilities of MLLMs. This
work conducts a first, in-depth study on…

[2503.24379] Any2Caption:Interpreting Any Condition to Caption for Controllable Video
  Generation
URL: https://sqwu.top/Any2Cap/
Summary: To address the bottleneck of accurate user intent interpretation within the
current video generation community, we present Any2Caption, a novel framework
for controllable video generation under any condition. The key idea is to
decouple various condition interpretation steps from the video synthesis step.
By leveraging modern multimodal large langu…

[2504.07491] Kimi-VL Technical Report
URL: https://huggingface.co/spaces/moonshotai/Kimi-VL-A3B-Thinking
Summary: We present Kimi-VL, an efficient open-source Mixture-of-Experts (MoE)
vision-language model (VLM) that offers advanced multimodal reasoning,
long-context understanding, and strong agent capabilities - all while
activating only 2.8B parameters in its language decoder (Kimi-VL-A3B). Kimi-VL
demonstrates strong performance across challenging domains: …

[2504.11346] Seedream 3.0 Technical Report
URL: https://team.doubao.com/zh/tech/seedream3_0
Summary: We present Seedream 3.0, a high-performance Chinese-English bilingual image
generation foundation model. We develop several technical improvements to
address existing challenges in Seedream 2.0, including alignment with
complicated prompts, fine-grained typography generation, suboptimal visual
aesthetics and fidelity, and limited image resolutions.…

[2504.02826] Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual
  Editing
Summary: Large Multi-modality Models (LMMs) have made significant progress in visual
understanding and generation, but they still face challenges in General Visual
Editing, particularly in following complex instructions, preserving appearance
consistency, and supporting flexible input formats. To address this gap, we
introduce RISEBench, the first benchmark…

[2504.02782] GPT-ImgEval: A Comprehensive Benchmark for Diagnosing GPT4o in Image
  Generation
Summary: The recent breakthroughs in OpenAI's GPT4o model have demonstrated
surprisingly good capabilities in image generation and editing, resulting in
significant excitement in the community. This technical report presents the
first-look evaluation benchmark (named GPT-ImgEval), quantitatively and
qualitatively diagnosing GPT-4o's performance across three…

[2503.23461] TextCrafter: Accurately Rendering Multiple Texts in Complex Visual
  Scenes
URL: https://dnknju.github.io/textcrafter-vue/
Summary: This paper explores the task of Complex Visual Text Generation (CVTG), which
centers on generating intricate textual content distributed across diverse
regions within visual images. In CVTG, image generation models often rendering
distorted and blurred visual text or missing some visual text. To tackle these
challenges, we propose TextCrafter, a no…

[2503.23307] MoCha: Towards Movie-Grade Talking Character Synthesis
URL: https://congwei1230.github.io/MoCha/
Summary: Recent advancements in video generation have achieved impressive motion
realism, yet they often overlook character-driven storytelling, a crucial task
for automated film, animation generation. We introduce Talking Characters, a
more realistic task to generate talking character animations directly from
speech and text. Unlike talking head, Talking C…

[2504.00999] MergeVQ: A Unified Framework for Visual Generation and Representation
  with Disentangled Token Merging and Quantization
URL: https://apexgen-x.github.io/MergeVQ/
Summary: Masked Image Modeling (MIM) with Vector Quantization (VQ) has achieved great
success in both self-supervised pre-training and image generation. However,
most existing methods struggle to address the trade-off in shared latent space
for generation quality vs. representation learning and efficiency. To push the
limits of this paradigm, we propose Mer…

[2504.17761] Step1X-Edit: A Practical Framework for General Image Editing
Summary: In recent years, image editing models have witnessed remarkable and rapid
development. The recent unveiling of cutting-edge multimodal models such as
GPT-4o and Gemini2 Flash has introduced highly promising image editing
capabilities. These models demonstrate an impressive aptitude for fulfilling a
vast majority of user-driven editing requirements,…

[2504.05979] An Empirical Study of GPT-4o Image Generation Capabilities
Summary: The landscape of image generation has rapidly evolved, from early GAN-based
approaches to diffusion models and, most recently, to unified generative
architectures that seek to bridge understanding and generation tasks. Recent
advances, especially the GPT-4o, have demonstrated the feasibility of
high-fidelity multimodal generation, their architectur…

[2504.15376] Towards Understanding Camera Motions in Any Video
URL: https://linzhiqiu.github.io/papers/camerabench/
Summary: We introduce CameraBench, a large-scale dataset and benchmark designed to
assess and improve camera motion understanding. CameraBench consists of ~3,000
diverse internet videos, annotated by experts through a rigorous multi-stage
quality control process. One of our contributions is a taxonomy of camera
motion primitives, designed in collaboration w…

[2504.16072] Describe Anything: Detailed Localized Image and Video Captioning
URL: https://describe-anything.github.io
Summary: Generating detailed and accurate descriptions for specific regions in images
and videos remains a fundamental challenge for vision-language models. We
introduce the Describe Anything Model (DAM), a model designed for detailed
localized captioning (DLC). DAM preserves both local details and global context
through two key innovations: a focal prompt,…

[2504.06263] OmniSVG: A Unified Scalable Vector Graphics Generation Model
URL: https://omnisvg.github.io/
Summary: Scalable Vector Graphics (SVG) is an important image format widely adopted in
graphic design because of their resolution independence and editability. The
study of generating high-quality SVG has continuously drawn attention from both
designers and researchers in the AIGC community. However, existing methods
either produces unstructured outputs wit…

[2504.01014] AnimeGamer: Infinite Anime Life Simulation with Next Game State
  Prediction
URL: https://howe125.github.io/AnimeGamer.github.io/
Summary: Recent advancements in image and video synthesis have opened up new promise
in generative games. One particularly intriguing application is transforming
characters from anime films into interactive, playable entities. This allows
players to immerse themselves in the dynamic anime world as their favorite
characters for life simulation through langua…

[2504.08685] Seaweed-7B: Cost-Effective Training of Video Generation Foundation Model
URL: https://seaweed.video/
Summary: This technical report presents a cost-efficient strategy for training a video
generation foundation model. We present a mid-sized research model with
approximately 7 billion parameters (7B) called Seaweed-7B trained from scratch
using 665,000 H100 GPU hours. Despite being trained with moderate computational
resources, Seaweed-7B demonstrates highly…

[2504.05298] One-Minute Video Generation with Test-Time Training
URL: https://test-time-training.github.io/video-dit/
Summary: Transformers today still struggle to generate one-minute videos because
self-attention layers are inefficient for long context. Alternatives such as
Mamba layers struggle with complex multi-scene stories because their hidden
states are less expressive. We experiment with Test-Time Training (TTT) layers,
whose hidden states themselves can be neural …

------------------------------------------------------------------------------------------
Cluster 3 | size=12

[2504.20571] Reinforcement Learning for Reasoning in Large Language Models with One
  Training Example
Summary: We show that reinforcement learning with verifiable reward using one training
example (1-shot RLVR) is effective in incentivizing the math reasoning
capabilities of large language models (LLMs). Applying RLVR to the base model
Qwen2.5-Math-1.5B, we identify a single example that elevates model performance
on MATH500 from 36.0% to 73.6%, and improve…

[2504.13837] Does Reinforcement Learning Really Incentivize Reasoning Capacity in
  LLMs Beyond the Base Model?
URL: https://limit-of-rlvr.github.io/
Summary: Reinforcement Learning with Verifiable Rewards (RLVR) has recently
demonstrated notable success in enhancing the reasoning capabilities of LLMs,
particularly in mathematics and programming tasks. It is widely believed that
RLVR enables LLMs to continuously self-improve, thus acquiring novel reasoning
abilities that exceed corresponding base models'…

[2503.20783] Understanding R1-Zero-Like Training: A Critical Perspective
Summary: DeepSeek-R1-Zero has shown that reinforcement learning (RL) at scale can
directly enhance the reasoning capabilities of LLMs without supervised
fine-tuning. In this work, we critically examine R1-Zero-like training by
analyzing its two core components: base models and RL. We investigate a wide
range of base models, including DeepSeek-V3-Base, to un…

[2504.16084] TTRL: Test-Time Reinforcement Learning
Summary: This paper investigates Reinforcement Learning (RL) on data without explicit
labels for reasoning tasks in Large Language Models (LLMs). The core challenge
of the problem is reward estimation during inference while not having access to
ground-truth information. While this setting appears elusive, we find that
common practices in Test-Time Scaling (…

[2504.02495] Inference-Time Scaling for Generalist Reward Modeling
Summary: Reinforcement learning (RL) has been widely adopted in post-training for
large language models (LLMs) at scale. Recently, the incentivization of
reasoning capabilities in LLMs from RL indicates that proper learning
methods could enable effective inference-time scalability. A key challenge of
RL is to obtain accurate reward signals for LLMs in vario…

[2504.00050] JudgeLRM: Large Reasoning Models as a Judge
URL: https://huggingface.co/spaces/nuojohnchen/JudgeLRMDemo
Summary: The rise of Large Language Models (LLMs) as evaluators offers a scalable
alternative to human annotation, yet existing Supervised Fine-Tuning (SFT) for
judges approaches often fall short in domains requiring complex reasoning. In
this work, we investigate whether LLM judges truly benefit from enhanced
reasoning capabilities. Through a detailed anal…

[2503.24290] Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement
  Learning on the Base Model
URL: https://huggingface.co/Open-Reasoner-Zero
Summary: We introduce Open-Reasoner-Zero, the first open source implementation of
large-scale reasoning-oriented RL training focusing on scalability, simplicity
and accessibility. Through extensive experiments, we demonstrate that a
minimalist approach, vanilla PPO with GAE (lambda=1, gamma=1) and
straightforward rule-based rewards, without any KL regulariz…

[2504.11536] ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
URL: https://retool-rl.github.io/
Summary: While reasoning models (e.g., DeepSeek R1) trained with reinforcement
learning (RL), excel in textual reasoning, they struggle in scenarios requiring
structured problem-solving, such as geometric reasoning, concise computation,
or complex equation solving-areas where computational tools like code
interpreters (CI) demonstrate distinct advantages. T…

[2504.07128] DeepSeek-R1 Thoughtology: Let's <think> about LLM Reasoning
URL: https://mcgill-nlp.github.io/thoughtology/
Summary: Large Reasoning Models like DeepSeek-R1 mark a fundamental shift in how LLMs
approach complex problems. Instead of directly producing an answer for a given
input, DeepSeek-R1 creates detailed multi-step reasoning chains, seemingly
"thinking" about a problem before providing an answer. This reasoning process
is publicly available to the user, creati…

[2504.14945] Learning to Reason under Off-Policy Guidance
Summary: Recent advances in large reasoning models (LRMs) demonstrate that
sophisticated behaviors such as multi-step reasoning and self-reflection can
emerge via reinforcement learning (RL) with simple rule-based rewards. However,
existing zero-RL approaches are inherently ``on-policy'', limiting learning to
a model's own outputs and failing to acquire rea…

[2504.10481] xVerify: Efficient Answer Verifier for Reasoning Model Evaluations
Summary: With the release of the o1 model by OpenAI, reasoning models adopting slow
thinking strategies have gradually emerged. As the responses generated by such
models often include complex reasoning, intermediate steps, and
self-reflection, existing evaluation methods are often inadequate. They
struggle to determine whether the LLM output is truly equiva…

[2504.13146] Antidistillation Sampling
URL: https://antidistillation.com
Summary: Frontier models that generate extended reasoning traces inadvertently produce
rich token sequences that can facilitate model distillation. Recognizing this
vulnerability, model owners may seek sampling strategies that limit the
effectiveness of distillation without compromising model performance.
Antidistillation sampling provides exactly this capa…

------------------------------------------------------------------------------------------
Cluster 0 | size=8

[2504.13161] CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for
  Language Model Pre-training
URL: https://research.nvidia.com/labs/lpr/climb/
Summary: Pre-training datasets are typically collected from web content and lack
inherent domain divisions. For instance, widely used datasets like Common Crawl
do not include explicit domain labels, while manually curating labeled datasets
such as The Pile is labor-intensive. Consequently, identifying an optimal
pre-training data mixture remains a challeng…

[2504.20734] UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with
  Diverse Modalities and Granularities
URL: https://universalrag.github.io
Summary: Retrieval-Augmented Generation (RAG) has shown substantial promise in
improving factual accuracy by grounding model responses with external knowledge
relevant to queries. However, most existing RAG approaches are limited to a
text-only corpus, and while recent efforts have extended RAG to other
modalities such as images and videos, they typically o…

[2504.15521] The Bitter Lesson Learned from 2,000+ Multilingual Benchmarks
Summary: As large language models (LLMs) continue to advance in linguistic
capabilities, robust multilingual evaluation has become essential for promoting
equitable technological progress. This position paper examines over 2,000
multilingual (non-English) benchmarks from 148 countries, published between
2021 and 2024, to evaluate past, present, and future p…

[2504.07964] C3PO: Critical-Layer, Core-Expert, Collaborative Pathway Optimization
  for Test-Time Expert Re-Mixing
Summary: Mixture-of-Experts (MoE) Large Language Models (LLMs) suffer from severely
sub-optimal expert pathways-our study reveals that naive expert selection
learned from pretraining leaves a surprising 10-20% accuracy gap for
improvement. Motivated by this observation, we develop a novel class of
test-time optimization methods to re-weight or "re-mixing" t…

[2504.00927] Multi-Token Attention
Summary: Soft attention is a critical mechanism powering LLMs to locate relevant parts
within a given context. However, individual attention weights are determined by
the similarity of only a single query and key token vector. This "single token
attention" bottlenecks the amount of information used in distinguishing a
relevant part from the rest of the cont…

[2504.07096] OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training
  Tokens
URL: https://playground.allenai.org
Summary: We present OLMoTrace, the first system that traces the outputs of language
models back to their full, multi-trillion-token training data in real time.
OLMoTrace finds and shows verbatim matches between segments of language model
output and documents in the training text corpora. Powered by an extended
version of infini-gram (Liu et al., 2024), our …

[2504.02507] ZClip: Adaptive Spike Mitigation for LLM Pre-Training
Summary: Training large language models (LLMs) presents numerous challenges, including
gradient instability and loss spikes. These phenomena can lead to catastrophic
divergence, requiring costly checkpoint restoration and data batch skipping.
Traditional gradient clipping techniques, such as constant or norm-based
methods, fail to address these issues effec…

[2504.15120] Kuwain 1.5B: An Arabic SLM via Language Injection
Summary: Enhancing existing models with new knowledge is a crucial aspect of AI
development. This paper introduces a novel method for integrating a new
language into a large language model (LLM). Our approach successfully
incorporates a previously unseen target language into an existing LLM without
compromising its prior knowledge. We trained a tiny model w…

------------------------------------------------------------------------------------------
Cluster 2 | size=4

[2504.08791] PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday
  Home Clusters
URL: https://github.com/Lizonghang/prima.cpp
Summary: Emergency of DeepSeek R1 and QwQ 32B have broken through performance barriers
for running frontier large language models (LLMs) on home devices. While
consumer hardware is getting stronger and model quantization is improving,
existing end-side solutions still demand GPU clusters, large RAM/VRAM, and high
bandwidth, far beyond what a common home clu…

[2504.12285] BitNet b1.58 2B4T Technical Report
Summary: We introduce BitNet b1.58 2B4T, the first open-source, native 1-bit Large
Language Model (LLM) at the 2-billion parameter scale. Trained on a corpus of 4
trillion tokens, the model has been rigorously evaluated across benchmarks
covering language understanding, mathematical reasoning, coding proficiency,
and conversational ability. Our results demo…

[2504.06261] Hogwild! Inference: Parallel LLM Generation via Concurrent Attention
URL: https://eqimp.github.io/hogwild_llm/
Summary: Large Language Models (LLMs) have demonstrated the ability to tackle
increasingly complex tasks through advanced reasoning, long-form content
generation, and tool use. Solving these tasks often involves long
inference-time computations. In human problem solving, a common strategy to
expedite work is collaboration: by dividing the problem into sub-t…

[2504.05741] DDT: Decoupled Diffusion Transformer
Summary: Diffusion transformers have demonstrated remarkable generation quality,
albeit requiring longer training iterations and numerous inference steps. In
each denoising step, diffusion transformers encode the noisy inputs to extract
the lower-frequency semantic component and then decode the higher frequency
with identical modules. This scheme creates an…

------------------------------------------------------------------------------------------
Cluster 4 | size=4

[2504.01990] Advances and Challenges in Foundation Agents: From Brain-Inspired
  Intelligence to Evolutionary, Collaborative, and Safe Systems
Summary: The advent of large language models (LLMs) has catalyzed a transformative
shift in artificial intelligence, paving the way for advanced intelligent
agents capable of sophisticated reasoning, robust perception, and versatile
action across diverse domains. As these agents increasingly drive AI research
and practical applications, their design, evalua…

[2504.20879] The Leaderboard Illusion
URL: https://cohere.com/research/lmarena
Summary: Measuring progress is fundamental to the advancement of any scientific field.
As benchmarks play an increasingly central role, they also grow more
susceptible to distortion. Chatbot Arena has emerged as the go-to leaderboard
for ranking the most capable AI systems. Yet, in this work we identify
systematic issues that have resulted in a distorted pl…

[2504.17192] Paper2Code: Automating Code Generation from Scientific Papers in Machine
  Learning
Summary: Despite the rapid growth of machine learning research, corresponding code
implementations are often unavailable, making it slow and labor-intensive for
researchers to reproduce results and build upon prior work. In the meantime,
recent Large Language Models (LLMs) excel at understanding scientific documents
and generating high-quality code. Inspire…

[2504.01724] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation
  with Hybrid Guidance
URL: https://grisoon.github.io/DreamActor-M1/
Summary: While recent image-based human animation methods achieve realistic body and
facial motion synthesis, critical gaps remain in fine-grained holistic
controllability, multi-scale adaptability, and long-term temporal coherence,
which leads to their lower expressiveness and robustness. We propose a
diffusion transformer (DiT) based framework, DreamActor…

==========================================================================================
# month=2025-05 BEST CLUSTERING (mode=B, k=5)

------------------------------------------------------------------------------------------
Cluster 2 | size=17

[2505.17667] QwenLong-L1: Towards Long-Context Large Reasoning Models with
  Reinforcement Learning
Summary: Recent large reasoning models (LRMs) have demonstrated strong reasoning
capabilities through reinforcement learning (RL). These improvements have
primarily been observed within the short-context reasoning tasks. In contrast,
extending LRMs to effectively process and reason on long-context inputs via RL
remains a critical unsolved challenge. To brid…

[2505.19641] SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning
  Logical Reasoning and Beyond
URL: https://huggingface.co/datasets/MiniMaxAI/SynLogic
Summary: Recent advances such as OpenAI-o1 and DeepSeek R1 have demonstrated the
potential of Reinforcement Learning (RL) to enhance reasoning abilities in
Large Language Models (LLMs). While open-source replication efforts have
primarily focused on mathematical and coding domains, methods and resources for
developing general reasoning capabilities remain u…

[2505.14810] Scaling Reasoning, Losing Control: Evaluating Instruction Following in
  Large Reasoning Models
Summary: Instruction-following is essential for aligning large language models (LLMs)
with user intent. While recent reasoning-oriented models exhibit impressive
performance on complex mathematical problems, their ability to adhere to
natural language instructions remains underexplored. In this work, we introduce
MathIF, a dedicated benchmark for evaluating…

[2505.10554] Beyond 'Aha!': Toward Systematic Meta-Abilities Alignment in Large
  Reasoning Models
Summary: Large reasoning models (LRMs) already possess a latent capacity for long
chain-of-thought reasoning. Prior work has shown that outcome-based
reinforcement learning (RL) can incidentally elicit advanced reasoning
behaviors such as self-correction, backtracking, and verification phenomena
often referred to as the model's "aha moment". However, the ti…

[2505.23621] Table-R1: Inference-Time Scaling for Table Reasoning
Summary: In this work, we present the first study to explore inference-time scaling on
table reasoning tasks. We develop and evaluate two post-training strategies to
enable inference-time scaling: distillation from frontier model reasoning
traces and reinforcement learning with verifiable rewards (RLVR). For
distillation, we introduce a large-scale dataset …

[2505.02387] RM-R1: Reward Modeling as Reasoning
Summary: Reward modeling is essential for aligning large language models (LLMs) with
human preferences, especially through reinforcement learning from human
feedback (RLHF). To provide accurate reward signals, a reward model (RM) should
stimulate deep thinking and conduct interpretable reasoning before assigning a
score or a judgment. However, existing RMs …

[2505.04921] Perception, Reason, Think, and Plan: A Survey on Large Multimodal
  Reasoning Models
URL: https://github.com/HITsz-TMG/Awesome-Large-Multimodal-Reasoning-Models
Summary: Reasoning lies at the heart of intelligence, shaping the ability to make
decisions, draw conclusions, and generalize across domains. In artificial
intelligence, as systems increasingly operate in open, uncertain, and
multimodal environments, reasoning becomes essential for enabling robust and
adaptive behavior. Large Multimodal Reasoning Models (LM…

[2505.07608] MiMo: Unlocking the Reasoning Potential of Language Model -- From
  Pretraining to Posttraining
Summary: We present MiMo-7B, a large language model born for reasoning tasks, with
optimization across both pre-training and post-training stages. During
pre-training, we enhance the data preprocessing pipeline and employ a
three-stage data mixing strategy to strengthen the base model's reasoning
potential. MiMo-7B-Base is pre-trained on 25 trillion tokens,…

[2505.03335] Absolute Zero: Reinforced Self-play Reasoning with Zero Data
URL: https://andrewzh112.github.io/absolute-zero-reasoner/
Summary: Reinforcement learning with verifiable rewards (RLVR) has shown promise in
enhancing the reasoning capabilities of large language models by learning
directly from outcome-based rewards. Recent RLVR works that operate under the
zero setting avoid supervision in labeling the reasoning process, but still
depend on manually curated collections of quest…

[2505.22617] The Entropy Mechanism of Reinforcement Learning for Reasoning Language
  Models
Summary: This paper aims to overcome a major obstacle in scaling RL for reasoning with
LLMs, namely the collapse of policy entropy. Such phenomenon is consistently
observed across vast RL runs without entropy intervention, where the policy
entropy dropped sharply at the early training stage, this diminished
exploratory ability is always accompanied with the…

[2505.13417] AdaptThink: Reasoning Models Can Learn When to Think
Summary: Recently, large reasoning models have achieved impressive performance on
various tasks by employing human-like deep thinking. However, the lengthy
thinking process substantially increases inference overhead, making efficiency
a critical bottleneck. In this work, we first demonstrate that NoThinking,
which prompts the reasoning model to skip thinkin…

[2505.03318] Unified Multimodal Chain-of-Thought Reward Model through Reinforcement
  Fine-Tuning
URL: https://codegoat24.github.io/UnifiedReward/think
Summary: Recent advances in multimodal Reward Models (RMs) have shown significant
promise in delivering reward signals to align vision models with human
preferences. However, current RMs are generally restricted to providing direct
responses or engaging in shallow reasoning processes with limited depth, often
leading to inaccurate reward signals. We posit t…

[2505.21327] MME-Reasoning: A Comprehensive Benchmark for Logical Reasoning in MLLMs
URL: https://alpha-innovator.github.io/mmereasoning.github.io/
Summary: Logical reasoning is a fundamental aspect of human intelligence and an
essential capability for multimodal large language models (MLLMs). Despite the
significant advancement in multimodal reasoning, existing benchmarks fail to
comprehensively evaluate their reasoning abilities due to the lack of explicit
categorization for logical reasoning types a…

[2504.20752] Grokking in the Wild: Data Augmentation for Real-World Multi-Hop
  Reasoning with Transformers
Summary: Transformers have achieved great success in numerous NLP tasks but continue
to exhibit notable gaps in multi-step factual reasoning, especially when
real-world knowledge is sparse. Recent advances in grokking have demonstrated
that neural networks can transition from memorizing to perfectly generalizing
once they detect underlying logical patterns …

[2505.17225] Reasoning Model is Stubborn: Diagnosing Instruction Overriding in
  Reasoning Models
URL: https://reasoningtrap.github.io/
Summary: Large language models have demonstrated remarkable proficiency in long and
complex reasoning tasks. However, they frequently exhibit a problematic
reliance on familiar reasoning patterns, a phenomenon we term reasoning
rigidity. Despite explicit instructions from users, these models often
override clearly stated conditions and default to habitual r…

[2505.18129] One RL to See Them All: Visual Triple Unified Reinforcement Learning
Summary: Reinforcement learning (RL) has significantly advanced the reasoning
capabilities of vision-language models (VLMs). However, the use of RL beyond
reasoning tasks remains largely unexplored, especially for perceptionintensive
tasks like object detection and grounding. We propose V-Triune, a Visual Triple
Unified Reinforcement Learning system that en…

[2505.11049] GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning
Summary: To enhance the safety of VLMs, this paper introduces a novel reasoning-based
VLM guard model dubbed GuardReasoner-VL. The core idea is to incentivize the
guard model to deliberatively reason before making moderation decisions via
online RL. First, we construct GuardReasoner-VLTrain, a reasoning corpus with
123K samples and 631K reasoning steps, spa…

------------------------------------------------------------------------------------------
Cluster 1 | size=12

[2505.02567] Unified Multimodal Understanding and Generation Models: Advances,
  Challenges, and Opportunities
Summary: Recent years have seen remarkable progress in both multimodal understanding
models and image generation models. Despite their respective successes, these
two domains have evolved independently, leading to distinct architectural
paradigms: While autoregressive-based architectures have dominated multimodal
understanding, diffusion-based models have b…

[2505.09568] BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture,
  Training and Dataset
Summary: Unifying image understanding and generation has gained growing attention in
recent research on multimodal models. Although design choices for image
understanding have been extensively studied, the optimal model architecture and
training recipe for a unified framework with image generation remain
underexplored. Motivated by the strong potential of a…

[2505.15809] MMaDA: Multimodal Large Diffusion Language Models
URL: https://huggingface.co/spaces/Gen-Verse/MMaDA
Summary: We introduce MMaDA, a novel class of multimodal diffusion foundation models
designed to achieve superior performance across diverse domains such as textual
reasoning, multimodal understanding, and text-to-image generation. The approach
is distinguished by three key innovations: (i) MMaDA adopts a unified diffusion
architecture with a shared probabi…

[2505.14683] Emerging Properties in Unified Multimodal Pretraining
URL: https://bagel-ai.org/
Summary: Unifying multimodal understanding and generation has shown impressive
capabilities in cutting-edge proprietary systems. In this work, we introduce
BAGEL, an open0source foundational model that natively supports multimodal
understanding and generation. BAGEL is a unified, decoder0only model pretrained
on trillions of tokens curated from large0scale …

[2505.19297] Alchemist: Turning Public Text-to-Image Data into Generative Gold
URL: https://huggingface.co/datasets/yandex/alchemist
Summary: Pre-training equips text-to-image (T2I) models with broad world knowledge,
but this alone is often insufficient to achieve high aesthetic quality and
alignment. Consequently, supervised fine-tuning (SFT) is crucial for further
refinement. However, its effectiveness highly depends on the quality of the
fine-tuning dataset. Existing public SFT datase…

[2505.07062] Seed1.5-VL Technical Report
URL: https://seed.bytedance.com/en/tech/seed1_5_vl
Summary: We present Seed1.5-VL, a vision-language foundation model designed to advance
general-purpose multimodal understanding and reasoning. Seed1.5-VL is composed
with a 532M-parameter vision encoder and a Mixture-of-Experts (MoE) LLM of 20B
active parameters. Despite its relatively compact architecture, it delivers
strong performance across a wide spect…

[2505.05470] Flow-GRPO: Training Flow Matching Models via Online RL
Summary: We propose Flow-GRPO, the first method integrating online reinforcement
learning (RL) into flow matching models. Our approach uses two key strategies:
(1) an ODE-to-SDE conversion that transforms a deterministic Ordinary
Differential Equation (ODE) into an equivalent Stochastic Differential Equation
(SDE) that matches the original model's marginal …

[2505.23747] Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial
  Intelligence
URL: https://diankun-wu.github.io/Spatial-MLLM/
Summary: Recent advancements in Multimodal Large Language Models (MLLMs) have
significantly enhanced performance on 2D visual tasks. However, improving their
spatial intelligence remains a challenge. Existing 3D MLLMs always rely on
additional 3D or 2.5D data to incorporate spatial awareness, restricting their
utility in scenarios with only 2D inputs, such …

[2505.21497] Paper2Poster: Towards Multimodal Poster Automation from Scientific
  Papers
URL: https://paper2poster.github.io/
Summary: Academic poster generation is a crucial yet challenging task in scientific
communication, requiring the compression of long-context interleaved documents
into a single, visually coherent page. To address this challenge, we introduce
the first benchmark and metric suite for poster generation, which pairs recent
conference papers with author-designed…

[2505.07747] Step1X-3D: Towards High-Fidelity and Controllable Generation of Textured
  3D Assets
URL: https://stepfun-ai.github.io/Step1X-3D/
Summary: While generative artificial intelligence has advanced significantly across
text, image, audio, and video domains, 3D generation remains comparatively
underdeveloped due to fundamental challenges such as data scarcity, algorithmic
limitations, and ecosystem fragmentation. To this end, we present Step1X-3D, an
open framework addressing these challeng…

[2505.18445] OmniConsistency: Learning Style-Agnostic Consistency from Paired
  Stylization Data
Summary: Diffusion models have advanced image stylization significantly, yet two core
challenges persist: (1) maintaining consistent stylization in complex scenes,
particularly identity, composition, and fine details, and (2) preventing style
degradation in image-to-image pipelines with style LoRAs. GPT-4o's exceptional
stylization consistency highlights th…

[2505.18125] TabSTAR: A Foundation Tabular Model With Semantically Target-Aware
  Representations
URL: https://eilamshapira.com/TabSTAR
Summary: While deep learning has achieved remarkable success across many domains, it
has historically underperformed on tabular learning tasks, which remain
dominated by gradient boosting decision trees (GBDTs). However, recent
advancements are paving the way for Tabular Foundation Models, which can
leverage real-world knowledge and generalize across divers…

------------------------------------------------------------------------------------------
Cluster 4 | size=12

[2505.04620] On Path to Multimodal Generalist: General-Level and General-Bench
URL: https://generalist.top/
Summary: The Multimodal Large Language Model (MLLM) is currently experiencing rapid
growth, driven by the advanced capabilities of LLMs. Unlike earlier
specialists, existing MLLMs are evolving towards a Multimodal Generalist
paradigm. Initially limited to understanding multiple modalities, these models
have advanced to not only comprehend but also generate …

[2505.21600] R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large
  Model Token Routing
URL: https://fuvty.github.io/R2R_Project_Page/
Summary: Large Language Models (LLMs) achieve impressive reasoning capabilities at the
cost of substantial inference overhead, posing substantial deployment
challenges. Although distilled Small Language Models (SLMs) significantly
enhance efficiency, their performance suffers as they fail to follow LLMs'
reasoning paths. Luckily, we reveal that only a small…

[2505.21189] Exploring the Latent Capacity of LLMs for One-Step Text Generation
Summary: A recent study showed that large language models (LLMs) can reconstruct
surprisingly long texts - up to thousands of tokens - via autoregressive
generation from just one specially trained input embedding. In this work, we
explore whether such reconstruction is possible without autoregression. We show
that frozen LLMs can generate hundreds of accura…

[2505.11820] Chain-of-Model Learning for Language Model
Summary: In this paper, we propose a novel learning paradigm, termed Chain-of-Model
(CoM), which incorporates the causal relationship into the hidden states of
each layer as a chain style, thereby introducing great scaling efficiency in
model training and inference flexibility in deployment. We introduce the
concept of Chain-of-Representation (CoR), which f…

[2505.09388] Qwen3 Technical Report
URL: https://qwenlm.github.io/blog/qwen3/
Summary: In this work, we present Qwen3, the latest version of the Qwen model family.
Qwen3 comprises a series of large language models (LLMs) designed to advance
performance, efficiency, and multilingual capabilities. The Qwen3 series
includes models of both dense and Mixture-of-Expert (MoE) architectures, with
parameter scales ranging from 0.6 to 235 bill…

[2505.04588] ZeroSearch: Incentivize the Search Capability of LLMs without Searching
URL: https://alibaba-nlp.github.io/ZeroSearch/
Summary: Effective information searching is essential for enhancing the reasoning and
generation capabilities of large language models (LLMs). Recent research has
explored using reinforcement learning (RL) to improve LLMs' search capabilities
by interacting with live search engines in real-world environments. While these
approaches show promising results, t…

[2505.17894] Mutarjim: Advancing Bidirectional Arabic-English Translation with a
  Small Language Model
Summary: We introduce Mutarjim, a compact yet powerful language model for
bidirectional Arabic-English translation. While large-scale LLMs have shown
impressive progress in natural language processing tasks, including machine
translation, smaller models. Leveraging this insight, we developed Mutarjim
based on Kuwain-1.5B , a language model tailored for both…

[2505.14302] Scaling Law for Quantization-Aware Training
Summary: Large language models (LLMs) demand substantial computational and memory
resources, creating deployment challenges. Quantization-aware training (QAT)
addresses these challenges by reducing model precision while maintaining
performance. However, the scaling behavior of QAT, especially at 4-bit
precision (W4A4), is not well understood. Existing QAT s…

[2505.09666] System Prompt Optimization with Meta-Learning
Summary: Large Language Models (LLMs) have shown remarkable capabilities, with
optimizing their input prompts playing a pivotal role in maximizing their
performance. However, while LLM prompts consist of both the task-agnostic
system prompts and task-specific user prompts, existing work on prompt
optimization has focused on user prompts specific to individu…

[2505.19457] BizFinBench: A Business-Driven Real-World Financial Benchmark for
  Evaluating LLMs
URL: https://hithink-research.github.io/BizFinBench/
Summary: Large language models excel in general tasks, yet assessing their reliability
in logic-heavy, precision-critical domains like finance, law, and healthcare
remains challenging. To address this, we introduce BizFinBench, the first
benchmark specifically designed to evaluate LLMs in real-world financial
applications. BizFinBench consists of 6,781 well…

[2505.02707] Voila: Voice-Language Foundation Models for Real-Time Autonomous
  Interaction and Voice Role-Play
URL: https://voila.maitrix.org
Summary: A voice AI agent that blends seamlessly into daily life would interact with
humans in an autonomous, real-time, and emotionally expressive manner. Rather
than merely reacting to commands, it would continuously listen, reason, and
respond proactively, fostering fluid, dynamic, and emotionally resonant
interactions. We introduce Voila, a family of la…

[2505.07916] MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable
  Speaker Encoder
URL: https://minimax-ai.github.io/tts_tech_report/
Summary: We introduce MiniMax-Speech, an autoregressive Transformer-based
Text-to-Speech (TTS) model that generates high-quality speech. A key innovation
is our learnable speaker encoder, which extracts timbre features from a
reference audio without requiring its transcription. This enables
MiniMax-Speech to produce highly expressive speech with timbre cons…

------------------------------------------------------------------------------------------
Cluster 3 | size=5

[2505.14669] Quartet: Native FP4 Training Can Be Optimal for Large Language Models
Summary: The rapid advancement of large language models (LLMs) has been paralleled by
unprecedented increases in computational demands, with training costs for
state-of-the-art models doubling every few months. Training models directly in
low-precision arithmetic offers a solution, by improving both computational
throughput and energy efficiency. Specifical…

[2505.09343] Insights into DeepSeek-V3: Scaling Challenges and Reflections on
  Hardware for AI Architectures
Summary: The rapid scaling of large language models (LLMs) has unveiled critical
limitations in current hardware architectures, including constraints in memory
capacity, computational efficiency, and interconnection bandwidth. DeepSeek-V3,
trained on 2,048 NVIDIA H800 GPUs, demonstrates how hardware-aware model
co-design can effectively address these challe…

[2505.19147] Shifting AI Efficiency From Model-Centric to Data-Centric Compression
URL: https://github.com/xuyang-liu16/Awesome-Token-level-Model-Compression
Summary: The rapid advancement of large language models (LLMs) and multi-modal LLMs
(MLLMs) has historically relied on model-centric scaling through increasing
parameter counts from millions to hundreds of billions to drive performance
gains. However, as we approach hardware limits on model size, the dominant
computational bottleneck has fundamentally shift…

[2505.11594] SageAttention3: Microscaling FP4 Attention for Inference and An
  Exploration of 8-Bit Training
URL: https://github.com/thu-ml/SageAttention
Summary: The efficiency of attention is important due to its quadratic time
complexity. We enhance the efficiency of attention through two key
contributions: First, we leverage the new FP4 Tensor Cores in Blackwell GPUs to
accelerate attention computation. Our implementation achieves 1038 TOPS on
RTX5090, which is a 5x speedup over the fastest FlashAttentio…

[2505.10475] Parallel Scaling Law for Language Models
Summary: It is commonly believed that scaling language models should commit a
significant space or time cost, by increasing the parameters (parameter
scaling) or output tokens (inference-time scaling). We introduce the third and
more inference-efficient scaling paradigm: increasing the model's parallel
computation during both training and inference time. We…

------------------------------------------------------------------------------------------
Cluster 0 | size=4

[2505.17612] Distilling LLM Agent into Small Models with Retrieval and Code Tools
Summary: Large language models (LLMs) excel at complex reasoning tasks but remain
computationally expensive, limiting their practical deployment. To address
this, recent works have focused on distilling reasoning capabilities into
smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher
LLMs. However, this approach struggles in scenar…

[2505.20411] SWE-rebench: An Automated Pipeline for Task Collection and
  Decontaminated Evaluation of Software Engineering Agents
Summary: LLM-based agents have shown promising capabilities in a growing range of
software engineering (SWE) tasks. However, advancing this field faces two
critical challenges. First, high-quality training data is scarce, especially
data that reflects real-world SWE scenarios, where agents must interact with
development environments, execute code and adapt …

[2505.15277] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents
Summary: Web navigation is a unique domain that can automate many repetitive real-life
tasks and is challenging as it requires long-horizon sequential decision making
beyond typical multimodal large language model (MLLM) tasks. Yet, specialized
reward models for web navigation that can be utilized during both training and
test-time have been absent until no…

[2505.16938] NovelSeek: When Agent Becomes the Scientist -- Building Closed-Loop
  System from Hypothesis to Verification
Summary: Artificial Intelligence (AI) is accelerating the transformation of scientific
research paradigms, not only enhancing research efficiency but also driving
innovation. We introduce NovelSeek, a unified closed-loop multi-agent framework
to conduct Autonomous Scientific Research (ASR) across various scientific
research fields, enabling researchers to t…

==========================================================================================
# month=2025-06 BEST CLUSTERING (mode=C, k=5)

------------------------------------------------------------------------------------------
Cluster 3 | size=18

[2506.07900] MiniCPM4: Ultra-Efficient LLMs on End Devices
URL: https://huggingface.co/collections/openbmb/minicpm4-6841ab29d180257e940baa9b
Summary: This paper introduces MiniCPM4, a highly efficient large language model (LLM)
designed explicitly for end-side devices. We achieve this efficiency through
systematic innovation in four key dimensions: model architecture, training
data, training algorithms, and inference systems. Specifically, in terms of
model architecture, we propose InfLLM v2, a …

[2506.18841] LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement
  Learning
URL: https://huggingface.co/THU-KEG/
Summary: Ultra-long generation by large language models (LLMs) is a widely demanded
scenario, yet it remains a significant challenge due to their maximum
generation length limit and overall quality degradation as sequence length
increases. Previous approaches, exemplified by LongWriter, typically rely on
''teaching'', which involves supervised fine-tuning (…

[2506.05209] The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly
  Licensed Text
URL: https://huggingface.co/common-pile
Summary: Large language models (LLMs) are typically trained on enormous quantities of
unlicensed text, a practice that has led to scrutiny due to possible
intellectual property infringement and ethical concerns. Training LLMs on
openly licensed text presents a first step towards addressing these issues, but
prior data collection efforts have yielded dataset…

[2506.16406] Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights
URL: https://jerryliang24.github.io/DnD/
Summary: Modern Parameter-Efficient Fine-Tuning (PEFT) methods such as low-rank
adaptation (LoRA) reduce the cost of customizing large language models (LLMs),
yet still require a separate optimization run for every downstream dataset. We
introduce Drag-and-Drop LLMs (\textit{DnD)}, a prompt-conditioned
parameter generator that eliminates per-task training b…

[2506.09991] Multiverse: Your Language Models Secretly Decide How to Parallelize and
  Merge Generation
URL: https://multiverse4fm.github.io/
Summary: Autoregressive Large Language Models (AR-LLMs) frequently exhibit implicit
parallelism in sequential generation. Inspired by this, we introduce
Multiverse, a new generative model that enables natively parallel generation.
Multiverse internalizes a MapReduce paradigm, generating automatically through
three stages: (i) a Map stage for adaptive task d…

[2506.20920] FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data
  Processing to Every Language
URL: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
Summary: Pre-training state-of-the-art large language models (LLMs) requires vast
amounts of clean and diverse text data. While the open development of large
high-quality English pre-training datasets has seen substantial recent
progress, training performant multilingual LLMs remains a challenge, in large
part due to the inherent difficulty of tailoring fil…

[2506.12928] Scaling Test-time Compute for LLM Agents
Summary: Scaling test time compute has shown remarkable success in improving the
reasoning abilities of large language models (LLMs). In this work, we conduct
the first systematic exploration of applying test-time scaling methods to
language agents and investigate the extent to which it improves their
effectiveness. Specifically, we explore different test-t…

[2506.14028] MultiFinBen: A Multilingual, Multimodal, and Difficulty-Aware Benchmark
  for Financial LLM Evaluation
Summary: Recent advances in large language models (LLMs) have accelerated progress in
financial NLP and applications, yet existing benchmarks remain limited to
monolingual and unimodal settings, often over-relying on simple tasks and
failing to reflect the complexity of real-world financial communication. We
introduce MultiFinBen, the first multilingual and…

[2506.13585] MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning
  Attention
URL: https://huggingface.co/MiniMaxAI/MiniMax-M1-80k
Summary: We introduce MiniMax-M1, the world's first open-weight, large-scale
hybrid-attention reasoning model. MiniMax-M1 is powered by a hybrid
Mixture-of-Experts (MoE) architecture combined with a lightning attention
mechanism. The model is developed based on our previous MiniMax-Text-01 model,
which contains a total of 456 billion parameters with 45.9 bi…

[2506.06444] Saffron-1: Towards an Inference Scaling Paradigm for LLM Safety
  Assurance
URL: https://q-rz.github.io/p/saffron
Summary: Existing safety assurance research has primarily focused on training-phase
alignment to instill safe behaviors into LLMs. However, recent studies have
exposed these methods' susceptibility to diverse jailbreak attacks.
Concurrently, inference scaling has significantly advanced LLM reasoning
capabilities but remains unexplored in the context of safe…

[2506.07044] Lingshu: A Generalist Foundation Model for Unified Multimodal Medical
  Understanding and Reasoning
URL: https://alibaba-damo-academy.github.io/lingshu/
Summary: Multimodal Large Language Models (MLLMs) have demonstrated impressive
capabilities in understanding common visual elements, largely due to their
large-scale datasets and advanced training strategies. However, their
effectiveness in medical applications remains limited due to the inherent
discrepancies between data and tasks in medical scenarios and…

[2506.05176] Qwen3 Embedding: Advancing Text Embedding and Reranking Through
  Foundation Models
URL: https://qwenlm.github.io/blog/qwen3-embedding/
Summary: In this work, we introduce the Qwen3 Embedding series, a significant
advancement over its predecessor, the GTE-Qwen series, in text embedding and
reranking capabilities, built upon the Qwen3 foundation models. Leveraging the
Qwen3 LLMs' robust capabilities in multilingual text understanding and
generation, our innovative multi-stage training pipeli…

[2505.21115] Will It Still Be True Tomorrow? Multilingual Evergreen Question
  Classification to Improve Trustworthy QA
URL: https://s-nlp.github.io/Evergreen-classification/
Summary: Large Language Models (LLMs) often hallucinate in question answering (QA)
tasks. A key yet underexplored factor contributing to this is the temporality
of questions -- whether they are evergreen (answers remain stable over time) or
mutable (answers change). In this work, we introduce EverGreenQA, the first
multilingual QA dataset with evergreen lab…

[2505.24863] AlphaOne: Reasoning Models Thinking Slow and Fast at Test Time
URL: https://alphaone-project.github.io/
Summary: This paper presents AlphaOne (alpha1), a universal framework for
modulating reasoning progress in large reasoning models (LRMs) at test time.
alpha1 first introduces alpha moment, which represents the scaled
thinking phase with a universal parameter alpha. Within this scaled
pre-alpha moment phase, it dynamically schedules slow thinking transitions…

[2506.11930] Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback
Summary: Recent studies have shown LLMs possess some ability to improve their
responses when given external feedback. However, it remains unclear how
effectively and thoroughly these models can incorporate extrinsic feedback. In
an ideal scenario, if LLMs receive near-perfect and complete feedback, we would
expect them to fully integrate the feedback and ch…

[2506.09513] ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical
  Reasoning
Summary: Though reasoning-based large language models (LLMs) have excelled in
mathematics and programming, their capabilities in knowledge-intensive medical
question answering remain underexplored. To address this, we introduce
ReasonMed, the largest medical reasoning dataset, comprising 370k high-quality
examples distilled from 1.7 million initial reasonin…

[2506.16035] Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal
  Document Understanding
Summary: Retrieval-Augmented Generation (RAG) systems have revolutionized information
retrieval and question answering, but traditional text-based chunking methods
struggle with complex document structures, multi-page tables, embedded figures,
and contextual dependencies across page boundaries. We present a novel
multimodal document chunking approach that l…

[2506.12285] CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction
  Following
Summary: Recent advances in audio-text large language models (LLMs) have opened new
possibilities for music understanding and generation. However, existing
benchmarks are limited in scope, often relying on simplified tasks or
multi-choice evaluations that fail to reflect the complexity of real-world
music analysis. We reinterpret a broad range of traditiona…

------------------------------------------------------------------------------------------
Cluster 1 | size=14

[2506.09113] Seedance 1.0: Exploring the Boundaries of Video Generation Models
URL: https://seed.bytedance.com/seedance
Summary: Notable breakthroughs in diffusion modeling have propelled rapid improvements
in video generation, yet current foundational model still face critical
challenges in simultaneously balancing prompt following, motion plausibility,
and visual quality. In this report, we introduce Seedance 1.0, a
high-performance and inference-efficient video foundation…

[2506.18095] ShareGPT-4o-Image: Aligning Multimodal Models with GPT-4o-Level Image
  Generation
Summary: Recent advances in multimodal generative models have unlocked photorealistic,
instruction-aligned image generation, yet leading systems like GPT-4o-Image
remain proprietary and inaccessible. To democratize these capabilities, we
present ShareGPT-4o-Image, the first dataset comprising 45K text-to-image and
46K text-and-image-to-image data, all synth…

[2506.03147] UniWorld: High-Resolution Semantic Encoders for Unified Visual
  Understanding and Generation
Summary: Although existing unified models deliver strong performance on
vision-language understanding and text-to-image generation, their models are
limited in exploring image perception and manipulation tasks, which are
urgently desired by users for wide applications. Recently, OpenAI released
their powerful GPT-4o-Image model for comprehensive image perce…

[2506.05573] PartCrafter: Structured 3D Mesh Generation via Compositional Latent
  Diffusion Transformers
URL: https://wgsxm.github.io/projects/partcrafter
Summary: We introduce PartCrafter, the first structured 3D generative model that
jointly synthesizes multiple semantically meaningful and geometrically distinct
3D meshes from a single RGB image. Unlike existing methods that either produce
monolithic 3D shapes or follow two-stage pipelines, i.e., first segmenting an
image and then reconstructing each segmen…

[2506.17450] BlenderFusion: 3D-Grounded Visual Editing and Generative Compositing
URL: https://blenderfusion.github.io/
Summary: We present BlenderFusion, a generative visual compositing framework that
synthesizes new scenes by recomposing objects, camera, and background. It
follows a layering-editing-compositing pipeline: (i) segmenting and converting
visual inputs into editable 3D entities (layering), (ii) editing them in
Blender with 3D-grounded control (editing), and (ii…

[2506.16054] PAROAttention: Pattern-Aware ReOrdering for Efficient Sparse and
  Quantized Attention in Visual Generation Models
URL: https://a-suozhang.xyz/paroattn.github.io/
Summary: In visual generation, the quadratic complexity of attention mechanisms
results in high memory and computational costs, especially for longer token
sequences required in high-resolution image or multi-frame video generation. To
address this, prior research has explored techniques such as sparsification and
quantization. However, these techniques fac…

[2506.05284] Video World Models with Long-term Spatial Memory
URL: https://spmem.github.io/
Summary: Emerging world models autoregressively generate video frames in response to
actions, such as camera movements and text prompts, among other control
signals. Due to limited temporal context window sizes, these models often
struggle to maintain scene consistency during revisits, leading to severe
forgetting of previously generated environments. Inspi…

[2506.17201] Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with
  Hybrid History Condition
URL: https://hunyuan-gamecraft.github.io/
Summary: Recent advances in diffusion-based and controllable video generation have
enabled high-quality and temporally coherent video synthesis, laying the
groundwork for immersive interactive gaming experiences. However, current
methods face limitations in dynamics, generality, long-term consistency, and
efficiency, which limit the ability to create variou…

[2506.18701] Matrix-Game: Interactive World Foundation Model
URL: https://matrix-game-homepage.github.io
Summary: We introduce Matrix-Game, an interactive world foundation model for
controllable game world generation. Matrix-Game is trained using a two-stage
pipeline that first performs large-scale unlabeled pretraining for environment
understanding, followed by action-labeled training for interactive video
generation. To support this, we curate Matrix-Game-MC…

[2506.18871] OmniGen2: Exploration to Advanced Multimodal Generation
URL: https://vectorspacelab.github.io/OmniGen2/
Summary: In this work, we introduce OmniGen2, a versatile and open-source generative
model designed to provide a unified solution for diverse generation tasks,
including text-to-image, image editing, and in-context generation. Unlike
OmniGen v1, OmniGen2 features two distinct decoding pathways for text and image
modalities, utilizing unshared parameters and…

[2506.15675] Sekai: A Video Dataset towards World Exploration
URL: https://lixsp11.github.io/sekai-project/
Summary: Video generation techniques have made remarkable progress, promising to be
the foundation of interactive world exploration. However, existing video
generation datasets are not well-suited for world exploration training as they
suffer from some limitations: limited locations, short duration, static scenes,
and a lack of annotations about exploration…

[2506.19851] AnimaX: Animating the Inanimate in 3D with Joint Video-Pose Diffusion
  Models
URL: https://anima-x.github.io/
Summary: We present AnimaX, a feed-forward 3D animation framework that bridges the
motion priors of video diffusion models with the controllable structure of
skeleton-based animation. Traditional motion synthesis methods are either
restricted to fixed skeletal topologies or require costly optimization in
high-dimensional deformation spaces. In contrast, Ani…

[2506.05301] SeedVR2: One-Step Video Restoration via Diffusion Adversarial
  Post-Training
URL: https://iceclear.github.io/projects/seedvr2/
Summary: Recent advances in diffusion-based video restoration (VR) demonstrate
significant improvement in visual quality, yet yield a prohibitive
computational cost during inference. While several distillation-based
approaches have exhibited the potential of one-step image restoration,
extending existing approaches to VR remains challenging and underexplore…

[2506.18882] Light of Normals: Unified Feature Representation for Universal
  Photometric Stereo
Summary: Universal photometric stereo (PS) aims to recover high-quality surface
normals from objects under arbitrary lighting conditions without relying on
specific illumination models. Despite recent advances such as SDM-UniPS and Uni
MS-PS, two fundamental challenges persist: 1) the deep coupling between varying
illumination and surface normal features, w…

------------------------------------------------------------------------------------------
Cluster 2 | size=8

[2506.01939] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
  Reinforcement Learning for LLM Reasoning
URL: https://shenzhi-wang.github.io/high-entropy-minority-tokens-rlvr/
Summary: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a
powerful approach to enhancing the reasoning capabilities of Large Language
Models (LLMs), while its mechanisms are not yet well understood. In this work,
we undertake a pioneering exploration of RLVR through the novel perspective of
token entropy patterns, comprehensively analy…

[2505.24864] ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in
  Large Language Models
URL: https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B
Summary: Recent advances in reasoning-centric language models have highlighted
reinforcement learning (RL) as a promising method for aligning models with
verifiable rewards. However, it remains contentious whether RL truly expands a
model's reasoning capabilities or merely amplifies high-reward outputs already
latent in the base model's distribution, and wh…

[2506.08007] Reinforcement Pre-Training
Summary: In this work, we introduce Reinforcement Pre-Training (RPT) as a new scaling
paradigm for large language models and reinforcement learning (RL).
Specifically, we reframe next-token prediction as a reasoning task trained
using RL, where it receives verifiable rewards for correctly predicting the
next token for a given context. RPT offers a scalable …

[2506.10910] Magistral
Summary: We introduce Magistral, Mistral's first reasoning model and our own scalable
reinforcement learning (RL) pipeline. Instead of relying on existing
implementations and RL traces distilled from prior models, we follow a ground
up approach, relying solely on our own models and infrastructure. Notably, we
demonstrate a stack that enabled us to explore t…

[2505.24726] Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning
URL: https://writer.com/research/
Summary: We explore a method for improving the performance of large language models
through self-reflection and reinforcement learning. By incentivizing the model
to generate better self-reflections when it answers incorrectly, we demonstrate
that a model's ability to solve complex, verifiable tasks can be enhanced even
when generating synthetic data is inf…

[2506.06395] Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models
Summary: Large language models (LLMs) excel at reasoning, yet post-training remains
critical for aligning their behavior with task goals. Existing reinforcement
learning (RL) methods often depend on costly human annotations or external
reward models. We propose Reinforcement Learning via Self-Confidence (RLSC),
which uses the model's own confidence as rewar…

[2505.24760] REASONING GYM: Reasoning Environments for Reinforcement Learning with
  Verifiable Rewards
Summary: We introduce Reasoning Gym (RG), a library of reasoning environments for
reinforcement learning with verifiable rewards. It provides over 100 data
generators and verifiers spanning multiple domains including algebra,
arithmetic, computation, cognition, geometry, graph theory, logic, and various
common games. Its key innovation is the ability to gen…

[2506.08343] Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves
  Reasoning Efficiency
Summary: Recent advances in large reasoning models have enabled complex, step-by-step
reasoning but often introduce significant overthinking, resulting in verbose
and redundant outputs that hinder efficiency. In this study, we examine whether
explicit self-reflection, signaled by tokens such as "Wait" and "Hmm", is
necessary for advanced reasoning. We propo…

------------------------------------------------------------------------------------------
Cluster 4 | size=7

[2506.02387] VS-Bench: Evaluating VLMs for Strategic Reasoning and Decision-Making in
  Multi-Agent Environments
URL: https://vs-bench.github.io
Summary: Recent advancements in Vision Language Models (VLMs) have expanded their
capabilities to interactive agent tasks, yet existing benchmarks remain limited
to single-agent or text-only environments. In contrast, real-world scenarios
often involve multiple agents interacting within rich visual and linguistic
contexts, posing challenges with both multim…

[2506.01844] SmolVLA: A Vision-Language-Action Model for Affordable and Efficient
  Robotics
URL: https://huggingface.co/blog/smolvla
Summary: Vision-language models (VLMs) pretrained on large-scale multimodal datasets
encode rich visual and linguistic knowledge, making them a strong foundation
for robotics. Rather than training robotic policies from scratch, recent
approaches adapt VLMs into vision-language-action (VLA) models that enable
natural language-driven perception and control. H…

[2506.03569] MiMo-VL Technical Report
Summary: We open-source MiMo-VL-7B-SFT and MiMo-VL-7B-RL, two powerful vision-language
models delivering state-of-the-art performance in both general visual
understanding and multimodal reasoning. MiMo-VL-7B-RL outperforms Qwen2.5-VL-7B
on 35 out of 40 evaluated tasks, and scores 59.4 on OlympiadBench, surpassing
models with up to 78B parameters. For GUI gr…

[2506.10521] Scientists' First Exam: Probing Cognitive Abilities of MLLM via
  Perception, Understanding, and Reasoning
URL: https://prismax.opencompass.org.cn/
Summary: Scientific discoveries increasingly rely on complex multimodal reasoning
based on information-intensive scientific data and domain-specific expertise.
Empowered by expert-level scientific benchmarks, scientific Multimodal Large
Language Models (MLLMs) hold the potential to significantly enhance this
discovery process in realistic workflows. However…

[2505.24867] Time Blindness: Why Video-Language Models Can't See What Humans Can?
URL: https://timeblindness.github.io
Summary: Recent advances in vision-language models (VLMs) have made impressive strides
in understanding spatio-temporal relationships in videos. However, when spatial
information is obscured, these models struggle to capture purely temporal
patterns. We introduce SpookyBench, a benchmark where information is
encoded solely in temporal sequences of noise-lik…

[2506.11763] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents
URL: https://deepresearch-bench.github.io
Summary: Deep Research Agents are a prominent category of LLM-based agents. By
autonomously orchestrating multistep web exploration, targeted retrieval, and
higher-order synthesis, they transform vast amounts of online information into
analyst-grade, citation-rich reports--compressing hours of manual desk research
into minutes. However, a comprehensive benc…

[2506.06751] Geopolitical biases in LLMs: what are the "good" and the "bad" countries
  according to contemporary language models
URL: https://airi-institute.github.io/geopolitical_llm_bias
Summary: This paper evaluates geopolitical biases in LLMs with respect to various
countries though an analysis of their interpretation of historical events with
conflicting national perspectives (USA, UK, USSR, and China). We introduce a
novel dataset with neutral event descriptions and contrasting viewpoints from
different countries. Our findings show sign…

------------------------------------------------------------------------------------------
Cluster 0 | size=3

[2506.05010] ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow
  Development
URL: https://x.com/wangly0229/status/1923515826713526583
Summary: We introduce ComfyUI-Copilot, a large language model-powered plugin designed
to enhance the usability and efficiency of ComfyUI, an open-source platform for
AI-driven art creation. Despite its flexibility and user-friendly interface,
ComfyUI can present challenges to newcomers, including limited documentation,
model misconfigurations, and the compl…

[2506.09790] ComfyUI-R1: Exploring Reasoning Models for Workflow Generation
URL: https://github.com/AIDC-AI/ComfyUI-Copilot
Summary: AI-generated content has evolved from monolithic models to modular workflows,
particularly on platforms like ComfyUI, enabling customization in creative
pipelines. However, crafting effective workflows requires great expertise to
orchestrate numerous specialized components, presenting a steep learning curve
for users. To address this challenge, we …

[2506.17612] JarvisArt: Liberating Human Artistic Creativity via an Intelligent Photo
  Retouching Agent
URL: https://jarvisart.vercel.app/
Summary: Photo retouching has become integral to contemporary visual storytelling,
enabling users to capture aesthetics and express creativity. While professional
tools such as Adobe Lightroom offer powerful capabilities, they demand
substantial expertise and manual effort. In contrast, existing AI-based
solutions provide automation but often suffer from li…

==========================================================================================
# month=2025-07 BEST CLUSTERING (mode=C, k=5)

------------------------------------------------------------------------------------------
Cluster 2 | size=15

[2507.01006] GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable
  Reinforcement Learning
Summary: We present GLM-4.1V-Thinking, a vision-language model (VLM) designed to
advance general-purpose multimodal reasoning. In this report, we share our key
findings in the development of the reasoning-centric training framework. We
first develop a capable vision foundation model with significant potential
through large-scale pre-training, which arguably…

[2507.05255] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for
  Visual Reasoning
URL: https://weiyana.github.io/Open-Vision-Reasoner/
Summary: The remarkable reasoning capability of large language models (LLMs) stems
from cognitive behaviors that emerge through reinforcement with verifiable
rewards. This work investigates how to transfer this principle to Multimodal
LLMs (MLLMs) to unlock advanced visual reasoning. We introduce a two-stage
paradigm built on Qwen2.5-VL-7B: a massive lingui…

[2507.07966] Scaling RL to Long Videos
URL: https://github.com/NVlabs/Long-RL
Summary: We introduce a full-stack framework that scales up reasoning in
vision-language models (VLMs) to long videos, leveraging reinforcement
learning. We address the unique challenges of long video reasoning by
integrating three critical components: (1) a large-scale dataset,
LongVideo-Reason, comprising 52K long video QA pairs with high-quality
reasonin…

[2507.06167] Skywork-R1V3 Technical Report
Summary: We introduce Skywork-R1V3, an advanced, open-source vision-language model
(VLM) that pioneers a new approach to visual reasoning. Its key innovation lies
in effectively transferring reasoning skills from text-only Large Language
Models (LLMs) to visual tasks. The strong performance of Skywork-R1V3 primarily
stems from our elaborate post-training RL…

[2507.01949] Kwai Keye-VL Technical Report
URL: https://kwai-keye.github.io/
Summary: While Multimodal Large Language Models (MLLMs) demonstrate remarkable
capabilities on static images, they often fall short in comprehending dynamic,
information-dense short-form videos, a dominant medium in today's digital
landscape. To bridge this gap, we introduce Kwai Keye-VL, an
8-billion-parameter multimodal foundation model engineered for lea…

[2507.13348] VisionThink: Smart and Efficient Vision Language Model via Reinforcement
  Learning
Summary: Recent advancements in vision-language models (VLMs) have improved
performance by increasing the number of visual tokens, which are often
significantly longer than text tokens. However, we observe that most real-world
scenarios do not require such an extensive number of visual tokens. While the
performance drops significantly in a small subset of O…

[2507.14683] MiroMind-M1: An Open-Source Advancement in Mathematical Reasoning via
  Context-Aware Multi-Stage Policy Optimization
Summary: Large language models have recently evolved from fluent text generation to
advanced reasoning across diverse domains, giving rise to reasoning language
models. Among these domains, mathematical reasoning serves as a representative
benchmark as it requires precise multi-step logic and abstract reasoning, which
can be generalized to other tasks. Whil…

[2507.19849] Agentic Reinforced Policy Optimization
URL: https://github.com/dongguanting/ARPO
Summary: Large-scale reinforcement learning with verifiable rewards (RLVR) has
demonstrated its effectiveness in harnessing the potential of large language
models (LLMs) for single-turn reasoning tasks. In realistic reasoning
scenarios, LLMs can often utilize external tools to assist in task-solving
processes. However, current RL algorithms inadequately bal…

[2507.22448] Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency
  and Performance
URL: https://tiiuae.github.io/Falcon-H1/
Summary: In this report, we introduce Falcon-H1, a new series of large language models
(LLMs) featuring hybrid architecture designs optimized for both high
performance and efficiency across diverse use cases. Unlike earlier Falcon
models built solely on Transformer or Mamba architectures, Falcon-H1 adopts a
parallel hybrid approach that combines Transformer…

[2507.14843] The Invisible Leash: Why RLVR May Not Escape Its Origin
Summary: Recent advances in large reasoning models highlight Reinforcement Learning
with Verifiable Rewards (RLVR) as a promising method for enhancing AI's
capabilities, particularly in solving complex logical tasks. However, it
remains unclear whether RLVR truly expands a model's reasoning boundary or
merely amplifies high-reward outputs that the base mode…

[2507.18071] Group Sequence Policy Optimization
Summary: This paper introduces Group Sequence Policy Optimization (GSPO), our stable,
efficient, and performant reinforcement learning algorithm for training large
language models. Unlike previous algorithms that adopt token-level importance
ratios, GSPO defines the importance ratio based on sequence likelihood and
performs sequence-level clipping, rewardin…

[2507.02592] WebSailor: Navigating Super-human Reasoning for Web Agent
URL: https://github.com/Alibaba-NLP/WebAgent
Summary: Transcending human cognitive limitations represents a critical frontier in
LLM training. Proprietary agentic systems like DeepResearch have demonstrated
superhuman capabilities on extremely complex information-seeking benchmarks
such as BrowseComp, a feat previously unattainable. We posit that their success
hinges on a sophisticated reasoning patte…

[2507.15846] GUI-G^2: Gaussian Reward Modeling for GUI Grounding
URL: https://zju-real.github.io/GUI-G2/
Summary: Graphical User Interface (GUI) grounding maps natural language instructions
to precise interface locations for autonomous interaction. Current
reinforcement learning approaches use binary rewards that treat elements as
hit-or-miss targets, creating sparse signals that ignore the continuous nature
of spatial interactions. Motivated by human clicking…

[2507.16632] Step-Audio 2 Technical Report
URL: https://www.stepfun.com/docs/en/step-audio2
Summary: This paper presents Step-Audio~2, an end-to-end multi-modal large language
model designed for industry-strength audio understanding and speech
conversation. By integrating a latent audio encoder and reasoning-centric
reinforcement learning (RL), Step-Audio 2 achieves promising performance in
automatic speech recognition (ASR) and audio understandin…

[2507.06261] Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality,
  Long Context, and Next Generation Agentic Capabilities
Summary: In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and
Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite
models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA
performance on frontier coding and reasoning benchmarks. In addition to its
incredible coding and reasoning skills, Gemini 2.5 Pro …

------------------------------------------------------------------------------------------
Cluster 0 | size=13

[2507.16784] Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning
URL: https://www.subconscious.dev/
Summary: To break the context limits of large language models (LLMs) that bottleneck
reasoning accuracy and efficiency, we propose the Thread Inference Model (TIM),
a family of LLMs trained for recursive and decompositional problem solving, and
TIMRUN, an inference runtime enabling long-horizon structured reasoning beyond
context limits. Together, TIM hoste…

[2507.06203] A Survey on Latent Reasoning
Summary: Large Language Models (LLMs) have demonstrated impressive reasoning
capabilities, especially when guided by explicit chain-of-thought (CoT)
reasoning that verbalizes intermediate steps. While CoT improves both
interpretability and accuracy, its dependence on natural language reasoning
limits the model's expressive bandwidth. Latent reasoning tackle…

[2507.09477] Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning
  Systems in LLMs
Summary: Retrieval-Augmented Generation (RAG) lifts the factuality of Large Language
Models (LLMs) by injecting external knowledge, yet it falls short on problems
that demand multi-step inference; conversely, purely reasoning-oriented
approaches often hallucinate or mis-ground facts. This survey synthesizes both
strands under a unified reasoning-retrieval p…

[2507.13334] A Survey of Context Engineering for Large Language Models
Summary: The performance of Large Language Models (LLMs) is fundamentally determined
by the contextual information provided during inference. This survey introduces
Context Engineering, a formal discipline that transcends simple prompt design
to encompass the systematic optimization of information payloads for LLMs. We
present a comprehensive taxonomy decom…

[2507.00432] Does Math Reasoning Improve General LLM Capabilities? Understanding
  Transferability of LLM Reasoning
Summary: Math reasoning has become the poster child of progress in large language
models (LLMs), with new models rapidly surpassing human-level performance on
benchmarks like MATH and AIME. But as math leaderboards improve week by week,
it is worth asking: do these gains reflect broader problem-solving ability or
just narrow overfitting? To answer this ques…

[2507.10532] Reasoning or Memorization? Unreliable Results of Reinforcement Learning
  Due to Data Contamination
Summary: The reasoning capabilities of large language models (LLMs) have been a
longstanding focus of research. Recent works have further enhanced these
capabilities using reinforcement learning (RL), with many new methods claiming
significant improvements with minimal or no external supervision. Surprisingly,
some studies even suggest that random or incorr…

[2507.16075] Deep Researcher with Test-Time Diffusion
Summary: Deep research agents, powered by Large Language Models (LLMs), are rapidly
advancing; yet, their performance often plateaus when generating complex,
long-form research reports using generic test-time scaling algorithms. Drawing
inspiration from the iterative nature of human research, which involves cycles
of searching, reasoning, and revision, we p…

[2507.03724] MemOS: A Memory OS for AI System
URL: https://memos.openmem.net/
Summary: Large Language Models (LLMs) have become an essential infrastructure for
Artificial General Intelligence (AGI), yet their lack of well-defined memory
management systems hinders the development of long-context reasoning, continual
personalization, and knowledge consistency.Existing models mainly rely on
static parameters and short-lived contextual s…

[2507.20984] SmallThinker: A Family of Efficient Large Language Models Natively
  Trained for Local Deployment
Summary: While frontier large language models (LLMs) continue to push capability
boundaries, their deployment remains confined to GPU-powered cloud
infrastructure. We challenge this paradigm with SmallThinker, a family of LLMs
natively designed - not adapted - for the unique constraints of local devices:
weak computational power, limited memory, and slow st…

[2507.01951] Test-Time Scaling with Reflective Generative Model
Summary: We introduce our first reflective generative model MetaStone-S1, which
obtains OpenAI o3's performance via the self-supervised process reward model
(SPRM). Through sharing the backbone network and using task-specific heads for
next token prediction and process scoring respectively, SPRM successfully
integrates the policy model and process reward mo…

[2507.16812] MegaScience: Pushing the Frontiers of Post-Training Datasets for Science
  Reasoning
URL: https://huggingface.co/MegaScience
Summary: Scientific reasoning is critical for developing AI scientists and supporting
human researchers in advancing the frontiers of natural science discovery.
However, the open-source community has primarily focused on mathematics and
coding while neglecting the scientific domain, largely due to the absence of
open, large-scale, high-quality, verifiable s…

[2507.02092] Energy-Based Transformers are Scalable Learners and Thinkers
URL: https://energy-based-transformers.github.io/
Summary: Inference-time computation techniques, analogous to human System 2 Thinking,
have recently become popular for improving model performances. However, most
existing approaches suffer from several limitations: they are modality-specific
(e.g., working only in text), problem-specific (e.g., verifiable domains like
math and coding), or require additiona…

[2507.21046] A Survey of Self-Evolving Agents: On Path to Artificial Super
  Intelligence
Summary: Large Language Models (LLMs) have demonstrated strong capabilities but remain
fundamentally static, unable to adapt their internal parameters to novel tasks,
evolving knowledge domains, or dynamic interaction contexts. As LLMs are
increasingly deployed in open-ended, interactive environments, this static
nature has become a critical bottleneck, nec…

------------------------------------------------------------------------------------------
Cluster 3 | size=13

[2507.02813] LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with
  TriMap Video Diffusion
URL: https://liuff19.github.io/LangScene-X/
Summary: Recovering 3D structures with open-vocabulary scene understanding from 2D
images is a fundamental but daunting task. Recent developments have achieved
this by performing per-scene optimization with embedded language information.
However, they heavily rely on the calibrated dense-view reconstruction
paradigm, thereby suffering from severe rendering …

[2507.08441] Vision Foundation Models as Effective Visual Tokenizers for
  Autoregressive Image Generation
Summary: Leveraging the powerful representations of pre-trained vision foundation
models -- traditionally used for visual comprehension -- we explore a novel
direction: building an image tokenizer directly atop such models, a largely
underexplored area. Specifically, we employ a frozen vision foundation model as
the encoder of our tokenizer. To enhance its …

[2506.23044] Ovis-U1 Technical Report
Summary: In this report, we introduce Ovis-U1, a 3-billion-parameter unified model
that integrates multimodal understanding, text-to-image generation, and image
editing capabilities. Building on the foundation of the Ovis series, Ovis-U1
incorporates a diffusion-based visual decoder paired with a bidirectional token
refiner, enabling image generation tasks …

[2507.05964] T-LoRA: Single Image Diffusion Model Customization Without Overfitting
Summary: While diffusion model fine-tuning offers a powerful approach for customizing
pre-trained models to generate specific objects, it frequently suffers from
overfitting when training samples are limited, compromising both generalization
capability and output diversity. This paper tackles the challenging yet most
impactful task of adapting a diffusion m…

[2507.06165] OmniPart: Part-Aware 3D Generation with Semantic Decoupling and
  Structural Cohesion
URL: https://omnipart.github.io/
Summary: The creation of 3D assets with explicit, editable part structures is crucial
for advancing interactive applications, yet most generative methods produce
only monolithic shapes, limiting their utility. We introduce OmniPart, a novel
framework for part-aware 3D object generation designed to achieve high semantic
decoupling among components while main…

[2507.14119] NoHumansRequired: Autonomous High-Quality Image Editing Triplet Mining
URL: https://riko0.github.io/No-Humans-Required/
Summary: Recent advances in generative modeling enable image editing assistants that
follow natural language instructions without additional user input. Their
supervised training requires millions of triplets: original image, instruction,
edited image. Yet mining pixel-accurate examples is hard. Each edit must affect
only prompt-specified regions, preserve …

[2507.01945] LongAnimation: Long Animation Generation with Dynamic Global-Local
  Memory
URL: https://cn-makers.github.io/long_animation_web/
Summary: Animation colorization is a crucial part of real animation industry
production. Long animation colorization has high labor costs. Therefore,
automated long animation colorization based on the video generation model has
significant research value. Existing studies are limited to short-term
colorization. These studies adopt a local paradigm, fusing o…

[2507.13546] nablaNABLA: Neighborhood Adaptive Block-Level Attention
Summary: Recent progress in transformer-based architectures has demonstrated
remarkable success in video generation tasks. However, the quadratic complexity
of full attention mechanisms remains a critical bottleneck, particularly for
high-resolution and long-duration video sequences. In this paper, we propose
NABLA, a novel Neighborhood Adaptive Block-Level…

[2507.17744] Yume: An Interactive World Generation Model
URL: https://stdstu12.github.io/YUME-Project/
Summary: Yume aims to use images, text, or videos to create an interactive, realistic,
and dynamic world, which allows exploration and control using peripheral
devices or neural signals. In this report, we present a preview version of
\method, which creates a dynamic world from an input image and allows
exploration of the world using keyboard actions. To ac…

[2507.21493] BANG: Dividing 3D Assets via Generative Exploded Dynamics
URL: https://sites.google.com/view/bang7355608
Summary: 3D creation has always been a unique human strength, driven by our ability to
deconstruct and reassemble objects using our eyes, mind and hand. However,
current 3D design tools struggle to replicate this natural process, requiring
considerable artistic expertise and manual labor. This paper introduces BANG, a
novel generative approach that bridges …

[2507.13347] π^3: Scalable Permutation-Equivariant Visual Geometry Learning
URL: https://yyfz.github.io/pi3/
Summary: We introduce pi^3, a feed-forward neural network that offers a novel
approach to visual geometry reconstruction, breaking the reliance on a
conventional fixed reference view. Previous methods often anchor their
reconstructions to a designated viewpoint, an inductive bias that can lead to
instability and failures if the reference is suboptimal. In c…

[2507.05566] SingLoRA: Low Rank Adaptation Using a Single Matrix
Summary: Low-Rank Adaptation (LoRA) has significantly advanced parameter-efficient
fine-tuning of large pretrained models. LoRA augments the pre-trained weights
of a model by adding the product of two smaller matrices that together form a
low-rank matrix update. Recent research has shown that scale disparities
between these two matrices often cause unstable…

[2507.07105] 4KAgent: Agentic Any Image to 4K Super-Resolution
URL: https://4kagent.github.io/
Summary: We present 4KAgent, a unified agentic super-resolution generalist system
designed to universally upscale any image to 4K resolution (and even higher, if
applied iteratively). Our system can transform images from extremely low
resolutions with severe degradations, for example, highly distorted inputs at
256x256, into crystal-clear, photorealistic 4K…

------------------------------------------------------------------------------------------
Cluster 1 | size=6

[2507.16863] Pixels, Patterns, but No Poetry: To See The World like Humans
URL: https://turingeyetest.github.io/
Summary: Achieving human-like perception and reasoning in Multimodal Large Language
Models (MLLMs) remains a central challenge in artificial intelligence. While
recent research has primarily focused on enhancing reasoning capabilities in
MLLMs, a fundamental question persists: Can Multimodal Large Language Models
truly perceive the world as humans do? This …

[2506.23918] Thinking with Images for Multimodal Reasoning: Foundations, Methods, and
  Future Frontiers
Summary: Recent progress in multimodal reasoning has been significantly advanced by
textual Chain-of-Thought (CoT), a paradigm where models conduct reasoning
within language. This text-centric approach, however, treats vision as a
static, initial context, creating a fundamental "semantic gap" between rich
perceptual data and discrete symbolic thought. Human…

[2507.22827] ScreenCoder: Advancing Visual-to-Code Generation for Front-End
  Automation via Modular Multimodal Agents
URL: https://huggingface.co/spaces/Jimmyzheng-10/ScreenCoder
Summary: Automating the transformation of user interface (UI) designs into front-end
code holds significant promise for accelerating software development and
democratizing design workflows. While recent large language models (LLMs) have
demonstrated progress in text-to-code generation, many existing approaches rely
solely on natural language prompts, limiti…

[2507.07957] MIRIX: Multi-Agent Memory System for LLM-Based Agents
URL: https://mirix.io/
Summary: Although memory capabilities of AI agents are gaining increasing attention,
existing solutions remain fundamentally limited. Most rely on flat, narrowly
scoped memory components, constraining their ability to personalize, abstract,
and reliably recall user-specific information over time. To this end, we
introduce MIRIX, a modular, multi-agent memor…

[2507.21809] HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D
  Worlds from Words or Pixels
URL: https://3d-models.hunyuan.tencent.com/world/
Summary: Creating immersive and playable 3D worlds from texts or images remains a
fundamental challenge in computer vision and graphics. Existing world
generation approaches typically fall into two categories: video-based methods
that offer rich diversity but lack 3D consistency and rendering efficiency, and
3D-based methods that provide geometric consisten…

[2507.08800] NeuralOS: Towards Simulating Operating Systems via Neural Generative
  Models
URL: https://neural-os.com/
Summary: We introduce NeuralOS, a neural framework that simulates graphical user
interfaces (GUIs) of operating systems by directly predicting screen frames in
response to user inputs such as mouse movements, clicks, and keyboard events.
NeuralOS combines a recurrent neural network (RNN), which tracks computer
state, with a diffusion-based neural renderer t…

------------------------------------------------------------------------------------------
Cluster 4 | size=3

[2507.11097] The Devil behind the mask: An emergent safety vulnerability of Diffusion
  LLMs
Summary: Diffusion-based large language models (dLLMs) have recently emerged as a
powerful alternative to autoregressive LLMs, offering faster inference and
greater interactivity via parallel decoding and bidirectional modeling.
However, despite strong performance in code generation and text infilling, we
identify a fundamental safety concern: existing alig…

[2507.00994] Should We Still Pretrain Encoders with Masked Language Modeling?
URL: https://huggingface.co/MLMvsCLM
Summary: Learning high-quality text representations is fundamental to a wide range of
NLP tasks. While encoder pretraining has traditionally relied on Masked
Language Modeling (MLM), recent evidence suggests that decoder models
pretrained with Causal Language Modeling (CLM) can be effectively repurposed as
encoders, often surpassing traditional encoders on …

[2507.10524] Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive
  Token-Level Computation
Summary: Scaling language models unlocks impressive capabilities, but the accompanying
computational and memory demands make both training and deployment expensive.
Existing efficiency efforts typically target either parameter sharing or
adaptive computation, leaving open the question of how to attain both
simultaneously. We introduce Mixture-of-Recursions …

==========================================================================================
# month=2025-08 BEST CLUSTERING (mode=C, k=5)

------------------------------------------------------------------------------------------
Cluster 4 | size=20

[2508.20453] MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World
  Tasks via MCP Servers
URL: https://huggingface.co/spaces/mcpbench/mcp-bench
Summary: We introduce MCP-Bench, a benchmark for evaluating large language models
(LLMs) on realistic, multi-step tasks that demand tool use, cross-tool
coordination, precise parameter control, and planning/reasoning for solving
tasks. Built on the Model Context Protocol (MCP), MCP-Bench connects LLMs to 28
representative live MCP servers spanning 250 tools…

[2508.10874] SSRL: Self-Search Reinforcement Learning
URL: https://huggingface.co/collections/TsinghuaC3I/ssrl-6899957a64d4a31f7f43bc88
Summary: We investigate the potential of large language models (LLMs) to serve as
efficient simulators for agentic search tasks in reinforcement learning (RL),
thereby reducing dependence on costly interactions with external search
engines. To this end, we first quantify the intrinsic search capability of LLMs
via structured prompting and repeated sampling,…

[2508.16153] AgentFly: Fine-tuning LLM Agents without Fine-tuning LLMs
Summary: In this paper, we introduce a novel learning paradigm for adaptive Large
Language Model (LLM) agents that eliminates the need for fine-tuning the
underlying LLMs. Existing approaches are often either rigid, relying on static,
handcrafted reflection workflows, or computationally intensive, requiring
gradient updates of LLM model parameters. In contr…

[2508.13167] Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent
  Distillation and Agentic RL
URL: https://chain-of-agents-afm.github.io/
Summary: Recent advances in large language models (LLMs) and multi-agent systems have
demonstrated remarkable capabilities in complex problem-solving tasks such as
deep research, vibe coding, and mathematical reasoning. However, most existing
multi-agent systems are built upon manual prompt/workflow engineering with
sophisticated agent frameworks, making th…

[2508.11987] FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction
URL: https://futurex-ai.github.io/
Summary: Future prediction is a complex task for LLM agents, requiring a high level of
analytical thinking, information gathering, contextual understanding, and
decision-making under uncertainty. Agents must not only gather and interpret
vast amounts of dynamic information but also integrate diverse data sources,
weigh uncertainties, and adapt predictions b…

[2508.07999] WideSearch: Benchmarking Agentic Broad Info-Seeking
URL: https://widesearch-seed.github.io/
Summary: From professional research to everyday planning, many tasks are bottlenecked
by wide-scale information seeking, which is more repetitive than cognitively
complex. With the rapid development of Large Language Models (LLMs), automated
search agents powered by LLMs offer a promising solution to liberate humans
from this tedious work. However, the capa…

[2508.02694] Efficient Agents: Building Effective Agents While Reducing Cost
Summary: The remarkable capabilities of Large Language Model (LLM)-driven agents have
enabled sophisticated systems to tackle complex, multi-step tasks, but their
escalating costs threaten scalability and accessibility. This work presents the
first systematic study of the efficiency-effectiveness trade-off in modern
agent systems, addressing the critical ne…

[2508.09736] Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with
  Long-Term Memory
URL: https://m3-agent.github.io/
Summary: We introduce M3-Agent, a novel multimodal agent framework equipped with
long-term memory. Like humans, M3-Agent can process real-time visual and
auditory inputs to build and update its long-term memory. Beyond episodic
memory, it also develops semantic memory, enabling it to accumulate world
knowledge over time. Its memory is organized in an entity…

[2508.01191] Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens
Summary: Chain-of-Thought (CoT) prompting has been shown to improve Large Language
Model (LLM) performance on various tasks. With this approach, LLMs appear to
produce human-like reasoning steps before providing answers (a.k.a., CoT
reasoning), which often leads to the perception that they engage in deliberate
inferential processes. However, some initial fi…

[2508.07050] ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability
URL: https://github.com/8421BCD/ReasonRank
Summary: Large Language Model (LLM) based listwise ranking has shown superior
performance in many passage ranking tasks. With the development of Large
Reasoning Models, many studies have demonstrated that step-by-step reasoning
during test-time helps improve listwise ranking performance. However, due to
the scarcity of reasoning-intensive training data, exi…

[2508.05629] On the Generalization of SFT: A Reinforcement Learning Perspective with
  Reward Rectification
Summary: We present a simple yet theoretically motivated improvement to Supervised
Fine-Tuning (SFT) for the Large Language Model (LLM), addressing its limited
generalization compared to reinforcement learning (RL). Through mathematical
analysis, we reveal that standard SFT gradients implicitly encode a problematic
reward structure that may severely restric…

[2508.04026] VeriGUI: Verifiable Long-Chain GUI Dataset
Summary: Recent studies have delved into constructing autonomous agents capable of
performing complex Graphical User Interface (GUI)-based computer tasks, with
the potential to revolutionize human-computer interaction. Despite encouraging
results, existing efforts mainly focus on short-term interactions and rely on
outcome-only verification, thereby limitin…

[2508.10419] ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long
  Narrative Reasoning
Summary: Narrative comprehension on long stories and novels has been a challenging
domain attributed to their intricate plotlines and entangled, often evolving
relations among characters and entities. Given the LLM's diminished reasoning
over extended context and high computational cost, retrieval-based approaches
remain a pivotal role in practice. However,…

[2508.17445] TreePO: Bridging the Gap of Policy Optimization and Efficacy and
  Inference Efficiency with Heuristic Tree-based Modeling
URL: https://m-a-p.ai/TreePO
Summary: Recent advancements in aligning large language models via reinforcement
learning have achieved remarkable gains in solving complex reasoning problems,
but at the cost of expensive on-policy rollouts and limited exploration of
diverse reasoning paths. In this work, we introduce TreePO, involving a
self-guided rollout algorithm that views sequence ge…

[2508.14460] DuPO: Enabling Reliable LLM Self-Verification via Dual Preference
  Optimization
Summary: We present DuPO, a dual learning-based preference optimization framework that
generates annotation-free feedback via a generalized duality. DuPO addresses
two key limitations: Reinforcement Learning with Verifiable Rewards (RLVR)'s
reliance on costly labels and applicability restricted to verifiable tasks, and
traditional dual learning's restrictio…

[2508.07407] A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm
  Bridging Foundation Models and Lifelong Agentic Systems
URL: https://huggingface.co/spaces/X-iZhang/Awesome-Self-Evolving-Agents
Summary: Recent advances in large language models have sparked growing interest in AI
agents capable of solving complex, real-world tasks. However, most existing
agent systems rely on manually crafted configurations that remain static after
deployment, limiting their ability to adapt to dynamic and evolving
environments. To this end, recent research has exp…

[2508.05405] DeepPHY: Benchmarking Agentic VLMs on Physical Reasoning
URL: https://github.com/XinrunXu/DeepPHY
Summary: Although Vision Language Models (VLMs) exhibit strong perceptual abilities
and impressive visual reasoning, they struggle with attention to detail and
precise action planning in complex, dynamic environments, leading to subpar
performance. Real-world tasks typically require complex interactions, advanced
spatial reasoning, long-term planning, and c…

[2508.15260] Deep Think with Confidence
URL: https://jiaweizzhao.github.io/deepconf/
Summary: Large Language Models (LLMs) have shown great potential in reasoning tasks
through test-time scaling methods like self-consistency with majority voting.
However, this approach often leads to diminishing returns in accuracy and high
computational overhead. To address these challenges, we introduce Deep Think
with Confidence (DeepConf), a simple yet …

[2508.09848] PRELUDE: A Benchmark Designed to Require Global Comprehension and
  Reasoning over Long Contexts
URL: https://gorov.github.io/prelude
Summary: We introduce PRELUDE, a benchmark for evaluating long-context understanding
through the task of determining whether a character's prequel story is
consistent with the canonical narrative of the original book. Our task poses a
stronger demand for global comprehension and deep reasoning than existing
benchmarks -- as the prequels are not part of the …

[2508.13491] From Scores to Skills: A Cognitive Diagnosis Framework for Evaluating
  Financial Large Language Models
Summary: Large Language Models (LLMs) have shown promise for financial applications,
yet their suitability for this high-stakes domain remains largely unproven due
to inadequacies in existing benchmarks. Existing benchmarks solely rely on
score-level evaluation, summarizing performance with a single score that
obscures the nuanced understanding of what mode…

------------------------------------------------------------------------------------------
Cluster 1 | size=10

[2508.11737] Ovis2.5 Technical Report
Summary: We present Ovis2.5, a successor to Ovis2 designed for native-resolution
visual perception and strong multimodal reasoning. Ovis2.5 integrates a
native-resolution vision transformer that processes images at their native,
variable resolutions, avoiding the degradation from fixed-resolution tiling and
preserving both fine detail and global layout -- c…

[2508.03320] Skywork UniPic: Unified Autoregressive Modeling for Visual Understanding
  and Generation
Summary: We introduce Skywork UniPic, a 1.5 billion-parameter autoregressive model
that unifies image understanding, text-to-image generation, and image editing
within a single architecture-eliminating the need for task-specific adapters or
inter-module connectors-and demonstrate that compact multimodal systems can
achieve state-of-the-art performance on co…

[2508.11630] Thyme: Think Beyond Images
URL: https://thyme-vl.github.io/
Summary: Following OpenAI's introduction of the ``thinking with images'' concept,
recent efforts have explored stimulating the use of visual information in the
reasoning process to enhance model performance in perception and reasoning
tasks. However, to the best of our knowledge, no open-source work currently
offers a feature set as rich as proprietary mode…

[2508.13154] 4DNeX: Feed-Forward 4D Generative Modeling Made Easy
URL: https://4dnex.github.io/
Summary: We present 4DNeX, the first feed-forward framework for generating 4D (i.e.,
dynamic 3D) scene representations from a single image. In contrast to existing
methods that rely on computationally intensive optimization or require
multi-frame video inputs, 4DNeX enables efficient, end-to-end image-to-4D
generation by fine-tuning a pretrained video diffu…

[2508.14041] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos
URL: https://linjohnss.github.io/longsplat/
Summary: LongSplat addresses critical challenges in novel view synthesis (NVS) from
casually captured long videos characterized by irregular camera motion, unknown
camera poses, and expansive scenes. Current methods often suffer from pose
drift, inaccurate geometry initialization, and severe memory limitations. To
address these issues, we introduce LongSpla…

[2508.10104] DINOv3
URL: https://ai.meta.com/blog/dinov3-self-supervised-vision-model/
Summary: Self-supervised learning holds the promise of eliminating the need for manual
data annotation, enabling models to scale effortlessly to massive datasets and
larger architectures. By not being tailored to specific tasks or domains, this
training paradigm has the potential to learn visual representations from
diverse sources, ranging from natural to …

[2508.20751] Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable
  Text-to-Image Reinforcement Learning
URL: https://codegoat24.github.io/UnifiedReward/Pref-GRPO
Summary: Recent advancements highlight the importance of GRPO-based reinforcement
learning methods and benchmarking in enhancing text-to-image (T2I) generation.
However, current methods using pointwise reward models (RM) for scoring
generated images are susceptible to reward hacking. We reveal that this happens
when minimal score differences between images …

[2508.02324] Qwen-Image Technical Report
Summary: We present Qwen-Image, an image generation foundation model in the Qwen
series that achieves significant advances in complex text rendering and precise
image editing. To address the challenges of complex text rendering, we design a
comprehensive data pipeline that includes large-scale data collection,
filtering, annotation, synthesis, and balancing…

[2508.19652] Self-Rewarding Vision-Language Model via Reasoning Decomposition
Summary: Vision-Language Models (VLMs) often suffer from visual hallucinations, saying
things that are not actually in the image, and language shortcuts, where they
skip the visual part and just rely on text priors. These issues arise because
most post-training methods for VLMs rely on simple verifiable answer matching
and supervise only final outputs, leav…

[2508.14879] MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds
URL: https://daibingquan.github.io/MeshCoder
Summary: Reconstructing 3D objects into editable programs is pivotal for applications
like reverse engineering and shape editing. However, existing methods often
rely on limited domain-specific languages (DSLs) and small-scale datasets,
restricting their ability to model complex geometries and structures. To
address these challenges, we introduce MeshCoder,…

------------------------------------------------------------------------------------------
Cluster 0 | size=9

[2508.05004] R-Zero: Self-Evolving Reasoning LLM from Zero Data
URL: https://chengsong-huang.github.io/R-Zero.github.io/
Summary: Self-evolving Large Language Models (LLMs) offer a scalable path toward
super-intelligence by autonomously generating, refining, and learning from
their own experiences. However, existing methods for training such models still
rely heavily on vast human-curated tasks and labels, typically via fine-tuning
or reinforcement learning, which poses a fun…

[2508.20722] rStar2-Agent: Agentic Reasoning Technical Report
Summary: We introduce rStar2-Agent, a 14B math reasoning model trained with agentic
reinforcement learning to achieve frontier-level performance. Beyond current
long CoT, the model demonstrates advanced cognitive behaviors, such as thinking
carefully before using Python coding tools and reflecting on code execution
feedback to autonomously explore, verify, …

[2508.06471] GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models
Summary: We present GLM-4.5, an open-source Mixture-of-Experts (MoE) large language
model with 355B total parameters and 32B activated parameters, featuring a
hybrid reasoning method that supports both thinking and direct response modes.
Through multi-stage training on 23T tokens and comprehensive post-training with
expert model iteration and reinforcement …

[2508.00414] Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent
  Foundation Models Training
Summary: General AI Agents are increasingly recognized as foundational frameworks for
the next generation of artificial intelligence, enabling complex reasoning, web
interaction, coding, and autonomous research capabilities. However, current
agent systems are either closed-source or heavily reliant on a variety of paid
APIs and proprietary tools, limiting a…

[2508.14029] Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains
  RLVR
URL: https://mastervito.github.io/SvS.github.io/
Summary: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as
a key paradigm for post-training Large Language Models (LLMs), particularly for
complex reasoning tasks. However, vanilla RLVR training has been shown to
improve Pass@1 performance at the expense of policy entropy, leading to reduced
generation diversity and limiting the …

[2508.10433] We-Math 2.0: A Versatile MathBook System for Incentivizing Visual
  Mathematical Reasoning
URL: https://we-math2.github.io/
Summary: Multimodal Large Language Models (MLLMs) have demonstrated impressive
capabilities across various tasks, but still struggle with complex mathematical
reasoning. Existing research primarily focuses on dataset construction and
method optimization, often overlooking two critical aspects: comprehensive
knowledge-driven design and model-centric data spa…

[2508.03680] Agent Lightning: Train ANY AI Agents with Reinforcement Learning
URL: https://www.microsoft.com/en-us/research/project/agent-lightning/
Summary: We present Agent Lightning, a flexible and extensible framework that enables
Reinforcement Learning (RL)-based training of Large Language Models (LLMs) for
any AI agent. Unlike existing methods that tightly couple RL training with
agent or rely on sequence concatenation with masking, Agent Lightning achieves
complete decoupling between agent execut…

[2508.05748] WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent
URL: https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
Summary: Web agents such as Deep Research have demonstrated superhuman cognitive
abilities, capable of solving highly challenging information-seeking problems.
However, most research remains primarily text-centric, overlooking visual
information in the real world. This makes multimodal Deep Research highly
challenging, as such agents require much stronger r…

[2507.23726] Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving
Summary: LLMs have demonstrated strong mathematical reasoning abilities by leveraging
reinforcement learning with long chain-of-thought, yet they continue to
struggle with theorem proving due to the lack of clear supervision signals when
solely using natural language. Dedicated domain-specific languages like Lean
provide clear supervision via formal verific…

------------------------------------------------------------------------------------------
Cluster 3 | size=8

[2508.02193] Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed
  Inference
URL: https://seed.bytedance.com/en/seed_diffusion
Summary: We present Seed Diffusion Preview, a large-scale language model based on
discrete-state diffusion, offering remarkably fast inference speed. Thanks to
non-sequential, parallel generation, discrete diffusion models provide a
notable speedup to mitigate the inherent latency of token-by-token decoding, as
demonstrated recently (e.g., Mercury Coder, Ge…

[2508.00819] Beyond Fixed: Variable-Length Denoising for Diffusion Large Language
  Models
Summary: Diffusion Large Language Models (DLLMs) are emerging as a powerful
alternative to the dominant Autoregressive Large Language Models, offering
efficient parallel generation and capable global context modeling. However, the
practical application of DLLMs is hindered by a critical architectural
constraint: the need for a statically predefined generati…

[2508.10975] BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale
  Pretraining
Summary: Recent advances in large language model (LLM) pretraining have shown that
simply scaling data quantity eventually leads to diminishing returns, hitting a
data wall. In response, the use of synthetic data for pretraining has emerged
as a promising paradigm for pushing the frontier of performance. Despite this,
the factors affecting synthetic data qu…

[2508.10711] NextStep-1: Toward Autoregressive Image Generation with Continuous
  Tokens at Scale
URL: https://stepfun.ai/research/en/nextstep1
Summary: Prevailing autoregressive (AR) models for text-to-image generation either
rely on heavy, computationally-intensive diffusion models to process continuous
image tokens, or employ vector quantization (VQ) to obtain discrete tokens with
quantization loss. In this paper, we push the autoregressive paradigm forward
with NextStep-1, a 14B autoregressive …

[2508.19205] VibeVoice Technical Report
URL: https://microsoft.github.io/VibeVoice/
Summary: This report presents VibeVoice, a novel model designed to synthesize
long-form speech with multiple speakers by employing next-token diffusion,
which is a unified method for modeling continuous data by autoregressively
generating latent vectors via diffusion. To enable this, we introduce a novel
continuous speech tokenizer that, when compared to th…

[2508.01959] SitEmb-v1.5: Improved Context-Aware Dense Retrieval for Semantic
  Association and Long Story Comprehension
URL: https://huggingface.co/SituatedEmbedding
Summary: Retrieval-augmented generation (RAG) over long documents typically involves
splitting the text into smaller chunks, which serve as the basic units for
retrieval. However, due to dependencies across the original document,
contextual information is often essential for accurately interpreting each
chunk. To address this, prior work has explored encodi…

[2508.09983] Story2Board: A Training-Free Approach for Expressive Storyboard
  Generation
URL: https://daviddinkevich.github.io/Story2Board/
Summary: We present Story2Board, a training-free framework for expressive storyboard
generation from natural language. Existing methods narrowly focus on subject
identity, overlooking key aspects of visual storytelling such as spatial
composition, background evolution, and narrative pacing. To address this, we
introduce a lightweight consistency framework c…

[2508.15882] Beyond Transcription: Mechanistic Interpretability in ASR
Summary: Interpretability methods have recently gained significant attention,
particularly in the context of large language models, enabling insights into
linguistic representations, error detection, and model behaviors such as
hallucinations and repetitions. However, these techniques remain underexplored
in automatic speech recognition (ASR), despite their…

------------------------------------------------------------------------------------------
Cluster 2 | size=3

[2508.18265] InternVL3.5: Advancing Open-Source Multimodal Models in Versatility,
  Reasoning, and Efficiency
URL: https://chat.intern-ai.org.cn/
Summary: We introduce InternVL 3.5, a new family of open-source multimodal models that
significantly advances versatility, reasoning capability, and inference
efficiency along the InternVL series. A key innovation is the Cascade
Reinforcement Learning (Cascade RL) framework, which enhances reasoning through
a two-stage process: offline RL for stable converg…

[2508.15763] Intern-S1: A Scientific Multimodal Foundation Model
Summary: In recent years, a plethora of open-source foundation models have emerged,
achieving remarkable progress in some widely attended fields, with performance
being quite close to that of closed-source models. However, in high-value but
more challenging scientific professional fields, either the fields still rely
on expert models, or the progress of gen…

[2508.05635] Genie Envisioner: A Unified World Foundation Platform for Robotic
  Manipulation
URL: https://genie-envisioner.github.io/
Summary: We introduce Genie Envisioner (GE), a unified world foundation platform for
robotic manipulation that integrates policy learning, evaluation, and
simulation within a single video-generative framework. At its core, GE-Base is
a large-scale, instruction-conditioned video diffusion model that captures the
spatial, temporal, and semantic dynamics of re…

==========================================================================================
# month=2025-09 BEST CLUSTERING (mode=B, k=5)

------------------------------------------------------------------------------------------
Cluster 3 | size=13

[2509.01055] VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use
Summary: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated
success in enhancing LLM reasoning capabilities, but remains limited to
single-turn interactions without tool integration. While recent Agentic
Reinforcement Learning with Tool use (ARLT) approaches have emerged to address
multi-turn tool interactions, existing works develop tas…

[2509.02547] The Landscape of Agentic Reinforcement Learning for LLMs: A Survey
Summary: The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm
shift from conventional reinforcement learning applied to large language models
(LLM RL), reframing LLMs from passive sequence generators into autonomous,
decision-making agents embedded in complex, dynamic worlds. This survey
formalizes this conceptual shift by contrasti…

[2509.02479] SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn
  Tool-Integrated Reasoning
URL: https://simpletir.notion.site/report
Summary: Large Language Models (LLMs) can significantly improve their reasoning
capabilities by interacting with external tools, a paradigm known as
Tool-Integrated Reasoning (TIR). However, extending TIR to multi-turn scenarios
using Reinforcement Learning (RL) is often hindered by training instability and
performance collapse. We identify that such instab…

[2509.21268] MMR1: Enhancing Multimodal Reasoning with Variance-Aware Sampling and
  Open Resources
URL: https://huggingface.co/MMR1
Summary: Large multimodal reasoning models have achieved rapid progress, but their
advancement is constrained by two major limitations: the absence of open,
large-scale, high-quality long chain-of-thought (CoT) data, and the instability
of reinforcement learning (RL) algorithms in post-training. Group Relative
Policy Optimization (GRPO), the standard framew…

[2509.15207] FlowRL: Matching Reward Distributions for LLM Reasoning
Summary: We propose FlowRL: matching the full reward distribution via flow balancing
instead of maximizing rewards in large language model (LLM) reinforcement
learning (RL). Recent advanced reasoning models adopt reward-maximizing methods
(\eg, PPO and GRPO), which tend to over-optimize dominant reward signals while
neglecting less frequent but valid reason…

[2509.21240] Tree Search for LLM Agent Reinforcement Learning
Summary: Recent advances in reinforcement learning (RL) have significantly enhanced
the agentic capabilities of large language models (LLMs). In long-term and
multi-turn agent tasks, existing approaches driven solely by outcome rewards
often suffer from the problem of sparse supervision. To address the challenge,
we propose Tree-based Group Relative Policy …

[2509.22611] Quantile Advantage Estimation for Entropy-Safe Reasoning
Summary: Reinforcement Learning with Verifiable Rewards (RLVR) strengthens LLM
reasoning, but training often oscillates between {entropy collapse} and
{entropy explosion}. We trace both hazards to the mean baseline used in
value-free RL (e.g., GRPO and DAPO), which improperly penalizes
negative-advantage samples under reward outliers. We propose {Quantile
A…

[2509.00676] LLaVA-Critic-R1: Your Critic Model is Secretly a Strong Policy Model
Summary: In vision-language modeling, critic models are typically trained to evaluate
outputs -- assigning scalar scores or pairwise preferences -- rather than to
generate responses. This separation from policy models, which produce the
responses, is so entrenched that critics are rarely considered for direct
policy use. In this work, we challenge this conv…

[2509.19803] VCRL: Variance-based Curriculum Reinforcement Learning for Large
  Language Models
Summary: Policy-based reinforcement learning currently plays an important role in
improving LLMs on mathematical reasoning tasks. However, existing rollout-based
reinforcement learning methods (GRPO, DAPO, GSPO, etc.) fail to explicitly
consider LLMs' learning ability for samples of different difficulty levels,
which is contrary to the human cognitive proce…

[2509.22576] EPO: Entropy-regularized Policy Optimization for LLM Agents
  Reinforcement Learning
Summary: Training LLM agents in multi-turn environments with sparse rewards, where
completing a single task requires 30+ turns of interaction within an episode,
presents a fundamental challenge for reinforcement learning. We identify a
critical failure mode unique to this setting: the exploration-exploitation
cascade failure. This cascade begins with early-…

[2509.02544] UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn
  Reinforcement Learning
URL: https://seed-tars.com/showcase/ui-tars-2/
Summary: The development of autonomous agents for graphical user interfaces (GUIs)
presents major challenges in artificial intelligence. While recent advances in
native agent models have shown promise by unifying perception, reasoning,
action, and memory through end-to-end learning, open problems remain in data
scalability, multi-turn reinforcement learning…

[2509.17567] LIMI: Less is More for Agency
URL: https://github.com/GAIR-NLP/LIMI
Summary: We define Agency as the emergent capacity of AI systems to function as
autonomous agents actively discovering problems, formulating hypotheses, and
executing solutions through self-directed engagement with environments and
tools. This fundamental capability marks the dawn of the Age of AI Agency,
driven by a critical industry shift: the urgent need…

[2508.18106] A.S.E: A Repository-Level Benchmark for Evaluating Security in
  AI-Generated Code
URL: https://aicgseceval.tencent.com/home
Summary: The increasing adoption of large language models (LLMs) in software
engineering necessitates rigorous security evaluation of their generated code.
However, existing benchmarks often lack relevance to real-world AI programming
scenarios, making them inadequate for assessing the practical security risks
associated with AI-generated code in production…

------------------------------------------------------------------------------------------
Cluster 2 | size=12

[2509.09372] VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action
  Model
URL: https://vla-adapter.github.io/
Summary: Vision-Language-Action (VLA) models typically bridge the gap between
perceptual and action spaces by pre-training a large-scale Vision-Language
Model (VLM) on robotic data. While this approach greatly enhances performance,
it also incurs significant training costs. In this paper, we investigate how to
effectively bridge vision-language (VL) represe…

[2509.07979] Visual Representation Alignment for Multimodal Large Language Models
URL: https://cvlab-kaist.github.io/VIRAL/
Summary: Multimodal large language models (MLLMs) trained with visual instruction
tuning have achieved strong performance across diverse tasks, yet they remain
limited in vision-centric tasks such as object counting or spatial reasoning.
We attribute this gap to the prevailing text-only supervision paradigm, which
provides only indirect guidance for the vis…

[2508.21112] EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for
  General Robot Control
URL: https://eo-robotics.ai/eo-1
Summary: The human ability to seamlessly perform multimodal reasoning and physical
interaction in the open world is a core goal for general-purpose embodied
intelligent systems. Recent vision-language-action (VLA) models, which are
co-trained on large-scale robot and visual-text data, have demonstrated notable
progress in general robot control. However, the…

[2509.17765] Qwen3-Omni Technical Report
Summary: We present Qwen3-Omni, a single multimodal model that, for the first time,
maintains state-of-the-art performance across text, image, audio, and video
without any degradation relative to single-modal counterparts. Qwen3-Omni
matches the performance of same-sized single-modal models within the Qwen
series and excels particularly on audio tasks. Acro…

[2509.20328] Video models are zero-shot learners and reasoners
URL: https://video-zero-shot.github.io/
Summary: The remarkable zero-shot capabilities of Large Language Models (LLMs) have
propelled natural language processing from task-specific models to unified,
generalist foundation models. This transformation emerged from simple
primitives: large, generative models trained on web-scale data. Curiously, the
same primitives apply to today's generative video …

[2509.09674] SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning
Summary: Vision-Language-Action (VLA) models have recently emerged as a powerful
paradigm for robotic manipulation. Despite substantial progress enabled by
large-scale pretraining and supervised fine-tuning (SFT), these models face two
fundamental challenges: (i) the scarcity and high cost of large-scale
human-operated robotic trajectories required for SFT …

[2509.14008] Hala Technical Report: Building Arabic-Centric Instruction & Translation
  Models at Scale
Summary: We present Hala, a family of Arabic-centric instruction and translation
models built with our translate-and-tune pipeline. We first compress a strong
ARleftrightarrowEN teacher to FP8 (yielding sim2times higher
throughput with no quality loss) and use it to create high-fidelity bilingual
supervision. A lightweight language model LFM2-1.2B is then f…

[2509.18174] Baseer: A Vision-Language Model for Arabic Document-to-Markdown OCR
URL: https://oinsight.ai/
Summary: Arabic document OCR remains a challenging task due to the language's cursive
script, diverse fonts, diacritics, and right-to-left orientation. While modern
Multimodal Large Language Models (MLLMs) have advanced document understanding
for high-resource languages, their performance on Arabic remains limited. In
this work, we introduce Baseer, a visio…

[2509.22186] MinerU2.5: A Decoupled Vision-Language Model for Efficient
  High-Resolution Document Parsing
URL: https://opendatalab.github.io/MinerU/
Summary: We introduce MinerU2.5, a 1.2B-parameter document parsing vision-language
model that achieves state-of-the-art recognition accuracy while maintaining
exceptional computational efficiency. Our approach employs a coarse-to-fine,
two-stage parsing strategy that decouples global layout analysis from local
content recognition. In the first stage, the mo…

[2509.15221] ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform
  Data
Summary: Vision-Language Models (VLMs) have enabled computer use agents (CUAs) that
operate GUIs autonomously, showing great potential, yet progress is limited by
the lack of large-scale, open-source computer use data and foundation models.
In this work, we introduce ScaleCUA, a step toward scaling open-source CUAs. It
offers a large-scale dataset spanning …

[2509.12201] OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling
URL: https://yangzhou24.github.io/OmniWorld/
Summary: The field of 4D world modeling - aiming to jointly capture spatial geometry
and temporal dynamics - has witnessed remarkable progress in recent years,
driven by advances in large-scale generative models and multimodal learning.
However, the development of truly general 4D world models remains fundamentally
constrained by the availability of high-qu…

[2509.03867] Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth
URL: https://huggingface.co/datasets/extraordinarylab/drivel-hub
Summary: We introduce Drivelology, a unique linguistic phenomenon characterised as
"nonsense with depth", utterances that are syntactically coherent yet
pragmatically paradoxical, emotionally loaded, or rhetorically subversive.
While such expressions may resemble surface-level nonsense, they encode
implicit meaning requiring contextual inference, moral reas…

------------------------------------------------------------------------------------------
Cluster 1 | size=9

[2509.08827] A Survey of Reinforcement Learning for Large Reasoning Models
URL: https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs
Summary: In this paper, we survey recent advances in Reinforcement Learning (RL) for
reasoning with Large Language Models (LLMs). RL has achieved remarkable success
in advancing the frontier of LLM capabilities, particularly in addressing
complex logical tasks such as mathematics and coding. As a result, RL has
emerged as a foundational methodology for tran…

[2509.04419] Towards a Unified View of Large Language Model Post-Training
Summary: Two major sources of training data exist for post-training modern language
models: online (model-generated rollouts) data, and offline (human or
other-model demonstrations) data. These two types of data are typically used by
approaches like Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT),
respectively. In this paper, we show that these…

[2509.07980] Parallel-R1: Towards Parallel Thinking via Reinforcement Learning
Summary: Parallel thinking has emerged as a novel approach for enhancing the reasoning
capabilities of large language models (LLMs) by exploring multiple reasoning
paths concurrently. However, activating such capabilities through training
remains challenging, as existing methods predominantly rely on supervised
fine-tuning (SFT) over synthetic data, which e…

[2508.21113] R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs
  via Bi-Mode Annealing and Reinforce Learning
Summary: Multimodal Large Language Models (MLLMs) equipped with step-by-step thinking
capabilities have demonstrated remarkable performance on complex reasoning
problems. However, this thinking process is redundant for simple problems
solvable without complex reasoning. To address this inefficiency, we propose
R-4B, an auto-thinking MLLM, which can adaptive…

[2509.08721] Sharing is Caring: Efficient LM Post-Training with Collective RL
  Experience Sharing
URL: https://blog.gensyn.ai/sapo-efficient-lm-post-training-with-collective-rl/
Summary: Post-training language models (LMs) with reinforcement learning (RL) can
enhance their complex reasoning capabilities without supervised fine-tuning, as
demonstrated by DeepSeek-R1-Zero. However, effectively utilizing RL for LMs
requires significant parallelization to scale-up inference, which introduces
non-trivial technical challenges (e.g. laten…

[2509.06160] Reverse-Engineered Reasoning for Open-Ended Generation
URL: https://m-a-p.ai/REER_DeepWriter/
Summary: While the ``deep reasoning'' paradigm has spurred significant advances in
verifiable domains like mathematics, its application to open-ended, creative
generation remains a critical challenge. The two dominant methods for
instilling reasoning -- reinforcement learning (RL) and instruction
distillation -- falter in this area; RL struggles with the ab…

[2509.22638] Language Models Can Learn from Verbal Feedback Without Scalar Rewards
Summary: LLMs are often trained with RL from human or AI feedback, yet such methods
typically compress nuanced feedback into scalar rewards, discarding much of
their richness and inducing scale imbalance. We propose treating verbal
feedback as a conditioning signal. Inspired by language priors in text-to-image
generation, which enable novel outputs from uns…

[2509.21320] SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines
Summary: We present a scientific reasoning foundation model that aligns natural
language with heterogeneous scientific representations. The model is pretrained
on a 206B-token corpus spanning scientific text, pure sequences, and
sequence-text pairs, then aligned via SFT on 40M instructions, annealed
cold-start bootstrapping to elicit long-form chain-of-thou…

[2509.04664] Why Language Models Hallucinate
Summary: Like students facing hard exam questions, large language models sometimes
guess when uncertain, producing plausible yet incorrect statements instead of
admitting uncertainty. Such "hallucinations" persist even in state-of-the-art
systems and undermine trust. We argue that language models hallucinate because
the training and evaluation procedures re…

------------------------------------------------------------------------------------------
Cluster 4 | size=9

[2509.13313] ReSum: Unlocking Long-Horizon Search Intelligence via Context
  Summarization
URL: https://tongyi-agent.github.io/blog/
Summary: Large Language Model (LLM)-based web agents demonstrate strong performance on
knowledge-intensive tasks but are hindered by context window limitations in
paradigms like ReAct. Complex queries involving multiple entities, intertwined
relationships, and high uncertainty demand extensive search cycles that rapidly
exhaust context budgets before reachi…

[2509.06501] WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents
Summary: The paradigm of Large Language Models (LLMs) has increasingly shifted toward
agentic applications, where web browsing capabilities are fundamental for
retrieving information from diverse online sources. However, existing
open-source web agents either demonstrate limited information-seeking abilities
on complex tasks or lack transparent implementati…

[2508.21148] A Survey of Scientific Large Language Models: From Data Foundations to
  Agent Frontiers
Summary: Scientific Large Language Models (Sci-LLMs) are transforming how knowledge is
represented, integrated, and applied in scientific research, yet their progress
is shaped by the complex nature of scientific data. This survey presents a
comprehensive, data-centric synthesis that reframes the development of Sci-LLMs
as a co-evolution between models and …

[2509.13305] WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic
  Data and Scalable Reinforcement Learning
URL: https://tongyi-agent.github.io/blog/
Summary: Transcending human cognitive limitations represents a critical frontier in
LLM training. Proprietary agentic systems like DeepResearch have demonstrated
superhuman capabilities on extremely complex information-seeking benchmarks
such as BrowseComp, a feat previously unattainable. We posit that their success
hinges on a sophisticated reasoning patte…

[2509.00375] Open Data Synthesis For Deep Research
Summary: Large language models (LLMs) are increasingly expected to go beyond simple
factual queries toward Deep Research-tasks that require decomposing questions
into sub-problems, coordinating multi-step reasoning, and synthesizing evidence
from diverse sources. We formalize Deep Research tasks with verifiable answers
as Hierarchical Constraint Satisfactio…

[2509.13312] WebWeaver: Structuring Web-Scale Evidence with Dynamic Outlines for
  Open-Ended Deep Research
URL: https://tongyi-agent.github.io/blog/
Summary: This paper tackles open-ended deep research (OEDR), a complex challenge where
AI agents must synthesize vast web-scale information into insightful reports.
Current approaches are plagued by dual-fold limitations: static research
pipelines that decouple planning from evidence acquisition and one-shot
generation paradigms that easily suffer from long…

[2509.13310] Scaling Agents via Continual Pre-training
URL: https://tongyi-agent.github.io/blog/
Summary: Large language models (LLMs) have evolved into agentic systems capable of
autonomous tool use and multi-step reasoning for complex problem-solving.
However, post-training approaches building upon general-purpose foundation
models consistently underperform in agentic tasks, particularly in open-source
implementations. We identify the root cause: the…

[2509.13311] Towards General Agentic Intelligence via Environment Scaling
Summary: Advanced agentic intelligence is a prerequisite for deploying Large Language
Models in practical, real-world applications. Diverse real-world APIs demand
precise, robust function-calling intelligence, which needs agents to develop
these capabilities through interaction in varied environments. The breadth of
function-calling competence is closely ti…

[2509.16198] RPG: A Repository Planning Graph for Unified and Scalable Codebase
  Generation
Summary: Large language models excel at function- and file-level code generation, yet
generating complete repositories from scratch remains a fundamental challenge.
This process demands coherent and reliable planning across proposal- and
implementation-level stages, while natural language, due to its ambiguity and
verbosity, is ill-suited for faithfully rep…

------------------------------------------------------------------------------------------
Cluster 0 | size=7

[2509.20427] Seedream 4.0: Toward Next-generation Multimodal Image Generation
URL: https://seed.bytedance.com/en/seedream4_0
Summary: We introduce Seedream 4.0, an efficient and high-performance multimodal image
generation system that unifies text-to-image (T2I) synthesis, image editing,
and multi-image composition within a single framework. We develop a highly
efficient diffusion transformer with a powerful VAE which also can reduce the
number of image tokens considerably. This …

[2509.22622] LongLive: Real-time Interactive Long Video Generation
URL: https://nvlabs.github.io/LongLive/
Summary: We present LongLive, a frame-level autoregressive (AR) framework for
real-time and interactive long video generation. Long video generation presents
challenges in both efficiency and quality. Diffusion and Diffusion-Forcing
models can produce high-quality videos but suffer from low efficiency due to
bidirectional attention. Causal attention AR mode…

[2508.20470] Droplet3D: Commonsense Priors from Videos Facilitate 3D Generation
URL: https://dropletx.github.io/
Summary: Scaling laws have validated the success and promise of large-data-trained
models in creative generation across text, image, and video domains. However,
this paradigm faces data scarcity in the 3D domain, as there is far less of it
available on the internet compared to the aforementioned modalities.
Fortunately, there exist adequate videos that inhe…

[2509.04338] From Editor to Dense Geometry Estimator
URL: https://amap-ml.github.io/FE2E/
Summary: Leveraging visual priors from pre-trained text-to-image (T2I) generative
models has shown success in dense prediction. However, dense prediction is
inherently an image-to-image task, suggesting that image editing models, rather
than T2I generative models, may be a more suitable foundation for fine-tuning.
  Motivated by this, we conduct a systemati…

[2509.08826] RewardDance: Reward Scaling in Visual Generation
Summary: Reward Models (RMs) are critical for improving generation models via
Reinforcement Learning (RL), yet the RM scaling paradigm in visual generation
remains largely unexplored. It primarily due to fundamental limitations in
existing approaches: CLIP-based RMs suffer from architectural and input
modality constraints, while prevalent Bradley-Terry loss…

[2509.24006] SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable
  Sparse-Linear Attention
URL: https://github.com/thu-ml/SLA
Summary: In Diffusion Transformer (DiT) models, particularly for video generation,
attention latency is a major bottleneck due to the long sequence length and the
quadratic complexity. We find that attention weights can be separated into two
parts: a small fraction of large weights with high rank and the remaining
weights with very low rank. This naturally …

[2509.08519] HuMo: Human-Centric Video Generation via Collaborative Multi-Modal
  Conditioning
URL: https://phantom-video.github.io/HuMo/
Summary: Human-Centric Video Generation (HCVG) methods seek to synthesize human videos
from multimodal inputs, including text, image, and audio. Existing methods
struggle to effectively coordinate these heterogeneous modalities due to two
challenges: the scarcity of training data with paired triplet conditions and
the difficulty of collaborating the sub-tas…

==========================================================================================
# month=2025-10 BEST CLUSTERING (mode=B, k=4)

------------------------------------------------------------------------------------------
Cluster 3 | size=21

[2509.25454] DeepSearch: Overcome the Bottleneck of Reinforcement Learning with
  Verifiable Rewards via Monte Carlo Tree Search
URL: https://github.com/smiles724/DeepSearch
Summary: Although RLVR has become an essential component for developing advanced
reasoning skills in LLMs, contemporary studies have documented training
plateaus that emerge following thousands of optimization steps, demonstrating
notable decreases in performance gains despite increased computational
investment. This limitation stems from the sparse explora…

[2510.05592] In-the-Flow Agentic System Optimization for Effective Planning and Tool
  Use
URL: https://agentflow.stanford.edu/
Summary: Outcome-driven reinforcement learning has advanced reasoning in large
language models (LLMs), but prevailing tool-augmented approaches train a
single, monolithic policy that interleaves thoughts and tool calls under full
context; this scales poorly with long horizons and diverse tools and
generalizes weakly to new scenarios. Agentic systems offer a…

[2510.01051] GEM: A Gym for Agentic LLMs
URL: https://axon-rl.github.io/
Summary: The training paradigm for large language models (LLMs) is moving from static
datasets to experience-based learning, where agents acquire skills via
interacting with complex environments. To facilitate this transition we
introduce GEM (General Experience Maker), an open-source environment simulator
designed for the age of LLMs. Analogous to OpenAI-G…

[2510.08540] MM-HELIX: Boosting Multimodal Long-Chain Reflective Reasoning with
  Holistic Platform and Adaptive Hybrid Policy Optimization
URL: https://mm-helix.github.io/
Summary: While current Multimodal Large Language Models (MLLMs) have demonstrated
proficiency in reasoning tasks such as mathematics and logic, their capacity
for long-chain reflective reasoning, a prerequisite for solving complex
real-world problems, remains largely underexplored. In this work, we first
conduct an extensive empirical investigation to evalu…

[2510.18927] BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via
  Balanced Policy Optimization with Adaptive Clipping
URL: https://github.com/WooooDyy/BAPO
Summary: Reinforcement learning (RL) has recently become the core paradigm for
aligning and strengthening large language models (LLMs). Yet, applying RL in
off-policy settings--where stale data from past policies are used for
training--improves sample efficiency, but remains challenging: policy entropy
declines sharply, optimization often becomes unstable a…

[2510.21618] DeepAgent: A General Reasoning Agent with Scalable Toolsets
Summary: Large reasoning models have demonstrated strong problem-solving abilities,
yet real-world tasks often require external tools and long-horizon
interactions. Existing agent frameworks typically follow predefined workflows,
which limit autonomous and global task completion. In this paper, we introduce
DeepAgent, an end-to-end deep reasoning agent that…

[2510.04618] Agentic Context Engineering: Evolving Contexts for Self-Improving
  Language Models
Summary: Large language model (LLM) applications such as agents and domain-specific
reasoning increasingly rely on context adaptation -- modifying inputs with
instructions, strategies, or evidence, rather than weight updates. Prior
approaches improve usability but often suffer from brevity bias, which drops
domain insights for concise summaries, and from co…

[2510.08558] Agent Learning via Early Experience
Summary: A long-term goal of language agents is to learn and improve through their own
experience, ultimately outperforming humans in complex, real-world tasks.
However, training agents from experience data with reinforcement learning
remains difficult in many environments, which either lack verifiable rewards
(e.g., websites) or require inefficient long-ho…

[2509.25848] More Thought, Less Accuracy? On the Dual Nature of Reasoning in
  Vision-Language Models
URL: https://xytian1008.github.io/VAPO/
Summary: Reasoning has emerged as a pivotal capability in Large Language Models
(LLMs). Through Reinforcement Learning (RL), typically Group Relative Policy
Optimization (GRPO), these models are able to solve complex tasks such as
mathematics and code generation. Building on these advances, recent research
has sought to extend reasoning to Vision-Language M…

[2510.02245] ExGRPO: Learning to Reason from Experience
Summary: Reinforcement learning from verifiable rewards (RLVR) is an emerging paradigm
for improving the reasoning ability of large language models. However, standard
on-policy training discards rollout experiences after a single update, leading
to computational inefficiency and instability. While prior work on RL has
highlighted the benefits of reusing pas…

[2510.19338] Every Attention Matters: An Efficient Hybrid Architecture for
  Long-Context Reasoning
Summary: In this technical report, we present the Ring-linear model series,
specifically including Ring-mini-linear-2.0 and Ring-flash-linear-2.0.
Ring-mini-linear-2.0 comprises 16B parameters and 957M activations, while
Ring-flash-linear-2.0 contains 104B parameters and 6.1B activations. Both
models adopt a hybrid architecture that effectively integrates l…

[2510.24701] Tongyi DeepResearch Technical Report
URL: https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
Summary: We present Tongyi DeepResearch, an agentic large language model, which is
specifically designed for long-horizon, deep information-seeking research
tasks. To incentivize autonomous deep research agency, Tongyi DeepResearch is
developed through an end-to-end training framework that combines agentic
mid-training and agentic post-training, enabling sc…

[2510.11696] QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning
  for LLMs
URL: https://github.com/NVlabs/QeRL
Summary: We propose QeRL, a Quantization-enhanced Reinforcement Learning framework for
large language models (LLMs). While RL is essential for LLMs' reasoning
capabilities, it is resource-intensive, requiring substantial GPU memory and
long rollout durations. QeRL addresses these issues by combining NVFP4
quantization with Low-Rank Adaptation (LoRA), accele…

[2509.25541] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified
  Self-Play
Summary: Although reinforcement learning (RL) can effectively enhance the reasoning
capabilities of vision-language models (VLMs), current methods remain heavily
dependent on labor-intensive datasets that require extensive manual
construction and verification, leading to extremely high training costs and
consequently constraining the practical deployment of…

[2510.16872] DeepAnalyze: Agentic Large Language Models for Autonomous Data Science
URL: https://ruc-deepanalyze.github.io/
Summary: Autonomous data science, from raw data sources to analyst-grade deep research
reports, has been a long-standing challenge, and is now becoming feasible with
the emergence of powerful large language models (LLMs). Recent workflow-based
data agents have shown promising results on specific data tasks but remain
fundamentally limited in achieving fully…

[2510.23564] ReCode: Unify Plan and Action for Universal Granularity Control
Summary: Real-world tasks require decisions at varying granularities, and humans excel
at this by leveraging a unified cognitive representation where planning is
fundamentally understood as a high-level form of action. However, current Large
Language Model (LLM)-based agents lack this crucial capability to operate
fluidly across decision granularities. This…

[2510.18135] World-in-World: World Models in a Closed-Loop World
URL: https://world-in-world.github.io/
Summary: Generative world models (WMs) can now simulate worlds with striking visual
realism, which naturally raises the question of whether they can endow embodied
agents with predictive perception for decision making. Progress on this
question has been limited by fragmented evaluation: most existing benchmarks
adopt open-loop protocols that emphasize visua…

[2510.23473] Video-Thinker: Sparking "Thinking with Videos" via Reinforcement
  Learning
Summary: Recent advances in image reasoning methods, particularly "Thinking with
Images", have demonstrated remarkable success in Multimodal Large Language
Models (MLLMs); however, this dynamic reasoning paradigm has not yet been
extended to video reasoning tasks. In this paper, we propose Video-Thinker,
which empowers MLLMs to think with videos by autonomo…

[2510.14545] Agentic Entropy-Balanced Policy Optimization
Summary: Recently, Agentic Reinforcement Learning (Agentic RL) has made significant
progress in incentivizing the multi-turn, long-horizon tool-use capabilities of
web agents. While mainstream agentic RL algorithms autonomously explore
high-uncertainty tool-call steps under the guidance of entropy, excessive
reliance on entropy signals can impose further co…

[2509.24002] MCPMark: A Benchmark for Stress-Testing Realistic and Comprehensive MCP
  Use
URL: https://mcpmark.ai/
Summary: MCP standardizes how LLMs interact with external systems, forming the
foundation for general agents. However, existing MCP benchmarks remain narrow
in scope: they focus on read-heavy tasks or tasks with limited interaction
depth, and fail to capture the complexity and realism of real-world workflows.
To address this gap, we propose MCPMark, a bench…

[2510.24668] InteractComp: Evaluating Search Agents With Ambiguous Queries
Summary: Language agents have demonstrated remarkable potential in web search and
information retrieval. However, these search agents assume user queries are
complete and unambiguous, an assumption that diverges from reality where users
begin with incomplete queries requiring clarification through interaction. Yet
most agents lack interactive mechanisms dur…

------------------------------------------------------------------------------------------
Cluster 1 | size=16

[2510.09426] KORMo: Korean Open Reasoning Model for Everyone
Summary: This work presents the first large-scale investigation into constructing a
fully open bilingual large language model (LLM) for a non-English language,
specifically Korean, trained predominantly on synthetic data. We introduce
KORMo-10B, a 10.8B-parameter model trained from scratch on a Korean-English
corpus in which 68.74% of the Korean portion is …

[2510.18121] Efficient Long-context Language Model Training by Core Attention
  Disaggregation
Summary: We present core attention disaggregation (CAD), a technique that improves
long-context large language model training by decoupling the core attention
computation, softmax(QK^T)V, from the rest of the model and executing it on a
separate pool of devices. In existing systems, core attention is colocated with
other layers; at long context lengths, its…

[2510.26697] The End of Manual Decoding: Towards Truly End-to-End Language Models
Summary: The "end-to-end" label for LLMs is a misnomer. In practice, they depend on a
non-differentiable decoding process that requires laborious, hand-tuning of
hyperparameters like temperature and top-p. This paper introduces AutoDeco, a
novel architecture that enables truly "end-to-end" generation by learning to
control its own decoding strategy. We augm…

[2510.03215] Cache-to-Cache: Direct Semantic Communication Between Large Language
  Models
URL: https://fuvty.github.io/C2C_Project_Page/
Summary: Multi-LLM systems harness the complementary strengths of diverse Large
Language Models, achieving performance and efficiency gains unattainable by a
single model. In existing designs, LLMs communicate through text, forcing
internal representations to be transformed into output token sequences. This
process both loses rich semantic information and i…

[2510.15444] A Theoretical Study on Bridging Internal Probability and
  Self-Consistency for LLM Reasoning
URL: https://wnjxyk.github.io/RPC
Summary: Test-time scaling seeks to improve the reasoning performance of large
language models (LLMs) by adding computational resources. A prevalent approach
within the field is sampling-based test-time scaling methods, which enhance
reasoning by generating multiple reasoning paths for a given input during
inference. However, despite its practical success, …

[2510.25741] Scaling Latent Reasoning via Looped Language Models
URL: https://ouro-llm.github.io/
Summary: Modern LLMs are trained to "think" primarily via explicit text generation,
such as chain-of-thought (CoT), which defers reasoning to post-training and
under-leverages pre-training data. We present and open-source Ouro, named after
the recursive Ouroboros, a family of pre-trained Looped Language Models
(LoopLM) that instead build reasoning into the …

[2510.18866] LightMem: Lightweight and Efficient Memory-Augmented Generation
Summary: Despite their remarkable capabilities, Large Language Models (LLMs) struggle
to effectively leverage historical interaction information in dynamic and
complex environments. Memory systems enable LLMs to move beyond stateless
interactions by introducing persistent information storage, retrieval, and
utilization mechanisms. However, existing memory s…

[2510.00446] LongCodeZip: Compress Long Context for Code Language Models
Summary: Code generation under long contexts is becoming increasingly critical as
Large Language Models (LLMs) are required to reason over extensive information
in the codebase. While recent advances enable code LLMs to process long inputs,
high API costs and generation latency remain substantial bottlenecks. Existing
context pruning techniques, such as LLM…

[2510.26692] Kimi Linear: An Expressive, Efficient Attention Architecture
Summary: We introduce Kimi Linear, a hybrid linear attention architecture that, for
the first time, outperforms full attention under fair comparisons across
various scenarios -- including short-context, long-context, and reinforcement
learning (RL) scaling regimes. At its core lies Kimi Delta Attention (KDA), an
expressive linear attention module that exten…

[2510.15870] OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding
  LLM
URL: https://nvlabs.github.io/OmniVinci/
Summary: Advancing machine intelligence requires developing the ability to perceive
across multiple modalities, much as humans sense the world. We introduce
OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We
carefully study the design choices across model architecture and data curation.
For model architecture, we present three key i…

[2509.26507] The Dragon Hatchling: The Missing Link between the Transformer and
  Models of the Brain
Summary: The relationship between computing systems and the brain has served as
motivation for pioneering theoreticians since John von Neumann and Alan Turing.
Uniform, scale-free biological networks, such as the brain, have powerful
properties, including generalizing over time, which is the main barrier for
Machine Learning on the path to Universal Reasoni…

[2510.14528] PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model
Summary: In this report, we propose PaddleOCR-VL, a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This in…

[2510.09116] DITING: A Multi-Agent Evaluation Framework for Benchmarking Web Novel
  Translation
Summary: Large language models (LLMs) have substantially advanced machine translation
(MT), yet their effectiveness in translating web novels remains unclear.
Existing benchmarks rely on surface-level metrics that fail to capture the
distinctive traits of this genre. To address these gaps, we introduce DITING,
the first comprehensive evaluation framework fo…

[2510.18234] DeepSeek-OCR: Contexts Optical Compression
Summary: We present DeepSeek-OCR as an initial investigation into the feasibility of
compressing long contexts via optical 2D mapping. DeepSeek-OCR consists of two
components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically,
DeepEncoder serves as the core engine, designed to maintain low activations
under high-resolution input while achiev…

[2510.04849] When Models Lie, We Learn: Multilingual Span-Level Hallucination
  Detection with PsiloQA
Summary: Hallucination detection remains a fundamental challenge for the safe and
reliable deployment of large language models (LLMs), especially in applications
requiring factual accuracy. Existing hallucination benchmarks often operate at
the sequence level and are limited to English, lacking the fine-grained,
multilingual supervision needed for a compreh…

[2510.04871] Less is More: Recursive Reasoning with Tiny Networks
URL: https://alexiajm.github.io/2025/09/29/tiny_recursive_models.html#
Summary: Hierarchical Reasoning Model (HRM) is a novel approach using two small neural
networks recursing at different frequencies. This biologically inspired method
beats Large Language models (LLMs) on hard puzzle tasks such as Sudoku, Maze,
and ARC-AGI while trained with small models (27M parameters) on small data
(around 1000 examples). HRM holds great …

------------------------------------------------------------------------------------------
Cluster 2 | size=10

[2510.26583] Emu3.5: Native Multimodal Models are World Learners
URL: https://emu.world/
Summary: We introduce Emu3.5, a large-scale multimodal world model that natively
predicts the next state across vision and language. Emu3.5 is pre-trained
end-to-end with a unified next-token prediction objective on a corpus of
vision-language interleaved data containing over 10 trillion tokens, primarily
derived from sequential frames and transcripts of in…

[2510.12586] Advancing End-to-End Pixel Space Generative Modeling via Self-supervised
  Pre-training
Summary: Pixel-space generative models are often more difficult to train and generally
underperform compared to their latent-space counterparts, leaving a persistent
performance and efficiency gap. In this paper, we introduce a novel two-stage
training framework that closes this gap for pixel-space diffusion and
consistency models. In the first stage, we pr…

[2510.11690] Diffusion Transformers with Representation Autoencoders
URL: https://rae-dit.github.io/
Summary: Latent generative modeling, where a pretrained autoencoder maps pixels into a
latent space for the diffusion process, has become the standard strategy for
Diffusion Transformers (DiT); however, the autoencoder component has barely
evolved. Most DiTs continue to rely on the original VAE encoder, which
introduces several limitations: outdated backbon…

[2510.02283] Self-Forcing++: Towards Minute-Scale High-Quality Video Generation
URL: https://self-forcing-plus-plus.github.io/
Summary: Diffusion models have revolutionized image and video generation, achieving
unprecedented visual quality. However, their reliance on transformer
architectures incurs prohibitively high computational costs, particularly when
extending generation to long videos. Recent work has explored autoregressive
formulations for long video generation, typically …

[2510.08673] Thinking with Camera: A Unified Multimodal Model for Camera-Centric
  Understanding and Generation
URL: https://kangliao929.github.io/projects/puffin/
Summary: Camera-centric understanding and generation are two cornerstones of spatial
intelligence, yet they are typically studied in isolation. We present Puffin, a
unified camera-centric multimodal model that extends spatial awareness along
the camera dimension. Puffin integrates language regression and diffusion-based
generation to interpret and create sc…

[2510.11693] Scaling Language-Centric Omnimodal Representation Learning
URL: https://huggingface.co/LCO-Embedding
Summary: Recent multimodal embedding approaches leveraging multimodal large language
models (MLLMs) fine-tuned with contrastive learning (CL) have shown promising
results, yet the underlying reasons behind their superiority remain
underexplored. This work argues that a crucial advantage of MLLM-based
approaches stems from implicit cross-modal alignment achi…

[2510.14975] WithAnyone: Towards Controllable and ID Consistent Image Generation
URL: https://doby-xu.github.io/WithAnyone/
Summary: Identity-consistent generation has become an important focus in text-to-image
research, with recent models achieving notable success in producing images
aligned with a reference identity. Yet, the scarcity of large-scale paired
datasets containing multiple images of the same individual forces most
approaches to adopt reconstruction-based training. …

[2510.23607] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial
  Representations
URL: https://pointcept.github.io/Concerto/
Summary: Humans learn abstract concepts through multisensory synergy, and once formed,
such representations can often be recalled from a single modality. Inspired by
this principle, we introduce Concerto, a minimalist simulation of human concept
learning for spatial cognition, combining 3D intra-modal self-distillation with
2D-3D cross-modal joint embedding…

[2510.23538] JanusCoder: Towards a Foundational Visual-Programmatic Interface for
  Code Intelligence
Summary: The scope of neural code intelligence is rapidly expanding beyond text-based
source code to encompass the rich visual outputs that programs generate. This
visual dimension is critical for advanced applications like flexible content
generation and precise, program-driven editing of visualizations. However,
progress has been impeded by the scarcity o…

[2510.05096] Paper2Video: Automatic Video Generation from Scientific Papers
URL: https://showlab.github.io/Paper2Video/
Summary: Academic presentation videos have become an essential medium for research
communication, yet producing them remains highly labor-intensive, often
requiring hours of slide design, recording, and editing for a short 2 to 10
minutes video. Unlike natural video, presentation video generation involves
distinctive challenges: inputs from research papers,…

------------------------------------------------------------------------------------------
Cluster 0 | size=3

[2510.05684] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to
  Embodied AI
URL: https://worv-ai.github.io/d2e/
Summary: Large language models leverage internet-scale text data, yet embodied AI
remains constrained by the prohibitive costs of physical trajectory collection.
Desktop environments -- particularly gaming -- offer a compelling alternative:
they provide rich sensorimotor interactions at scale while maintaining the
structured observation-action coupling esse…

[2510.12276] Spatial Forcing: Implicit Spatial Representation Alignment for
  Vision-language-action Model
URL: https://spatial-forcing.github.io/
Summary: Vision-language-action (VLA) models have recently shown strong potential in
enabling robots to follow language instructions and execute precise actions.
However, most VLAs are built upon vision-language models pretrained solely on
2D data, which lack accurate spatial awareness and hinder their ability to
operate in the 3D physical world. Existing s…

[2510.12403] Robot Learning: A Tutorial
URL: https://huggingface.co/spaces/lerobot/robot-learning-tutorial
Summary: Robot learning is at an inflection point, driven by rapid advancements in
machine learning and the growing availability of large-scale robotics data.
This shift from classical, model-based methods to data-driven, learning-based
paradigms is unlocking unprecedented capabilities in autonomous systems. This
tutorial navigates the landscape of modern r…

==========================================================================================
# month=2025-11 BEST CLUSTERING (mode=C, k=5)

------------------------------------------------------------------------------------------
Cluster 0 | size=13

[2511.03773] Scaling Agent Learning via Experience Synthesis
Summary: While reinforcement learning (RL) can empower large language model (LLM)
agents by enabling self-improvement through interaction, its practical adoption
remains challenging due to costly rollouts, limited task diversity, unreliable
reward signals, and infrastructure complexity, all of which obstruct the
collection of scalable experience data. To ad…

[2511.19304] AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning
Summary: Humans naturally adapt to diverse environments by learning underlying rules across worlds with different dynamics, observations, and reward structures. In contrast, existing agents typically demonstrate improvements via self-evolving within a single domain, implicitly assuming a fixed environment distribution. Cross-environment learning has remaine…

[2511.16043] Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning
Summary: Large Language Model (LLM) Agents, often trained with Reinforcement Learning (RL), are constrained by a dependency on human-curated data, limiting scalability and tethering AI to human knowledge. Existing self-evolution frameworks offer an alternative but are typically restricted by the model's inherent capabilities and single-round interactions, h…

[2511.20639] Latent Collaboration in Multi-Agent Systems
Summary: Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce …

[2511.11793] MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling
URL: https://dr.miromind.ai/
Summary: We present MiroThinker v1.0, an open-source research agent designed to advance tool-augmented reasoning and information-seeking capabilities. Unlike previous agents that only scale up model size or context length, MiroThinker explores interaction scaling at the model level, systematically training the model to handle deeper and more frequent agent-…

[2510.25889] π_RL: Online RL Fine-tuning for Flow-based
  Vision-Language-Action Models
URL: https://rlinf.readthedocs.io/en/latest/rst_source/examples/pi0.html
Summary: Vision-Language-Action (VLA) models enable robots to understand and perform
complex tasks from multimodal input. Although recent work explores using
reinforcement learning (RL) to automate the laborious data collection process
in scaling supervised fine-tuning (SFT), applying large-scale RL to flow-based
VLAs (e.g., pi_0, pi_{0.5}) remains challeng…

[2511.07332] Grounding Computer Use Agents on Human Demonstrations
URL: https://groundcua.github.io/
Summary: Building reliable computer-use agents requires grounding: accurately
connecting natural language instructions to the correct on-screen elements.
While large datasets exist for web and mobile interactions, high-quality
resources for desktop environments are limited. To address this gap, we
introduce GroundCUA, a large-scale desktop grounding dataset…

[2511.09057] PAN: A World Model for General, Interactable, and Long-Horizon World Simulation
URL: https://ifm.mbzuai.ac.ae/pan/
Summary: A world model enables an intelligent agent to imagine, predict, and reason about how the world evolves in response to its actions, and accordingly to plan and strategize. While recent video generation models produce realistic visual sequences, they typically operate in the prompt-to-full-video manner without causal control, interactivity, or long-h…

[2511.08892] Lumine: An Open Recipe for Building Generalist Agents in 3D Open Worlds
URL: https://www.lumine-ai.org/
Summary: We introduce Lumine, the first open recipe for developing generalist agents capable of completing hours-long complex missions in real time within challenging 3D open-world environments. Lumine adopts a human-like interaction paradigm that unifies perception, reasoning, and action in an end-to-end manner, powered by a vision-language model. It proce…

[2511.18423] General Agentic Memory Via Deep Research
URL: https://github.com/VectorSpaceLab/general-agentic-memory
Summary: Memory is critical for AI agents, yet the widely-adopted static memory, aiming to create readily available memory in advance, is inevitably subject to severe information loss. To address this limitation, we propose a novel framework called general agentic memory (GAM). GAM follows the principle of "just-in time (JIT) compilation" where it focuses o…

[2511.19399] DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research
URL: https://github.com/rlresearch/dr-tulu
Summary: Deep research models perform multi-step research to produce long-form, well-attributed answers. However, most open deep research models are trained on easily verifiable short-form QA tasks via reinforcement learning with verifiable rewards (RLVR), which does not extend to realistic long-form tasks. We address this with Reinforcement Learning with E…

[2511.15705] GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization
URL: https://ekonwang.github.io/geo-vista/
Summary: Current research on agentic visual reasoning enables deep multimodal understanding but primarily focuses on image manipulation tools, leaving a gap toward more general-purpose agentic models. In this work, we revisit the geolocalization task, which requires not only nuanced visual grounding but also web search to confirm or refine hypotheses during…

[2510.24411] OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid
  Validation in Realistic Workflows
Summary: Computer-using agents powered by Vision-Language Models (VLMs) have
demonstrated human-like capabilities in operating digital environments like
mobile platforms. While these agents hold great promise for advancing digital
automation, their potential for unsafe operations, such as system compromise
and privacy leakage, is raising significant concern…

------------------------------------------------------------------------------------------
Cluster 1 | size=13

[2511.04460] V-Thinker: Interactive Thinking with Images
Summary: Empowering Large Multimodal Models (LMMs) to deeply integrate image
interaction with long-horizon reasoning capabilities remains a long-standing
challenge in this field. Recent advances in vision-centric reasoning explore a
promising "Thinking with Images" paradigm for LMMs, marking a shift from
image-assisted reasoning to image-interactive thinkin…

[2511.04570] Thinking with Video: Video Generation as a Promising Multimodal
  Reasoning Paradigm
URL: https://thinking-with-video.github.io/
Summary: "Thinking with Text" and "Thinking with Images" paradigm significantly
improve the reasoning ability of large language models (LLMs) and Vision
Language Models (VLMs). However, these paradigms have inherent limitations. (1)
Images capture only single moments and fail to represent dynamic processes or
continuous changes, and (2) The separation of te…

[2511.02779] When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for
  Visual Chain-of-Thought
URL: https://mira-benchmark.github.io/
Summary: We propose MIRA, a new benchmark designed to evaluate models in scenarios
where generating intermediate visual images is essential for successful
reasoning. Unlike traditional CoT methods that rely solely on text, tasks in
MIRA require models to generate and utilize intermediate images - such as
sketches, structural diagrams, or path drawings - to …

[2510.27492] ThinkMorph: Emergent Properties in Multimodal Interleaved
  Chain-of-Thought Reasoning
URL: https://thinkmorph.github.io/
Summary: Multimodal reasoning requires iterative coordination between language and
vision, yet it remains unclear what constitutes a meaningful interleaved chain
of thought. We posit that text and image thoughts should function as
complementary, rather than isomorphic, modalities that mutually advance
reasoning. Guided by this principle, we build ThinkMorph…

[2511.16334] OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe
URL: https://evolvinglmms-lab.github.io/OpenMMReasoner/
Summary: Recent advancements in large reasoning models have fueled growing interest in extending such capabilities to multimodal domains. However, despite notable progress in visual reasoning, the lack of transparent and reproducible data curation and training strategies remains a major barrier to scalable research. In this work, we introduce OpenMMReasoner…

[2511.15065] Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks
URL: https://imyangc7.github.io/VRBench_Web/
Summary: Video Models have achieved remarkable success in high-fidelity video generation with coherent motion dynamics. Analogous to the development from text generation to text-based reasoning in language modeling, the development of video models motivates us to ask: Can video models reason via video generation? Compared with the discrete text corpus, vide…

[2511.02778] VCode: a Multimodal Coding Benchmark with SVG as Symbolic Visual
  Representation
URL: https://csu-jpg.github.io/VCode/
Summary: Code has emerged as a precise and executable medium for reasoning and action
in the agent era. Yet, progress has largely focused on language-centric tasks
such as program synthesis and debugging, leaving visual-centric coding
underexplored. Inspired by how humans reason over sketches, we advocate SVG
code as a compact, interpretable, and executable…

[2511.12609] Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data
URL: https://idealistxy.github.io/Uni-MoE-v2.github.io/
Summary: We present Uni-MoE 2.0 from the Lychee family. As a fully open-source omnimodal large model (OLM), it substantially advances Lychee's Uni-MoE series in language-centric multimodal understanding, reasoning, and generating. Based on the Qwen2.5-7B dense architecture, we build Uni-MoE-2.0-Omni from scratch through three core contributions: dynamic-cap…

[2511.11113] VIDEOP2R: Video Understanding from Perception to Reasoning
Summary: Reinforcement fine-tuning (RFT), a two-stage framework consisting of supervised fine-tuning (SFT) and reinforcement learning (RL) has shown promising results on improving reasoning ability of large language models (LLMs). Yet extending RFT to large video language models (LVLMs) remains challenging. We propose VideoP2R, a novel process-aware video R…

[2511.09611] MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation
URL: https://tyfeld.github.io/mmadaparellel.github.io/
Summary: While thinking-aware generation aims to improve performance on complex tasks, we identify a critical failure mode where existing sequential, autoregressive approaches can paradoxically degrade performance due to error propagation. To systematically analyze this issue, we propose ParaBench, a new benchmark designed to evaluate both text and image ou…

[2510.25616] Don't Blind Your VLA: Aligning Visual Representations for OOD
  Generalization
URL: https://blind-vla-paper.github.io
Summary: The growing success of Vision-Language-Action (VLA) models stems from the
promise that pretrained Vision-Language Models (VLMs) can endow agents with
transferable world knowledge and vision-language (VL) grounding, laying a
foundation for action models with broader generalization. Yet when these VLMs
are adapted to the action modality, it remains u…

[2511.03506] HaluMem: Evaluating Hallucinations in Memory Systems of Agents
Summary: Memory systems are key components that enable AI systems such as LLMs and AI
agents to achieve long-term learning and sustained interaction. However, during
memory storage and retrieval, these systems frequently exhibit memory
hallucinations, including fabrication, errors, conflicts, and omissions.
Existing evaluations of memory hallucinations are …

[2511.16719] SAM 3: Segment Anything with Concepts
URL: https://ai.meta.com/sam3/
Summary: We present Segment Anything Model (SAM) 3, a unified model that detects, segments, and tracks objects in images and videos based on concept prompts, which we define as either short noun phrases (e.g., "yellow school bus"), image exemplars, or a combination of both. Promptable Concept Segmentation (PCS) takes such prompts and returns segmentation ma…

------------------------------------------------------------------------------------------
Cluster 4 | size=13

[2510.27688] Continuous Autoregressive Language Models
URL: https://shaochenze.github.io/blog/2025/CALM/
Summary: The efficiency of large language models (LLMs) is fundamentally limited by
their sequential, token-by-token generation process. We argue that overcoming
this bottleneck requires a new design axis for LLM scaling: increasing the
semantic bandwidth of each generative step. To this end, we introduce
Continuous Autoregressive Language Models (CALM), a …

[2511.08577] Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models
Summary: Improving reasoning capabilities of Large Language Models (LLMs), especially under parameter constraints, is crucial for real-world applications. Prior work proposes recurrent transformers, which allocate a fixed number of extra iterations per token to improve generation quality. After the first, standard forward pass, instead of verbalization, las…

[2511.08923] TiDAR: Think in Diffusion, Talk in Autoregression
URL: https://tidarlm.github.io
Summary: Diffusion language models hold the promise of fast parallel generation, while autoregressive (AR) models typically excel in quality due to their causal structure aligning naturally with language modeling. This raises a fundamental question: can we achieve a synergy with high throughput, higher GPU utilization, and AR level quality? Existing methods…

[2511.03276] Diffusion Language Models are Super Data Learners
URL: https://github.com/JinjieNi/dlms-are-super-data-learners
Summary: Under strictly controlled pre-training settings, we observe a Crossover: when
unique data is limited, diffusion language models (DLMs) consistently surpass
autoregressive (AR) models by training for more epochs. The crossover shifts
later with more or higher-quality data, earlier with larger models, and
persists across dense and sparse architecture…

[2511.13254] Souper-Model: How Simple Arithmetic Unlocks State-of-the-Art LLM Performance
Summary: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their training remains resource- and time-intensive, requiring massive compute power and careful orchestration of training procedures. Model souping-the practice of averaging weights from multiple models of the same architecture-has emerged as a promi…

[2511.15552] Multimodal Evaluation of Russian-language Architectures
URL: https://mera.a-ai.ru/en/multi
Summary: Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, w…

[2511.20626] ROOT: Robust Orthogonalized Optimizer for Neural Network Training
URL: https://github.com/huawei-noah/noah-research/tree/master/ROOT
Summary: The optimization of large language models (LLMs) remains a critical challenge, particularly as model scaling exacerbates sensitivity to algorithmic imprecision and training instability. Recent advances in optimizers have improved convergence efficiency through momentum orthogonalization, but suffer from two key robustness limitations: dimensional f…

[2511.13647] Part-X-MLLM: Part-aware 3D Multimodal Large Language Model
URL: https://chunshi.wang/Part-X-MLLM/
Summary: We introduce Part-X-MLLM, a native 3D multimodal large language model that unifies diverse 3D tasks by formulating them as programs in a structured, executable grammar. Given an RGB point cloud and a natural language prompt, our model autoregressively generates a single, coherent token sequence encoding part-level bounding boxes, semantic descripti…

[2511.14295] AraLingBench A Human-Annotated Benchmark for Evaluating Arabic Linguistic Capabilities of Large Language Models
Summary: We present AraLingBench: a fully human annotated benchmark for evaluating the Arabic linguistic competence of large language models (LLMs). The benchmark spans five core categories: grammar, morphology, spelling, reading comprehension, and syntax, through 150 expert-designed multiple choice questions that directly assess structural language underst…

[2510.25602] INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization
  Formats
Summary: Modern AI hardware, such as Nvidia's Blackwell architecture, is increasingly
embracing low-precision floating-point (FP) formats to handle the pervasive
activation outliers in Large Language Models (LLMs). Despite this industry
trend, a unified comparison of FP and integer (INT) quantization across varying
granularities has been missing, leaving al…

[2511.14993] Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation
URL: https://kandinskylab.ai/
Summary: This report introduces Kandinsky 5.0, a family of state-of-the-art foundation models for high-resolution image and 10-second video synthesis. The framework comprises three core line-up of models: Kandinsky 5.0 Image Lite - a line-up of 6B parameter image generation models, Kandinsky 5.0 Video Lite - a fast and lightweight 2B parameter text-to-video…

[2511.15210] Unveiling Intrinsic Dimension of Texts: from Academic Abstract to Creative Story
Summary: Intrinsic dimension (ID) is an important tool in modern LLM analysis, informing studies of training dynamics, scaling behavior, and dataset structure, yet its textual determinants remain underexplored. We provide the first comprehensive study grounding ID in interpretable text properties through cross-encoder analysis, linguistic features, and spar…

[2511.10555] A Style is Worth One Code: Unlocking Code-to-Style Image Generation with Discrete Style Space
URL: https://Kwai-Kolors.github.io/CoTyle/
Summary: Innovative visual stylization is a cornerstone of artistic creation, yet generating novel and consistent visual styles remains a significant challenge. Existing generative approaches typically rely on lengthy textual prompts, reference images, or parameter-efficient fine-tuning to guide style-aware image generation, but often struggle with style co…

------------------------------------------------------------------------------------------
Cluster 3 | size=6

[2511.10629] One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models
Summary: Diffusion models struggle to scale beyond their training resolutions, as direct high-resolution sampling is slow and costly, while post-hoc image super-resolution (ISR) introduces artifacts and additional latency by operating after decoding. We present the Latent Upscaler Adapter (LUA), a lightweight module that performs super-resolution directly o…

[2511.19365] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation
URL: https://zehong-ma.github.io/DeCo/
Summary: Pixel diffusion aims to generate images directly in pixel space in an end-to-end fashion. This approach avoids the limitations of VAE in the two-stage latent diffusion, offering higher model capacity. Existing pixel diffusion models suffer from slow training and inference, as they usually model both high-frequency signals and low-frequency semantic…

[2511.13720] Back to Basics: Let Denoising Generative Models Denoise
Summary: Today's denoising diffusion models do not "denoise" in the classical sense, i.e., they do not directly predict clean images. Rather, the neural networks predict noise or a noised quantity. In this paper, we suggest that predicting clean data and predicting noised quantities are fundamentally different. According to the manifold assumption, natural …

[2511.16624] SAM 3D: 3Dfy Anything in Images
URL: https://ai.meta.com/sam3d/
Summary: We present SAM 3D, a generative model for visually grounded 3D object reconstruction, predicting geometry, texture, and layout from a single image. SAM 3D excels in natural images, where occlusion and scene clutter are common and visual recognition cues from context play a larger role. We achieve this with a human- and model-in-the-loop pipeline fo…

[2511.10647] Depth Anything 3: Recovering the Visual Space from Any Views
URL: https://depth-anything-3.github.io/
Summary: We present Depth Anything 3 (DA3), a model that predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses. In pursuit of minimal modeling, DA3 yields two key insights: a single plain transformer (e.g., vanilla DINO encoder) is sufficient as a backbone without architectural specialization, a…

[2511.09146] DoPE: Denoising Rotary Position Embedding
URL: https://The-physical-picture-of-LLMs.github.io
Summary: Rotary Position Embedding (RoPE) in Transformer models has inherent limits that weaken length extrapolation. We reinterpret the attention map with positional encoding as a noisy feature map, and propose Denoising Positional Encoding (DoPE), a training-free method based on truncated matrix entropy to detect outlier frequency bands in the feature map…

------------------------------------------------------------------------------------------
Cluster 2 | size=5

[2511.07327] IterResearch: Rethinking Long-Horizon Agents via Markovian State
  Reconstruction
Summary: Recent advances in deep-research agents have shown promise for autonomous
knowledge construction through dynamic reasoning over external sources.
However, existing approaches rely on a mono-contextual paradigm that
accumulates all information in a single, expanding context window, leading to
context suffocation and noise contamination that limit th…

[2511.06221] Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model
  Reasoning Ability in VibeThinker-1.5B
URL: https://github.com/WeiboAI/VibeThinker
Summary: Challenging the prevailing consensus that small models inherently lack robust
reasoning, this report introduces VibeThinker-1.5B, a 1.5B-parameter dense
model developed via our Spectrum-to-Signal Principle (SSP). This challenges the
prevailing approach of scaling model parameters to enhance capabilities, as
seen in models like DeepSeek R1 (671B) an…

[2510.22115] Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open
  Language Foundation
Summary: We introduce Ling 2.0, a series reasoning-oriented language foundation built
upon the principle that every activation boosts reasoning capability. Designed
to scale from tens of billions to one trillion parameters under a unified
Mixture-of-Experts (MoE) paradigm, Ling 2.0 emphasizes high sparsity,
cross-scale consistency, and efficiency guided by …

[2511.13612] P1: Mastering Physics Olympiads with Reinforcement Learning
URL: https://prime-rl.github.io/P1/
Summary: Recent progress in large language models (LLMs) has moved the frontier from puzzle-solving to science-grade reasoning-the kind needed to tackle problems whose answers must stand against nature, not merely fit a rubric. Physics is the sharpest test of this shift, which binds symbols to reality in a fundamental way, serving as the cornerstone of most…

[2511.17592] GigaEvo: An Open Source Optimization Framework Powered By LLMs And Evolution Algorithms
URL: https://airi-institute.github.io/gigaevo-cover/
Summary: Recent advances in LLM-guided evolutionary computation, particularly AlphaEvolve (Novikov et al., 2025; Georgiev et al., 2025), have demonstrated remarkable success in discovering novel mathematical constructions and solving challenging optimization problems. However, the high-level descriptions in published work leave many implementation details u…

==========================================================================================
# month=2025-12 BEST CLUSTERING (mode=C, k=4)

------------------------------------------------------------------------------------------
Cluster 0 | size=25

[2511.21689] ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration
URL: https://research.nvidia.com/labs/lpr/ToolOrchestra/
Summary: Large language models are powerful generalists, yet solving deep and complex problems such as those of the Humanity's Last Exam (HLE) remains both conceptually challenging and computationally expensive. We show that small orchestrators managing other models and a variety of tools can both push the upper bound of intelligence and improve efficiency …

[2512.02038] Deep Research: A Systematic Survey
URL: https://deep-research-survey.github.io/
Summary: Large language models (LLMs) have rapidly evolved from text generators into powerful problem solvers. Yet, many open tasks demand critical thinking, multi-source, and verifiable outputs, which are beyond single-shot prompting or standard retrieval-augmented generation. Recently, numerous studies have explored Deep Research (DR), which aims to combi…

[2512.12967] QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management
Summary: We introduce QwenLong-L1.5, a model that achieves superior long-context reasoning capabilities through systematic post-training innovations. The key technical breakthroughs of QwenLong-L1.5 are as follows: (1) Long-Context Data Synthesis Pipeline: We develop a systematic synthesis framework that generates challenging reasoning tasks requiring multi…

[2512.02556] DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
Summary: We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance. The key technical breakthroughs of DeepSeek-V3.2 are as follows: (1) DeepSeek Sparse Attention (DSA): We introduce DSA, an efficient attention mechanism that substantially reduces computational complexity while preserving…

[2511.18538] From Code Foundation Models to Agents and Applications: A Practical Guide to Code Intelligence
Summary: Large language models (LLMs) have fundamentally transformed automated software development by enabling direct translation of natural language descriptions into functional code, driving commercial adoption through tools like Github Copilot (Microsoft), Cursor (Anysphere), Trae (ByteDance), and Claude Code (Anthropic). While the field has evolved dra…

[2512.07461] Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning
URL: https://bigai-nlco.github.io/Native-Parallel-Reasoner/
Summary: We introduce Native Parallel Reasoner (NPR), a teacher-free framework that enables Large Language Models (LLMs) to self-evolve genuine parallel reasoning capabilities. NPR transforms the model from sequential emulation to native parallel cognition through three key innovations: 1) a self-distilled progressive training paradigm that transitions from…

[2512.16676] DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI
URL: https://github.com/OpenDCAI/DataFlow
Summary: The rapidly growing demand for high-quality data in Large Language Models (LLMs) has intensified the need for scalable, reliable, and semantically rich data preparation pipelines. However, current practices remain dominated by ad-hoc scripts and loosely specified workflows, which lack principled abstractions, hinder reproducibility, and offer limit…

[2512.04987] Nex-N1: Agentic Models Trained via a Unified Ecosystem for Large-Scale Environment Construction
Summary: The evolution of Large Language Models (LLMs) from passive responders to autonomous agents necessitates a fundamental shift in learning paradigms -- from static imitation to incentive-driven decision making. However, this transition is significantly impeded by the lack of scalable infrastructure capable of constructing high-quality interaction sign…

[2512.01374] Stabilizing Reinforcement Learning with LLMs: Formulation and Practices
Summary: This paper proposes a novel formulation for reinforcement learning (RL) with large language models, explaining why and under what conditions the true sequence-level reward can be optimized via a surrogate token-level objective in policy gradient methods such as REINFORCE. Specifically, through a first-order approximation, we show that this surrogat…

[2512.20491] Step-DeepResearch Technical Report
Summary: As LLMs shift toward autonomous agents, Deep Research has emerged as a pivotal metric. However, existing academic benchmarks like BrowseComp often fail to meet real-world demands for open-ended research, which requires robust skills in intent recognition, long-horizon decision-making, and cross-source verification. To address this, we introduce Ste…

[2512.19673] Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies
Summary: Existing reinforcement learning (RL) approaches treat large language models (LLMs) as a single unified policy, overlooking their internal mechanisms. Understanding how policy evolves across layers and modules is therefore crucial for enabling more targeted optimization and raveling out complex reasoning mechanisms. In this paper, we decompose the l…

[2512.15745] LLaDA2.0: Scaling Up Diffusion Language Models to 100B
Summary: This paper presents LLaDA2.0 -- a tuple of discrete diffusion large language models (dLLM) scaling up to 100B total parameters through systematic conversion from auto-regressive (AR) models -- establishing a new paradigm for frontier-scale deployment. Instead of costly training from scratch, LLaDA2.0 upholds knowledge inheritance, progressive adapt…

[2512.16969] Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows
URL: https://internscience.github.io/SGI-Page/
Summary: Despite advances in scientific AI, a coherent framework for Scientific General Intelligence (SGI)-the ability to autonomously conceive, investigate, and reason across scientific domains-remains lacking. We present an operational SGI definition grounded in the Practical Inquiry Model (PIM: Deliberation, Conception, Action, Perception) and operationa…

[2511.22570] DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning
Summary: Large language models have made significant progress in mathematical reasoning, which serves as an important testbed for AI and could impact scientific research if further advanced. By scaling reasoning with reinforcement learning that rewards correct final answers, LLMs have improved from poor performance to saturating quantitative reasoning compe…

[2511.21631] Qwen3-VL Technical Report
URL: https://chat.qwen.ai
Summary: We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. The model family includes both dense (2B/4B/8B/32B) and mixture-of-exp…

[2512.13564] Memory in the Age of AI Agents
Summary: Memory has emerged, and will continue to remain, a core capability of foundation model-based agents. As research on agent memory rapidly expands and attracts unprecedented attention, the field has also become increasingly fragmented. Existing works that fall under the umbrella of agent memory often differ substantially in their motivations, impleme…

[2512.20605] Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning
Summary: Large-scale autoregressive models pretrained on next-token prediction and finetuned with reinforcement learning (RL) have achieved unprecedented success on many problem domains. During RL, these models explore by generating new outputs, one token at a time. However, sampling actions token-by-token can result in highly inefficient learning, particul…

[2512.10430] T-pro 2.0: An Efficient Russian Hybrid-Reasoning Model and Playground
URL: https://t-pro-2-0.streamlit.app
Summary: We introduce T-pro 2.0, an open-weight Russian LLM for hybrid reasoning and efficient inference. The model supports direct answering and reasoning-trace generation, using a Cyrillic-dense tokenizer and an adapted EAGLE speculative-decoding pipeline to reduce latency. To enable reproducible and extensible research, we release the model weights, the …

[2512.17532] Robust-R1: Degradation-Aware Reasoning for Robust Visual Understanding
URL: https://jqt.me/index.html
Summary: Multimodal Large Language Models struggle to maintain reliable performance under extreme real-world visual degradations, which impede their practical robustness. Existing robust MLLMs predominantly rely on implicit training/adaptation that focuses solely on visual encoder generalization, suffering from limited interpretability and isolated optimiza…

[2512.16301] Adaptation of Agentic AI
Summary: Cutting-edge agentic AI systems are built on foundation models that can be adapted to plan, reason, and interact with external tools to perform increasingly complex and specialized tasks. As these systems grow in capability and scope, adaptation becomes a central mechanism for improving performance, reliability, and generalization. In this paper, w…

[2512.15431] Step-GUI Technical Report
URL: https://opengelab.github.io/
Summary: Recent advances in multimodal large language models unlock unprecedented opportunities for GUI automation. However, a fundamental challenge remains: how to efficiently acquire high-quality training data while maintaining annotation reliability? We introduce a self-evolving training pipeline powered by the Calibrated Step Reward System, which conver…

[2512.02589] PaperDebugger: A Plugin-Based Multi-Agent System for In-Editor Academic Writing, Review, and Editing
URL: https://www.paperdebugger.com/
Summary: Large language models are increasingly embedded into academic writing workflows, yet existing assistants remain external to the editor, preventing deep interaction with document state, structure, and revision history. This separation makes it impossible to support agentic, context-aware operations directly within LaTeX editors such as Overleaf. We …

[2512.17220] Mindscape-Aware Retrieval Augmented Generation for Improved Long Context Understanding
Summary: Humans understand long and complex texts by relying on a holistic semantic representation of the content. This global view helps organize prior knowledge, interpret new information, and integrate evidence dispersed across a document, as revealed by the Mindscape-Aware Capability of humans in psychology. Current Retrieval-Augmented Generation (RAG) …

[2512.04324] DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle
URL: https://da-comp.github.io/
Summary: Real-world enterprise data intelligence workflows encompass data engineering that turns raw sources into analytical-ready tables and data analysis that convert those tables into decision-oriented insights. We introduce DAComp, a benchmark of 210 tasks that mirrors these complex workflows. Data engineering (DE) tasks require repository-level enginee…

[2512.23447] Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss
Summary: Mixture-of-Experts (MoE) models lack explicit constraints to ensure the router's decisions align well with the experts' capabilities, which ultimately limits model performance. To address this, we propose expert-router coupling (ERC) loss, a lightweight auxiliary loss that tightly couples the router's decisions with expert capabilities. Our approac…

------------------------------------------------------------------------------------------
Cluster 3 | size=13

[2512.13604] LongVie 2: Multimodal Controllable Ultra-Long Video World Model
URL: https://vchitect.github.io/LongVie2-project/
Summary: Building video world models upon pretrained video generation systems represents an important yet challenging step toward general spatiotemporal intelligence. A world model should possess three essential properties: controllability, long-term visual quality, and temporal consistency. To this end, we take a progressive approach-first enhancing contro…

[2512.23576] LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation
Summary: Real-time video generation via diffusion is essential for building general-purpose multimodal interactive AI systems. However, the simultaneous denoising of all video frames with bidirectional attention via an iterative process in diffusion models prevents real-time interaction. While existing distillation methods can make the model autoregressive …

[2512.14691] MMGR: Multi-Modal Generative Reasoning
URL: https://zefan-cai.github.io/MMGR.github.io/
Summary: Video foundation models generate visually realistic and temporally coherent content, but their reliability as world simulators depends on whether they capture physical, logical, and spatial constraints. Existing metrics such as Frechet Video Distance (FVD) emphasize perceptual quality and overlook reasoning failures, including violations of causali…

[2512.14614] WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling
URL: https://3d-models.hunyuan.tencent.com/world/
Summary: This paper presents WorldPlay, a streaming video diffusion model that enables real-time, interactive world modeling with long-term geometric consistency, resolving the trade-off between speed and memory that limits current methods. WorldPlay draws power from three key innovations. 1) We use a Dual Action Representation to enable robust action contr…

[2512.08765] Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance
URL: https://wan-move.github.io/
Summary: We present Wan-Move, a simple and scalable framework that brings motion control to video generative models. Existing motion-controllable methods typically suffer from coarse control granularity and limited scalability, leaving their outputs insufficient for practical use. We narrow this gap by achieving precise and high-quality motion control. Our …

[2512.20619] SemanticGen: Video Generation in Semantic Space
URL: https://jianhongbai.github.io/SemanticGen/
Summary: State-of-the-art video generative models typically learn the distribution of video latents in the VAE space and map them to pixels using a VAE decoder. While this approach can generate high-quality videos, it suffers from slow convergence and is computationally expensive when generating long videos. In this paper, we introduce SemanticGen, a novel …

[2512.01816] Envision: Benchmarking Unified Understanding & Generation for Causal World Process Insights
URL: https://opendatalab-raiser.github.io/Envision/
Summary: Current multimodal models aim to transcend the limitations of single-modality representations by unifying understanding and generation, often using text-to-image (T2I) tasks to calibrate semantic consistency. However, their reliance on static, single-image generation in training and evaluation leads to overfitting to static pattern matching and sem…

[2511.20785] LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling
URL: https://evolvinglmms-lab.github.io/LongVT/
Summary: Large multimodal models (LMMs) have shown great potential for video reasoning with textual Chain-of-Thought. However, they remain vulnerable to hallucinations, especially when processing long-form videos where evidence is sparse and temporally dispersed. Inspired by how humans comprehend long videos - by first skimming globally and then examining r…

[2512.08269] EgoX: Egocentric Video Generation from a Single Exocentric Video
URL: https://keh0t0.github.io/EgoX/
Summary: Egocentric perception enables humans to experience and understand the world directly from their own point of view. Translating exocentric (third-person) videos into egocentric (first-person) videos opens up new possibilities for immersive understanding but remains highly challenging due to extreme camera pose variations and minimal view overlap. Th…

[2512.13281] Video Reality Test: Can AI-Generated ASMR Videos fool VLMs and Humans?
URL: https://video-reality-test.github.io/
Summary: Recent advances in video generation have produced vivid content that are often indistinguishable from real videos, making AI-generated video detection an emerging societal challenge. Prior AIGC detection benchmarks mostly evaluate video without audio, target broad narrative domains, and focus on classification solely. Yet it remains unclear whether…

[2512.09363] StereoWorld: Geometry-Aware Monocular-to-Stereo Video Generation
URL: https://ke-xing.github.io/StereoWorld/
Summary: The growing adoption of XR devices has fueled strong demand for high-quality stereo video, yet its production remains costly and artifact-prone. To address this challenge, we present StereoWorld, an end-to-end framework that repurposes a pretrained video generator for high-fidelity monocular-to-stereo video generation. Our framework jointly conditi…

[2512.03041] MultiShotMaster: A Controllable Multi-Shot Video Generation Framework
URL: https://qinghew.github.io/MultiShotMaster/
Summary: Current video generation techniques excel at single-shot clips but struggle to produce narrative multi-shot videos, which require flexible shot arrangement, coherent narrative, and controllability beyond text prompts. To tackle these challenges, we propose MultiShotMaster, a framework for highly controllable multi-shot video generation. We extend a…

[2512.16793] PhysBrain: Human Egocentric Data as a Bridge from Vision Language Models to Physical Intelligence
URL: https://zgc-embodyai.github.io/PhysBrain/
Summary: Robotic generalization relies on physical intelligence: the ability to reason about state changes, contact-rich interactions, and long-horizon planning under egocentric perception and action. However, most VLMs are trained primarily on third-person data, creating a fundamental viewpoint mismatch for humanoid robots. Scaling robot egocentric data co…

------------------------------------------------------------------------------------------
Cluster 2 | size=8

[2511.22699] Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer
URL: https://tongyi-mai.github.io/Z-Image-blog/
Summary: The landscape of high-performance image generation models is currently dominated by proprietary systems, such as Nano Banana Pro and Seedream 4.0. Leading open-source alternatives, including Qwen-Image, Hunyuan-Image-3.0 and FLUX.2, are characterized by massive parameter counts (20B to 80B), making them impractical for inference, and fine-tuning on…

[2512.04677] Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length
URL: https://liveavatar.github.io/
Summary: Existing diffusion-based video generation methods are fundamentally constrained by sequential computation and long-horizon inconsistency, limiting their practical adoption in real-time, streaming audio-driven avatar synthesis. We present Live Avatar, an algorithm-system co-designed framework that enables efficient, high-fidelity, and infinite-lengt…

[2512.05150] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows
URL: https://zhenglin-cheng.com/twinflow
Summary: Recent advances in large multi-modal generative models have demonstrated impressive capabilities in multi-modal generation, including image and video generation. These models are typically built upon multi-step frameworks like diffusion and flow matching, which inherently limits their inference efficiency (requiring 40-100 Number of Function Evalua…

[2512.13687] Towards Scalable Pre-training of Visual Tokenizers for Generation
Summary: The quality of the latent space in visual tokenizers (e.g., VAEs) is crucial for modern generative models. However, the standard reconstruction-based training paradigm produces a latent space that is biased towards low-level information, leading to a foundation flaw: better pixel-level accuracy does not lead to higher-quality generation. This impli…

[2512.16093] TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times
URL: https://github.com/thu-ml/TurboDiffusion
Summary: We introduce TurboDiffusion, a video generation acceleration framework that can speed up end-to-end diffusion generation by 100-200x while maintaining video quality. TurboDiffusion mainly relies on several components for acceleration: (1) Attention acceleration: TurboDiffusion uses low-bit SageAttention and trainable Sparse-Linear Attention (SLA) t…

[2512.17504] InsertAnywhere: Bridging 4D Scene Geometry and Diffusion Models for Realistic Video Object Insertion
URL: https://myyzzzoooo.github.io/InsertAnywhere/
Summary: Recent advances in diffusion-based video generation have opened new possibilities for controllable video editing, yet realistic video object insertion (VOI) remains challenging due to limited 4D scene understanding and inadequate handling of occlusion and lighting effects. We present InsertAnywhere, a new VOI framework that achieves geometrically c…

[2512.15603] Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition
Summary: Recent visual generative models often struggle with consistency during image editing due to the entangled nature of raster images, where all visual content is fused into a single canvas. In contrast, professional design tools employ layered representations, allowing isolated edits while preserving consistency. Motivated by this, we propose Qwen-Ima…

[2512.13586] ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding
Summary: Autoregressive models (ARMs) are hindered by slow sequential inference. While masked diffusion models (MDMs) offer a parallel alternative, they suffer from critical drawbacks: high computational overhead from precluding Key-Value (KV) caching, and incoherent generation arising from learning dependencies over an intractable space of token combinatio…

------------------------------------------------------------------------------------------
Cluster 1 | size=4

[2512.02014] TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models
URL: https://tuna-ai.org/
Summary: Unified multimodal models (UMMs) aim to jointly perform multimodal understanding and generation within a single framework. We present TUNA, a native UMM that builds a unified continuous visual representation by cascading a VAE encoder with a representation encoder. This unified representation space allows end-to-end processing of images and videos …

[2512.19693] The Prism Hypothesis: Harmonizing Semantic and Pixel Representations via Unified Autoencoding
Summary: Deep representations across modalities are inherently intertwined. In this paper, we systematically analyze the spectral characteristics of various semantic and pixel encoders. Interestingly, our study uncovers a highly inspiring and rarely explored correspondence between an encoder's feature spectrum and its functional role: semantic encoders prim…

[2512.16922] Next-Embedding Prediction Makes Strong Vision Learners
URL: https://sihanxu.me/nepa
Summary: Inspired by the success of generative pretraining in natural language, we ask whether the same principles can yield strong self-supervised visual learners. Instead of training models to output features for downstream use, we train them to generate embeddings to perform predictive tasks directly. This work explores such a shift from learning represe…

[2512.16776] Kling-Omni Technical Report
Summary: We present Kling-Omni, a generalist generative framework designed to synthesize high-fidelity videos directly from multimodal visual language inputs. Adopting an end-to-end perspective, Kling-Omni bridges the functional separation among diverse video generation, editing, and intelligent reasoning tasks, integrating them into a holistic system. Unli…
