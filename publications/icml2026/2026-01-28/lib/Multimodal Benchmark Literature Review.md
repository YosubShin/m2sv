# **Multimodal Spatial Intelligence: A Comprehensive Review of Map-View Reasoning, Geo-Localization, and Benchmark Design**

## **1\. Introduction: The Imperative for Allocentric-Egocentric Alignment**

The evolution of Multimodal Large Language Models (MLLMs) has precipitated a paradigm shift in computer vision and artificial intelligence, moving the field from static, classification-based tasks toward dynamic, reasoning-intensive challenges. While contemporary models such as GPT-4o, Gemini, and their open-weight counterparts have demonstrated profound capabilities in general visual question answering (VQA), optical character recognition (OCR), and semantic scene understanding, a critical cognitive gap remains: the ability to align allocentric (world-centered) and egocentric (viewer-centered) spatial representations. This dissonance is most acutely observed in navigation and orientation tasks where an agent must reconcile a top-down orthogonal map view with a ground-level perspective view to determine its heading or location. The proposed benchmark, "map to street view," which tasks models with identifying directional orientation (a, b, c, d) at an intersection given a map and a street view image, addresses this precise theoretical and practical lacuna.  
This report provides an exhaustive literature review designed to contextualize the "map to street view" benchmark within the broader tapestry of spatial AI research. It synthesizes developments from diverse yet converging fields: spatial mental modeling and cognitive mapping, cross-view geo-localization (CVGL), planetary-scale coordinate prediction, symbolic geometric reasoning, and the theoretical science of benchmark design in an era of rapid model saturation. By examining the methodologies and limitations of existing benchmarks—ranging from the mental simulation tasks of **MindCube** to the navigational challenges of **Touchdown** and the global localization of **PIGEON**—we establish the necessity for a specialized evaluation framework that rigorously tests fine-grained directional reasoning at the intersection of map reading and visual perception.

### **1.1 The Cognitive Gap in Current Architectures**

At the heart of the proposed benchmark lies the challenge of spatial transduction. Human navigation relies on the continuous, almost subconscious integration of an internal "cognitive map"—an allocentric representation of the environment—with immediate sensory perception, which is inherently egocentric.1 Cognitive science suggests that this process involves "mental rotation" and "perspective taking," mechanisms that allow a human navigator to look at a 2D map, mentally rotate it to align with the forward visual field, and deduce that "left on the map" corresponds to "straight ahead" in the real world depending on their orientation.  
Current Vision-Language Models (VLMs), despite their massive parameter counts and extensive training on internet-scale data, often treat spatial queries as pattern-matching retrieval tasks rather than geometric reasoning problems. Research utilizing the **MindCube** benchmark indicates that even state-of-the-art proprietary models achieve performance levels barely exceeding random guessing on tasks requiring complex spatial mental modeling.1 This suggests that the "physics engine" or "geometric core" required to perform mental rotations and align disparate coordinate systems is either absent or severely underdeveloped in current transformer-based architectures. The "map to street view" task serves as a direct probe of this specific deficiency, isolating the variable of orientation in a way that generic geo-localization or VQA tasks do not.

### **1.2 The Trajectory of Spatial Benchmarking**

The historical trajectory of spatial benchmarking has moved from simple object localization (e.g., bounding boxes) to semantic relation detection (e.g., Visual Genome) and now towards embodied and reasoning-based tasks. Early benchmarks focused on "what" and "where" in a static 2D plane. The new frontier, represented by benchmarks like **MindCube**, **MapBench**, and the proposed task, asks "how" spaces relate to one another across different conceptual domains (symbolic map vs. photorealistic view). This shift is driven by the realization that high performance on static datasets does not translate to robust spatial intelligence in real-world scenarios, such as autonomous driving or robotic navigation, where the agent must actively interpret a map to make decisions.4  
Furthermore, the rapid saturation of general-purpose benchmarks like **MMLU** (Massive Multitask Language Understanding) and **GSM8K** (Grade School Math) has created a crisis of evaluation in the AI community. When models routinely score above 90% on standard tests, those tests lose their discriminatory power.5 The literature emphasizes the urgent need for "unsaturated" benchmarks—tests that are sufficiently complex, novel, or nuanced to expose the remaining limitations of frontier models. The "map to street view" benchmark, with its requirement for cross-modal geometric alignment and mental rotation, fits the criteria for an unsaturated test that can drive the next wave of innovation in spatial AI.7

## **2\. Spatial Reasoning and Mental Modeling**

To understand the theoretical basis for the "map to street view" task, one must first examine the state of research in spatial mental modeling. This domain moves beyond identifying objects to understanding the structural and geometric relationships between them, especially when those relationships are not immediately visible.

### **2.1 The MindCube Benchmark: A Precursor in Cognitive Simulation**

The most significant recent contribution to this domain is **MindCube**, a comprehensive benchmark designed to evaluate how well VLMs can form robust spatial mental models from limited visual views.1 MindCube challenges models not just to see, but to imagine—to perform "cognitive mapping" (representing positions), "perspective-taking" (understanding orientations), and "mental simulation" (predicting the dynamics of "what-if" movements).9

#### **2.1.1 Dataset Composition and Task Taxonomy**

MindCube comprises over 21,000 spatial reasoning questions across 3,268 images, organized into a taxonomy that closely parallels the cognitive requirements of reading a map.9 The tasks are divided into categories such as:

* **Perspective Taking:** Inferring spatial relations from a viewpoint different from the camera's.  
* **Mental Rotation:** Predicting the appearance of a scene or object after a rotational transformation.  
* **Look-Around Dynamics:** Hypothetical scenarios where the model must predict what would be visible if the observer turned left, right, or moved forward.11

This taxonomy is directly relevant to the "map to street view" proposal. Guessing a direction at an intersection is fundamentally a "Perspective Taking" and "Mental Rotation" task. The model must essentially place itself inside the map (perspective taking) and rotate its mental view to match the street image (mental rotation).

#### **2.1.2 Performance Metrics and the "Reasoning Gap"**

Quantitative analysis from MindCube studies reveals a stark "reasoning gap." While models perform adequately on simple object recognition, their ability to manipulate these spatial representations is poor. For example, on the spatial mental modeling benchmark, leading proprietary models like GPT-4o achieve only \~38-44% overall accuracy, a figure that borders on random guessing for 4-way multiple-choice questions.1 This performance floor is critical evidence supporting the need for the new benchmark; it proves that the capability to align and rotate spatial views is far from solved.  
Specifically, the literature highlights that relying on a single viewpoint leads to "partial observation" errors. Models fail to "complete" the scene in their "mind's eye." In the context of map reading, this implies that models struggle to infer the existence of landmarks that might be just out of frame or occluded, a skill necessary for robust orientation guessing.1

#### **2.1.3 The "Map-Then-Reason" Paradigm**

To mitigate these failures, MindCube researchers introduced the "map-then-reason" paradigm. This approach involves a two-stage process: first, the model is supervised to generate an explicit "cognitive map" or structural representation of the scene; second, it reasons over this map to answer the question.2 Experiments show that this method significantly boosts accuracy, in some cases raising performance from \~37% to over 60%.2  
**Implication for Proposed Benchmark:** This finding suggests that the "map to street view" task might benefit from, or even require, an intermediate step where the model explicitly aligns features (e.g., "The red building in the image corresponds to this gray square on the map"). Future iterations of the benchmark could include "reasoning traces" or auxiliary supervision that asks the model to identify these correspondences before guessing the direction, thereby testing the validity of its internal cognitive map.

### **2.2 Symbolic and Geometric Reasoning**

While MindCube focuses on 3D and photorealistic environments, another stream of research evaluates spatial reasoning through the lens of mathematics and planar geometry. Benchmarks such as **DynaMath**, **MathVista**, and **Geometry3K** test a model's ability to interpret diagrams, which are essentially simplified, symbolic maps.13

#### **2.2.1 DynaMath and Reasoning Robustness**

**DynaMath** introduces the critical concept of "reasoning robustness." Rather than using static questions that can be memorized, DynaMath generates dynamic variants of seed questions—changing visual values, rotating figures, or altering graph shapes—to test if the model truly grasps the underlying geometric principles.13

* **Relevance:** A map is a dynamic geometric system. The relationship between a street intersection and its map representation is governed by strict geometric rules (topology, angle). The "robustness" metrics from DynaMath—checking if a model stays consistent when the map is rotated or the street view changes slightly—are highly applicable. If a model can identify North, it should logically identify South given the opposite view; DynaMath’s methodology suggests testing this consistency explicitly.13

#### **2.2.2 MathVista and Integrated Reasoning**

**MathVista** combines various datasets to evaluate algebraic, arithmetic, and geometric reasoning in visual contexts.14 It highlights that current VLMs often fail at "multimodal interleaved" tasks—where text and image must be processed together to solve a logical puzzle. Map reading is the quintessential interleaved task: the text (street names, cardinal directions) and the image (road layout, buildings) are mutually dependent. MathVista's low baselines on geometry problems further reinforce the difficulty of the symbolic interpretation required for map reading.14

## **3\. Cross-View Geo-Localization (CVGL) and Orientation**

The technical core of the proposed benchmark—matching a ground view to an overhead view—is the subject of the field known as Cross-View Geo-Localization (CVGL). This mature field provides the foundational datasets, metrics, and architectures upon which the new benchmark will inevitably build.

### **3.1 The Evolution of CVGL Benchmarks**

The CVGL field has progressed through several generations of benchmarks, each attempting to address the limitations of the last.

#### **3.1.1 First Generation: CVUSA and CVACT**

**CVUSA** (Cross-View USA) and **CVACT** (Cross-View ACT) established the baseline for this task.16 These datasets consist of massive pairs of ground-level panoramic images and corresponding satellite image patches.

* **Methodology:** The standard approach is "one-to-one retrieval." The model learns a shared embedding space where the feature vector of the ground image is close to the feature vector of the matching satellite image.18  
* **Limitation \- Saturation:** Recent literature argues that CVUSA and CVACT are approaching saturation. Top models now achieve Recall@1 scores exceeding 90% in some settings.16  
* **Limitation \- Alignment Assumptions:** Crucially, these benchmarks often simplify the orientation problem. In many standard splits, the ground-level panoramas are north-aligned, removing the need for the model to estimate orientation. The proposed "map to street view" benchmark specifically reintroduces this difficulty, making it a distinct and necessary evolution from standard CVGL.17

#### **3.1.2 Second Generation: VIGOR and University-1652**

To address the artificiality of perfect alignment, **VIGOR** (Cross-View Image Geo-localization beyond One-to-one Retrieval) introduced a setting where the ground-view image is not perfectly centered in the satellite image.18 This forces the model to reason about spatial offsets. **University-1652** expanded the modality to include **drone imagery**.20 This is pivotal because drone views represent an intermediate perspective between the orthogonal satellite view and the frontal street view. The literature on University-1652 demonstrates that models trained with this intermediate view generalize better, suggesting that "map to street view" reasoning might be aided by understanding oblique aerial perspectives.20

### **3.2 Orientation Estimation: The "Hidden" Variable**

Within CVGL, a sub-field specifically targets **Orientation Estimation**—the exact task of the user's benchmark.

#### **3.2.1 OriLoc: Conquering the Limited Field of View**

The **OriLoc** framework addresses the "Limited Field of View" (FOV) problem.19 While benchmarks like CVUSA use 360-degree panoramas, real-world street views (and likely the images in the proposed benchmark) are often limited-FOV crops (e.g., a 90-degree view from a dashboard camera).

* **Mechanism:** OriLoc employs a "sliding window" convolution strategy. It slides the limited-FOV feature map over the satellite embedding (effectively rotating it) to find the angle of maximum correlation.  
* **Metric:** Instead of just retrieval accuracy, OriLoc and similar papers evaluate **Angular Error** (in degrees). This is a critical recommendation for the new benchmark: rather than just A/B/C/D classification, providing a fine-grained angular error metric allows for more nuanced model comparison.17

#### **3.2.2 Mathematical Techniques for Orientation**

Literature reviews distinct approaches to determining direction:

1. **Classification/Binning:** Dividing the compass into discrete bins (e.g., 4 bins for N/E/S/W, or 360 bins for degrees) and treating it as a classification problem.25  
2. **Polar Transformation:** Many successful CVGL models use a polar transform to warp the satellite image into a pseudo-panorama, making the matching process a simple linear alignment. However, this often assumes the center of the satellite image is the camera location. The proposed benchmark's "intersection" setting complicates this if the camera is not perfectly centered.17  
3. **Activation Matching:** Methods utilizing Grad-CAM to visualize which parts of the satellite image trigger the street view match. These techniques reveal that models often focus on road topology and building corners—features that are orientation-dependent.17

## **4\. Planetary-Scale Coordinate Prediction: The "Where" vs. "Which Way"**

While CVGL focuses on matching a specific pair, "Planetary-Scale Coordinate Prediction" models like **PIGEON** attempt to guess the location of an image from the entire world. This literature provides insight into the visual features models use for geolocation, which are often the same features needed for orientation (e.g., sun position, driving side).

### **4.1 PIGEON and Semantic Geocells**

**PIGEON** (Predicting Image Geolocations) represents the state-of-the-art in this domain.27

* **Semantic Geocells:** Unlike early models like **PlaNet** 29 that used arbitrary grid cells, PIGEON clusters the world into "semantic geocells" based on density and political boundaries.  
* **Contrastive Learning:** PIGEON leverages a multi-task objective that combines location prediction with visual attribute classification (e.g., "urban," "forest"). This auxiliary task helps the model learn robust visual representations.28  
* **Connection to Orientation:** Although PIGEON's primary output is coordinates (latitude/longitude), the literature notes that it implicitly learns orientation cues. For instance, knowing that shadows fall to the south implies the image is in the northern hemisphere and helps constrain the camera heading relative to the sun. The "map to street view" benchmark could explicitly test this by including "sun compass" tasks.27

### **4.2 The GeoGuessr Baseline**

PIGEON was benchmarked against top human players of *GeoGuessr*. In location guessing, AI has achieved parity with experts. However, in fine-grained orientation and "pinpointing" (finding the exact meter-perfect location on a map), humans often still excel due to their ability to read text and cross-reference map symbols.28 This validates the "map to street view" task as a necessary "next step" challenge where AI dominance is not yet assured.

## **5\. Map Understanding and Navigation**

Beyond matching pixels, there is the problem of "reading" the map as a semantic document. This is covered by MapQA and MapBench.

### **5.1 MapQA: The VQA of Cartography**

**MapQA** benchmarks the ability of models to answer questions based on map images.31

* **Symbol Grounding:** Questions like "What is the nearest clinic?" or "Which state is to the west?" require the model to ground text labels to spatial locations.  
* **Limitations:** Current VLMs struggle with these tasks, particularly with "multi-hop" spatial reasoning (e.g., finding a location, then finding what is next to it). The literature indicates that retrieval-based approaches (finding the text in the image) often outperform generative reasoning for these tasks, suggesting that pure "reasoning" is a bottleneck.33  
* **Implication:** The proposed benchmark should ensure that the map component is not just a visual texture but contains semantic information (road shapes, maybe labels) that must be "read" to solve the orientation puzzle.

### **5.2 Touchdown: Embodied Navigation**

**Touchdown** brings NLP into the mix. An agent navigates a Google Street View environment to find a hidden object ("Touchdown") based on instructions.35

* **SDR (Spatial Description Resolution):** This sub-task is the inverse of the user's proposal. Instead of "Image \+ Map \-\> Orientation," it is "Text Description \-\> Location in Panorama."  
* **Visual-Linguistic Alignment:** Touchdown proves that "grounding" spatial prepositions (left, right, behind) in complex visual environments is extremely difficult. The low success rates of baseline models on Touchdown (often \<20% for precise placement) reinforce that the "map to street view" task will be a rigorous challenge.36

### **5.3 MapBench and Robustness**

**MapBench** focuses on the robustness of map construction algorithms.4

* **Sensor Corruption:** A key finding is that performance degrades catastrophically with sensor noise (rain, fog). A robust benchmark should essentially include a "hard mode" with degraded imagery to prevent models from relying on fragile high-frequency texture cues.  
* **Route Planning:** MapBench also tests path planning. The inability of MLLMs to plan valid routes without "short-circuiting" obstacles suggests that their internal representation of the map's topology is flawed. The "map to street view" task tests this topology understanding from a different angle (orientation rather than pathfinding).4

## **6\. Benchmarking Theory: The Crisis of Saturation**

The user's request highlights the importance of "unsaturated tests." This is a dominant theme in 2024-2025 AI literature.

### **6.1 The Saturation of General Benchmarks**

Benchmarks that were once considered grand challenges, such as **MMLU** (general knowledge) and **GSM8K** (math), have been effectively "solved" by frontier models, with scores compressing into the 90-99% range.5

* **Loss of Signal:** When a benchmark saturates, it loses its ability to distinguish between models. A 98% score vs. a 99% score may just represent noise or training data contamination rather than a real difference in intelligence.7  
* **Contamination:** "Memorization" is a huge issue. Models often memorize the test set of famous benchmarks.

### **6.2 Designing for the Future: FrontierMath and Beyond**

New benchmarks like **FrontierMath** are designed to be impossibly hard for current models (success rates \<2%) to provide a runway for future progress.8

* **Complexity:** They require multi-step, novel reasoning that cannot be pattern-matched.  
* **Unsaturated Nature:** The "map to street view" task fits this paradigm. Because it relies on the *relationship* between two inputs (map and image) rather than a single static answer, and because the geometric permutations are infinite, it is harder to "memorize" than a static fact.  
* **Human Baseline:** Effective unsaturated benchmarks often exhibit a large gap between human expert performance and AI performance. In spatial reasoning (MindCube), this gap is massive. Establishing a human baseline for "map to street view" (which is likely high for humans with map-reading training) will be crucial to validating its difficulty.2

## **7\. Comparative Analysis of Relevant Benchmarks**

To systematically situate the proposed "map to street view" benchmark, we compare it against the key benchmarks identified in the literature.

| Benchmark | Core Task | Input Modalities | Key Metric | Relevance | Gap Addressed by Proposal |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **MindCube** 1 | Spatial Mental Modeling | Images (Limited View) | Accuracy (MCQ) | **High** | Adds explicit map-reading to mental simulation. |
| **CVUSA/CVACT** 16 | Geo-Localization | Satellite \+ Street View | Recall@K | **Medium** | Shifts from retrieval to fine-grained orientation reasoning. |
| **OriLoc** 19 | Orientation Estimation | Satellite \+ Street View (Crop) | Angular Error | **Very High** | Formalizes the "orientation" task in a benchmark setting. |
| **PIGEON** 27 | Coordinate Prediction | Street View (Panorama) | Distance (km) | **Medium** | Focuses on local orientation vs. global coordinates. |
| **Touchdown** 35 | Embodied Navigation | Street View \+ Text | Success Rate | **High** | Isolates the "orientation" step from the full nav policy. |
| **MapQA** 31 | Map VQA | Map Image \+ Text | Accuracy | **Medium** | Integrates the street view perspective into map reading. |
| **MapBench** 4 | Map Construction/Nav | Map/Sensor Data | IOU / Path Score | **Medium** | Tests perception-to-map alignment in a static VQA format. |

## **8\. Conclusion and Strategic Recommendations**

The literature review confirms that the "map to street view" benchmark occupies a distinct and scientifically valuable niche. It lies at the intersection of **Cross-View Geo-Localization** (which provides the data and metrics) and **Spatial Mental Modeling** (which provides the cognitive theory). While benchmarks for both exist, few rigorously test the specific reasoning process of aligning a map's symbolic topology with a street view's perspective geometry to deduce orientation.  
Existing benchmarks are either saturated (CVUSA), focus on global coordinates rather than local orientation (PIGEON), or lack the map modality entirely (MindCube). The proposed benchmark addresses the "Allocentric-Egocentric Gap" that plagues current MLLMs, offering a "Frontier"-style challenge that is resistant to simple pattern matching and memorization.

### **8.1 Recommendations for Benchmark Design**

Based on the synthesis of methodologies from OriLoc, MindCube, and DynaMath, the following recommendations are made for the "map to street view" paper:

1. **Adopt Angular Error Metrics:** Do not rely solely on A/B/C/D classification. As seen in **OriLoc**, using Mean Absolute Error (MAE) in degrees provides a finer signal of progress and distinguishes between "180-degree errors" (complete confusion) and "10-degree errors" (minor misalignment).17  
2. **Incorporate Limited FOV:** Follow the **OriLoc** and **MindCube** findings that limited field-of-view images are harder and more realistic than panoramas. This forces the model to "hallucinate" or infer the missing context.1  
3. **Include "Reasoning Trace" Evaluation:** Inspired by the "map-then-reason" success in **MindCube**, the benchmark should ideally support evaluating the *intermediate* steps (e.g., "Identify the landmark in both views") to diagnose *why* models fail.2  
4. **Emphasize Robustness:** Drawing from **MapBench** and **DynaMath**, include "corrupted" or "dynamic" splits (e.g., night images, rotated maps) to ensure the model is learning invariant geometric rules rather than texture matching.4  
5. **Leverage Drone Views as Scaffolding:** Consider including a subset of **University-1652** style drone images as a "training scaffold" or an easier difficulty tier, bridging the gap between the hard map-to-street transfer.20

By integrating these elements, the "map to street view" benchmark can establish itself as a definitive test of spatial intelligence in the multimodal era.

#### **Works cited**

1. SpatialDreamer: Incentivizing Spatial Reasoning via Active Mental Imagery \- arXiv, accessed January 23, 2026, [https://arxiv.org/html/2512.07733v1](https://arxiv.org/html/2512.07733v1)  
2. Spatial Mental Modeling from Limited Views \- Emergent Mind, accessed January 23, 2026, [https://www.emergentmind.com/papers/2506.21458](https://www.emergentmind.com/papers/2506.21458)  
3. MindJourney: Test-Time Scaling with World Models for Spatial Reasoning | OpenReview, accessed January 23, 2026, [https://openreview.net/forum?id=L2W4wQsNkY](https://openreview.net/forum?id=L2W4wQsNkY)  
4. MapBench: Spatial Reasoning Benchmark \- Emergent Mind, accessed January 23, 2026, [https://www.emergentmind.com/topics/mapbench](https://www.emergentmind.com/topics/mapbench)  
5. AI Benchmarks Hit Saturation | Stanford HAI, accessed January 23, 2026, [https://hai.stanford.edu/news/ai-benchmarks-hit-saturation](https://hai.stanford.edu/news/ai-benchmarks-hit-saturation)  
6. Top 50 AI Model Benchmarks & Evaluation Metrics (2025 Guide) | Articles | O-mega, accessed January 23, 2026, [https://o-mega.ai/articles/top-50-ai-model-evals-full-list-of-benchmarks-october-2025](https://o-mega.ai/articles/top-50-ai-model-evals-full-list-of-benchmarks-october-2025)  
7. Mapping global dynamics of benchmark creation and saturation in artificial intelligence, accessed January 23, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9649641/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9649641/)  
8. AI Models Are Getting Smarter. New Tests Are Racing to Catch Up | TIME, accessed January 23, 2026, [https://time.com/7203729/ai-evaluations-safety/](https://time.com/7203729/ai-evaluations-safety/)  
9. MLL-Lab/MindCube · Datasets at Hugging Face, accessed January 23, 2026, [https://huggingface.co/datasets/MLL-Lab/MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube)  
10. \[2506.21458\] Spatial Mental Modeling from Limited Views \- arXiv, accessed January 23, 2026, [https://arxiv.org/abs/2506.21458](https://arxiv.org/abs/2506.21458)  
11. Spatial Mental Modeling from Limited Views | OpenReview, accessed January 23, 2026, [https://openreview.net/pdf/e0abae061388ee27ed3b39b215ba3c56eafe3531.pdf](https://openreview.net/pdf/e0abae061388ee27ed3b39b215ba3c56eafe3531.pdf)  
12. Spatial Mental Modeling from Limited Views \- ResearchGate, accessed January 23, 2026, [https://www.researchgate.net/publication/393066179\_Spatial\_Mental\_Modeling\_from\_Limited\_Views](https://www.researchgate.net/publication/393066179_Spatial_Mental_Modeling_from_Limited_Views)  
13. DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models | OpenReview, accessed January 23, 2026, [https://openreview.net/forum?id=VOAMTA8jKu](https://openreview.net/forum?id=VOAMTA8jKu)  
14. MathVista: Evaluating Math Reasoning in Visual Contexts, accessed January 23, 2026, [https://mathvista.github.io/](https://mathvista.github.io/)  
15. Self-Rewarding Vision-Language Model via Reasoning Decomposition \- arXiv, accessed January 23, 2026, [https://arxiv.org/html/2508.19652v1](https://arxiv.org/html/2508.19652v1)  
16. Advancing Cross-View Geo-Localization in Global Cities \- IEEE Xplore, accessed January 23, 2026, [https://ieeexplore.ieee.org/iel8/4609443/10766875/10758289.pdf](https://ieeexplore.ieee.org/iel8/4609443/10766875/10758289.pdf)  
17. Revisiting Street-to-Aerial View Image Geo-Localization and Orientation Estimation \- CVF Open Access, accessed January 23, 2026, [https://openaccess.thecvf.com/content/WACV2021/papers/Zhu\_Revisiting\_Street-to-Aerial\_View\_Image\_Geo-Localization\_and\_Orientation\_Estimation\_WACV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/WACV2021/papers/Zhu_Revisiting_Street-to-Aerial_View_Image_Geo-Localization_and_Orientation_Estimation_WACV_2021_paper.pdf)  
18. VIGOR: Cross-View Image Geo-Localization Beyond One-to-One Retrieval \- CVF Open Access, accessed January 23, 2026, [https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu\_VIGOR\_Cross-View\_Image\_Geo-Localization\_Beyond\_One-to-One\_Retrieval\_CVPR\_2021\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_VIGOR_Cross-View_Image_Geo-Localization_Beyond_One-to-One_Retrieval_CVPR_2021_paper.pdf)  
19. OriLoc: Unlimited-FoV and Orientation-Free Cross-View Geolocalization \- IEEE Xplore, accessed January 23, 2026, [http://ieeexplore.ieee.org/document/11037236/](http://ieeexplore.ieee.org/document/11037236/)  
20. \[2002.12186\] University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization \- arXiv, accessed January 23, 2026, [https://arxiv.org/abs/2002.12186](https://arxiv.org/abs/2002.12186)  
21. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization \- ResearchGate, accessed January 23, 2026, [https://www.researchgate.net/profile/Zhedong-Zheng-2/publication/346200144\_University-1652\_A\_Multi-view\_Multi-source\_Benchmark\_for\_Drone-based\_Geo-localization/links/61b80253fd2cbd72009b7ade/University-1652-A-Multi-view-Multi-source-Benchmark-for-Drone-based-Geo-localization.pdf](https://www.researchgate.net/profile/Zhedong-Zheng-2/publication/346200144_University-1652_A_Multi-view_Multi-source_Benchmark_for_Drone-based_Geo-localization/links/61b80253fd2cbd72009b7ade/University-1652-A-Multi-view-Multi-source-Benchmark-for-Drone-based-Geo-localization.pdf)  
22. 3D Positioning of Drones through Images \- PMC \- PubMed Central, accessed January 23, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11397779/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11397779/)  
23. OriLoc: Unlimited-FoV and Orientation-Free Cross-View Geolocalization \- IEEE Xplore, accessed January 23, 2026, [http://ieeexplore.ieee.org/iel8/4609443/10766875/11037236.pdf](http://ieeexplore.ieee.org/iel8/4609443/10766875/11037236.pdf)  
24. Uncertainty-Aware Vision-Based Metric Cross-View Geolocalization | Request PDF, accessed January 23, 2026, [https://www.researchgate.net/publication/373312588\_Uncertainty-Aware\_Vision-Based\_Metric\_Cross-View\_Geolocalization](https://www.researchgate.net/publication/373312588_Uncertainty-Aware_Vision-Based_Metric_Cross-View_Geolocalization)  
25. SpaGBOL: Spatial-Graph-Based Orientated Localisation \- CVF Open Access, accessed January 23, 2026, [https://openaccess.thecvf.com/content/WACV2025/papers/Shore\_SpaGBOL\_Spatial-Graph-Based\_Orientated\_Localisation\_WACV\_2025\_paper.pdf](https://openaccess.thecvf.com/content/WACV2025/papers/Shore_SpaGBOL_Spatial-Graph-Based_Orientated_Localisation_WACV_2025_paper.pdf)  
26. Where Am I Looking At? Joint Location and Orientation Estimation by Cross-View Matching | Request PDF \- ResearchGate, accessed January 23, 2026, [https://www.researchgate.net/publication/343461342\_Where\_Am\_I\_Looking\_At\_Joint\_Location\_and\_Orientation\_Estimation\_by\_Cross-View\_Matching](https://www.researchgate.net/publication/343461342_Where_Am_I_Looking_At_Joint_Location_and_Orientation_Estimation_by_Cross-View_Matching)  
27. PIGEON: Predicting Image Geolocations, accessed January 23, 2026, [https://lukashaas.github.io/PIGEON-CVPR24/](https://lukashaas.github.io/PIGEON-CVPR24/)  
28. PIGEON: Predicting Image Geolocations \- CVF Open Access, accessed January 23, 2026, [https://openaccess.thecvf.com/content/CVPR2024/papers/Haas\_PIGEON\_Predicting\_Image\_Geolocations\_CVPR\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Haas_PIGEON_Predicting_Image_Geolocations_CVPR_2024_paper.pdf)  
29. PlaNet \- Photo Geolocation with Convolutional Neural Networks | Request PDF \- ResearchGate, accessed January 23, 2026, [https://www.researchgate.net/publication/319770340\_PlaNet\_-\_Photo\_Geolocation\_with\_Convolutional\_Neural\_Networks](https://www.researchgate.net/publication/319770340_PlaNet_-_Photo_Geolocation_with_Convolutional_Neural_Networks)  
30. PIGEON: Predicting Image Geolocations \- DEV Community, accessed January 23, 2026, [https://dev.to/aimodels-fyi/pigeon-predicting-image-geolocations-1clc](https://dev.to/aimodels-fyi/pigeon-predicting-image-geolocations-1clc)  
31. MAPWise: Evaluating Vision-Language Models for Advanced Map Queries \- Cognitive Computation Group, accessed January 23, 2026, [https://cogcomp.seas.upenn.edu/papers/MRKSRG25.pdf](https://cogcomp.seas.upenn.edu/papers/MRKSRG25.pdf)  
32. MAPWise: Evaluating Vision-Language Models for Advanced Map Queries \- ACL Anthology, accessed January 23, 2026, [https://aclanthology.org/2025.naacl-long.473.pdf](https://aclanthology.org/2025.naacl-long.473.pdf)  
33. (PDF) MapQA: Open-domain Geospatial Question Answering on Map Data \- ResearchGate, accessed January 23, 2026, [https://www.researchgate.net/publication/389749241\_MapQA\_Open-domain\_Geospatial\_Question\_Answering\_on\_Map\_Data](https://www.researchgate.net/publication/389749241_MapQA_Open-domain_Geospatial_Question_Answering_on_Map_Data)  
34. \[2503.07871\] MapQA: Open-domain Geospatial Question Answering on Map Data \- arXiv, accessed January 23, 2026, [https://arxiv.org/abs/2503.07871](https://arxiv.org/abs/2503.07871)  
35. Retouchdown: Releasing Touchdown on StreetLearn as a Public Resource for Language Grounding Tasks in Street View \- Semantic Scholar, accessed January 23, 2026, [https://semanticscholar.org/paper/ce3b1b492cc9f41592930e98576089fc8c7c7060](https://semanticscholar.org/paper/ce3b1b492cc9f41592930e98576089fc8c7c7060)  
36. TOUCHDOWN: Natural Language Navigation and Spatial Reasoning in Visual Street Environments \- CVF Open Access, accessed January 23, 2026, [https://openaccess.thecvf.com/content\_CVPR\_2019/papers/Chen\_TOUCHDOWN\_Natural\_Language\_Navigation\_and\_Spatial\_Reasoning\_in\_Visual\_Street\_CVPR\_2019\_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_TOUCHDOWN_Natural_Language_Navigation_and_Spatial_Reasoning_in_Visual_Street_CVPR_2019_paper.pdf)  
37. Cornell Touchdown natural language navigation and spatial reasoning dataset. \- GitHub, accessed January 23, 2026, [https://github.com/lil-lab/touchdown](https://github.com/lil-lab/touchdown)  
38. Can Large Vision Language Models Read Maps like a Human? \- arXiv, accessed January 23, 2026, [https://arxiv.org/pdf/2503.14607](https://arxiv.org/pdf/2503.14607)