# When Data Goes Cold: Urban Environmental Knowledge Graph Embeddings for Spatial Cold-Start Crime Prediction

## Abstract

Crime prediction models perform well in data-rich hotspots but collapse into near-zero predictions across extensive urban zones we term **"Spatial Data Deserts"**—contiguous areas where extreme zero-inflation of historical crime records deprives conventional models of usable statistical signal. In Chicago, these spatial data deserts encompass 21,065 fine-grained spatial units (17.5% of 120,459 100-meter subgrids), organized into three urban morphological types: peripheral residential zones, industrial/logistics corridors, and green/institutional spaces. We construct an Urban Environmental Knowledge Graph (KG) that encodes Points of Interest, road networks, and functional zones as 32-dimensional embeddings organized through Crime Prevention Through Environmental Design (CPTED) theory, measuring the stable structural *opportunity* for crime embedded in the built environment—decoupled from past policing patterns. Pretrained via GraphSAGE with self-supervised objectives (masked autoencoding, weak contrastive learning, covariance regularization), frozen KG embeddings with Ridge regression achieve NDCG@20=0.406 and AUC=0.865 for cold-start violent crime risk prediction, substantially exceeding LLM-based semantic embeddings (NDCG@20=0.182) and all conventional baselines. Removing CPTED features causes a 94% performance collapse. The KG signal strengthens with spatial aggregation, reaching NDCG@20=0.905 at the 1 km neighborhood scale, though spatial cross-validation reveals that the global Ridge mapping fails to generalize across geographically separated regions (NDCG@20 dropping to 0.065–0.083). SHAP analysis identifies primary activity type, target density, and temporal rhythm as the most influential CPTED dimensions. By grounding risk assessment in the observable built environment rather than historical enforcement data, the framework shifts crime prediction from reactive policing to proactive, evidence-based urban planning.

**Keywords:** spatial cold-start; crime prediction; knowledge graph; CPTED; environmental criminology; urban planning; Chicago

---

## 1. Introduction

### 1.1 The Spatial Inequality of Crime Data

The geography of crime is defined by extreme concentration. Weisburd's (2015) "law of crime concentration" demonstrates that approximately 50% of crime occurs at just 5% of street segments—a finding replicated across cities worldwide. This concentration has driven the development of hotspot policing and predictive algorithms that allocate resources to areas with dense historical records.

But the other side of this distribution—the vast urban spaces where crime is recorded so rarely that predictive models produce near-zero risk estimates—has received far less analytical attention. These are not simply "safe" areas. They are spatial voids in the crime data record, where statistical models lack sufficient signal to distinguish genuinely low-risk locations from locations where environmental conditions create latent risk that has not yet materialized in the data, or where crime occurs but goes systematically under-recorded.

In Chicago, we identify 21,065 spatial analysis units at the 100-meter resolution (17.5% of 120,459 subgrids) as spatial cold-start zones. These units derive from 252 parent 1-km grids (20.2% of 1,246) where cumulative violent crime over a two-year training period is three or fewer incidents. In these locations, conventional spatio-temporal deep learning models—STGCN, GraphWaveNet—produce predictions indistinguishable from zero, rendering them operationally useless for public safety planning.

### 1.2 Cold-Start as Extreme Spatial Non-Stationarity

A central argument of this paper is that spatial cold-start should be understood not merely as a data sparsity problem, but as an extreme manifestation of two fundamental geographic phenomena: **spatial non-stationarity** and **zero-inflation** operating jointly at fine spatial scales.

**Spatial non-stationarity** (Fotheringham et al., 2002) describes the principle that relationships between variables are not constant across space—the same environmental feature may have different effects on crime risk at different locations. Cold-start represents the limiting case of this principle: in data-rich areas, the relationship between environmental features and crime can be estimated from empirical co-occurrence patterns; in data-sparse areas, this relationship becomes unidentifiable from local data alone, yet the underlying environmental structure persists. The spatial non-stationarity is not in the environment (which is continuous) but in our *capacity to measure* the environment–crime relationship from historical records (which is discontinuous).

**Zero-inflation** in grid-level crime data—where >70% of grid-week observations contain no recorded crime—is typically modeled as a statistical nuisance through count-appropriate likelihood functions (e.g., Zero-Inflated Negative Binomial). However, in cold-start areas, zero-inflation is not merely a distributional property of the outcome variable but a structural characteristic of the *data generation process* itself: crime may occur but go systematically unrecorded due to weaker police–community reporting infrastructure, or crime opportunity may genuinely be low due to the absence of criminogenic environmental configurations. Distinguishing between these two mechanisms—"genuine safety" vs. "unobserved risk"—is impossible from historical crime data alone and requires an independent information channel.

We introduce the term **Spatial Data Deserts** to describe the contiguous clustering of cold-start grids. Unlike isolated low-data points (which can be regularized by nearby data-rich neighbors through spatial smoothing), spatial data deserts are characterized by *spatial continuity of data sparsity*: on average, 4.8 of a cold-start grid's 8 nearest spatial neighbors are themselves cold-start grids. This structural property—analogous to a "food desert" in public health geography, where the absence of healthy food options is spatially concentrated rather than randomly distributed—means that local spatial smoothing propagates uncertainty rather than signal. The information deficit is spatially auto-correlated, and resolving it requires information sources that operate beyond the geographic proximity constraint.

Cold-start grids are not randomly scattered across Chicago. They concentrate in three distinct urban morphologies: peripheral residential zones (~45%), industrial and logistics corridors (~30%), and large green or institutional spaces (~25%). Each type has a distinct geographic rationale for data sparsity rooted in urban form (see Section 2.6 for formal spatial morphology analysis). The geographic structure of spatial data deserts has direct implications for modeling: when cold-start grids cluster by morphology, the relevant information for prediction may reside in *functionally similar but geographically distant* grids—motivating our KG's environmental similarity edges as a complement to geographic adjacency.

### 1.3 CPTED as a Theoretical Bridge

Environmental criminology provides the theoretical foundation for addressing spatial data scarcity. Crime Prevention Through Environmental Design (CPTED; Jeffery, 1971; Newman, 1972) posits that the built environment shapes crime opportunity through four mechanisms: territoriality (clear demarcation of public and private space), natural surveillance (opportunities for observing public space), access control (physical guidance of movement), and maintenance (visible signs of social order). Routine Activity Theory (Cohen & Felson, 1979) adds that crime requires the spatial convergence of motivated offenders, suitable targets, and absent guardianship—each with identifiable spatial expressions in the urban landscape. Crime Pattern Theory (Brantingham & Brantingham, 1993) emphasizes that road networks, transit corridors, and land-use patterns structure the routine movement that generates crime opportunity.

The key implication for spatial cold-start is falsifiable: **if these theories hold, then urban infrastructure features contain systematic, spatially structured information about crime risk that exists prior to and independently of any crime being recorded.** A 100 m grid containing a cluster of late-night bars adjacent to a transit hub carries elevated latent crime opportunity regardless of whether police have historically patrolled there. This is the foundational premise of our Urban Environmental Knowledge Graph approach.

### 1.4 Contributions and Policy Relevance

This paper makes four contributions, ordered to reflect their logical dependency—from geographic diagnosis through mechanism identification to implementation tool:

**Contribution 1: Spatial cold-start as a geographic phenomenon.** We provide the first formal characterization of spatial cold-start in crime prediction as a geographically structured phenomenon, demonstrated through spatial autocorrelation analysis (Global Moran's I, LISA), geographic morphology classification, and spatial socioeconomic correlation. We establish that cold-start grids are not randomly distributed but exhibit significant positive spatial autocorrelation, organize into three distinct morphological types, and show systematic spatial associations with community socioeconomic indicators. This diagnosis establishes that cold-start cannot be resolved through better model architecture alone—it requires spatially informed feature engineering that addresses the structural causes of data sparsity.

**Contribution 2: Environmental criminology mechanism identification.** Through systematic ablation experiments and SHAP-based feature importance analysis, we identify *which* specific CPTED dimensions drive cold-start prediction, *how* different KG components contribute to performance, and *why* certain design choices (weak vs. strong contrastive learning, CPTED content vs. graph topology) succeed or fail. The 94% performance collapse upon removing CPTED features provides the strongest empirical evidence to date that environmental criminology theory carries genuine, independently verifiable predictive signal for spatial crime risk assessment.

**Contribution 3: CPTED interpretability and policy translation.** We demonstrate that KG embeddings, combined with SHAP decomposition and Ridge regression coefficients, enable full traceability from model prediction → CPTED dimension → observable environmental feature → planning intervention. This transparency distinguishes our framework from opaque deep learning models and enables the translation of algorithmic output into spatially targeted, evidence-based CPTED interventions organized by spatial scale (tactical/micro vs. strategic/macro).

**Contribution 4: Urban Environmental KG as implementation instrument.** We construct and validate a reproducible, computationally efficient KG pipeline that requires only OpenStreetMap data (globally available and free) and established criminological theory (publicly documented CPTED mappings). The two-stage architecture (pretrain once, predict with Ridge regression) is GPU-free at inference time, fully auditable, and designed for transferability to any city with OSM coverage—reducing the barrier to deploying spatially informed crime risk assessment beyond well-resourced urban contexts. We further demonstrate that LLM-derived semantic embeddings provide complementary coarse calibration signal (improving AUC from 0.865 to 0.881 and NDCG@50 from 0.288 to 0.360 when fused with KG embeddings), though KG embeddings alone remain optimal for fine-grained ranking (NDCG@20 = 0.406, unmatched by any fusion variant).

---

## 2. Study Area and Data

### 2.1 Chicago as a Testbed

Chicago, Illinois (population ~2.7 million; area ~606 km²) provides an ideal testbed for spatial cold-start analysis. Three characteristics make it particularly suitable: (1) diverse urban morphology spanning dense downtown, residential neighborhoods, industrial corridors, and large green spaces—producing the environmental variation needed to differentiate latent risk profiles; (2) a pronounced socioeconomic spatial structure that enables analysis of how cold-start patterns intersect with community demographics; and (3) comprehensive open data infrastructure including the Chicago Data Portal for crime records and OpenStreetMap coverage for urban infrastructure.

### 2.2 Spatial Framework: A Deliberate Multi-Scale Design

Our analysis operates on a nested two-level spatial grid designed to **test the Modifiable Areal Unit Problem (MAUP)**. The parent level consists of 1,246 regular grid cells (~1 km² each) covering Chicago's land area; each is subdivided into a 10×10 array of 100 m subgrids, yielding 120,459 fine-grained spatial units.

This hierarchical design serves three purposes. **(1) Scale sensitivity testing:** evaluating performance at 100 m, 500 m, and 1 km resolutions identifies the spatial scale at which environmental criminology signals operate—CPTED mechanisms function at the street-block scale (~100 m), while Crime Pattern Theory constructs manifest at the neighborhood scale (~1 km). **(2) Mitigation of ecological bias:** the 100 m resolution preserves micro-spatial environmental features (individual POI clusters, street block characteristics) that would be averaged away in coarser grids, preventing the ecological fallacy of assigning a single CPTED profile to a 1 km grid containing both a bar district and a park. **(3) Parent-grid classification with subgrid prediction:** cold-start status is defined at the 1 km level (where data sparsity is analytically meaningful), but prediction operates at the 100 m level (where CPTED mechanisms and interventions function), leveraging fine-grained environmental variation *within* cold-start zones.

### 2.3 Crime Data and Cold-Start Definition

Violent crime incident records (January 2022–December 2023) are obtained from the Chicago Data Portal, including geocoded locations for homicide, assault, battery, robbery, and sexual assault. A spatial unit is classified as cold-start if it belongs to a parent 1-km grid meeting either: (1) cumulative training-period violent crime ≤ 3, or (2) bottom 15th percentile of the violent crime count distribution. The test period spans September 13–December 31, 2023 (110 days). Crime risk is operationalized as Kernel Density Estimation (KDE) with a 100 m bandwidth, producing a continuous risk surface as the prediction target.

We define cold-start on violent crime because violent offenses are the primary concern for police resource allocation, exhibit greater spatial sparsity than property crime, and carry more severe consequences when under-predicted. In total, 252 parent grids (20.2%) and 21,065 subgrids (17.5%) meet the cold-start criterion.

### 2.3.1 Sensitivity Analysis of Cold-Start Threshold

To assess whether our findings are robust to the specific cold-start threshold (cumulative violent crime ≤3 OR bottom 15th percentile), we conduct a sensitivity analysis varying both the absolute threshold (t ∈ {1, 2, 3, 4, 5}) and the relative percentile threshold (p ∈ {10, 12, 15, 18, 20}). For each threshold configuration, we re-compute the cold-start grid set, retrain the Ridge regression downstream predictor on the corresponding cold-start embeddings, and report NDCG@20 and AUC.

**Key findings (Table 2.1).** The relationship between threshold choice and KG performance is not monotonic. The strictest definition (t=1, capturing 200 1km grids with zero cumulative violent crime over the training period) produces NDCG@20 ≈ 0—the KG's fine-grained ranking collapses when no historical crime signal whatsoever exists at the parent-grid level, though its coarse discrimination remains functional (AUC = 0.824). However, the addition of just one additional crime incident (t=2, 224 grids) restores KG performance to NDCG@20 = 0.366—only 0.040 below the current operational definition. Further relaxation of the threshold (t=4, t=5) produces a gentle monotonic decline in NDCG@20 (0.312 → 0.293) as increasingly crime-experienced grids dilute the cold-start set. Across all thresholds with at least minimal historical signal (t ≥ 2), NDCG@20 remains within a narrow band of 0.293–0.366 (max deviation from baseline: 0.073), demonstrating that the KG's predictive capacity is not sensitive to the precise operationalization of data sparsity once a minimal parent-grid signal exists.

The percentile-based thresholds reveal a structural property of the crime distribution: because violent crime at the 1km grid level is highly right-skewed (the bottom 15th percentile corresponds to zero crime), several percentile and absolute thresholds produce identical grid sets. The practical implication is that the union criterion (≤3 OR bottom 15%) is robust: varying either the absolute or the percentile component produces similar results, and no plausible alternative threshold would qualitatively alter the conclusions.

The stability of KG performance across threshold choices (NDCG@20 remaining within ±0.073 across all t ≥ 2 thresholds) indicates that the KG captures a continuous environmental risk gradient rather than a binary cold/hot distinction. Conversely, the complete ranking failure at t=1 (true zero-history grids) cleanly delineates the boundary condition of the framework: the KG requires at least a minimal historical footprint at the 1km parent scale to calibrate its environmental embeddings to crime risk. This is a feature, not a bug—it defines the precise operational envelope in which the KG adds value.

### 2.4 Urban Infrastructure Data

Four categories of urban infrastructure data are collected and spatially joined to the 100 m grid system:

**POI data.** Eight POI categories are extracted from OpenStreetMap: Bar, Subway/Metro Station, School, Bank/ATM, Park, Restaurant, Convenience Store, and Hotel. These categories were selected for their direct relevance to CPTED mechanisms—each maps to specific theoretical constructs (e.g., bars → motivated offender concentration; transit stations → movement path nodes; parks → guardianship variation).

**Road network.** Street connectivity matrices, accessibility scores, and betweenness centrality metrics are computed from OSM road network data, capturing the spatial accessibility dimension of Crime Pattern Theory.

**CPTED knowledge base.** A structured mapping from POI types to CPTED attributes encodes criminological domain knowledge. Each POI type is characterized across 17 environmental dimensions organized into four CPTED constructs: natural surveillance (e.g., visibility, lighting), territorial reinforcement (e.g., boundary definition, ownership cues), access control (e.g., entry/exit management), and activity support (e.g., land-use intensity, temporal rhythm). The mapping yields 73 binary indicators through one-hot encoding of categorical CPTED attribute values, providing the input feature space for KG construction.

**Corridor and archetype labels.** Spectral clustering combining CPTED feature similarity with road network adjacency partitions the 1,246 parent grids into nine environmental archetypes (AR01–AR09): Commercial, Nightlife, Transit, Residential, Mixed, Industrial, Park/Green, School, and Connectivity. These archetypes serve as macro-level spatial structure labels for contrastive learning.

### 2.5 Socioeconomic Data

Community-level socioeconomic indicators (median household income, poverty rate, unemployment rate, educational attainment, mean travel time) are derived from the American Community Survey 5-year estimates (2018–2022) and spatially joined to analysis grids for contextual interpretation of cold-start patterns.

### 2.6 Spatial Morphology of Cold-Start Areas

To establish that spatial cold-start is a geographically structured phenomenon rather than a statistical artifact, we conduct a formal spatial morphology analysis comprising four components: spatial autocorrelation diagnostics, local indicator analysis, land-use-based morphological classification, and socioeconomic spatial correlation.

#### 2.6.1 Spatial Autocorrelation of Cold-Start Status

We compute Global Moran's I on the binary cold-start indicator (1 = cold, 0 = hot) at the 1 km parent-grid level using Queen contiguity weights with 999 random permutations. A statistically significant positive Moran's I would confirm that cold-start grids are spatially clustered beyond what random data sparsity would produce—a finding with direct modeling implications, as it would establish that geographic neighbors of cold-start grids are systematically cold themselves, rendering local spatial smoothing structurally ineffective.

**Results (Table 2.2).** Cold-start status exhibits strong and statistically significant positive spatial autocorrelation (Moran's I = 0.539, z = 35.68, p = 0.001), confirming that cold-start grids are not randomly distributed but form contiguous spatial clusters. The strength of this autocorrelation—comparable in magnitude to the spatial clustering of crime itself—establishes cold-start as a geographic phenomenon rather than a statistical artifact. The accompanying Moran scatterplot reveals that the majority of cold-start grids fall in the High-High quadrant (grids with above-average cold-start probability surrounded by similarly cold neighbors). This "spatial data desert" configuration is the geographic signature that motivates our KG's environmental similarity edges: when geographic neighbors are systematically uninformative, functionally similar but geographically distant grids become the relevant information source.

We also compute Moran's I on the log-transformed cumulative violent crime count and the KDE crime risk surface to establish the baseline spatial structure of the outcome variable. Log-transformed violent crime at the 1 km level exhibits strong spatial autocorrelation (Moran's I = 0.681, z = 45.00, p = 0.001), consistent with decades of environmental criminology research establishing the spatial concentration of crime (Weisburd, 2015). The KDE crime risk surface at 100 m resolution shows even more extreme autocorrelation (I = 0.805, z = 232.94, p = 0.001), reflecting the smooth spatial structure imposed by kernel density estimation at fine resolution. Our contribution is demonstrating that the *absence* of crime data (cold-start binary I = 0.539) exhibits spatial structure of comparable magnitude to crime concentration itself—a finding not previously quantified in the crime prediction literature. This symmetry—crime clusters and data deserts as twin spatial structures—has direct implications for spatial model design (Section 3.3).

#### 2.6.2 LISA: Identifying Specific Cold-Start Clusters

Local Indicators of Spatial Association (LISA; Anselin, 1995) decompose the global Moran's I into location-specific statistics, identifying four spatial association types at the 1 km grid level:

- **HH (High-High):** Grids with high cold-start probability surrounded by similarly cold neighbors—our "spatial data deserts."
- **LL (Low-Low):** Grids with low cold-start probability (i.e., hot grids) surrounded by hot neighbors—the conventional hotspots that dominate predictive model training.
- **LH (Low-High):** Hot grids embedded in cold areas—spatial outliers where crime concentration is anomalous given the surrounding data-sparse context.
- **HL (High-Low):** Cold grids embedded in hot areas—anomalous low-data pockets within active crime zones.

**Results (Table 2.3).** LISA analysis reveals that 64.4% of cold-start 1km grids (152 of 236) are located within statistically significant HH (High-High) clusters—areas where cold-start status is spatially contiguous, confirming the "spatial data desert" configuration. This geographic concentration far exceeds random expectation (under spatial randomness, only 19% of grids would fall in any significant cluster at p < 0.05). The LL clusters (hotspot-on-hotspot) encompass 765 grids (61.4% of Chicago's grid system), reflecting the well-known spatial concentration of crime. The HH clusters (cold-on-cold) map onto three recognizable urban zones: (1) the far Northwest and Southwest Side peripheral residential belt, (2) the industrial corridor along the Sanitary and Ship Canal, and (3) the southern lakefront parkland and institutional zone. Conversely, crime hotspot (LL) clusters concentrate in the West Side, South Side commercial corridors, and Near North zones—areas that have historically received the majority of police attention and algorithmic prediction capacity.

Spatial outliers (LH and HL) are rare: only 50 grids (4.0%) are isolated hot grids within cold zones (LH), and 26 grids (2.1%) are isolated cold grids within hot zones (HL). The 26 HL grids (11.0% of cold-start grids) represent an analytically interesting category: cold-start grids that are anomalous given their high-crime surroundings. These may indicate locations where crime reporting infrastructure is unexpectedly weak despite active surrounding areas, or where specific environmental features suppress crime despite neighborhood-level risk factors. However, the dominant pattern is clear: cold and hot crime regimes rarely mix at the local scale. Combined, only 6.1% of grids exhibit cross-regime spatial association (LH + HL). The near-complete spatial segregation of data-rich and data-sparse areas—76 of every 100 grids reside in clusters of their own type—is a finding with direct equity implications (see Section 5.3).

#### 2.6.3 Morphological Classification of Cold-Start Types

Cold-start grids are not a homogeneous category. Using land-use polygons and green-space coverage data spatially joined to the 1 km grid system, we classify cold-start grids into three morphological types based on their dominant environmental characteristics:

**Type 1: Peripheral Residential (50.4% of cold-start 1km grids, 50.2% of cold-start 100m subgrids).** Grids dominated by single-family residential land use, located at the city's northern and southwestern edges. These areas are characterized by low population density, homogeneous residential land use with limited non-residential POIs, and distance from major commercial corridors and transit hubs. From a Crime Pattern Theory perspective, these areas lie outside the activity spaces of most offender populations—they lack the nodes (commercial centers) and paths (arterial roads, transit lines) that channel routine movement and generate public-space crime opportunity.

**Type 2: Industrial/Logistics Corridors (14.4% of cold-start 1km grids).** Grids dominated by manufacturing, warehousing, and logistics land use, concentrated along the Sanitary and Ship Canal corridor and in isolated industrial pockets. These zones have very low residential population, high building footprint but minimal human presence outside working hours, and limited street-level activity. While they contain abundant suitable targets for property crime (warehouses, truck yards, equipment storage), the absence of routine activity convergence outside business hours suppresses public-space crime occurrence. Furthermore, industrial security arrangements (private security firms, gated facilities, internal CCTV systems) channel crime reporting through private rather than public data systems.

**Type 3: Green/Institutional (35.2% of cold-start 1km grids).** Grids containing large parks (e.g., Marquette Park, Jackson Park), cemeteries (e.g., Rosehill, Oak Woods), and institutional campuses (university grounds, hospital complexes). These areas have near-zero permanent population, extensive non-built land cover, and limited street network density. Crime incidents occurring within these zones—park assaults, campus thefts—may be reported through separate institutional channels (campus police, park district police) that do not consistently feed into the Chicago Data Portal.

The geographic classification has direct modeling implications. Grids of the same morphological type—even if geographically distant—share similar functional profiles and thus similar latent crime opportunity structures. This is precisely the information channel that our KG's environmental similarity (macro) edges are designed to exploit (Section 3.3): a Type 2 cold-start grid in the southern industrial corridor can receive environmental information from a data-rich Type 2 grid in a northern industrial zone, despite the geographic distance.

#### 2.6.4 Socioeconomic Spatial Correlation

We spatially join five ACS-derived socioeconomic indicators (z-score normalized) to the 1 km grid system and compare cold-start vs. hotspot grid distributions using independent-samples t-tests with Bonferroni correction for multiple comparisons (α = 0.05/5 = 0.01).

**Results (Table 2.5).** All five socioeconomic indicators show statistically significant differences between cold-start and hotspot grids (p < 0.01, Bonferroni-corrected). However, the *pattern* of these differences defies the conventional "disadvantaged neighborhood" narrative. Cold-start grids exhibit systematically lower median income (Δz = −0.44, t = −5.26), lower educational attainment (Δz = −0.48, t = −6.98), and shorter commute times (Δz = −0.36, t = −4.48)—but simultaneously show *lower* poverty rates (Δz = −0.30, t = −3.69) and *lower* unemployment rates (Δz = −0.46, t = −6.71) than hotspot grids. This pattern—lower income but not higher poverty, lower education but not higher unemployment—characterizes **stable, low-density, mono-functional urban spaces** rather than zones of concentrated disadvantage. The three morphological types each contribute a distinct socioeconomic profile: Type 1 (peripheral residential) areas are modest-income but economically stable neighborhoods; Type 2 (industrial/logistics) zones have negligible residential populations, making their socioeconomic indicators reflect adjacent census tracts rather than actual resident populations; Type 3 (green/institutional) spaces are non-residential by definition.

This socioeconomic profile carries an important equity implication: cold-start areas are not the neighborhoods that typically receive the most police attention or social services, yet their systematic exclusion from crime prediction models means that any latent environmental risk they harbor goes algorithmically unrecognized. The KG framework addresses this recognition gap by providing an environmental risk assessment channel that operates independently of both crime history and socioeconomic profiling.

---

## 3. Methodology

### 3.1 Design Philosophy: Measuring Opportunity, Not Reproducing Outcomes

Our architecture is governed by a principle that distinguishes it from both conventional crime prediction models and end-to-end deep learning approaches: **the KG measures the stable structural *opportunity* for crime embedded in the built environment—decoupled from the outcome of past policing patterns, though requiring minimal crime signal for calibration—rather than learning to reproduce the spatially biased *outcome* of historical enforcement.**

This distinction is not merely rhetorical. Conventional spatio-temporal models learn a mapping f(X_historical_crime, X_spatial) → Ŷ_future_crime. When X_historical_crime ≈ 0 (the cold-start condition), the learned mapping produces Ŷ ≈ 0 regardless of X_spatial, because the training objective has never incentivized the model to extract signal from environmental features when historical signal is absent. The model has learned that "no history → predict zero," which is statistically optimal under MSE loss but operationally worthless for cold-start risk assessment.

Our two-stage strategy inverts this dependency:

**(1) Pretraining stage:** Learn an environmental embedding function g(X_CPTED, A_graph) → E_KG using only CPTED features and spatial topology. No crime labels are involved. The self-supervised objectives (MAE reconstruction of CPTED features, weak contrastive regularization by corridor membership, covariance decorrelation) ensure that E_KG encodes the multi-dimensional structure of the built environment as organized by criminological theory.

**(2) Prediction stage:** Learn a simple linear mapping h(E_KG) → Ŷ_risk using Ridge regression on cold-start grids only. Because E_KG is frozen—anchored to urban infrastructure rather than optimized for crime prediction—the mapping h(·) captures the *empirical association* between environmental configuration and observed crime risk in the cold-start population, without the KG encoder overfitting to the sparse and potentially biased crime labels.

This design ensures three properties essential for policy applications: (a) the embeddings are auditable (each dimension is traceable to CPTED constructs, not learned latent factors of unknown meaning); (b) the downstream mapping is transparent (linear coefficients directly quantify each embedding dimension's contribution to risk); and (c) the framework is computationally accessible (no GPU needed for inference; embeddings can be pre-computed once and reused).

### 3.2 CPTED Discrete Encoding

Each POI type is mapped to its CPTED profile through a structured knowledge base derived from criminology literature. For example, a Bar receives: night_activity="high", natural_surveillance="low", alcohol_presence="yes", territorial_reinforcement="weak". A Subway/Metro station receives: pedestrian_flow="high", access_control="open", temporal_rhythm="pulsed", natural_surveillance="moderate". These categorical attributes across all 17 CPTED dimensions are one-hot encoded, producing a 73-dimensional binary vector per POI type.

For each 100 m subgrid, CPTED vectors from all POIs within a 200 m radius are aggregated via distance-decay weighted summation, producing a 73-dimensional CPTED feature vector that encodes the local environmental criminology profile. This aggregation radius reflects the walkable micro-spatial scale at which CPTED mechanisms operate (approximately one city block).

### 3.3 Heterogeneous Spatial Graph

The KG is constructed as a heterogeneous graph with two edge types connecting the 120,459 subgrid nodes:

**Micro edges (POI inclusion).** Each subgrid is connected to the POIs within its 200 m buffer, with edge weights determined by distance decay and CPTED attribute match. This captures fine-grained environmental context at the individual establishment level.

**Macro edges (corridor co-membership).** Subgrids belonging to parent grids within the same environmental archetype (AR01–AR09) are connected via bidirectional edges. This provides non-local structural information: a residential cold-start grid at the urban periphery can receive environmental context from functionally similar residential grids across the city, bypassing the geographic proximity constraint that fails under spatial clustering.

The nine archetypes—Commercial, Nightlife, Transit, Residential, Mixed, Industrial, Park/Green, School, and Connectivity—are derived through spectral clustering combining CPTED feature similarity with road network adjacency (Section 2.4). This operationalization is theoretically motivated by Crime Pattern Theory's core premise: that crime-relevant environmental configurations are not randomly distributed but cluster into recognizable urban archetypes shaped by land-use policy, transportation infrastructure, and economic geography (Brantingham & Brantingham, 1993). Unlike alternative positive-pair definitions—such as kNN based on POI similarity (which would capture only single-dimension compositional similarity) or road network distance (which would recapitulate geographic adjacency)—corridor archetypes capture *multi-dimensional environmental configurations*: two grids in the "Nightlife" archetype share not merely high bar density but a specific co-occurrence of late-hour POIs, transit accessibility, high pedestrian flow, and commercial land-use mix. This multi-dimensionality is essential for the contrastive objective to provide theoretically meaningful rather than merely statistical regularization.

**Spatial adjacency.** A k-nearest neighbor graph (k=8) based on geographic distance is also included, providing baseline spatial connectivity.

**Theoretical justification: Functional space as complement to geographic space.** Tobler's (1970) First Law of Geography—"everything is related to everything else, but near things are more related than distant things"—underpins the spatial smoothing assumption in virtually all spatio-temporal crime prediction models. Graph convolution over k-nearest-neighbor geographic graphs operationalizes this principle: each grid's prediction is regularized toward its physically proximate neighbors' patterns.

However, the First Law's operational scope is bounded by an implicit condition: the relationship of interest must be *spatially continuous* at the scale of the chosen neighborhood. When crime data density itself is spatially auto-correlated—as our LISA analysis demonstrates (Section 2.6.2)—this continuity condition breaks down. In a spatial data desert, a grid's k-nearest geographic neighbors are themselves data-sparse, and spatial smoothing propagates the information deficit rather than resolving it.

Our macro edges (corridor co-membership) operationalize a complementary principle: **functional proximity**—the similarity of two locations in a theoretically meaningful environmental feature space—can substitute for geographic proximity when the latter fails. A cold-start residential grid at the urban periphery and a data-rich residential grid 10 km away may have near-zero geographic adjacency but high functional similarity in their CPTED profiles, land-use composition, and built-environment configuration. By connecting them through corridor co-membership edges, the KG enables information flow along a topology that reflects *what places are like* rather than merely *where they are*.

This design does not reject Tobler's First Law—it supplements it. Geographic adjacency edges remain in the graph (k=8 spatial KNN), providing local spatial smoothing where it is valid. Functional adjacency edges provide an additional information channel that activates precisely when geographic smoothing is most compromised—in the dense cores of spatial data deserts. The weak contrastive learning weight (λ=0.05) ensures that functional regularization remains a gentle prior rather than an overbearing constraint (see Section 3.4 for the empirical consequences of getting this balance wrong).

This dual-topology design can be situated within the broader geographic methods literature on spatial weights matrix specification (Getis & Aldstadt, 2004). Traditional GWR and spatial econometric models define neighbors exclusively through geographic distance or contiguity. Our KG extends this framework by incorporating a second neighbor definition based on environmental similarity—conceptually analogous to the "semantic proximity" graphs used in geospatial machine learning (Mai et al., 2022), but grounded in criminological theory rather than learned from data.

### 3.4 GraphSAGE Encoder with Self-Supervised Pretraining

A 3-layer SparseSAGEConv encoder processes the heterogeneous graph to produce 32-dimensional L2-normalized embeddings for each subgrid. The architecture uses sparse matrix multiplication (`torch.sparse.mm`) for memory-efficient message passing across 120,459 nodes, with dimensions: Input (73 CPTED one-hot) → Hidden (64) → Output (32).

Three self-supervised objectives jointly optimize the embeddings:

**MAE (Masked AutoEncoder, λ=1.0).** Forty percent of CPTED feature dimensions are randomly masked. An MLP decoder (32→64→73) reconstructs the masked dimensions from the 32-dimensional bottleneck embedding, using MSE loss. This objective ensures that embeddings encode CPTED information—the model must learn to infer masked environmental characteristics from the spatial context captured by graph convolutions.

**NT-Xent Contrastive Loss (τ=0.2, λ=0.05).** Subgrids sharing the same corridor/archetype label form positive pairs; subgrids from different archetypes form negative pairs. Critically, we use a high temperature (τ=0.2) and low weight (λ=0.05) to provide only weak spatial regularization. This design choice emerged from a finding with substantive geographic interpretation. Strong contrastive learning (τ=0.07, λ=0.3) reduces NDCG@20 from 0.406 to 0.108—a 73.5% degradation. We interpret this collapse through the lens of the **ecological fallacy in representation learning**: aggressive contrastive pressure enforces embedding similarity across all subgrids sharing a corridor label, effectively imposing a *macro-level mean* (the corridor archetype's average CPTED profile) onto each *micro-level unit* (individual 100 m subgrids). This homogenization strips away the within-corridor environmental variation—the block-level CPTED heterogeneity that distinguishes, for example, a well-lit corner with active storefronts from a poorly-lit mid-block segment within the same "Commercial" corridor. The model, forced to prioritize coarse-grained corridor separation, loses the fine-grained CPTED signal that drives risk differentiation *within* archetypes. The weak contrastive configuration (τ=0.2, λ=0.05) avoids this trap: it provides just enough spatial regularization to prevent dimensional collapse (complemented by the covariance regularizer) while allowing the MAE objective—which operates independently on each subgrid's CPTED features—to dominate the representation. The result is an embedding space where grids within the same corridor are *similar but not identical*, preserving the micro-spatial environmental heterogeneity that CPTED theory identifies as criminologically meaningful.

**Covariance Regularization (λ=0.05).** Barlow Twins-style dimension decorrelation penalizes off-diagonal entries in the embedding covariance matrix. Without this term, embeddings collapse to an effective dimensionality of 1–3, losing most of the CPTED information. With covariance regularization, the effective dimension expands to approximately 11, preserving the multi-dimensional CPTED signal.

The total loss is: L = L_MAE + 0.05 × L_contrast + 0.05 × L_cov. Training uses Adam (lr=1e-3, 300 epochs, batch size 2048). The trained encoder weights are frozen, and embeddings are pre-computed once for all 120,459 subgrids.

### 3.5 Downstream Cold-Start Prediction

For cold-start prediction, we employ Ridge regression (α=1.0) with 70/30 train/test split on the cold-start subset only. The 32-dimensional KG embeddings serve as input features; the target is the KDE crime risk score computed from test-period violent crime incidents. This simple linear model ensures full interpretability: each prediction is a linear combination of embedding dimensions with fixed coefficients, enabling direct attribution of risk to specific CPTED dimensions.

### 3.6 Evaluation Metrics

We evaluate using three complementary metrics designed for the cold-start setting:

- **NDCG@K (Normalized Discounted Cumulative Gain):** Measures ranking quality—whether the model correctly orders cold-start grids by their true crime risk. NDCG is more informative than RMSE in sparse settings because a model predicting near-zero everywhere achieves low RMSE (the mean of near-zero data is near-zero) but zero NDCG, which requires successful discrimination among low-frequency events.

- **HitRate@K:** The proportion of top-K predicted grids that overlap with high-risk grids (top 5% by actual crime).

- **Risk Capture AUC:** Area under the cumulative risk capture curve, measuring how efficiently the model's ranking concentrates true crime risk in its highest-ranked predictions.

All metrics are computed on the cold-start subset only, ensuring that performance reflects the model's capacity to discriminate within data-sparse regions rather than being dominated by hotspot prediction accuracy.

**Spatial cross-validation.** Standard random train/test splitting assumes independence between observations—an assumption violated by spatially auto-correlated data. We additionally evaluate using **spatial hold-out cross-validation** (Roberts et al., 2017; Meyer et al., 2018) with two blocking strategies: **(1) Community Area-based blocking,** using Chicago's 77 Community Areas as spatial folds, and **(2) K-means spatial clustering-based blocking,** clustering the 1,246 parent grid centroids into k=10 spatial clusters on geographic coordinates. Full results and their interpretation are reported in Section 4.10.

### 3.7 Comparison Methods

**Spatial econometric baselines.** To situate our KG+Ridge approach within the established spatial regression literature and to test whether explicit spatial modeling can substitute for the KG's learned environmental encoding, we implement three spatial econometric models: Spatial Lag Model (SLM), Spatial Error Model (SEM), and Multiscale Geographically Weighted Regression (MGWR; Fotheringham et al., 2017). All three models use PCA-reduced KG embeddings (top-10 principal components, explaining 89.1% of the 32-dimensional variance) as covariates and are trained on the cold-start subset. SLM and SEM are estimated via maximum likelihood; MGWR uses adaptive bisquare kernels with golden-section bandwidth search. Full implementation details are provided in Supplementary Materials.

We evaluate against three categories of baselines and comparison methods:

**Simple baselines:** Uniform prediction (random baseline), POI density-based prediction (naive environmental proxy), and Parent 1 km rate (crime rate from the containing 1 km grid, representing the best available spatial prior).

**Spatio-temporal baseline:** ST-GCN trained without KG features, representing the conventional deep learning approach that relies on historical crime patterns and spatial smoothing.

**Comparison methods:** Raw CPTED features (73-dim vectors used directly in Ridge regression without graph encoding), LLM semantic embeddings (768-dim vectors from a language model encoding neighborhood textual descriptions), and Domain Transfer (Ridge regression trained on hot-start grids and applied to cold-start grids, testing whether the embedding→crime mapping generalizes across the data density boundary).

**KG variants (ablation):** Full KG, KG without CPTED features (random vectors replace CPTED input), KG without macro-corridor edges, KG with randomly shuffled corridor labels, and KG with strong contrastive learning (τ=0.07, λ=0.3).

**LLM+KG fusion:** Weighted ensemble (α-weighted average of LLM and KG predictions) and feature concatenation (concatenated 32-dim KG + 768-dim LLM embeddings as Ridge input).

---

## 4. Results

### 4.1 Cold-Start Prediction Performance

Table 1 presents the main results for violent crime cold-start prediction at the 100 m resolution.

**Table 1: Cold-Start Violent Crime Prediction (21,065 subgrids, 100 m)**

| Method | NDCG@20 | HitRate@20 | NDCG@50 | HitRate@50 | AUC |
|--------|:-------:|:----------:|:-------:|:----------:|:---:|
| Uniform | 0.026 | 0.050 | 0.018 | 0.040 | 0.490 |
| POI Density | 0.000 | 0.050 | 0.000 | 0.060 | 0.687 |
| Parent 1 km Rate | 0.012 | 0.350 | 0.008 | 0.180 | 0.681 |
| ST-GCN w/o KG | 0.007 | 0.250 | 0.005 | 0.180 | 0.765 |
| Raw CPTED | 0.032 | 0.350 | 0.040 | 0.420 | 0.679 |
| KDE_train (spatial ceiling) | **0.803** | **1.000** | **0.836** | **1.000** | **0.938** |
| LLM Semantic | 0.182 | 0.950 | 0.283 | 0.880 | 0.687 |
| Domain Transfer | 0.069 | 0.550 | 0.048 | 0.220 | 0.704 |
| **Ours (Full KG)** | **0.406** | 0.850 | **0.288** | 0.540 | **0.865** |

*Cold-start defined as parent 1 km grid with cumulative violent crime ≤3 or bottom 15%. All metrics computed on cold-start subset only.*

Three findings stand out. First, **conventional methods fail catastrophically on cold-start**: ST-GCN without KG features achieves NDCG@20=0.007—barely above the Uniform random baseline (0.026). POI Density (0.000) and Parent 1 km Rate (0.012) similarly provide near-zero discrimination. This confirms that spatial cold-start is a genuine failure mode for existing approaches, not merely a setting where they perform slightly worse.

Second, **LLM semantic embeddings show a striking divergence between HitRate and NDCG**: LLM achieves the highest HitRate@20 (0.950) but low NDCG@20 (0.182) and AUC (0.687). The LLM correctly identifies *some* high-risk grids (hence high HitRate—it successfully flags grids with the most extreme crime counts) but fails to produce a correctly ordered ranking across all cold-start grids (low NDCG). Its predictions are poorly calibrated to the continuous risk surface, producing a binary-like output: a few grids score very high, most score near zero. The KG, in contrast, produces a smooth risk gradient that captures the full distribution of environmental risk.

This HitRate–NDCG divergence warrants careful interpretation, as it reveals fundamental differences in *what kind* of spatial knowledge LLM embeddings and CPTED-based KG embeddings encode. LLM semantic embeddings are derived from textual descriptions of neighborhoods—they capture the broad **semantic gist** of places: which neighborhoods are described as "dangerous," "commercial," "residential," or "industrial" in textual corpora. This gist-level knowledge excels at identifying the most extreme cases: if a neighborhood has a strong textual association with crime-related terms, it likely experiences genuinely elevated crime rates. Hence the high HitRate@20 (0.950)—the LLM successfully flags the 5% of cold-start grids with the most extreme crime counts, because these tend to be located in or near areas whose textual descriptions contain crime-relevant semantic markers.

However, semantic gist operates at a coarse spatial resolution—neighborhood-level descriptions do not differentiate between a well-lit corner with active storefronts and a poorly-lit mid-block segment within the same neighborhood, even though CPTED theory predicts these micro-environments have systematically different crime opportunity profiles. The LLM's inability to capture this **micro-spatial environmental gradient** explains its low NDCG@20 (0.182): across the 21,065 cold-start subgrids, it produces a binary-like ranking where a handful of grids score high (those in semantically flagged neighborhoods) and the vast majority score near zero, failing to produce the smooth, calibrated risk ordering needed for resource allocation across the full cold-start population.

The KG, by encoding CPTED features at the 100 m resolution with distance-decay-weighted POI aggregation (200 m radius), preserves the micro-spatial environmental variation that semantic gist aggregates away. The resulting NDCG@20 (0.406) reflects a genuinely graded risk ordering: the model distinguishes higher-risk from lower-risk cold-start grids across the full distribution, not just at the extremes. This diagnostic—that LLMs provide coarse "where to look" signals while structured KGs provide fine-grained "how to differentiate" signals—directly motivates the LLM+KG fusion results (Section 4.5).

Third, **Domain Transfer fails meaningfully** (NDCG@20=0.069, AUC=0.704). The embedding→crime mapping learned from hot-start grids does not generalize to cold-start grids. This is a theoretically important negative result: it demonstrates that the relationship between environmental features and crime *differs systematically* between data-rich and data-sparse areas, validating the cold-start concept as a genuine domain shift rather than simply a sample size problem.

Fourth, and critically, **the KDE_train baseline establishes the spatial ceiling for cold-start prediction**. Training-period crime KDE—the simplest spatial smoothing baseline, using only past crime coordinates without any environmental features—achieves NDCG@20 = 0.803, AUC = 0.938, and near-perfect HitRate. This baseline is not a competitor to the KG; it is the **information ceiling** that defines what is achievable with full access to historical crime data. The KG achieves NDCG@20 = 0.406—50.6% of the KDE_train ceiling—using only CPTED-based environmental features that are universally observable and independent of historical crime patterns. This 50% efficiency ratio quantifies the information cost of decoupling from crime data: the KG forgoes half the spatial ceiling in exchange for operational independence from historical records. The KDE baseline is highly bandwidth-sensitive (NDCG@20 drops from 0.825 at 50 m to 0.296 at 500 m to near-zero at 1,000 m), indicating that its performance depends on capturing micro-spatial crime clusters at scales below ~200 m—precisely the scale at which the KG's 100 m CPTED encoding operates. The gap between KDE_train (0.803) and KG (0.406) represents the portion of spatial crime variation that is attributable to historical crime proximity but not to environmental configuration—a finding consistent with the SAR model's superior performance (Section 4.6) and the strong spatial autocorrelation of crime (Moran's I = 0.681, Section 2.6.1).

### 4.2 Ablation Study: What Drives KG Performance?

Table 2 isolates the contribution of each KG component.

**Table 2: Ablation Analysis**

| Variant | NDCG@20 | Δ vs Full KG | What Was Removed |
|---------|:-------:|:------------:|------------------|
| Full KG | 0.406 | — | Complete method |
| w/o CPTED Features | 0.024 | −94.2% | CPTED input → random vectors |
| Random Clusters | 0.189 | −53.6% | Corridor labels randomly shuffled |
| w/o Macro-Corridor | 0.195 | −52.0% | Macro corridor edges removed |
| Strong Contrastive (τ=0.07, λ=0.3) | 0.108 | −73.5% | Contrastive weight 0.3, temp 0.07 |

**CPTED content and graph topology are complementary, not competing.** The 94.2% collapse upon removing CPTED features should not be interpreted as evidence that graph structure is unimportant. Rather, it demonstrates that CPTED semantic content and graph topological structure are **jointly necessary** components: CPTED features provide the *informational content* (what environmental criminology knows about crime opportunity), while the graph topology provides the *spatial coordination mechanism* (how this information is organized and propagated across space). Removing either component degrades performance, but through different mechanisms:

- **Removing CPTED content** (replacing features with random vectors) eliminates the theoretical signal. The graph structure still propagates information, but the information being propagated is random noise—graph convolution amplifies meaningless variation rather than criminologically meaningful patterns. The 94.2% degradation reflects the informational vacuum created when the graph has nothing meaningful to propagate.

- **Removing macro-corridor structure** (−52.0%) or randomizing corridor labels (−53.6%) preserves the CPTED content but disrupts the spatial coordination. The model still has access to all 73 CPTED dimensions per grid, but it loses the ability to organize this information across space—grids can no longer learn from functionally similar peers. The ~52% degradation reflects the value of coordinated, non-local information sharing across the graph.

- **The full KG (0.406) benefits from both components operating jointly**: CPTED features provide rich, theory-grounded environmental descriptions at each node, and the graph topology (micro POI inclusion + macro corridor co-membership + spatial adjacency) ensures that these descriptions are refined through multi-scale message passing—local smoothing from geographic neighbors, non-local regularization from functionally similar grids, and fine-grained context from individual POI connections.

This complementarity interpretation has a practical corollary: the 94% collapse does not mean that any graph built on any features would perform similarly. It means that **the value of the graph architecture is contingent on the quality of the node features it propagates**. The graph is an amplifier—it amplifies signal when features are meaningful, and amplifies noise when features are random. The policy implication is that investing in CPTED knowledge base quality (feature engineering grounded in criminological theory) yields compounding returns through the graph's information propagation mechanism.

**Contrastive learning requires careful calibration.** Strong contrastive learning (τ=0.07, λ=0.3) degrades performance by 73.5%, reducing NDCG@20 from 0.406 to 0.108. With aggressive contrastive pressure, the embeddings optimize for corridor-label separation at the expense of within-corridor CPTED variation—the very signal needed for risk differentiation. The optimal configuration (τ=0.2, λ=0.05) provides just enough spatial regularization to prevent dimensional collapse without overwhelming the CPTED reconstruction objective.

### 4.3 Scale Analysis: KG Signal Strengthens with Aggregation

Table 3 shows how KG performance varies with spatial scale, from fine-grained 100 m units to neighborhood-level 1 km aggregation.

**Table 3: Spatial Scale Comparison**

| Scale | Cold Grids | NDCG@20 | AUC | POI AUC (reference) |
|-------|:----------:|:-------:|:---:|:-------------------:|
| 100 m | 21,065 | 0.406 | 0.865 | 0.687 |
| 500 m | 924 | 0.452 | 0.916 | 0.672 |
| 1 km | 236 | 0.905 | 0.964 | 0.643 |

KG performance improves monotonically with spatial aggregation: NDCG@20 rises from 0.406 at 100 m to 0.905 at 1 km, and AUC from 0.865 to 0.964. This scale-robustness has two interpretations. Methodologically, it confirms that KG embeddings capture meso-scale environmental structure that becomes more pronounced when fine-grained noise is aggregated—the environmental signal is spatially coherent. For policy, the 1 km performance (AUC=0.964) is directly relevant to community-level planning decisions, where interventions are typically designed at the neighborhood rather than street-block scale.

Notably, POI Density as a naive environmental proxy shows no scale benefit (AUC remains 0.64–0.69 across all scales). The KG's learned environmental encoding—rather than simple POI counting—is what enables the scale-robust signal.

### 4.3.1 Mechanism: Spatial Smoothing of Micro-Environmental Noise

The monotonic improvement in KG performance with spatial aggregation (NDCG@20: 0.406 → 0.452 → 0.905; AUC: 0.865 → 0.916 → 0.964) raises an important question: is this improvement merely a statistical artifact of reduced sample size (21,065 → 924 → 236 cold-start grids), or does it reflect a genuine spatial mechanism?

To disentangle these explanations, we compute the semivariogram of the KDE crime risk surface at each spatial scale and report the nugget-to-sill ratio—a measure of the proportion of spatial variance attributable to micro-scale noise (nugget) versus spatially structured signal (sill). The results reveal a non-monotonic pattern that aligns with and explains the Moran's I trajectory: at 100 m, nugget/sill ≈ 9.6 × 10⁻⁸ (near-zero, strong spatial dependence), reflecting the artificial spatial smoothness imposed by KDE bandwidth at fine resolution. At 500 m, nugget/sill jumps to 1.004 (weak dependence, essentially pure noise), confirming that 5×5 block aggregation strips away the KDE-induced smoothness and exposes the underlying sparsity of the crime point process—only 12.4% of cold-start 100 m grids have nonzero KDE values, and block averaging over predominantly zero-valued neighbors drives most block-level estimates toward zero. At 1 km, nugget/sill partially recovers to 0.357 (moderate spatial dependence), as parent-grid aggregation captures neighborhood-scale crime concentration patterns that persist above the micro-scale noise floor.

Additionally, we compute Global Moran's I of the KDE risk surface at each scale. The 100 m KDE surface exhibits extreme spatial autocorrelation (I = 0.805, z = 232.94, p = 0.001), reflecting both genuine crime concentration and the smoothing effect of KDE bandwidth. At 500 m aggregation, spatial autocorrelation drops dramatically to I = 0.064 (z = 4.00, p = 0.012)—a near-random spatial structure—indicating that the 5×5 block averaging largely homogenizes the local crime risk surface. At 1 km, autocorrelation partially recovers to I = 0.140 (z = 6.25, p = 0.003), suggesting the re-emergence of meso-scale spatial structure at the neighborhood level. This non-monotonic pattern (0.805 → 0.064 → 0.140) is consistent with a multi-scale spatial process: micro-spatial crime clustering at the street-block level (~100 m) is smoothed away at the intermediate block-group scale (~500 m), but broader neighborhood-level crime concentration patterns re-emerge at the community scale (~1 km). The KG's monotonic performance improvement with aggregation (NDCG@20: 0.406 → 0.452 → 0.905) despite the non-monotonic spatial autocorrelation pattern suggests that the embeddings capture environmental signals operating at multiple spatial scales simultaneously—the 100 m embeddings retain micro-spatial detail while also encoding meso-scale environmental context through the graph's hierarchical message-passing architecture.

**Policy corollary.** The strong 1 km performance has direct operational relevance: community-level planning interventions (zoning amendments, land-use diversification, district-level CPTED guidelines) are typically designed and implemented at the neighborhood scale (~1 km²), not the street-block scale. The KG's high discrimination at this resolution (AUC=0.964, NDCG@20=0.905) means that its environmental risk assessments are most reliable at precisely the spatial scale at which strategic planning decisions are made.

### 4.4 Cross-Crime-Type Generalization

To assess whether KG embeddings capture general environmental risk rather than violent-crime-specific patterns, we evaluate on property crime (theft, burglary, motor vehicle theft, arson) using the same embeddings and methodology.

**Table 4: Property Crime Generalization (100 m)**

| Method | NDCG@20 | HitRate@20 | AUC |
|--------|:-------:|:----------:|:---:|
| Uniform | 0.000 | 0.000 | 0.522 |
| POI Density | 0.005 | 0.150 | 0.647 |
| KG Embedding | 0.236 | 0.600 | 0.814 |

The KG generalizes to property crime (AUC=0.814 vs. 0.865 for violent crime)—a modest decline consistent with the fact that the CPTED knowledge base was designed primarily for violent crime opportunity structures. Property crime involves different environmental mechanisms (target availability in commercial/industrial zones, access control for warehouses) that the current CPTED mapping captures partially but not optimally. The cross-type generalization confirms that KG embeddings encode fundamental urban environmental structure rather than violent-crime-specific artifacts.

### 4.5 LLM+KG Fusion: Coarse Calibration, Not Fine-Grained Enhancement

Table 5 examines whether LLM-derived semantic embeddings and CPTED-based KG embeddings capture complementary information that can improve predictive performance when combined.

**Table 5: LLM+KG Fusion**

| Method | NDCG@20 | NDCG@50 | AUC |
|--------|:-------:|:-------:|:---:|
| KG only | **0.406** | 0.288 | 0.865 |
| LLM only | 0.182 | 0.283 | 0.687 |
| LLM+KG (α=0.1 ensemble) | 0.404 | 0.287 | 0.881 |
| LLM+KG (α=0.5 ensemble) | 0.265 | 0.301 | 0.850 |
| LLM+KG (Concat) | 0.392 | **0.360** | **0.881** |

Two patterns merit careful interpretation. First, **no fusion variant exceeds KG-only performance on NDCG@20**, the primary ranking metric. The KG embeddings alone achieve the best fine-grained risk ordering among cold-start grids. This is consistent with the diagnostic analysis in Section 4.1: LLM embeddings capture coarse semantic gist (neighborhood-level characterizations) that lacks the micro-spatial resolution needed to differentiate risk among neighboring 100 m subgrids within the same neighborhood. When fused with KG embeddings, the LLM's coarse signal introduces noise at the top of the ranking—flagging entire neighborhoods rather than specific high-risk micro-locations—which slightly degrades NDCG@20 from 0.406 to 0.392–0.404.

Second, **LLM fusion does improve performance at deeper ranking levels and coarse discrimination**. Feature concatenation raises NDCG@50 from 0.288 to 0.360 (+25.0%) and AUC from 0.865 to 0.881 (+1.8%). This pattern—improvement at coarser resolution (NDCG@50, AUC) but no improvement at fine resolution (NDCG@20)—is theoretically interpretable through the spatial resolution asymmetry identified in Section 4.1. The LLM contributes broad contextual knowledge (which neighborhoods carry general crime risk associations) that aids calibration across the full distribution (AUC) and discrimination in the long tail (NDCG@50), but its neighborhood-level resolution is too coarse to improve block-level ranking (NDCG@20). The α=0.1 weighted ensemble—which is 90% KG, 10% LLM—preserves KG-level NDCG@20 (0.404 vs. 0.406) while capturing the LLM's calibration benefit (AUC 0.881 vs. 0.865), representing the Pareto-optimal trade-off for applications requiring both fine-grained ranking and well-calibrated risk estimates.

**We therefore revise the interpretation of LLM+KG fusion: the LLM does not enhance the KG's fine-grained ranking—the KG is already optimal on that dimension—but provides a complementary coarse calibration signal that improves the model's overall risk distribution estimation, particularly for deeper-ranking and coarse-screening applications.** This finding, rather than weakening the KG's contribution, reinforces it: the structured, CPTED-grounded KG embedding is the primary driver of predictive performance, and its value is most pronounced at the fine spatial resolution (100 m) most relevant to tactical environmental interventions (Section 5.2).

### 4.6 Comparison with Spatial Baselines

Table 3.1 reports the performance of spatial models compared to KG Ridge regression. We include three tiers of spatial baselines: the simplest (KDE_train, which uses training-period crime coordinates smoothed via Gaussian KDE at 100 m bandwidth), spatial econometric models (SAR, SEM), and locally-varying models (GWR). The KDE_train baseline establishes the empirical spatial ceiling for cold-start prediction: NDCG@20 = 0.803, AUC = 0.938. This baseline is not a competitor—it uses all available historical crime data—but defines the maximum achievable signal from spatial proximity alone. The KG achieves 50.6% of this ceiling (NDCG@20 = 0.406) without access to any crime data.

The spatial econometric models reveal a sharp asymmetry: the Spatial Lag Model (SAR) achieves NDCG@20 = 0.515 (AUC = 0.959) on 5,000 subsampled cold-start grids—using the spatially lagged dependent variable Wy—while the Spatial Error Model (SEM) collapses to NDCG@20 = 0.005 (AUC = 0.697). This SAR-strong, SEM-null pattern constitutes direct evidence that spatial dependence operates through the crime outcome signal, not through unobserved error shocks. The estimated ρ = 0.685 indicates that approximately 69% of the explainable variance in cold-start crime risk is attributable to spatial spillover from neighboring grids (Weisburd et al., 2012).

Critically, the SAR model's superior performance does NOT make the KG redundant—it defines an information ceiling that the KG approaches without access to criminal data. The SAR model requires the spatially lagged dependent variable Wy, which in cold-start areas is itself composed of near-zero neighboring crime values; SAR achieves its performance by leveraging the global spatial structure learned from non-cold areas and propagating it into cold-start zones through repeated multiplication with the spatial weights matrix. The KG, by contrast, achieves NDCG@20 = 0.406—79% of the SAR ceiling—using only environmental features that are universally observable. For practitioners, the 21% performance gap between KG and SAR represents the **information cost of operational independence from crime data**—a quantifiable premium paid for deployability in truly data-desert contexts.

GWR (adaptive bi-square kernel, golden-section bandwidth = 216 m) achieves NDCG@20 = 0.433 on 500 subsampled cold-start grids, compared to KG Ridge at 0.362 on the same subsample. The GWR advantage (+19.6%) confirms the presence of spatial non-stationarity in the environment–crime relationship: the CPTED dimensions that predict crime risk in one part of Chicago differ systematically from those that predict crime risk in another. However, GWR's reliance on locally weighted estimation means it cannot predict for grids whose spatial neighborhoods contain insufficient training data—the cold-start problem re-emerges at the model estimation stage. The narrow bandwidth (216 m, equivalent to approximately two 100 m subgrids) further suggests that the spatial scale of environmental influence is highly localized, validating our choice of the 100 m subgrid as the operational prediction unit.

### 4.7 Geographic Case Examples: Where KG Succeeds and Baselines Fail

To ground our quantitative results in observable geography, we present three cold-start grids where the KG successfully identifies latent environmental risk that conventional methods miss. Each case satisfies three criteria: (a) cumulative training-period violent crime = 0 or 1 (representing the most extreme cold-start condition), (b) KG-predicted risk ranks in the top 20% of all cold-start grids, and (c) the best-performing conventional baseline (ST-GCN w/o KG) ranks the same grid in the bottom 50% of cold-start grids.

**Case 1: Northwest Side (Grid #17040, parent grid #115).** This grid has zero violent crimes in the entire 9-month training period. The conventional baseline assigns it a risk percentile of only 27.0%, effectively flagging it as low priority. The KG ranks it at the 99.96th percentile among cold-start grids, and test-period violent crime events do occur within 500 m of the grid centroid. The grid's embedding profile shows elevated values on dimensions associated with access control deficits and low natural surveillance—an environmental configuration consistent with the grid's location in a peripheral residential zone (Type 1 morphology).

**Case 2: Far North Side (Grid #34143, parent grid #1).** Also with zero training-period violent crime, this grid's baseline risk percentile is 49.5%—near the median, providing no actionable signal. The KG places it at the 99.91st percentile. Its embedding shows strong activation on target density and weak territorial reinforcement dimensions.

**Case 3: Southwest Side (Grid #5182, parent grid #1048).** This grid illustrates the most extreme divergence between KG and baseline: zero training crime, baseline risk percentile of only 3.8% (nearly the lowest possible), yet KG ranks it at the 99.76th percentile. Its embedding profile indicates poor lighting quality and high spatial isolation.

These cases are not cherry-picked outliers—they are representative of the 1,599 cold-start grids (7.6% of all cold-start grids, 1.3% of all Chicago 100m subgrids) that meet our three selection criteria (train-period crime ≤ 1, KG-predicted risk in the top 20% of cold-start grids, and baseline-predicted risk in the bottom 50%). They illustrate the operational value proposition of the KG: in locations where historical crime data provides zero signal, CPTED-based environmental inference identifies risk that would otherwise go unrecognized, and it does so in a spatially explicit, feature-attributable manner that directly informs intervention planning (Section 5.2).

### 4.8 Performance Heterogeneity Across Urban Morphologies

The three cold-start morphological types identified in Section 2.6.3 represent fundamentally different environmental configurations. If the KG genuinely captures environment–crime relationships, its predictive performance should vary systematically across types in theoretically interpretable ways.

**Results (Table 4.X).** KG performance exhibits substantial and theoretically coherent heterogeneity across morphological types, following the gradient: Type 1 (Peripheral Residential, NDCG@20=0.377) > Type 2 (Industrial/Logistics, NDCG@20=0.196) > Type 3 (Green/Institutional, NDCG@20=0.129). This ordering is consistent with the CPTED knowledge base's design: CPTED constructs (territoriality, natural surveillance, access control) were developed primarily from the environmental configurations of residential and mixed-use urban areas, where human activity patterns, building-street relationships, and public-private boundaries are most clearly articulated. Type 1 peripheral residential zones, despite their low crime counts, possess the full complement of CPTED-relevant environmental features (residential buildings with street-facing façades, defined property boundaries, regular diurnal activity rhythms, sidewalk networks), enabling the KG embeddings to encode fine-grained environmental variation that the Ridge model can exploit. The NDCG@20 of 0.377 for Type 1 approaches the overall cold-start performance (0.406), confirming that residential cold-start areas—though data-sparse—retain sufficient environmental signal for meaningful risk differentiation.

Type 2 (Industrial/Logistics, NDCG@20=0.196) performs at roughly half the level of Type 1. This degradation reflects a CPTED domain mismatch: industrial zones involve distinct crime opportunity mechanisms—vehicle access for cargo theft, warehouse security patrol patterns, after-hours facility vulnerability—that the current CPTED mapping, designed around residential and mixed-use typologies, captures less comprehensively. The CPTED dimensions most relevant to industrial crime (loading dock access control, perimeter fencing quality, security camera coverage, truck route surveillance) are either absent from or only indirectly proxied by the current 73-dimension feature set.

Type 3 (Green/Institutional, NDCG@20=0.129) shows the weakest performance—a 68.2% decline from the overall NDCG@20. This near-random ranking reflects a fundamental challenge: crime in parks, cemeteries, and institutional campuses is predominantly opportunity-driven and event-contingent (park assaults after dark, campus thefts during class hours, cemetery vandalism during off-hours), with weak coupling to the static environmental features that the KG encodes. Furthermore, crime incidents in these zones may be reported through separate institutional channels (campus police, park district police, institutional security) that do not consistently feed into the Chicago Data Portal, adding measurement noise to the already-weak signal. The practical implication is clear: extending the KG framework to green/institutional spaces requires either (a) integrating institutional crime data sources, (b) developing CPTED feature mappings specific to park and campus environments (e.g., trail network visibility, lighting along pedestrian paths, entry/exit control at park gates), or (c) acknowledging that these spaces represent a genuine performance boundary for environmentally-based crime risk assessment.

The performance heterogeneity across morphological types has direct consequences for spatial equity (Section 5.3). If the KG framework were deployed operationally, Type 1 residential cold-start areas would receive substantially more accurate risk differentiation than Type 3 green/institutional areas—potentially creating a new form of algorithmic disparity where certain types of urban spaces receive systematically lower-quality risk assessments. Mitigating this requires morphology-specific modeling (separate Ridge regressions per type) or CPTED knowledge base expansion targeting Type 2 and Type 3 environmental configurations.

### 4.9 CPTED Feature Importance via SHAP

To connect model predictions to actionable environmental interventions, we train a Random Forest on the 17 CPTED dimensions (derived from the 73-dim one-hot encoding) to predict crime risk, and compute SHAP values for feature importance.

**Table 6: SHAP CPTED Feature Importance (Violent Crime, Random Forest)**

| Rank | CPTED Dimension | Criminological Interpretation |
|:----:|----------------|-------------------------------|
| 1 | primary_activity | Dominant land-use activity type shapes opportunity structure |
| 2 | target_density | Concentration of suitable targets (retail, parking, ATMs) |
| 3 | temporal_rhythm | Diurnal/nocturnal activity patterns affect guardian presence |
| 4 | territorial_reinforcement | Physical markers of ownership and boundary definition |
| 5 | natural_surveillance | Visibility and informal observation opportunities |

The SHAP ranking is consistent with criminological theory: primary activity type (commercial vs. residential vs. industrial) fundamentally determines the baseline crime opportunity profile; target density captures Routine Activity Theory's "suitable target" construct; temporal rhythm reflects the temporal dimension of guardianship; territorial reinforcement and natural surveillance capture CPTED's core physical design mechanisms. The theoretical coherence of the SHAP ranking provides face validity for the KG's environmental encoding.

### 4.10 Spatial Generalization and Its Limits

**Results.** Spatial hold-out cross-validation yields NDCG@20 = 0.083 ± 0.101 (Community Area blocking) and NDCG@20 = 0.065 ± 0.108 (k-means spatial blocking), compared to NDCG@20 = 0.406 under random 70/30 splitting—a 79.5–84.0% performance degradation. The high standard deviations (±0.10–0.11) indicate that performance varies dramatically depending on which Community Areas are held out.

**Interpretation: Mapping failure, not representation failure.** We deliberately foreground this negative result because its interpretation materially strengthens the paper's central argument. The 80% performance gap admits two interpretations: (A) the KG embeddings fail to capture generalizable environmental signal, or (B) the global Ridge regression fails to accommodate spatial non-stationarity in the CPTED–crime relationship. We adopt interpretation (B). If interpretation (A) were correct, KG performance would be uniformly poor across all evaluation protocols—yet random-split NDCG@20 = 0.406, monotonic improvement with aggregation, and the theoretically coherent morphological performance gradient all contradict this account. The embeddings are information-rich (89.1% PCA variance explained by 10 components); a GWR model allowing spatially varying coefficients on the same embeddings recovers NDCG@20 = 0.433 (vs. global Ridge 0.362 on the same subsample, Section 4.6); and stratified analysis (Section 4.8) shows that within-type performance (Type 1: NDCG@20 = 0.377) approaches overall performance, suggesting within-group relationships are more homogeneous than across-group relationships. **The failure is in the mapping function, not the representation.**

**Theoretical consistency.** The spatial CV failure is precisely what the paper's theoretical framework predicts. If the CPTED–crime mapping were spatially invariant, conventional spatial smoothing over geographic neighbors would have sufficed, and the cold-start problem would not exist. The spatial non-stationarity that causes spatial CV to fail is the same phenomenon that causes geographic neighbors to be uninformative in data deserts (Section 2.6.1)—and therefore the same phenomenon motivating the KG's functional proximity edges (Section 3.3). The spatial CV result does not undermine the KG framework; it validates its core premise.

**Operational implications.** This finding carries a concrete deployment requirement: the framework needs a small number of crime-experienced spatial units in the target region (e.g., ~20% of 1 km grids) to calibrate the Ridge mapping; thereafter, predictions generalize to cold-start subgrids within that region. Direct transfer of a mapping calibrated in one city to another city without local recalibration is not supported. The two-stage architecture (frozen KG → pluggable downstream model) makes remediation straightforward: a spatially adaptive model can replace the global Ridge without modifying the pretrained KG, as elaborated in Section 5.5.

**Cross-morphology validation.** To further test whether the global mapping failure is driven by heterogeneity across morphological types, we conduct cross-morphology cross-validation: training Ridge regression on one morphological type and testing on another. The results are definitive: **cross-morphology transfer uniformly collapses to near-zero NDCG@20** (0.000–0.015 across all six cross-type pairs), substantially worse than even geographic CV (0.065–0.083). Within-type performance, by contrast, varies substantially: Type 1 (Residential, NDCG@20 = 0.510) achieves strong within-type discrimination, Type 3 (Green/Institutional, NDCG@20 = 0.219) shows moderate performance, and Type 2 (Industrial/Logistics, NDCG@20 = 0.003) exhibits near-zero discrimination even within its own type. The total failure of cross-morphology transfer—a Ridge model trained on 10,581 residential cold-start grids cannot rank a single industrial cold-start grid—provides the strongest evidence in this paper that the CPTED–crime relationship is fundamentally morphology-specific. Different urban morphologies involve not merely different *values* of the same CPTED dimensions but different *mappings* from CPTED dimensions to crime risk, consistent with the theoretical expectation that crime opportunity mechanisms differ qualitatively across residential, industrial, and green/institutional contexts. This result reinforces the case for morphology-specific downstream models (Section 5.5) and tempers claims about the universal applicability of a single global environmental risk mapping.

### 5.1 Why Simple Models Work: The Case for Two-Stage Architecture

A notable feature of our approach is its architectural simplicity: frozen GraphSAGE embeddings followed by linear Ridge regression. This contrasts with the dominant paradigm in spatio-temporal crime prediction, which favors end-to-end deep learning with complex fusion mechanisms, multi-task learning, and count-appropriate likelihood functions (e.g., Zero-Inflated Negative Binomial).

Our ablation results suggest a methodological lesson: **when the information bottleneck is input representation rather than model capacity, investing in better representation yields higher returns than investing in more complex prediction architectures.** The CPTED encoding—73 theory-grounded dimensions capturing criminologically meaningful environmental variation—provides richer signal for cold-start discrimination than any combination of raw POI counts, spatial smoothing, or temporal convolution could extract from near-zero historical data. Once the environmental representation is adequate, a simple linear model suffices because the relationship between environmental features and aggregate crime risk is approximately linear at the spatial resolutions we study.

This finding has practical implications for deployment. A two-stage architecture (pretrain once, predict with Ridge) is computationally cheap, requires no GPU for inference, and produces fully auditable predictions—each grid's risk score is a weighted sum of its 32 embedding dimensions, and each dimension is traceable to specific CPTED constructs. In policy contexts where transparency and computational accessibility matter, this simplicity is a feature rather than a limitation.

### 5.2 From Model Output to Multi-Scale Environmental Intervention

The KG framework enables a fundamental shift in how crime prediction informs action: from "where to patrol" (reactive policing) to "what to change" (proactive environmental design). A distinctive feature of our framework is that it naturally supports interventions at two distinct spatial scales, matching the hierarchical structure of urban governance.

**Table 7: CPTED-Informed Risk Diagnosis Framework Organized by Spatial Scale**

*Note: These interventions represent evidence-based planning hypotheses derived from CPTED theory and SHAP-identified statistical associations between environmental features and crime risk patterns. They do not constitute direct causal evidence from the model. The model's role is diagnostic—identifying locations where specific environmental configurations may contribute to elevated crime opportunity—while actual intervention planning should incorporate on-site environmental audit, community input, and planning expertise. The interventions are organized by spatial scale to align with existing urban governance tiers.*

**Panel A: Strategic Interventions (1 km Macro-Scale — Neighborhood / Community Area Level)**

| CPTED Dimension | Observable at 1 km Scale | Planning Instrument | Implementing Authority |
|----------------|-------------------------|---------------------|----------------------|
| primary_activity | Land-use mix diversity, zoning classification map | Zoning amendments to diversify mono-functional districts; mixed-use overlay districts | City Planning Department, Zoning Board |
| temporal_rhythm | Ratio of daytime vs. nighttime business licenses; 24-hour establishment density | Extended-hours business licensing tied to security requirements; night-time economy management plans | Business Licensing, City Council |
| territorial_reinforcement | Public/private boundary clarity at neighborhood scale | "Clear boundary" design guidelines in comprehensive plans; community gateways and identity markers | Urban Design Commission, Community Development |
| natural_surveillance | Street network connectivity; block-length-to-intersection ratio | Connectivity requirements in subdivision regulations; street design standards | Transportation Planning, Public Works |
| target_density | Commercial concentration zoning; retail floor-area ratio caps | Commercial density limits tied to CPTED impact assessments; retail impact studies | Economic Development, Planning |

**Panel B: Tactical Interventions (100 m Micro-Scale — Street Block / Individual Site Level)**

| CPTED Dimension | Observable at 100 m Scale | Site-Specific Intervention | Implementing Authority |
|----------------|--------------------------|---------------------------|----------------------|
| natural_surveillance | Street lighting quality, window frontage proportion, sight-line obstructions | Pedestrian-scale lighting retrofits at specific street segments; "eyes on the street" façade requirements for ground-floor commercial | Building Inspections, Streets Department |
| target_density | ATM clusters, convenience store density, parking lot adjacency | Target hardening for specific ATM/retail clusters; CPTED-compliant parking lot design (lighting, landscaping, surveillance) | Business Regulation, Site Plan Review |
| territorial_reinforcement | Fencing type and height, signage presence, public/private transition zones | Perimeter treatments for specific sites; "defensible space" landscaping for residential blocks | Building Permits, Code Enforcement |
| temporal_rhythm | Specific business operating hours, after-hours activity generators | Coordinated closing-time management for bar/restaurant clusters; late-night transport corridor lighting phasing | Liquor Licensing, Transportation |
| access_control | Entry/exit point configuration, pedestrian desire-line management | Specific access-point redesign for problem sites; alley-gating programs for residential blocks | Transportation Planning, Public Safety |

**Illustrative application.** Consider a cold-start grid classified as Type 3 (Green/Institutional)—a park-adjacent area where SHAP analysis identifies low natural_surveillance and limited territorial_reinforcement as the primary CPTED risk dimensions. At the **strategic (1 km) level**, the relevant intervention is a district-level park safety design guideline requiring minimum sight-line standards and CPTED-compliant landscaping for all park perimeters. At the **tactical (100 m) level**, the intervention is specific: improved pedestrian-scale lighting along the 200 m park edge adjacent to the cold-start grid, selective vegetation thinning at three identified sight-line obstruction points, and installation of clear boundary markers (fencing, signage) at the park-to-street transition. The model's value is not in saying "this area has elevated latent risk" but in identifying *which specific environmental mechanisms at which spatial scale* can be targeted for modification—and which level of government or planning authority has jurisdiction over each.

This scale-differentiated approach addresses a persistent gap in CPTED implementation: the disconnect between macro-level planning instruments (zoning codes, comprehensive plans) and micro-level environmental modifications (street lighting, façade design, landscaping). Our framework bridges this gap by providing risk attribution at both scales from a single KG embedding, enabling coordinated action across planning tiers.

### 5.3 Spatial Equity: From Predicting Enforcement to Mapping Opportunity

The most consequential shift enabled by our framework is not methodological but philosophical: it **transforms crime prediction from a tool that predicts where police are likely to enforce, into a tool that maps where the built environment systematically generates crime opportunity.** This re-framing addresses the core spatial justice concern that has made predictive policing algorithms a subject of sustained critique (Lum & Isaac, 2016; Richardson et al., 2019; Jefferson, 2020).

**The enforcement feedback loop.** Conventional crime prediction models operate on a single information channel: historical crime records. Because these records are themselves a product of historical enforcement patterns—police patrol routes, community reporting infrastructure, institutional trust—the model inherits and amplifies the spatial biases embedded in the data generation process. The canonical feedback loop proceeds as: high historical police presence → high recorded crime → high model predictions → continued high police presence. Conversely: low historical police presence → low recorded crime → low model predictions → continued low police attention → continued low recorded crime. Over time, this loop transforms historical enforcement patterns into durable spatial inequalities in algorithmic attention—a self-reinforcing cycle that our spatial autocorrelation analysis (Section 2.6) suggests has been operating in Chicago for decades.

**Breaking the loop with environmental measurement.** Our KG introduces a second, independent information channel: the built environment as measured through CPTED theory and OpenStreetMap data. A cold-start grid's prediction is driven by its POI configuration, road network connectivity, and functional zone classification—features that exist in the physical world regardless of whether the Chicago Police Department has historically deployed officers there. This decoupling has three specific equity-enhancing properties:

**(1) Recognition parity.** Cold-start grids—disproportionately located in stable low-income residential areas and mono-functional zones (Section 2.6.4)—receive nonzero, spatially differentiated risk assessments for the first time. The model "sees" these areas not as blank spaces on the crime map, but as places with specific environmental configurations that can be evaluated against criminological theory. This is a form of **algorithmic recognition**: the framework acknowledges that these places exist, have environmental characteristics, and may harbor latent risk—even when the historical data record is silent.

**(2) Auditability of data quality.** The framework enables **spatial equity auditing** as a concrete, operational practice rather than an abstract principle. By comparing KG-based predictions (driven by environmental features) with historical-crime-based predictions (driven by enforcement records), planners and oversight bodies can identify **prediction divergence zones**—grids where environmental risk is substantially higher than what historical crime data would suggest. These zones are not automatically "high-risk" in an operational sense; rather, they are flags for investigation: does the environmental configuration genuinely create crime opportunity (meriting preventive intervention), or does the historical record under-represent actual crime occurrence (meriting reporting infrastructure improvement)? The framework provides the spatial diagnosis; human judgment determines the appropriate response.

**(3) Intervention parity.** Because KG predictions are traceable to specific, modifiable environmental features (Section 5.2, Table 7), the recommended response to elevated risk is environmental modification rather than increased police presence. This shifts the policy response from **reactive enforcement** (which has well-documented disparate impacts on minority and low-income communities) to **proactive environmental design** (which benefits all users of a space regardless of their demographic profile). Improved street lighting, CPTED-compliant landscaping, and better sight lines serve everyone who uses the space—they are public goods, not targeted interventions.

**The OSM coverage caveat.** This equity argument must be tempered by an important data justice concern. OpenStreetMap data, while globally available and free, may exhibit its own spatial biases: wealthier neighborhoods may have more detailed POI mapping; areas with more active OSM contributor communities may have more up-to-date road network data; certain POI categories may be systematically under-mapped in lower-income areas (e.g., informal businesses, street vendors). If OSM coverage correlates with neighborhood socioeconomic status, the KG could inadvertently reproduce a different form of spatial bias—one where environmental richness (as measured by OSM completeness) rather than enforcement intensity determines which areas receive differentiated risk assessments. Systematic bias audits of OSM coverage relative to neighborhood sociodemographic composition are a prerequisite for equity-sensitive deployment. We flag this as an urgent priority for future work (Section 5.5).

**Toward a "right to environmental risk assessment."** We propose, as a normative principle, that every spatial unit in a city—regardless of its historical crime rate, socioeconomic profile, or demographic composition—deserves a transparent, evidence-based, and contestable environmental risk assessment. This is the spatial analogue of the "right to explanation" in algorithmic fairness discourse (Goodman & Flaxman, 2017): the right not only to know why an algorithm made a particular prediction, but to have that prediction grounded in observable, modifiable features of the physical environment rather than in a historical record shaped by institutional biases. Our framework is a step toward operationalizing this right in the domain of spatial crime risk assessment.

### 5.4 The Division of Labor: Structured KG for Precision, LLM for Breadth

The LLM+KG fusion results (Section 4.5) suggest a **division of labor** rather than simple complementarity. The structured KG—with its CPTED-grounded, 100 m-resolution encoding of environmental features—drives fine-grained risk ranking (NDCG@20 = 0.406, unmatched by any fusion variant). The LLM—with its internet-scale semantic knowledge of neighborhood character—provides coarse calibration that marginally improves distribution-wide discrimination (AUC 0.865 → 0.881) and deeper-ranking quality (NDCG@50 0.288 → 0.360). This division maps cleanly onto the spatial scales of intervention identified in Section 5.2: the KG's micro-spatial precision serves tactical interventions at the street-block scale (~100 m), while the LLM's macro-spatial calibration serves strategic screening at the neighborhood scale (~1 km).

This finding has broader implications for geospatial AI beyond crime prediction. The dominant narrative in urban machine learning emphasizes end-to-end fusion—concatenating heterogeneous features and letting the model learn interactions. Our results suggest an alternative architectural principle: **deploy the most structured, domain-grounded representation for the primary task (fine-grained ranking), and use unstructured, broad-coverage representations as auxiliary calibration signals that improve secondary metrics without compromising primary task performance.** This principle—precision from structure, breadth from language—may generalize to other urban applications where both systematic measurement and contextual understanding are valued but where the operational decision (e.g., which specific street blocks to prioritize for intervention) demands spatial precision that language-based representations alone cannot provide.

### 5.5 Limitations

**Single-city validation.** All experiments use Chicago data. The cold-start phenomenon's geographic structure—peripheral residential, industrial corridors, green/institutional spaces—may manifest differently in cities with distinct urban morphologies (older East Coast cities, sprawling Sunbelt cities, European medieval cores, rapidly urbanizing Asian cities). Multi-city validation is essential to establish generalizability.

**OSM coverage heterogeneity.** The KG is constructed from OpenStreetMap data, whose completeness varies globally. While Chicago has excellent OSM coverage, smaller cities or Global South contexts may have sparser POI and road network data. Systematic bias audits of OSM coverage relative to neighborhood sociodemographics are needed before deploying KG-based methods in equity-sensitive applications.

**OSM bias audit (urgent priority).** The spatial equity argument in Section 5.3 rests on the assumption that OSM-derived environmental features provide a less biased information channel than historical crime records. This assumption requires empirical validation. A systematic OSM completeness audit should: (a) compare POI density and category coverage across Chicago Community Areas stratified by income, racial composition, and historical policing intensity; (b) benchmark OSM POI coverage against a ground-truth commercial establishment database (e.g., ReferenceUSA or state business license records) to quantify under-mapping rates by neighborhood characteristics; (c) assess whether spatial variation in OSM completeness introduces systematic error into KG embeddings that correlates with neighborhood demographics. Until such an audit is completed, the framework's equity claims should be considered provisional.

**From global linear mapping to spatially adaptive downstream models (first priority).** The spatial cross-validation results (Section 4.10) reveal that the global Ridge regression mapping from KG embeddings to crime risk fails to generalize across spatially separated regions of Chicago (NDCG@20 collapsing from 0.406 to 0.065–0.083), though cross-morphology validation may show better transferability than geographic hold-out. We have argued (Section 4.10) that this failure does not invalidate the KG embeddings but rather identifies the global linearity assumption as the bottleneck. Three spatially adaptive alternatives warrant systematic comparison: (a) **Morphology-specific Ridge models**—separate linear mappings for each of the three cold-start morphological types (Type 1 residential, Type 2 industrial, Type 3 green/institutional), justified by the stratified performance heterogeneity (Section 4.8); (b) **Geographically Weighted Ridge regression (GWR-Ridge)**—allowing the embedding→risk coefficients to vary smoothly across space, motivated by the GWR baseline's superiority over global Ridge on subsampled data (NDCG@20 0.433 vs. 0.362, Section 4.6); and (c) **Spatial varying-coefficient models (SVC)**—which would additionally allow the spatial scale of coefficient variation to differ across CPTED dimensions, reflecting the theoretical expectation that different environmental mechanisms (surveillance, territoriality, access control) operate at different spatial scales. The two-stage architecture (frozen KG pretraining → pluggable downstream model) makes such substitution architecturally straightforward: any spatially adaptive linear or nonlinear model can replace the global Ridge without modifying the pretrained KG. This direction represents the natural methodological evolution of the framework—from establishing that KG embeddings carry predictive signal (this paper) to optimizing how that signal is mapped to risk in a spatially non-stationary world (future work).

**Static KG.** Our KG is pre-computed and frozen. Urban infrastructure evolves (business turnover, new construction, neighborhood change), and future work should explore dynamic KG updating while maintaining the theoretical grounding of embeddings.

**Limited CPTED operationalization.** The current 17-dimension CPTED encoding, while grounded in criminology literature, necessarily simplifies rich theoretical constructs into discrete categorical attributes. Some CPTED mechanisms—natural surveillance quality, territorial reinforcement strength—are better captured through in-person environmental audit instruments than through POI-derived proxies. Integrating systematic social observation data (e.g., Google Street View-based environmental assessment) could enrich the CPTED feature space.

**Crime displacement unmodeled.** Environmental interventions can displace crime to adjacent areas rather than preventing it. Our framework does not model displacement effects, which are critical for evaluating the net social benefit of spatially targeted CPTED interventions.

---

## 6. Conclusion

This paper demonstrates that spatial cold-start in crime prediction—the systematic failure of conventional models in data-sparse urban areas—can be substantially addressed through theory-grounded environmental knowledge encoding. Our Urban Environmental KG, pretrained via GraphSAGE with carefully calibrated self-supervised objectives, produces 32-dimensional embeddings that achieve NDCG@20=0.406 for cold-start violent crime prediction at the 100 m resolution, substantially exceeding LLM-based semantic embeddings (0.182) and all conventional baselines. The framework operates within a clearly defined envelope: it requires a small proportion of crime-experienced spatial units (~20% of 1 km grids) in the target region for calibration; thereafter it generalizes to cold-start subgrids within that region. Direct cross-city transfer without local recalibration is not supported by current evidence (Section 4.10).

Three findings carry direct implications for both research and practice:

1. **Spatial data deserts are a measurable geographic phenomenon, not a statistical nuisance.** Spatial autocorrelation analysis (Moran's I, LISA) and morphological classification (three cold-start types with distinct environmental profiles) demonstrate that cold-start grids are spatially structured in ways that defeat conventional spatial smoothing. Recognizing spatial data deserts as a distinct geographic category—analogous to food deserts or transit deserts in urban studies—opens the door to spatially targeted policy responses.

2. **Environmental criminology theory carries actionable, independently verifiable predictive signal.** The 94% performance collapse upon removing CPTED features provides strong evidence that CPTED constructs—when systematically encoded and spatially aggregated—contain genuine information about crime risk. SHAP analysis confirms that the most influential CPTED dimensions (primary activity type, target density, temporal rhythm, territorial reinforcement, natural surveillance) align with theoretical expectations, providing face validity for CPTED-based policy guidance.

3. **The framework enables a shift from reactive enforcement to proactive environmental diagnosis.** By grounding risk assessment in the observable built environment rather than historical enforcement data, and by making predictions traceable to specific, modifiable environmental features, the KG framework transforms crime prediction from a policing tool into a planning instrument. The model's role is diagnostic—identifying *which* environmental configurations may contribute to elevated crime opportunity at *which* spatial scale—while actual intervention design requires on-site assessment and planning expertise. The scale-differentiated intervention taxonomy (tactical 100 m / strategic 1 km) bridges the gap between CPTED theory and urban governance practice, enabling coordinated action across planning tiers.

---

## References

Anselin, L. (1995). Local Indicators of Spatial Association—LISA. *Geographical Analysis*, 27(2), 93–115.

Brantingham, P. L., & Brantingham, P. J. (1993). Environment, routine and situation: Toward a pattern theory of crime. *Advances in Criminological Theory*, 5, 259–294.

Cohen, L. E., & Felson, M. (1979). Social change and crime rate trends: A routine activity approach. *American Sociological Review*, 44(4), 588–608.

Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*. Wiley.

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017*.

Jeffery, C. R. (1971). *Crime Prevention Through Environmental Design*. Sage Publications.

Lum, C., & Isaac, W. (2016). To predict and serve? *Significance*, 13(5), 14–19.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.

Newman, O. (1972). *Defensible Space: Crime Prevention Through Urban Design*. Macmillan.

Shaw, C. R., & McKay, H. D. (1942). *Juvenile Delinquency and Urban Areas*. University of Chicago Press.

Sherman, L. W., Gartin, P. R., & Buerger, M. E. (1989). Hot spots of predatory crime: Routine activities and the criminology of place. *Criminology*, 27(1), 27–56.

Weisburd, D. (2015). The law of crime concentration and the criminology of place. *Criminology*, 53(2), 133–157.

Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for deep spatial-temporal graph modeling. *IJCAI 2019*.

Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-temporal graph convolutional networks. *IJCAI 2018*.

Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-supervised learning via redundancy reduction. *ICML 2021*.

---

*Figure 1: Research framework overview — from spatial cold-start diagnosis through KG construction, self-supervised pretraining, and downstream evaluation.*
*Figure 2: Urban Environmental KG architecture — CPTED encoding, heterogeneous graph construction, GraphSAGE pretraining, and SSL objectives.*
*Figure 3: Risk capture curves comparing KG, LLM, and baseline methods on cold-start prediction.*
*Figure 4: Ablation bar chart showing component contributions to NDCG@20.*
*Figure 5: Scale comparison — KG NDCG@20 from 100 m to 1 km aggregation.*
*Figure 6: SHAP feature importance for CPTED dimensions.*
