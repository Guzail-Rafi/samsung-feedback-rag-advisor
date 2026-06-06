export const kpis = {
  comments: 15000,
  totalComments: 15000,
  videos: 10,
  ragAccuracy: 0.93,
  topIssue: "Battery / Charging",
  sentiment: { positive: 0.48, neutral: 0.31, negative: 0.21 },
};

export const chartInitialDimension = { width: 520, height: 288 };

export const sentimentData = [
  { name: "Positive", value: 48, color: "#1428A0" },
  { name: "Neutral", value: 31, color: "#64748B" },
  { name: "Negative", value: 21, color: "#D14343" },
];

export const issueData = [
  {
    issue: "Battery / Charging",
    count: 846,
    share: "26%",
    sentiment: "Mixed",
    urgency: "High",
    recommendation: "Increase real-world endurance and charging value.",
  },
  {
    issue: "S-Pen / Features",
    count: 612,
    share: "19%",
    sentiment: "Negative",
    urgency: "High",
    recommendation: "Restore Bluetooth-style utility for Ultra buyers.",
  },
  {
    issue: "Camera Quality",
    count: 574,
    share: "18%",
    sentiment: "Mixed",
    urgency: "High",
    recommendation: "Prioritize creator-facing camera upgrades early.",
  },
  {
    issue: "AI / Gemini",
    count: 498,
    share: "15%",
    sentiment: "Mixed",
    urgency: "Medium",
    recommendation: "Tie AI features to daily tasks, not novelty demos.",
  },
  {
    issue: "Display / Durability",
    count: 421,
    share: "13%",
    sentiment: "Negative",
    urgency: "Medium",
    recommendation: "Make durability a visible proof point.",
  },
  {
    issue: "Price / Value",
    count: 337,
    share: "10%",
    sentiment: "Negative",
    urgency: "Medium",
    recommendation: "Defend flagship price with hardware differentiation.",
  },
];

export const topKeywords = [
  { keyword: "ai", score: 247.83 },
  { keyword: "fake", score: 247.57 },
  { keyword: "pen", score: 239.87 },
  { keyword: "better", score: 215.87 },
  { keyword: "buy", score: 214.36 },
  { keyword: "year", score: 197.57 },
  { keyword: "good", score: 164.59 },
  { keyword: "best", score: 161.92 },
  { keyword: "watching", score: 155.14 },
  { keyword: "battery", score: 151.42 },
];

export const categoryKeywords = [
  { category: "Battery / Charging", keyword: "battery", score: 58.36 },
  { category: "Battery / Charging", keyword: "charger", score: 38.81 },
  { category: "Battery / Charging", keyword: "charging", score: 28.32 },
  { category: "S-Pen / Features", keyword: "pen", score: 61.77 },
  { category: "S-Pen / Features", keyword: "bluetooth", score: 45.19 },
  { category: "S-Pen / Features", keyword: "feature", score: 36.06 },
  { category: "AI / Gemini", keyword: "ai", score: 72.48 },
  { category: "AI / Gemini", keyword: "gemini", score: 34.21 },
  { category: "Camera Quality", keyword: "camera", score: 49.84 },
  { category: "Price / Value", keyword: "worth", score: 27.44 },
];

export const topics = [
  {
    id: 0,
    name: "General Samsung / Apple Discussion",
    words: "apple, like, samsung, phones, video, people",
    signal: "Brand comparison drives a large share of casual discussion.",
  },
  {
    id: 1,
    name: "AI, Battery, Camera and Feature Discussion",
    words: "ultra, s25, s24, galaxy, watching",
    signal: "Flagship buyers evaluate bundled hardware and software together.",
  },
  {
    id: 2,
    name: "Galaxy Models / S23-S24-S25 Series",
    words: "samsung, phone, new, google, ai, buy",
    signal: "Upgrade intent clusters around model-to-model comparisons.",
  },
  {
    id: 3,
    name: "Buying Decision / Value Discussion",
    words: "pen, bluetooth, feature, spen, note",
    signal: "Removed features are treated as value loss by Ultra loyalists.",
  },
  {
    id: 4,
    name: "Fake Products / Screen / Camera Concerns",
    words: "phone, screen, best, year, privacy",
    signal: "Trust, display quality, and camera claims shape purchase confidence.",
  },
  {
    id: 5,
    name: "S-Pen / Bluetooth Removal",
    words: "iphone, better, pro, samsung, android",
    signal: "Competitor comparisons amplify feature-removal frustration.",
  },
  {
    id: 6,
    name: "Upgrade Cycle / Yearly Phone Changes",
    words: "fake, phone, ultra, need, worth, price",
    signal: "Users question whether yearly upgrades create enough visible change.",
  },
  {
    id: 7,
    name: "S25 vs S24 / iPhone Pro Max Comparison",
    words: "ai, battery, features, charging, camera",
    signal: "AI does not replace demand for battery, camera, and charging gains.",
  },
];

export const sentimentTopics = [
  { sentiment: "Positive", topic: "Brand / Product Praise", words: "samsung, phone, ultra, pen, best", color: "#1428A0" },
  { sentiment: "Positive", topic: "Camera and Design Feedback", words: "samsung, apple, iphone, galaxy, better", color: "#1428A0" },
  { sentiment: "Positive", topic: "AI / Feature Appreciation", words: "phone, better, good, ai, want", color: "#1428A0" },
  { sentiment: "Negative", topic: "Feature Removal Concerns", words: "phone, new, ai, battery, old", color: "#D14343" },
  { sentiment: "Negative", topic: "Battery / Charging Issues", words: "samsung, pen, bluetooth, features, use", color: "#D14343" },
  { sentiment: "Negative", topic: "Price / Value Concerns", words: "ultra, s25, s24, upgrade, screen", color: "#D14343" },
  { sentiment: "Neutral", topic: "Product Discussion", words: "apple, samsung, iphone, buy, phones", color: "#64748B" },
  { sentiment: "Neutral", topic: "Model / Upgrade Discussion", words: "phone, samsung, years, s23, android", color: "#64748B" },
];

export const entities = [
  { entity: "samsung", type: "Brand", mentions: 3984, sentiment: "Mixed" },
  { entity: "galaxy", type: "Product family", mentions: 1688, sentiment: "Positive" },
  { entity: "s25 ultra", type: "Product", mentions: 1296, sentiment: "Mixed" },
  { entity: "iphone", type: "Competitor", mentions: 1174, sentiment: "Mixed" },
  { entity: "apple", type: "Competitor", mentions: 1091, sentiment: "Mixed" },
  { entity: "s pen", type: "Feature", mentions: 842, sentiment: "Negative" },
  { entity: "gemini", type: "AI product", mentions: 426, sentiment: "Mixed" },
  { entity: "oneplus", type: "Competitor", mentions: 284, sentiment: "Neutral" },
];

export const entityExamples = [
  {
    entity: "s pen",
    comment: "Users repeatedly connect Ultra identity with S-Pen utility and Bluetooth-style controls.",
    issue: "S-Pen / Features",
  },
  {
    entity: "iphone",
    comment: "Competitor comparisons appear most often when users discuss price, camera, and upgrade value.",
    issue: "Price / Value",
  },
  {
    entity: "gemini",
    comment: "AI is discussed as interesting, but less decisive than battery, camera, and durability.",
    issue: "AI / Gemini",
  },
];

export const ragQuestions = [
  {
    query: "What are users saying about Samsung battery life?",
    confidence: "High",
    score: 0.676,
    answer:
      "Battery feedback is polarized. Some users praise endurance, while others describe one-day battery life as below flagship expectations.",
    evidence: ["battery life is amazing", "still aiming for 1-day battery life", "need larger battery"],
  },
  {
    query: "Why are users unhappy about the S-Pen?",
    confidence: "High",
    score: 0.717,
    answer:
      "Users see the Bluetooth S-Pen removal as a regression because it weakens the Ultra identity and reduces everyday utility.",
    evidence: ["removed bluetooth spen", "used the pen feature", "Ultra should keep Note features"],
  },
  {
    query: "What do users think about Galaxy AI and Gemini?",
    confidence: "High",
    score: 0.693,
    answer:
      "Users are curious about AI, but many comments ask Samsung to make AI practical and not use it as a substitute for hardware improvements.",
    evidence: ["galaxy ai", "gemini", "we don't care about galaxy ai"],
  },
  {
    query: "Are users comparing Samsung with Apple?",
    confidence: "High",
    score: 0.707,
    answer:
      "Yes. Apple and iPhone appear frequently when users evaluate camera quality, ecosystem value, and whether Samsung still feels meaningfully different.",
    evidence: ["better than iphone", "apple comparison", "pro max"],
  },
  {
    query: "What are users saying about Samsung camera quality?",
    confidence: "Medium",
    score: 0.689,
    answer:
      "Camera quality remains a key buying factor. Users want visible improvements, especially for creators and Pro Max comparisons.",
    evidence: ["camera quality", "creator", "pro max comparison"],
  },
];

export const retrievedEvidence = [
  {
    query: "Battery life",
    comment:
      "Aiming for 1 day battery life since 10 years is pathetic... is 2 day battery life in a phone too much to ask for?",
    sentiment: "negative",
    issue: "Battery / Charging",
    topic: "AI, Battery, Camera and Feature Discussion",
    score: 0.672,
  },
  {
    query: "S-Pen",
    comment: "Yeah idk why they removed bluetooth spen in the S25 Ultra.",
    sentiment: "positive",
    issue: "S-Pen / Features",
    topic: "Buying Decision / Value Discussion",
    score: 0.717,
  },
  {
    query: "AI / Gemini",
    comment:
      "Dear Samsung, for the upcoming Z Fold 8, we don't care about Galaxy AI... what we want is S Pen support, larger battery, and a more durable screen.",
    sentiment: "positive",
    issue: "AI / Gemini",
    topic: "Galaxy Models / S23-S24-S25 Series",
    score: 0.693,
  },
  {
    query: "Apple comparison",
    comment: "This is better than iPhone.",
    sentiment: "positive",
    issue: "Design / Build",
    topic: "S-Pen / Bluetooth Removal",
    score: 0.707,
  },
  {
    query: "Camera quality",
    comment: "People are complaining about battery life, not zoom cameras. Companies are trying to make phones for YouTubers, not everyday people.",
    sentiment: "positive",
    issue: "Camera",
    topic: "Buying Decision / Value Discussion",
    score: 0.689,
  },
];

export const ragEvaluation = [
  { query: "Battery life", precision: 1.0, weighted: 0.676, similarity: 0.686, lexical: 0.9 },
  { query: "S-Pen", precision: 1.0, weighted: 0.717, similarity: 0.714, lexical: 0.9 },
  { query: "AI / Gemini", precision: 1.0, weighted: 0.693, similarity: 0.718, lexical: 0.9 },
  { query: "Apple comparison", precision: 1.0, weighted: 0.707, similarity: 0.665, lexical: 0.8 },
  { query: "Camera quality", precision: 0.8, weighted: 0.689, similarity: 0.743, lexical: 0.75 },
  { query: "Display issues", precision: 0.8, weighted: 0.656, similarity: 0.677, lexical: 0.6 },
];

export const rerankerEvaluation = [
  { query: "Battery life", precision: 0.2, score: 0.005, max: 0.019 },
  { query: "S-Pen", precision: 0.8, score: 0.278, max: 0.986 },
  { query: "AI / Gemini", precision: 0.8, score: 0.196, max: 0.662 },
  { query: "Apple comparison", precision: 0.8, score: 0.207, max: 0.652 },
  { query: "Camera quality", precision: 0.0, score: 0.005, max: 0.009 },
  { query: "Display issues", precision: 0.2, score: 0.004, max: 0.012 },
];

export const strategyPriorities = [
  {
    phase: "Phase 1",
    title: "Restore flagship trust",
    focus: "Battery, camera, S-Pen utility",
    rationale: "These concerns carry the strongest satisfaction and value signals.",
  },
  {
    phase: "Phase 2",
    title: "Make AI useful",
    focus: "On-device editing, search, support, and creator workflows",
    rationale: "AI is visible in discussion but must solve concrete user tasks.",
  },
  {
    phase: "Phase 3",
    title: "Defend Ultra pricing",
    focus: "Durability, display confidence, trade-in messaging",
    rationale: "Price concerns fall when users can see premium hardware value.",
  },
];

export const strategyRecommendations = [
  {
    goal: "Customer satisfaction",
    recommendation:
      "Design the S27 Ultra as a no-compromise Ultra: stronger battery life, meaningful camera upgrades, restored S-Pen differentiation, and more visible display durability.",
    evidence: "Battery, S-Pen, camera, and display concerns appear across RAG and topic outputs.",
    score: 0.86,
  },
  {
    goal: "Competitive differentiation",
    recommendation:
      "Position Samsung against iPhone Pro Max by emphasizing multitasking, S-Pen workflows, Galaxy AI that saves time, and creator camera improvements.",
    evidence: "Apple and iPhone comparisons are frequent in entity and topic outputs.",
    score: 0.82,
  },
  {
    goal: "Business value",
    recommendation:
      "Bundle visible hardware upgrades with trade-in offers and AI services that create retention beyond the launch cycle.",
    evidence: "Price/value and yearly upgrade discussions show buyers need a clearer reason to upgrade.",
    score: 0.78,
  },
];

export const refinementExamples = [
  {
    feedback: "Phase 1 should also include camera improvement because creators care about camera quality.",
    result: "Camera moves into Phase 1 beside battery and S-Pen restoration.",
  },
  {
    feedback: "The roadmap should reduce risk around AI fatigue.",
    result: "AI shifts toward practical workflows and away from novelty-led messaging.",
  },
  {
    feedback: "Make the recommendation stronger for price-sensitive buyers.",
    result: "Trade-in, durability, and visible hardware value become launch proof points.",
  },
];

export const refinementDecisions = [
  {
    feedback: "Phase 1 should also include camera improvement because creators care about camera quality.",
    verdict: "Accepted",
    reason: "Camera complaints appear in RAG answers and strategy refinement output, so moving camera into Phase 1 is evidence-backed.",
    phase: "Phase 1",
  },
  {
    feedback: "Move premium AI features earlier for profit.",
    verdict: "Alternative suggested",
    reason: "AI can support profit, but the evidence says hardware value should come first. Put practical AI workflows in Phase 2.",
    phase: "Phase 2",
  },
  {
    feedback: "Add display durability and green-line prevention to Phase 1.",
    verdict: "Accepted",
    reason: "Display trust is a complaint and risk signal, so it belongs in the first reliability-focused phase.",
    phase: "Phase 1",
  },
  {
    feedback: "Remove S-Pen work to save cost.",
    verdict: "Rejected",
    reason: "The S-Pen is tied to Ultra identity. Removing it would conflict with the customer satisfaction evidence.",
    phase: "No change",
  },
];

export const agentRoutes = [
  { query: "Give me an overall summary of Samsung feedback.", agent: "summarization_agent" },
  { query: "What should Samsung prioritize for S27 Ultra?", agent: "strategy_agent" },
  { query: "Why are people upset about the S-Pen?", agent: "rag_answer_agent" },
  { query: "Which issues are most common?", agent: "issue_analysis_agent" },
];

export const pipelineStages = [
  { stage: "YouTube collection", output: "youtube_comments.csv", status: "Ready" },
  { stage: "Preprocessing", output: "clean_comments.csv", status: "Ready" },
  { stage: "Spell correction", output: "comments_with_spellcheck.csv", status: "Ready" },
  { stage: "Sentiment analysis", output: "comments_with_sentiment.csv", status: "Ready" },
  { stage: "Issue classification", output: "comments_with_categories.csv", status: "Ready" },
  { stage: "Keyword extraction", output: "top_keywords_overall.csv", status: "Ready" },
  { stage: "Topic modeling", output: "topic_keywords.csv", status: "Ready" },
  { stage: "Named entities", output: "ner_entities.csv", status: "Ready" },
  { stage: "RAG answers", output: "rag_answers.csv", status: "Ready" },
  { stage: "Strategy RAG", output: "strategy_rag_results.csv", status: "Ready" },
  { stage: "Agent routing", output: "agent_router_results.csv", status: "Ready" },
  { stage: "MLflow monitoring", output: "mlruns/", status: "Ready" },
];

export const monitoringMetrics = [
  { label: "Processed comments", value: "15,000", sub: "demo limit" },
  { label: "Unique videos", value: "10", sub: "logged by MLflow" },
  { label: "Manual Precision@5", value: "93%", sub: "average across eval queries" },
  { label: "RAG outputs", value: "6", sub: "logged answers generated" },
];

export const mlflowExperiment = {
  name: "Samsung_YouTube_RAG_Monitoring",
  id: "193329967239721503",
  runId: "27eaa48a176442558907ccb0fba088ba",
  trackingPath: "mlruns/193329967239721503",
  status: "active",
};

export const mlflowParams = [
  { name: "llm_provider", value: "OpenAI" },
  { name: "llm_model", value: "gpt-4o-mini" },
  { name: "embedding_model", value: "all-MiniLM-L6-v2" },
  { name: "reranker_model", value: "BAAI/bge-reranker-base" },
  { name: "rag_top_k", value: "5" },
  { name: "data_source", value: "YouTube comments" },
];

export const mlflowLoggedMetrics = [
  { name: "total_processed_comments", value: "15,000", group: "Dataset" },
  { name: "unique_videos", value: "10", group: "Dataset" },
  { name: "manual_checked_precision_at_5", value: "0.933", group: "Evaluation" },
  { name: "rag_answers_generated", value: "6", group: "RAG" },
  { name: "llm_summaries_generated", value: "5", group: "LLM" },
  { name: "bge_reranker_precision_at_5", value: "0.467", group: "Reranker" },
  { name: "sentiment_positive", value: "48%", group: "Sentiment" },
  { name: "sentiment_negative", value: "21%", group: "Sentiment" },
];

export const mlflowArtifacts = [
  "agent_router_results.csv",
  "comments_with_topics.csv",
  "llm_summaries.csv",
  "rag_answers.csv",
  "rag_bge_reranker_evaluation_results.csv",
  "rag_evaluation_results.csv",
  "rag_retrieval_results.csv",
  "topic_keywords.csv",
  "top_keywords_by_category.csv",
  "top_keywords_overall.csv",
];
