import { readFileSync } from "fs";
import path from "path";

export type Persona = {
  cluster_id: number;
  persona: string;
  size: number;
  share: number;
  dominant_sentiment: string;
  sentiment_share: number;
  top_issue: string;
  top_issue_share: number;
  top_topic: string;
  top_terms: string;
  recommendation: string;
  representative_comments: string[];
};

export type SegmentationEvaluation = {
  k: number;
  silhouette_score: number;
  calinski_harabasz_score: number;
  inertia: number;
  smallest_cluster: number;
  largest_cluster: number;
  smallest_cluster_share: number;
};

export type SegmentationPoint = {
  x: number;
  y: number;
  cluster_id: number;
  persona: string;
  sentiment: string;
};

export type SegmentationDashboard = {
  generated_at: string;
  method: {
    representation: string;
    max_features: number;
    dimensionality_reduction: string;
    svd_components: number;
    clustering: string;
    selected_k: number;
    random_state: number;
  };
  metrics: {
    comments_segmented: number;
    tfidf_features: number;
    explained_variance: number;
    silhouette_score: number;
    calinski_harabasz_score: number;
  };
  personas: Persona[];
  evaluation: SegmentationEvaluation[];
  scatter_points: SegmentationPoint[];
};

export function loadSegmentationDashboard(): SegmentationDashboard | null {
  try {
    const dataPath = path.join(process.cwd(), "data", "processed", "user_segmentation_dashboard.json");
    return JSON.parse(readFileSync(dataPath, "utf8")) as SegmentationDashboard;
  } catch {
    return null;
  }
}
