"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { chartInitialDimension } from "../lib/mock";
import type { SegmentationPoint } from "../lib/segmentation";

const personaColors = ["#1428A0", "#0F766E", "#7C3AED", "#C2410C", "#BE123C", "#0369A1", "#4D7C0F", "#A16207"];

export function SentimentDonut({
  data,
}: {
  data: Array<{ name: string; value: number; color: string }>;
}) {
  return (
    <div className="h-72 min-h-72">
      <ResponsiveContainer width="100%" height="100%" initialDimension={chartInitialDimension}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius={66} outerRadius={98} paddingAngle={3}>
            {data.map((item) => (
              <Cell key={item.name} fill={item.color} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export function IssueBarChart({
  data,
}: {
  data: Array<{ issue: string; count: number }>;
}) {
  return (
    <div className="h-80 min-h-80">
      <ResponsiveContainer width="100%" height="100%" initialDimension={chartInitialDimension}>
        <BarChart data={data} layout="vertical" margin={{ left: 16, right: 16 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="issue" width={128} tick={{ fontSize: 12 }} />
          <Tooltip cursor={{ fill: "#F1F5F9" }} />
          <Bar dataKey="count" fill="#1428A0" radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function KeywordBarChart({
  data,
}: {
  data: Array<{ keyword: string; score: number }>;
}) {
  return (
    <div className="h-80 min-h-80">
      <ResponsiveContainer width="100%" height="100%" initialDimension={chartInitialDimension}>
        <BarChart data={data} margin={{ top: 12, right: 16, left: 0, bottom: 24 }}>
          <XAxis dataKey="keyword" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip cursor={{ fill: "#F1F5F9" }} />
          <Bar dataKey="score" fill="#0F766E" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PrecisionChart({
  data,
}: {
  data: Array<{ query: string; precision: number }>;
}) {
  return (
    <div className="h-72 min-h-72">
      <ResponsiveContainer width="100%" height="100%" initialDimension={chartInitialDimension}>
        <BarChart data={data} margin={{ top: 12, right: 16, left: 0, bottom: 24 }}>
          <XAxis dataKey="query" tick={{ fontSize: 11 }} interval={0} angle={-18} textAnchor="end" height={64} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
          <Tooltip cursor={{ fill: "#F1F5F9" }} />
          <Bar dataKey="precision" fill="#7C3AED" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PersonaSizeChart({
  data,
}: {
  data: Array<{ persona: string; size: number; cluster_id: number }>;
}) {
  return (
    <div className="h-[420px] min-h-[420px]">
      <ResponsiveContainer width="100%" height="100%" initialDimension={chartInitialDimension}>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 18 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis type="category" dataKey="persona" width={190} tick={{ fontSize: 11 }} />
          <Tooltip cursor={{ fill: "#F1F5F9" }} formatter={(value) => Number(value).toLocaleString()} />
          <Bar dataKey="size" name="Comments" radius={[0, 6, 6, 0]}>
            {data.map((item) => (
              <Cell key={item.cluster_id} fill={personaColors[item.cluster_id % personaColors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PersonaScatterChart({ data }: { data: SegmentationPoint[] }) {
  const clusters = [...new Set(data.map((point) => point.cluster_id))].sort((a, b) => a - b);

  return (
    <div className="h-[420px] min-h-[420px]">
      <ResponsiveContainer width="100%" height="100%" initialDimension={chartInitialDimension}>
        <ScatterChart margin={{ top: 12, right: 12, bottom: 12, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="x" name="SVD 1" tick={{ fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="SVD 2" tick={{ fontSize: 11 }} />
          <ZAxis range={[18, 18]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          {clusters.map((clusterId) => (
            <Scatter
              key={clusterId}
              name={`Cluster ${clusterId}`}
              data={data.filter((point) => point.cluster_id === clusterId)}
              fill={personaColors[clusterId % personaColors.length]}
              fillOpacity={0.62}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
