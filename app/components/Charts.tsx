"use client";

import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { chartInitialDimension } from "../lib/mock";

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
