"use client";
import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

export function KpiCard({
  label, value, sub, icon: Icon, delay = 0,
}: {
  label: string; value: string; sub?: string; icon: LucideIcon; delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-100 hover:shadow-md transition-shadow"
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-500">{label}</span>
        <div className="rounded-lg bg-blue-50 p-2">
          <Icon className="h-5 w-5 text-[#1428A0]" />
        </div>
      </div>
      <div className="mt-3 text-3xl font-bold tracking-tight text-gray-900">{value}</div>
      {sub && <div className="mt-1 text-sm text-gray-400">{sub}</div>}
    </motion.div>
  );
}