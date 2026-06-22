"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Bot,
  Brain,
  FileQuestion,
  Gauge,
  GitBranch,
  Hash,
  Home,
  Lightbulb,
  MessageSquare,
  RefreshCw,
  Search,
  ShieldCheck,
  Tags,
  Target,
  UsersRound,
} from "lucide-react";

const navItems = [
  { href: "/", label: "Overview", icon: Home },
  { href: "/advisor", label: "Samsung Chat", icon: Bot },
  { href: "/sentiment", label: "Sentiment", icon: MessageSquare },
  { href: "/issues", label: "Issues", icon: Target },
  { href: "/topics", label: "Topics", icon: Brain },
  { href: "/keywords", label: "Keywords", icon: Hash },
  { href: "/entities", label: "Entities", icon: Tags },
  { href: "/segmentation", label: "User Personas", icon: UsersRound },
  { href: "/rag", label: "Feedback RAG", icon: FileQuestion },
  { href: "/strategy", label: "Strategy RAG", icon: Lightbulb },
  { href: "/refinement", label: "Roadmap Refinement", icon: RefreshCw },
  { href: "/pipeline", label: "Pipeline", icon: GitBranch },
  { href: "/monitoring", label: "Evaluation & MLflow", icon: Gauge },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[#F6F7F9] text-slate-950">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-72 border-r border-slate-200 bg-white lg:block">
        <div className="flex h-full flex-col">
          <div className="border-b border-slate-200 px-5 py-5">
            <Link href="/" className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#1428A0] text-white">
                <ShieldCheck className="h-5 w-5" />
              </span>
              <span>
                <span className="block text-sm font-semibold text-slate-950">Samsung Feedback</span>
                <span className="block text-xs text-slate-500">RAG Advisor</span>
              </span>
            </Link>
          </div>

          <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
            {navItems.map((item) => {
              const active = pathname === item.href;
              const Icon = item.icon;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition ${
                    active
                      ? "bg-[#1428A0] text-white"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-950"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="border-t border-slate-200 p-4">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-700">
                <Search className="h-3.5 w-3.5" />
                Data scope
              </div>
              <p className="mt-2 text-xs leading-5 text-slate-500">
                15,000 cleaned comments, RAG outputs, strategy evidence, and MLflow artifacts.
              </p>
            </div>
          </div>
        </div>
      </aside>

      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 px-4 py-3 backdrop-blur lg:hidden">
        <div className="flex items-center gap-3 overflow-x-auto">
          {navItems.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium ${
                  active ? "bg-[#1428A0] text-white" : "bg-slate-100 text-slate-700"
                }`}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </div>
      </header>

      <main className="lg:pl-72">{children}</main>
    </div>
  );
}
