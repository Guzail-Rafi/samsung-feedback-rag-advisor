"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Bot,
  Brain,
  FileQuestion,
  Files,
  Gauge,
  GitBranch,
  Hash,
  Home,
  Lightbulb,
  Menu,
  MessageSquare,
  RefreshCw,
  Search,
  ShieldCheck,
  Tags,
  Target,
  UsersRound,
  X,
} from "lucide-react";

const primaryNav = [
  { href: "/", label: "Overview", icon: Home },
  { href: "/advisor", label: "Samsung Chat", icon: Bot },
  { href: "/rag", label: "Feedback RAG", icon: FileQuestion },
  { href: "/strategy", label: "Strategy RAG", icon: Lightbulb },
  { href: "/refinement", label: "Roadmap", icon: RefreshCw },
];

const menuGroups = [
  {
    label: "Core workspace",
    items: [
      { href: "/", label: "Overview", description: "System summary and live signals", icon: Home },
      { href: "/advisor", label: "Samsung Chat", description: "Ask the routed Samsung advisor", icon: Bot },
      { href: "/rag", label: "Feedback RAG", description: "Grounded customer feedback answers", icon: FileQuestion },
      { href: "/strategy", label: "Strategy RAG", description: "Evidence-led product recommendations", icon: Lightbulb },
      {
        href: "/refinement",
        label: "Roadmap Refinement",
        description: "Negotiate and update roadmap phases",
        icon: RefreshCw,
      },
    ],
  },
  {
    label: "Feedback analytics",
    items: [
      { href: "/sentiment", label: "Sentiment", description: "Positive, neutral, and negative signals", icon: MessageSquare },
      { href: "/issues", label: "Issues", description: "Customer complaint categories", icon: Target },
      { href: "/topics", label: "Topics", description: "Themes across feedback", icon: Brain },
      { href: "/keywords", label: "Keywords", description: "Frequent terms and phrases", icon: Hash },
      { href: "/entities", label: "Entities", description: "Products, brands, and features", icon: Tags },
      { href: "/segmentation", label: "User Personas", description: "Feedback audience segments", icon: UsersRound },
    ],
  },
  {
    label: "Data and operations",
    items: [
      { href: "/documents", label: "Documents", description: "Uploaded Samsung knowledge", icon: Files },
      { href: "/pipeline", label: "Pipeline", description: "Processing stages and outputs", icon: GitBranch },
      { href: "/monitoring", label: "Evaluation & MLflow", description: "Quality metrics and traces", icon: Gauge },
    ],
  },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!menuOpen) return;

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };

    document.addEventListener("keydown", closeOnEscape);
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", closeOnEscape);
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  return (
    <div className="min-h-screen bg-[#F4F4F0] text-[#181818]">
      <header className="sticky top-0 z-40 border-b border-[#DDDCD5] bg-[#FAFAF7]/95 backdrop-blur-xl">
        <div className="mx-auto flex h-16 w-full max-w-[1480px] items-center gap-3 px-4 md:px-6">
          <Link href="/" className="mr-auto flex min-w-0 items-center gap-3">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#1428A0] text-white shadow-sm">
              <ShieldCheck className="h-4 w-4" />
            </span>
            <span className="truncate text-sm font-semibold tracking-tight text-[#181818] sm:text-base">
              Galaxy Insight
            </span>
          </Link>

          <nav aria-label="Primary navigation" className="hidden items-center gap-1 lg:flex">
            {primaryNav.map((item) => {
              const active = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                    active
                      ? "bg-[#181818] text-white shadow-sm"
                      : "text-[#5F5F5A] hover:bg-white hover:text-[#181818]"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <Link
            href="/advisor"
            aria-label="Open Samsung Chat"
            className={`hidden h-10 w-10 items-center justify-center rounded-lg border transition sm:flex lg:hidden ${
              pathname === "/advisor"
                ? "border-[#1428A0] bg-[#1428A0] text-white"
                : "border-[#D7D6CF] bg-white text-[#343430] hover:border-[#1428A0] hover:text-[#1428A0]"
            }`}
          >
            <Bot className="h-4 w-4" />
          </Link>

          <Link
            href="/refinement"
            aria-label="Open Roadmap Refinement"
            className={`hidden h-10 w-10 items-center justify-center rounded-lg border transition sm:flex lg:hidden ${
              pathname === "/refinement"
                ? "border-[#1428A0] bg-[#1428A0] text-white"
                : "border-[#D7D6CF] bg-white text-[#343430] hover:border-[#1428A0] hover:text-[#1428A0]"
            }`}
          >
            <RefreshCw className="h-4 w-4" />
          </Link>

          <button
            type="button"
            onClick={() => setMenuOpen(true)}
            aria-label="Open navigation menu"
            aria-expanded={menuOpen}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-[#D7D6CF] bg-white px-3 text-sm font-medium text-[#343430] shadow-sm transition hover:border-[#A8A79F] hover:bg-[#F7F7F3]"
          >
            <Menu className="h-4 w-4" />
            <span className="hidden sm:inline">Menu</span>
          </button>
        </div>
      </header>

      <main>{children}</main>

      <div
        className={`fixed inset-0 z-50 transition ${menuOpen ? "pointer-events-auto" : "pointer-events-none"}`}
        aria-hidden={!menuOpen}
      >
        <button
          type="button"
          aria-label="Close navigation menu"
          onClick={() => setMenuOpen(false)}
          className={`absolute inset-0 bg-black/35 backdrop-blur-[2px] transition-opacity duration-300 ${
            menuOpen ? "opacity-100" : "opacity-0"
          }`}
        />

        <aside
          role="dialog"
          aria-modal="true"
          aria-label="Navigation menu"
          className={`absolute inset-y-0 right-0 flex w-full max-w-md flex-col border-l border-[#DDDCD5] bg-[#FAFAF7] shadow-2xl transition-transform duration-300 ease-out ${
            menuOpen ? "translate-x-0" : "translate-x-full"
          }`}
        >
          <div className="flex h-16 items-center justify-between border-b border-[#DDDCD5] px-5">
            <div>
              <div className="text-sm font-semibold text-[#181818]">Galaxy Insight</div>
              <div className="text-xs text-[#777770]">Navigation and tools</div>
            </div>
            <button
              type="button"
              onClick={() => setMenuOpen(false)}
              aria-label="Close menu"
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-[#D7D6CF] bg-white text-[#343430] transition hover:bg-[#F0F0EB]"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-5">
            {menuGroups.map((group) => (
              <section key={group.label} className="mb-6 last:mb-0">
                <h2 className="px-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-[#85857D]">
                  {group.label}
                </h2>
                <div className="mt-2 grid gap-1">
                  {group.items.map((item) => {
                    const active = pathname === item.href;
                    const Icon = item.icon;

                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={`group flex items-center gap-3 rounded-xl px-3 py-3 transition ${
                          active ? "bg-[#181818] text-white" : "hover:bg-white"
                        }`}
                      >
                        <span
                          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                            active
                              ? "bg-white/12 text-white"
                              : "bg-[#E9EEFF] text-[#1428A0] group-hover:bg-[#DDE6FF]"
                          }`}
                        >
                          <Icon className="h-4 w-4" />
                        </span>
                        <span className="min-w-0">
                          <span className="block text-sm font-semibold">{item.label}</span>
                          <span
                            className={`mt-0.5 block truncate text-xs ${
                              active ? "text-white/65" : "text-[#777770]"
                            }`}
                          >
                            {item.description}
                          </span>
                        </span>
                      </Link>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>

          <div className="border-t border-[#DDDCD5] p-4">
            <div className="rounded-xl border border-[#DDDCD5] bg-white p-4">
              <div className="flex items-center gap-2 text-xs font-semibold text-[#343430]">
                <Search className="h-3.5 w-3.5 text-[#1428A0]" />
                Data scope
              </div>
              <p className="mt-2 text-xs leading-5 text-[#777770]">
                15,000 cleaned comments, two RAG systems, strategy evidence, and MLflow evaluation.
              </p>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
