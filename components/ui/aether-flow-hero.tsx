"use client";

import React from "react";
import Link from "next/link";
import { motion, type Variants } from "framer-motion";
import {
  ArrowRight,
  Bot,
  BrainCircuit,
  Database,
  MessageSquareText,
  Route,
  Sparkles,
} from "lucide-react";

type Point = {
  x: number;
  y: number;
};

type ParticleOptions = Point & {
  directionX: number;
  directionY: number;
  size: number;
  color: string;
  layer: number;
};

const pipelineSignals = [
  { label: "15K", caption: "comments", icon: MessageSquareText },
  { label: "2", caption: "RAG systems", icon: Database },
  { label: "93%", caption: "precision@5", icon: BrainCircuit },
];

const fadeUpVariants: Variants = {
  hidden: { opacity: 0, y: 22 },
  visible: (index: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: index * 0.14 + 0.25,
      duration: 0.7,
      ease: [0.22, 1, 0.36, 1],
    },
  }),
};

export default function AetherFlowHero() {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const canvasContext = canvas.getContext("2d");
    if (!canvasContext) return;
    const context: CanvasRenderingContext2D = canvasContext;

    let animationFrameId = 0;
    let particles: Particle[] = [];
    let width = 0;
    let height = 0;
    let devicePixelRatio = 1;
    const mouse: Point & { active: boolean; radius: number } = {
      x: 0,
      y: 0,
      active: false,
      radius: 170,
    };

    class Particle {
      x: number;
      y: number;
      directionX: number;
      directionY: number;
      size: number;
      color: string;
      layer: number;

      constructor(options: ParticleOptions) {
        this.x = options.x;
        this.y = options.y;
        this.directionX = options.directionX;
        this.directionY = options.directionY;
        this.size = options.size;
        this.color = options.color;
        this.layer = options.layer;
      }

      draw() {
        context.beginPath();
        context.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        context.fillStyle = this.color;
        context.shadowBlur = this.size > 1.5 ? 12 : 5;
        context.shadowColor = this.color;
        context.fill();
        context.shadowBlur = 0;
      }

      update() {
        if (this.x > width + 20 || this.x < -20) {
          this.directionX *= -1;
        }
        if (this.y > height + 20 || this.y < -20) {
          this.directionY *= -1;
        }

        if (mouse.active) {
          const deltaX = mouse.x - this.x;
          const deltaY = mouse.y - this.y;
          const distance = Math.hypot(deltaX, deltaY) || 1;

          if (distance < mouse.radius) {
            const force = (mouse.radius - distance) / mouse.radius;
            this.x -= (deltaX / distance) * force * 2.8;
            this.y -= (deltaY / distance) * force * 2.8;
          }
        }

        this.x += this.directionX * this.layer;
        this.y += this.directionY * this.layer;
        this.draw();
      }
    }

    const initialiseParticles = () => {
      const particleCount = Math.min(130, Math.max(45, Math.floor((width * height) / 10500)));
      particles = Array.from({ length: particleCount }, (_, index) => {
        const isAccentNode = index % 11 === 0;

        return new Particle({
          x: Math.random() * width,
          y: Math.random() * height,
          directionX: Math.random() * 0.34 - 0.17,
          directionY: Math.random() * 0.34 - 0.17,
          size: isAccentNode ? Math.random() * 1.8 + 1.8 : Math.random() * 1.1 + 0.55,
          color: isAccentNode
            ? "rgba(88, 150, 255, 0.95)"
            : "rgba(186, 133, 255, 0.82)",
          layer: Math.random() * 0.6 + 0.7,
        });
      });
    };

    const resizeCanvas = () => {
      const bounds = canvas.getBoundingClientRect();
      width = Math.max(bounds.width, 1);
      height = Math.max(bounds.height, 1);
      devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);

      canvas.width = Math.floor(width * devicePixelRatio);
      canvas.height = Math.floor(height * devicePixelRatio);
      context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
      initialiseParticles();
    };

    const connectParticles = () => {
      const connectionDistance = Math.min(150, Math.max(105, width / 9));

      for (let first = 0; first < particles.length; first += 1) {
        for (let second = first + 1; second < particles.length; second += 1) {
          const deltaX = particles[first].x - particles[second].x;
          const deltaY = particles[first].y - particles[second].y;
          const distance = Math.hypot(deltaX, deltaY);

          if (distance >= connectionDistance) continue;

          const opacity = (1 - distance / connectionDistance) * 0.52;
          const mouseDistance = mouse.active
            ? Math.hypot(particles[first].x - mouse.x, particles[first].y - mouse.y)
            : Number.POSITIVE_INFINITY;

          context.strokeStyle =
            mouseDistance < mouse.radius
              ? `rgba(115, 178, 255, ${Math.min(opacity + 0.22, 0.85)})`
              : `rgba(178, 126, 255, ${opacity})`;
          context.lineWidth = mouseDistance < mouse.radius ? 1.15 : 0.7;
          context.beginPath();
          context.moveTo(particles[first].x, particles[first].y);
          context.lineTo(particles[second].x, particles[second].y);
          context.stroke();
        }
      }
    };

    const animate = () => {
      context.clearRect(0, 0, width, height);
      const background = context.createRadialGradient(
        width * 0.62,
        height * 0.35,
        0,
        width * 0.5,
        height * 0.5,
        Math.max(width, height),
      );
      background.addColorStop(0, "#10153A");
      background.addColorStop(0.42, "#080A1D");
      background.addColorStop(1, "#02030A");
      context.fillStyle = background;
      context.fillRect(0, 0, width, height);

      particles.forEach((particle) => particle.update());
      connectParticles();
      animationFrameId = window.requestAnimationFrame(animate);
    };

    const updateMouse = (event: PointerEvent) => {
      const bounds = canvas.getBoundingClientRect();
      mouse.x = event.clientX - bounds.left;
      mouse.y = event.clientY - bounds.top;
      mouse.active = true;
    };

    const deactivateMouse = () => {
      mouse.active = false;
    };

    const resizeObserver = new ResizeObserver(resizeCanvas);
    resizeObserver.observe(canvas);
    canvas.addEventListener("pointermove", updateMouse);
    canvas.addEventListener("pointerleave", deactivateMouse);

    resizeCanvas();
    animate();

    return () => {
      resizeObserver.disconnect();
      canvas.removeEventListener("pointermove", updateMouse);
      canvas.removeEventListener("pointerleave", deactivateMouse);
      window.cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <section className="relative flex min-h-[640px] w-full items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-black text-white shadow-[0_26px_90px_rgba(6,8,30,0.3)] lg:min-h-[calc(100svh-8rem)]">
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className="absolute inset-0 h-full w-full"
      />

      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,rgba(2,3,10,0.78)_0%,rgba(2,3,10,0.28)_55%,rgba(2,3,10,0.55)_100%)]" />
      <div className="pointer-events-none absolute inset-0 opacity-25 [background-image:linear-gradient(rgba(255,255,255,0.055)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.055)_1px,transparent_1px)] [background-size:54px_54px]" />

      <div className="relative z-10 mx-auto grid w-full max-w-6xl items-center gap-12 px-6 py-16 md:px-10 lg:grid-cols-[1.1fr_0.9fr] lg:px-14">
        <div>
          <motion.div
            custom={0}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
            className="inline-flex items-center gap-2 rounded-full border border-purple-300/20 bg-purple-400/10 px-4 py-1.5 backdrop-blur-md"
          >
            <Sparkles className="h-4 w-4 text-purple-300" />
            <span className="text-sm font-medium text-gray-200">
              Neural feedback intelligence
            </span>
          </motion.div>

          <motion.h1
            custom={1}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
            className="mt-7 max-w-4xl bg-gradient-to-b from-white via-white to-slate-400 bg-clip-text text-5xl font-bold tracking-[-0.055em] text-transparent sm:text-6xl md:text-7xl"
          >
            Galaxy Insight RAG
          </motion.h1>

          <motion.p
            custom={2}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
            className="mt-6 max-w-2xl text-base leading-8 text-slate-300 sm:text-lg"
          >
            An NLP intelligence system that turns Samsung customer conversations into
            grounded answers, product strategy, and a roadmap that evolves with evidence.
          </motion.p>

          <motion.div
            custom={3}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
            className="mt-9 flex flex-wrap gap-3"
          >
            <Link
              href="/advisor"
              className="inline-flex items-center gap-2 rounded-lg bg-white px-6 py-3.5 text-sm font-semibold text-black shadow-lg shadow-purple-950/30 transition hover:bg-slate-200"
            >
              <Bot className="h-4 w-4" />
              Open Samsung Chat
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/refinement"
              className="inline-flex items-center gap-2 rounded-lg border border-white/15 bg-white/8 px-6 py-3.5 text-sm font-semibold text-white backdrop-blur-md transition hover:bg-white/14"
            >
              <Route className="h-4 w-4 text-purple-300" />
              Refine the Roadmap
            </Link>
          </motion.div>
        </div>

        <motion.div
          custom={4}
          variants={fadeUpVariants}
          initial="hidden"
          animate="visible"
          className="justify-self-stretch rounded-2xl border border-white/12 bg-slate-950/45 p-5 shadow-2xl backdrop-blur-xl sm:p-6 lg:max-w-md lg:justify-self-end"
        >
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-sm font-semibold text-white">NLP pipeline online</div>
              <div className="mt-1 text-xs text-slate-400">feedback to strategy flow</div>
            </div>
            <span className="flex items-center gap-2 rounded-full bg-emerald-400/10 px-3 py-1 text-xs font-medium text-emerald-300 ring-1 ring-emerald-300/15">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-300" />
              Ready
            </span>
          </div>

          <div className="mt-6 grid grid-cols-3 gap-2">
            {pipelineSignals.map((signal) => {
              const Icon = signal.icon;

              return (
                <div
                  key={signal.caption}
                  className="rounded-xl border border-white/8 bg-white/5 p-3"
                >
                  <Icon className="h-4 w-4 text-purple-300" />
                  <div className="mt-3 text-xl font-semibold text-white">{signal.label}</div>
                  <div className="mt-1 text-[11px] leading-4 text-slate-400">
                    {signal.caption}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-5 space-y-3">
            {[
              ["Customer comments", "Sentiment + issues"],
              ["Retrieved evidence", "Grounded RAG answer"],
              ["Strategy signals", "Living roadmap"],
            ].map(([source, output], index) => (
              <div
                key={source}
                className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 text-xs"
              >
                <span className="rounded-lg bg-white/5 px-3 py-2 text-slate-300 ring-1 ring-white/8">
                  {source}
                </span>
                <span className="relative flex h-5 w-8 items-center justify-center">
                  <span className="absolute h-px w-full bg-gradient-to-r from-purple-500/25 to-blue-400/80" />
                  <span
                    className="relative h-1.5 w-1.5 rounded-full bg-blue-300 shadow-[0_0_10px_rgba(125,190,255,0.9)]"
                    style={{ animationDelay: `${index * 180}ms` }}
                  />
                </span>
                <span className="rounded-lg bg-purple-400/8 px-3 py-2 text-purple-100 ring-1 ring-purple-300/10">
                  {output}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
