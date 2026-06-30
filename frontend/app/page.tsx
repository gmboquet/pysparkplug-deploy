import Link from "next/link";
import { NavBar } from "./components/NavBar";

function Feature({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <h3 className="font-semibold">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>
    </div>
  );
}

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      <NavBar />

      {/* Hero */}
      <section className="mx-auto max-w-6xl px-4 pb-12 pt-20 text-center">
        <span className="inline-block rounded-full border border-border bg-surface px-3 py-1 text-xs text-muted">
          One gateway · OpenAI-compatible · calibrated
        </span>
        <h1 className="mx-auto mt-6 max-w-3xl text-4xl font-semibold leading-tight sm:text-5xl">
          Host mixle probabilistic models{" "}
          <span className="text-accent">and open LLMs</span> behind one API.
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-lg leading-relaxed text-muted">
          mixle serves <strong className="text-fg">distributions and decisions</strong>,
          not just tokens — calibrated intervals, tail risk, abstention — and closes a{" "}
          <strong className="text-fg">real human-feedback loop</strong> with an actively
          elicited preference model. The chat looks like Claude or ChatGPT. The math underneath
          doesn&apos;t.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          <Link
            href="/chat"
            className="rounded-xl px-5 py-2.5 font-medium text-accent-fg"
            style={{ background: "var(--accent)" }}
          >
            Try the chat →
          </Link>
          <Link
            href="/signup"
            className="rounded-xl border border-border bg-surface px-5 py-2.5 font-medium hover:bg-surface-2"
          >
            Create an account
          </Link>
        </div>
        <p className="mt-3 text-xs text-muted">
          No backend yet? The gateway ships with an <code>echo</code> model so the chat works
          out of the box.
        </p>
      </section>

      {/* Features */}
      <section className="mx-auto grid max-w-6xl gap-4 px-4 pb-16 sm:grid-cols-3">
        <Feature
          title="Distributions + decisions"
          body="Every served model can speak calibrated predictive intervals, tail probabilities, and Bayes-optimal actions — advertised per-model via /v1/models and /capabilities."
        />
        <Feature
          title="A real feedback loop"
          body="👍 / 👎 / edit / regenerate aren't just logged. They fit a mixle preference-reward model with calibrated uncertainty and actively elicit the next most informative comparison."
        />
        <Feature
          title="mixle + open LLMs, composed"
          body="Proxy Llama / DeepSeek via vLLM or Ollama, serve native mixle artifacts, or compose them — a mixle model gating, reranking, and calibrating an LLM."
        />
      </section>

      {/* How it talks */}
      <section className="mx-auto max-w-6xl px-4 pb-24">
        <div className="rounded-2xl border border-border bg-surface p-6">
          <h2 className="text-lg font-semibold">OpenAI-compatible by default</h2>
          <p className="mt-2 text-sm text-muted">
            Point any OpenAI client at the gateway. This frontend uses the same routes you would.
          </p>
          <pre className="mt-4 overflow-x-auto rounded-xl border border-border bg-surface-2 p-4 text-xs text-fg">
            <code>{`POST /v1/chat/completions
{
  "model": "echo",
  "stream": true,
  "messages": [{ "role": "user", "content": "hello" }]
}`}</code>
          </pre>
        </div>
      </section>

      <footer className="border-t border-border py-8 text-center text-sm text-muted">
        mixle-mlops — all-in-one AI platform.
      </footer>
    </div>
  );
}
