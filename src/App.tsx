import { useMemo, useState } from "react";

type Prediction = {
  label: "Spam" | "Not spam";
  confidence: number;
  risk: number;
  signals: string[];
};

const spamSignals = [
  "free",
  "win",
  "winner",
  "claim",
  "urgent",
  "prize",
  "cash",
  "click",
  "link",
  "verify",
  "account",
  "limited",
  "offer",
  "loan",
  "gift",
  "congratulations",
  "selected",
  "stop",
  "unsubscribe",
];

const safeSignals = [
  "meeting",
  "appointment",
  "family",
  "home",
  "dinner",
  "thanks",
  "tomorrow",
  "today",
  "class",
  "office",
  "doctor",
];

const sampleMessages = [
  "URGENT! You have won a $1000 gift card. Click this link now to claim your prize.",
  "Can you pick up milk on your way home? Dinner is at 7.",
  "Your bank account needs verification. Reply with your PIN to avoid suspension.",
];

function classifyMessage(text: string): Prediction {
  const lower = text.toLowerCase();
  const tokens: string[] = lower.match(/[a-z0-9$]+/g) ?? [];
  const matches = spamSignals.filter((word) => tokens.includes(word) || lower.includes(word));
  const safeMatches = safeSignals.filter((word) => tokens.includes(word));
  const urlHits = (lower.match(/https?:\/\/|www\.|\.com|bit\.ly/g) ?? []).length;
  const moneyHits = (lower.match(/[$]|\b\d+\s?(cash|prize|gift|loan)\b/g) ?? []).length;
  const capsRatio = text.length ? (text.match(/[A-Z]/g)?.length ?? 0) / text.length : 0;
  const punctuationHits = (text.match(/!/g) ?? []).length;

  const rawScore =
    matches.length * 13 +
    urlHits * 18 +
    moneyHits * 13 +
    Math.min(punctuationHits * 3, 12) +
    (capsRatio > 0.18 ? 12 : 0) -
    safeMatches.length * 8;

  const risk = Math.max(3, Math.min(97, 34 + rawScore));
  const label = risk >= 58 ? "Spam" : "Not spam";
  const confidence = label === "Spam" ? risk : 100 - risk;
  const signals = [
    ...matches.slice(0, 4).map((word) => `keyword: ${word}`),
    ...(urlHits ? ["contains a link"] : []),
    ...(moneyHits ? ["money or prize language"] : []),
    ...(capsRatio > 0.18 ? ["unusual capitalization"] : []),
    ...(!matches.length && !urlHits ? ["ordinary conversational wording"] : []),
  ];

  return { label, confidence, risk, signals };
}

export default function App() {
  const [message, setMessage] = useState(sampleMessages[0]);
  const prediction = useMemo(() => classifyMessage(message), [message]);
  const isSpam = prediction.label === "Spam";

  return (
    <main className="min-h-screen overflow-hidden bg-[#07111f] text-white">
      <section className="relative flex min-h-screen items-center px-6 py-8 sm:px-10 lg:px-16">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_24%,rgba(59,130,246,0.35),transparent_28%),radial-gradient(circle_at_82%_18%,rgba(14,165,233,0.22),transparent_30%),linear-gradient(135deg,#07111f_0%,#0d1b2e_52%,#102239_100%)]" />
        <div className="scanline absolute inset-0 opacity-30" />
        <div className="orb-drift absolute -right-40 top-20 h-96 w-96 rounded-full bg-cyan-400/20 blur-3xl" />
        <div className="orb-drift-slow absolute -bottom-44 left-10 h-[30rem] w-[30rem] rounded-full bg-blue-600/20 blur-3xl" />

        <div className="relative z-10 mx-auto grid w-full max-w-7xl items-center gap-12 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="hero-copy max-w-2xl">
            <p className="mb-5 text-sm font-semibold uppercase tracking-[0.36em] text-cyan-200/80">
              SMS Spam Classifier
            </p>
            <h1 className="text-5xl font-semibold tracking-[-0.06em] text-white sm:text-7xl lg:text-8xl">
              Filter risky texts before they reach the inbox.
            </h1>
            <p className="mt-7 max-w-xl text-lg leading-8 text-slate-300">
              A compact machine-learning workflow for cleaning SMS text, vectorizing messages,
              comparing Naive Bayes with linear models, and serving predictions through Flask.
            </p>
            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
              <a
                href="#classifier"
                className="inline-flex items-center justify-center bg-cyan-300 px-6 py-3 text-sm font-bold uppercase tracking-[0.2em] text-slate-950 transition hover:bg-white"
              >
                Try the demo
              </a>
              <a
                href="#readme"
                className="inline-flex items-center justify-center border border-white/25 px-6 py-3 text-sm font-bold uppercase tracking-[0.2em] text-white transition hover:border-cyan-200 hover:text-cyan-100"
              >
                Start guide
              </a>
            </div>
          </div>

          <div id="classifier" className="phone-shell mx-auto w-full max-w-xl">
            <div className="border border-white/10 bg-slate-950/60 p-5 shadow-2xl shadow-cyan-950/40 backdrop-blur-xl sm:p-7">
              <div className="mb-7 flex items-center justify-between border-b border-white/10 pb-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.26em] text-cyan-200/70">Live classifier</p>
                  <h2 className="mt-1 text-2xl font-semibold tracking-tight">Message risk check</h2>
                </div>
                <div className={`h-3 w-3 rounded-full ${isSpam ? "bg-rose-400" : "bg-emerald-300"}`} />
              </div>

              <label htmlFor="sms" className="text-sm font-medium text-slate-200">
                Paste an SMS message
              </label>
              <textarea
                id="sms"
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                className="mt-3 min-h-40 w-full resize-none border border-white/10 bg-slate-900/80 p-4 text-base leading-7 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-200"
                placeholder="Type a text message to classify..."
              />

              <div className="mt-4 flex flex-wrap gap-2">
                {sampleMessages.map((sample, index) => (
                  <button
                    key={sample}
                    onClick={() => setMessage(sample)}
                    className="border border-white/10 px-3 py-2 text-left text-xs text-slate-300 transition hover:border-cyan-200 hover:text-white"
                  >
                    Sample {index + 1}
                  </button>
                ))}
              </div>

              <div className="mt-7 border-t border-white/10 pt-6">
                <div className="flex items-end justify-between gap-6">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Prediction</p>
                    <p className={`mt-2 text-4xl font-semibold ${isSpam ? "text-rose-200" : "text-emerald-200"}`}>
                      {prediction.label}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Confidence</p>
                    <p className="mt-2 text-4xl font-semibold text-white">{prediction.confidence.toFixed(0)}%</p>
                  </div>
                </div>

                <div className="mt-6 h-2 bg-white/10">
                  <div
                    className={`confidence-fill h-full ${isSpam ? "bg-rose-300" : "bg-emerald-300"}`}
                    style={{ width: `${prediction.confidence}%` }}
                  />
                </div>

                <div className="mt-5 flex flex-wrap gap-2">
                  {prediction.signals.map((signal) => (
                    <span key={signal} className="border border-white/10 px-3 py-1 text-xs text-slate-300">
                      {signal}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="readme" className="bg-slate-50 px-6 py-20 text-slate-950 sm:px-10 lg:px-16">
        <div className="mx-auto max-w-5xl">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-700">Project files included</p>
          <h2 className="mt-4 max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
            Train the real model locally, then launch the Flask app.
          </h2>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">
            The repository now includes Python training code, a Flask prediction app, dependency list,
            and a README that explains the dataset, commands, metrics, and deployment path.
          </p>

          <div className="mt-10 grid gap-8 md:grid-cols-3">
            {[
              ["1", "Train", "Download the UCI SMS Spam Collection, clean text, compare models, tune a threshold, and save the best pipeline."],
              ["2", "Serve", "Run Flask to classify messages from a small web form or JSON endpoint with label and confidence."],
              ["3", "Iterate", "Use precision, recall, F1, confusion matrices, and threshold settings to reduce missed spam."],
            ].map(([number, title, body]) => (
              <article key={title} className="border-t border-slate-300 pt-5">
                <span className="text-sm font-bold text-cyan-700">{number}</span>
                <h3 className="mt-4 text-2xl font-semibold">{title}</h3>
                <p className="mt-3 leading-7 text-slate-600">{body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
