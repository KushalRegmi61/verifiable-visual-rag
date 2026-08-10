"use client";

import { useCallback, useState } from "react";
import {
  API,
  ask,
  type Candidate,
  type ClaimEvent,
  type DoneEvent,
  type RetrievedEvent,
} from "@/lib/api";
import { toStyle } from "@/lib/overlay";

// Explicit hex values rather than Tailwind classes. The colour is chosen at
// runtime from the claim index, and Tailwind only ships classes it can see in
// the source, so a computed class name would compile to nothing.
const CLAIM_COLOURS = [
  "#2563eb",
  "#db2777",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#dc2626",
  "#4d7c0f",
];

function colourFor(index: number): string {
  return CLAIM_COLOURS[index % CLAIM_COLOURS.length];
}

function labelText(label: string | null): string {
  return label ? label.replace(/_/g, " ") : "unverified";
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** Everything one page's answer is made of. */
type PageResult = {
  retrieved: RetrievedEvent;
  expected: number | null;
  claims: ClaimEvent[];
  done: DoneEvent | null;
};

function keyOf(docSha: string, page: number): string {
  return `${docSha}:${page}`;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [pending, setPending] = useState(false);
  // Every page read for the current question, keyed doc_sha:page. Answering one
  // page costs a reader call plus a verifier call per claim on a serial GPU, so
  // re-reading a page the user has already seen just to get back to it is the
  // most expensive thing this UI could do. Keeping the results makes flipping
  // between retrieved pages free, and makes comparing what page 7 supports
  // against what page 12 supports possible at all, which is the only reason to
  // show more than one page.
  const [results, setResults] = useState<Record<string, PageResult>>({});
  const [active, setActive] = useState<string | null>(null);
  // Held separately from `results` and NOT cleared on a pinned re-ask. The
  // pinned branch of _choose_page returns candidates: [] by construction, so
  // reading the list off the current page made it single-use: click one
  // alternate page and the row vanishes, and getting back to the original page
  // means retyping the question. This is the UI's only affordance for the
  // retrieval-was-wrong case, which is exactly the case worth demonstrating.
  const [alternates, setAlternates] = useState<Candidate[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const run = useCallback(
    async (q: string, pin?: { doc: string; page: number }) => {
      const text = q.trim();
      if (!text) return;

      // Already read for this question: show it, spend nothing. Only reachable
      // from a candidate chip, which is always pinned, so a fresh question can
      // never be short-circuited by a page left over from the last one.
      if (pin && results[keyOf(pin.doc, pin.page)]) {
        setActive(keyOf(pin.doc, pin.page));
        setHovered(null);
        return;
      }

      setPending(true);
      setAsked(text);
      setError(null);
      setHovered(null);
      setActive(null);
      // Both the cache and the ranking belong to the question rather than to
      // the page on screen, so a new question drops both and a pinned re-ask
      // keeps both.
      if (!pin) {
        setResults({});
        setAlternates([]);
      }

      // Set by the retrieved frame, which always arrives first, and read by
      // every frame after it. A local rather than state, because the claim
      // handler needs the value inside this same stream.
      let key: string | null = null;
      const patch = (f: (r: PageResult) => PageResult) => {
        if (!key) return;
        const k = key;
        setResults((prev) => (prev[k] ? { ...prev, [k]: f(prev[k]) } : prev));
      };

      try {
        await ask(
          pin ? { question: text, doc: pin.doc, page: pin.page } : { question: text },
          {
            onRetrieved: (e) => {
              key = keyOf(e.doc_sha, e.page);
              setResults((prev) => ({
                ...prev,
                [key!]: { retrieved: e, expected: null, claims: [], done: null },
              }));
              setActive(key);
              // Only a fresh retrieval knows the ranking. A pinned re-ask
              // carries an empty list, and overwriting with it would discard
              // the only record of what else was considered.
              //
              // The top hit is prepended because the service sends candidates
              // as hits[1:]. Without it, clicking an alternate is one-way:
              // there would be no button for the page retrieval actually chose.
              if (!pin) {
                setAlternates([
                  {
                    doc_sha: e.doc_sha,
                    page: e.page,
                    score: e.score ?? 0,
                    doc_name: e.doc_name,
                  },
                  ...e.candidates,
                ]);
              }
            },
            onClaims: (n) => patch((r) => ({ ...r, expected: n })),
            // Appended rather than replaced: every claim arrives as its own
            // event, already verified, and the list grows as they land.
            onClaim: (c) => patch((r) => ({ ...r, claims: [...r.claims, c] })),
            onDone: (d) => patch((r) => ({ ...r, done: d })),
            onError: setError,
          },
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setPending(false);
      }
    },
    [results],
  );

  const currentPage = active ? (results[active] ?? null) : null;
  const retrieved = currentPage?.retrieved ?? null;
  const claims = currentPage?.claims ?? [];
  const expected = currentPage?.expected ?? null;
  const done = currentPage?.done ?? null;

  // The one line that guarantees a withheld claim can never reach the overlay.
  // The service already sends it with regions: [], and this makes the display
  // side agree rather than relying on that alone.
  const shown = claims.filter((c) => !c.withheld);
  const withheld = claims.filter((c) => c.withheld);
  const abstained = done?.abstained_overall ?? false;
  const pageImage = retrieved
    ? `${API}/documents/${retrieved.doc_sha}/pages/${retrieved.page}/image`
    : null;
  // A pinned re-ask does no retrieval, so its Retrieved event carries score:
  // null by construction. Reading the score off the ranking instead means the
  // header shows one for every page, which is what makes two pages comparable:
  // without it, switching to an alternate silently drops the only number that
  // says how well it matched.
  const rankScore =
    retrieved?.score ??
    alternates.find(
      (c) => c.doc_sha === retrieved?.doc_sha && c.page === retrieved?.page,
    )?.score ??
    null;

  return (
    <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Verifiable Visual RAG</h1>
        <p className="mt-1 text-sm text-black/60 dark:text-white/60">
          Every claim is checked by a second model and pinned to a region of the page it
          came from. Claims that fail the check are withheld, not shown.
        </p>
      </header>

      <form
        className="flex flex-col gap-2 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          void run(question);
        }}
      >
        <input
          className="flex-1 rounded-md border border-black/15 bg-white px-3 py-2 text-sm outline-none focus:border-black/40 dark:border-white/20 dark:bg-white/5 dark:focus:border-white/50"
          placeholder="Ask a question about an indexed document"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={pending}
        />
        <button
          type="submit"
          disabled={pending || !question.trim()}
          className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-white dark:text-black"
        >
          {pending ? "Working" : "Ask"}
        </button>
      </form>

      {error && (
        <div
          role="alert"
          className="mt-4 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-300"
        >
          {error}
        </div>
      )}

      {asked && (
        <div className="mt-6 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <section className="min-w-0">
            {retrieved?.warning && (
              <div className="mb-4 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-800 dark:text-amber-200">
                {retrieved.warning}
              </div>
            )}

            {abstained && (
              <div className="mb-4 rounded-xl border-2 border-amber-500/60 bg-amber-500/10 px-4 py-4">
                <p className="text-sm font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
                  No answer given
                </p>
                <p className="mt-1 text-sm text-black/70 dark:text-white/70">
                  The verifier rejected every claim the reader produced, so the system is
                  declining to answer this question rather than showing an answer it cannot
                  support.
                  {/* Without this, an unembedded page reads as a judgement about the
                      evidence when the real cause is that nobody ran `vvrag embed`. */}
                  {retrieved?.warning &&
                    " Note the warning above: this page has no embeddings, so that" +
                      " rejection is very likely a missing index rather than weak evidence."}
                </p>
              </div>
            )}

            {pending && expected === null && (
              <p className="text-sm text-black/50 dark:text-white/50">
                {retrieved ? "Reading the page" : "Retrieving a page"}
              </p>
            )}

            {expected !== null && claims.length < expected && (
              <p className="text-sm text-black/50 dark:text-white/50">
                Verifying claim {claims.length + 1} of {expected}
              </p>
            )}

            {/* The answer, which is the verified claims joined and nothing else.
                Deliberately NOT a second model call that summarises them: a
                synthesised sentence would be unverified text presented as the
                answer, which is the failure this whole system exists to
                prevent. reader.py states the rule ("the displayed answer is
                the claims joined, so nothing can drift between what is shown
                and what is verified"); this is where it becomes visible.
                Every span here passed the verifier and has a region on the
                page, and hovering one lights that region. */}
            {shown.length > 0 && (
              <div className="mt-3 rounded-xl border border-black/10 bg-black/[0.02] px-4 py-4 dark:border-white/10 dark:bg-white/[0.04]">
                <p className="text-xs font-medium uppercase tracking-wide text-black/45 dark:text-white/45">
                  Answer
                </p>
                <p className="mt-2 text-[15px] leading-relaxed">
                  {shown.map((c) => (
                    <span
                      key={c.index}
                      onMouseEnter={() => setHovered(c.index)}
                      onMouseLeave={() => setHovered(null)}
                      style={{
                        textDecorationColor: colourFor(c.index),
                        opacity: hovered !== null && hovered !== c.index ? 0.4 : 1,
                      }}
                      className="cursor-default underline decoration-2 underline-offset-4 transition-opacity"
                    >
                      {c.text}{" "}
                    </span>
                  ))}
                </p>
                <p className="mt-3 text-xs text-black/45 dark:text-white/45">
                  Every sentence above was checked by a second model and carries its own
                  evidence region. Hover one to find it on the page.
                </p>
              </div>
            )}

            {shown.length > 0 && (
              <p className="mt-5 text-xs font-medium uppercase tracking-wide text-black/45 dark:text-white/45">
                Evidence by claim
              </p>
            )}

            {shown.length > 0 && (
              <ol className="mt-3 space-y-3">
                {shown.map((c) => {
                  const colour = colourFor(c.index);
                  const dimmed = hovered !== null && hovered !== c.index;
                  return (
                    <li
                      key={c.index}
                      onMouseEnter={() => setHovered(c.index)}
                      onMouseLeave={() => setHovered(null)}
                      style={{ borderLeftColor: colour }}
                      className={`rounded-md border border-black/10 border-l-4 bg-black/[0.02] px-3 py-3 transition-opacity dark:border-white/10 dark:bg-white/[0.04] ${
                        dimmed ? "opacity-50" : "opacity-100"
                      }`}
                    >
                      <p className="text-sm leading-relaxed">{c.text}</p>
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                        <span
                          className="rounded-full px-2 py-0.5 font-medium text-white"
                          style={{ backgroundColor: colour }}
                        >
                          {labelText(c.label)}
                        </span>
                        <span className="text-black/50 dark:text-white/50">
                          confidence {pct(c.confidence)}
                        </span>
                        {c.compound && (
                          <span
                            className="rounded-full border border-black/20 px-2 py-0.5 text-black/60 dark:border-white/25 dark:text-white/60"
                            title="This claim asserts more than one thing, so a single region can only evidence part of it."
                          >
                            compound
                          </span>
                        )}
                        {c.regions.map((r, i) => (
                          <span
                            key={i}
                            className="rounded-full border border-black/20 px-2 py-0.5 text-black/60 dark:border-white/25 dark:text-white/60"
                            title={r.text ?? ""}
                          >
                            {r.modality}
                            {r.resolution ? ` / ${r.resolution}` : ""}
                          </span>
                        ))}
                      </div>
                      {c.reason && (
                        <p className="mt-2 text-xs text-black/50 dark:text-white/50">
                          {c.reason}
                        </p>
                      )}
                    </li>
                  );
                })}
              </ol>
            )}

            {withheld.length > 0 && (
              <details className="mt-5 rounded-md border border-black/10 bg-black/[0.02] px-3 py-2 dark:border-white/10 dark:bg-white/[0.04]">
                <summary className="cursor-pointer text-sm font-medium">
                  {withheld.length} claim{withheld.length === 1 ? "" : "s"} withheld by the
                  verifier
                </summary>
                <ul className="mt-3 space-y-3">
                  {withheld.map((c) => (
                    <li
                      key={c.index}
                      className="rounded-md border border-dashed border-black/20 px-3 py-2 dark:border-white/25"
                    >
                      <p className="text-sm text-black/60 line-through decoration-black/30 dark:text-white/60 dark:decoration-white/30">
                        {c.text}
                      </p>
                      <p className="mt-1 text-xs text-black/50 dark:text-white/50">
                        {labelText(c.label)}, confidence {pct(c.confidence)}
                      </p>
                      <p className="mt-1 text-xs text-black/50 dark:text-white/50">
                        {c.reason ?? "The verifier gave no reason."}
                      </p>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs text-black/40 dark:text-white/40">
                  Withheld claims carry no evidence region, so nothing is drawn for them on
                  the page.
                </p>
              </details>
            )}

            {done && !abstained && shown.length === 0 && (
              <p className="mt-3 text-sm text-black/50 dark:text-white/50">
                No claims to show.
              </p>
            )}
          </section>

          <section className="min-w-0">
            {retrieved ? (
              <>
                <div className="mb-2 flex flex-wrap items-baseline gap-2 text-xs text-black/60 dark:text-white/60">
                  <span className="font-medium text-black dark:text-white">
                    {retrieved.doc_name}
                  </span>
                  <span>page {retrieved.page}</span>
                  {rankScore !== null && <span>score {rankScore.toFixed(3)}</span>}
                </div>

                <div className="relative w-full overflow-hidden rounded-md border border-black/10 bg-white dark:border-white/15">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={pageImage!}
                    alt={`${retrieved.doc_name} page ${retrieved.page}`}
                    className="block w-full"
                  />
                  {shown.flatMap((c) =>
                    c.regions.map((r, i) => {
                      const colour = colourFor(c.index);
                      const dimmed = hovered !== null && hovered !== c.index;
                      return (
                        <div
                          key={`${c.index}-${i}`}
                          title={[
                            c.text,
                            c.reason ?? "",
                            r.resolution === "block"
                              ? "Block level: the heatmap could not separate the lines inside this block."
                              : "",
                          ]
                            .filter(Boolean)
                            .join("\n\n")}
                          style={{
                            ...toStyle(r.bbox),
                            position: "absolute",
                            borderColor: colour,
                            backgroundColor: `${colour}22`,
                            borderStyle: r.resolution === "block" ? "dashed" : "solid",
                            borderWidth: 2,
                            opacity: dimmed ? 0.25 : 1,
                          }}
                          className="pointer-events-auto rounded-[2px] transition-opacity"
                          onMouseEnter={() => setHovered(c.index)}
                          onMouseLeave={() => setHovered(null)}
                        />
                      );
                    }),
                  )}
                </div>

                <p className="mt-2 text-xs text-black/45 dark:text-white/45">
                  A dashed outline means the region stayed at block level because the
                  heatmap could not separate the lines inside it. Solid outlines are
                  line-level hits. Every box comes from the document text layer, never
                  drawn from the heatmap.
                </p>

                {alternates.length > 0 && (
                  <div className="mt-4">
                    <p className="text-xs font-medium text-black/60 dark:text-white/60">
                      Retrieved pages
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {alternates.map((cand: Candidate) => {
                        const isActive =
                          cand.doc_sha === retrieved.doc_sha && cand.page === retrieved.page;
                        // A read page comes back instantly and costs nothing;
                        // an unread one costs a reader call plus a verifier
                        // call per claim. Worth telling apart before the click
                        // rather than after it.
                        const read =
                          results[keyOf(cand.doc_sha, cand.page)] !== undefined;
                        // Retrieval searches the whole corpus, so a candidate is
                        // often in another document. Labelled only "page 24" it
                        // reads as page 24 of the document on screen, and
                        // clicking it swaps the document with no indication.
                        const elsewhere = cand.doc_sha !== retrieved.doc_sha;
                        return (
                          <button
                            key={`${cand.doc_sha}-${cand.page}`}
                            type="button"
                            disabled={pending || isActive}
                            title={`${cand.doc_name} page ${cand.page}${
                              read ? ", already read" : ", not read yet"
                            }`}
                            onClick={() =>
                              void run(asked, { doc: cand.doc_sha, page: cand.page })
                            }
                            className={`rounded-full border px-3 py-1 text-xs hover:border-black/40 dark:hover:border-white/50 ${
                              isActive
                                ? "border-black bg-black text-white dark:border-white dark:bg-white dark:text-black"
                                : read
                                  ? "border-black/40 dark:border-white/45"
                                  : "border-dashed border-black/20 dark:border-white/25"
                            } ${pending && !isActive ? "opacity-40" : ""}`}
                          >
                            {elsewhere && (
                              <span className={isActive ? "mr-1 opacity-70" : "mr-1 opacity-60"}>
                                {cand.doc_name}
                              </span>
                            )}
                            page {cand.page}
                            <span className={isActive ? "ml-1 opacity-70" : "ml-1 opacity-60"}>
                              {cand.score.toFixed(3)}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                    <p className="mt-2 text-xs text-black/45 dark:text-white/45">
                      Ranked by MaxSim over the page patch embeddings. A solid chip has
                      already been read and comes back instantly with its own claims and
                      boxes; a dashed one costs a reader call plus a verifier call per
                      claim.
                    </p>
                  </div>
                )}
              </>
            ) : (
              <p className="text-sm text-black/45 dark:text-white/45">
                The retrieved page appears here.
              </p>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
