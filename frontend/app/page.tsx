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
import { isAbstaining, pageForClaim, type PageRef } from "@/lib/claims";
import type { Region } from "@/lib/overlay";
import { AnswerPanel } from "@/components/AnswerPanel";
import { EvidenceVault } from "@/components/EvidenceVault";
import { PageRail } from "@/components/PageRail";
import { PageViewer } from "@/components/PageViewer";
import { ZoomOverlay } from "@/components/ZoomOverlay";
import { AlertIcon } from "@/components/icons";
import { AskComposer } from "@/components/AskComposer";

/** Everything one page's answer is made of. */
type PageResult = {
  retrieved: RetrievedEvent;
  expected: number | null;
  claims: ClaimEvent[];
  done: DoneEvent | null;
};

type Zoom = { claim: ClaimEvent; region: Region };

function keyOf(docSha: string, page: number): string {
  return `${docSha}:${page}`;
}

function imageFor(ref: PageRef): string {
  return `${API}/documents/${ref.doc_sha}/pages/${ref.page}/image`;
}

/** The page a retrieval opens on: the top of the list the reader was given. */
function topPageOf(e: RetrievedEvent): PageRef {
  // pages[0] is the top page, guaranteed server-side. The fallback is for a
  // response that carries no list at all rather than a wrong one, so that a
  // missing field costs the multi-page behaviour and not the viewer.
  return e.pages[0] ?? { doc_sha: e.doc_sha, page: e.page };
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
  // show more than one retrieved page.
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
  const [expanded, setExpanded] = useState<number | null>(null);
  const [zoom, setZoom] = useState<Zoom | null>(null);
  // Which of the pages the reader saw is in the viewer. Separate from `active`,
  // which is the page that was READ and keys the result cache: one reader call
  // covers three page images, so the answer belongs to the set while the viewer
  // shows one of them at a time. Null until the retrieved frame names a top
  // page.
  const [viewing, setViewing] = useState<PageRef | null>(null);
  // Measured from the rendered page image rather than assumed. Reset on a page
  // change so a landscape page cannot be cropped using the ratio of the
  // portrait one before it, which would distort every crop until the new image
  // finished loading.
  const [pageAspect, setPageAspect] = useState<number | null>(null);

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
        setExpanded(null);
        setPageAspect(null);
        // Back to the top of the cached read's own page list, not to whatever
        // page the last claim hovered before leaving happened to be on.
        setViewing(topPageOf(results[keyOf(pin.doc, pin.page)].retrieved));
        return;
      }

      setPending(true);
      setAsked(text);
      setError(null);
      setHovered(null);
      setExpanded(null);
      setZoom(null);
      setActive(null);
      setViewing(null);
      setPageAspect(null);
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
              setViewing(topPageOf(e));
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
  const expectedCount = currentPage?.expected ?? null;
  const done = currentPage?.done ?? null;

  // The one line that guarantees a withheld claim can never reach the overlay.
  // The service already sends it with regions: [], and this makes the display
  // side agree rather than relying on that alone.
  const shown = claims.filter((c) => !c.withheld);
  const withheld = claims.filter((c) => c.withheld);
  // The full `claims` array, not `shown`. A withheld lead is filtered out of
  // `shown` by construction, so the flag that says the answer is abstaining
  // only exists on the unfiltered list.
  const abstained = isAbstaining(claims, done);
  // Every page the reader saw, and the one currently in the viewer. `viewing`
  // is set from the same frame as `retrieved`, so falling back to the top page
  // here covers the render between the two rather than a missing value.
  const pages = retrieved?.pages ?? [];
  // Resolved per claim, not once. The vault crops a thumbnail per region, and a
  // claim grounded on another of the pages the reader saw needs THAT page's
  // image; handing every crop the retrieved page's image showed real ink from
  // the wrong page at the right coordinates, which looks correct and is not.
  //
  // pageAspect below is still measured from the viewer's image only. Every page
  // here belongs to one document, and a document's pages are the same size in
  // every corpus this has run on, so the aspect is shared in practice. If
  // cross-document reading ever lands, the crop for a page of a different shape
  // would be distorted and the aspect has to become per page too.
  const imageUrlFor = (c: ClaimEvent) =>
    retrieved ? imageFor(pageForClaim(c, pages, topPageOf(retrieved))) : null;
  const viewingRef = viewing ?? (retrieved ? topPageOf(retrieved) : null);
  const viewerUrl = viewingRef ? imageFor(viewingRef) : null;
  // True only while the viewer is on the page retrieval actually ranked. The
  // score is a property of that page, and leaving it on the header while
  // showing page 9 would attach page 4's number to page 9.
  const onRankedPage =
    viewingRef !== null &&
    viewingRef.doc_sha === retrieved?.doc_sha &&
    viewingRef.page === retrieved?.page;
  // A zoom crops the page ITS REGION is on, which is not always the page in the
  // viewer: the evidence vault can open a region belonging to another of the
  // pages the reader saw. Cropping the displayed page at those coordinates
  // would magnify unrelated text and caption it as the evidence.
  const zoomRef =
    zoom && viewingRef
      ? pageForClaim({ regions: [zoom.region] }, pages, viewingRef)
      : null;
  // A pinned re-ask does no retrieval, so its Retrieved event carries score:
  // null by construction. Reading the score off the ranking instead means the
  // header shows one for every page, which is what makes two pages comparable:
  // without it, switching to an alternate silently drops the only number that
  // says how well that page matched.
  const rankScore =
    retrieved?.score ??
    alternates.find(
      (c) => c.doc_sha === retrieved?.doc_sha && c.page === retrieved?.page,
    )?.score ??
    null;

  // The viewer follows the claim. Its evidence can be on any of the pages the
  // reader saw, and boxes are drawn only for the page on screen, so without
  // this a claim grounded on page 9 highlights nothing at all while page 4 is
  // displayed.
  //
  // `viewingRef` is the fallback, not the top page: a claim with no regions
  // must leave the viewer exactly where it is. Sending it home to the top page
  // would make hovering a withheld sentence yank the image away from the page
  // the user was reading.
  //
  // Plain functions rather than useCallback, unlike `run` above. They close
  // over `claims`, `pages` and `viewingRef`, which are derived from state on
  // every render, so a manual dependency list would be a new array every time
  // and buy nothing; the React Compiler rejects it outright with "existing
  // memoization could not be preserved" and then skips optimizing the whole
  // component, which costs more than the hook saves.
  const followClaim = (index: number) => {
    if (!viewingRef) return;
    const claim = claims.find((c) => c.index === index);
    if (!claim) return;
    const next = pageForClaim(claim, pages, viewingRef);
    // Same page, same object. Every box on screen calls this on mouse-enter,
    // and a fresh object each time would re-render the whole column for a
    // move that did not happen.
    setViewing((prev) =>
      prev && prev.doc_sha === next.doc_sha && prev.page === next.page ? prev : next,
    );
  };

  const hoverClaim = (index: number | null) => {
    setHovered(index);
    // Mouse-out deliberately does NOT return to the previous page. Snapping
    // back the instant the pointer leaves a sentence makes the viewer flick
    // between scans as the user reads down the answer, and nothing stays on
    // screen long enough to be read.
    if (index !== null) followClaim(index);
  };

  // Clicking a sentence in the answer opens its evidence and brings it into
  // view. Without the scroll the disclosure opens somewhere below the fold and
  // the click reads as having done nothing.
  const revealClaim = (index: number) => {
    setExpanded((prev) => (prev === index ? null : index));
    followClaim(index);
    requestAnimationFrame(() => {
      document
        .getElementById(`claim-${index}`)
        ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  };

  // `!done` as well as `pending`, because `pending` is cleared in run's
  // `finally`, one or more ticks after onDone lands. Without it there is a
  // commit where the decline card renders above a live "Verifying claim 3 of
  // 3" skeleton. It also composes with the early abstention: the decline now
  // appears on the lead's claim frame and SHOULD sit above a still-running
  // verify line until `done`.
  const status =
    pending && !done
      ? retrieved === null
        ? "Retrieving the best page"
        : expectedCount === null
          ? "Reading the page"
          : `Verifying claim ${Math.min(claims.length + 1, expectedCount)} of ${expectedCount}`
      : null;

  return (
    <>
      <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur-md">
        <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-3 px-5 py-3 lg:flex-row lg:items-center lg:gap-6 lg:px-8">
          <div className="shrink-0">
            <h1 className="text-sm font-semibold tracking-tight">Verifiable Visual RAG</h1>
            <p className="text-[11px] leading-tight text-faint">
              Region-level evidence, independently verified
            </p>
          </div>

          <AskComposer
            value={question}
            onChange={setQuestion}
            onSubmit={() => void run(question)}
            pending={pending}
          />
        </div>
        {pending && (
          <div className="h-0.5 w-full overflow-hidden bg-transparent">
            <div className="vv-sweep h-full w-1/4 bg-accent" />
          </div>
        )}
      </header>

      <main className="mx-auto w-full max-w-[1600px] flex-1 px-5 py-6 lg:px-8">
        {error && (
          <div
            role="alert"
            className="mb-5 flex gap-2.5 rounded-xl border border-danger/40 bg-danger/8 px-4 py-3 text-sm text-danger"
          >
            <AlertIcon className="mt-0.5 h-4 w-4 shrink-0" />
            <span className="min-w-0 break-words">{error}</span>
          </div>
        )}

        {!asked ? (
          <div className="mx-auto max-w-xl py-24 text-center">
            <h2 className="text-2xl font-semibold tracking-tight">
              Ask, and see exactly where the answer came from.
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              Every sentence of the answer is checked by a second model and pinned to a
              region of the page it came from. Claims that fail the check are withheld
              rather than shown.
            </p>
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-8">
            <div className="min-w-0">
              {/* The decline is otherwise invisible on this channel: announcing
                  "Answer complete" while the visible copy reads "I could not
                  answer this from this page" tells a screen reader user the
                  opposite of what is on screen, and this is the only place they
                  would hear about it. */}
              <p aria-live="polite" className="sr-only">
                {status ?? (done ? (abstained ? "No answer given" : "Answer complete") : "")}
              </p>

              {retrieved?.warning && (
                <div className="mb-4 flex gap-2.5 rounded-xl border border-warn/40 bg-warn-soft px-4 py-3 text-sm text-warn">
                  <AlertIcon className="mt-0.5 h-4 w-4 shrink-0" />
                  <span className="min-w-0">{retrieved.warning}</span>
                </div>
              )}

              {abstained && (
                <div className="mb-4 rounded-2xl border border-border bg-surface p-5">
                  <h2 className="text-base font-semibold">
                    I could not answer this from this page
                  </h2>
                  <p className="mt-2 text-sm leading-relaxed text-muted">
                    {/* Only ever a statement about this system's own process.
                        "I could not confirm" is true by construction whenever
                        this is shown. A sentence about what the page does or
                        does not contain would be a claim with no region behind
                        it, which is precisely what the rest of the system
                        exists to refuse. */}
                    {/* Names every page the reader saw, not just the one retrieval
                        ranked first. Saying "page 19" when three pages were read
                        understates what was checked, and a reader who knows the
                        answer is on page 14 would reasonably conclude the system
                        never looked. The list comes from the server, so it cannot
                        drift from what was actually read. */}
                    I read {retrieved?.doc_name}{" "}
                    {retrieved && retrieved.pages.length > 1
                      ? `pages ${retrieved.pages.map((p) => p.page).join(", ")}`
                      : `page ${retrieved?.page}`}
                    , but I was not able to confirm an answer to your question from what
                    is on {retrieved && retrieved.pages.length > 1 ? "them" : "it"}.
                    Rather than give you something I cannot stand behind, I have left it
                    out.
                    {/* Without this, an unembedded page reads as a judgement about
                        the evidence when the real cause is that nobody ran
                        `vvrag embed`. */}
                    {retrieved?.warning &&
                      " Note the warning above: this page has no embeddings, so this is" +
                        " very likely a missing index rather than weak evidence."}
                  </p>
                  {/* Under the lead rule, abstaining no longer means nothing
                      survived: a withheld first claim abstains the answer even
                      when later claims passed. Those survivors stay in the
                      Evidence Vault below, which already presents them as
                      evidence rather than as an answer. */}
                  {shown.length > 0 && (
                    <p className="mt-3 text-sm italic text-muted">
                      Here is what I was able to confirm from the page, in case it helps.
                    </p>
                  )}
                </div>
              )}

              {status && shown.length === 0 && (
                <div className="rounded-2xl border border-border bg-surface p-5">
                  <p className="text-sm text-muted">{status}</p>
                  <div className="mt-4 space-y-2.5" aria-hidden>
                    <div className="h-3 w-full rounded bg-sunken" />
                    <div className="h-3 w-[88%] rounded bg-sunken" />
                    <div className="h-3 w-[62%] rounded bg-sunken" />
                  </div>
                </div>
              )}

              {/* `!abstained` because the lead rule made `shown.length > 0 &&
                  abstained` reachable: a withheld first claim abstains the
                  answer while later claims survive. Without the guard the page
                  renders the decline and a panel headed "Answer" at the same
                  time, telling the user both that there is no answer and that
                  this is it. */}
              {shown.length > 0 && !abstained && (
                <AnswerPanel
                  shown={shown}
                  done={done}
                  withheldCount={withheld.length}
                  hovered={hovered}
                  onHover={hoverClaim}
                  onSelect={revealClaim}
                />
              )}

              {status && shown.length > 0 && (
                <p className="mt-3 px-1 text-xs text-faint">{status}</p>
              )}

              <EvidenceVault
                shown={shown}
                withheld={withheld}
                imageUrlFor={imageUrlFor}
                expanded={expanded}
                hovered={hovered}
                onToggle={revealClaim}
                onHover={hoverClaim}
                onZoom={(claim, region) => setZoom({ claim, region })}
                pageAspect={pageAspect}
              />

              {/* "No claims to show" used to live here and was unreachable by
                  construction: this client's `shown` filter is the same
                  predicate as the server's Answer.shown, and abstained_overall
                  is true whenever nothing survived, so `done && !abstained`
                  already implies shown.length > 0. It read as a live state and
                  was not one. The refusal is the card above. */}
            </div>

            {/* Sticky on desktop so the page stays in view while the evidence
                vault scrolls beside it. The two are meant to be read against
                each other, and a viewer that scrolls away breaks that. */}
            <div className="min-w-0 lg:sticky lg:top-[5.5rem] lg:self-start">
              {retrieved && viewingRef && viewerUrl ? (
                <>
                  <PageViewer
                    retrieved={retrieved}
                    imageUrl={viewerUrl}
                    page={viewingRef.page}
                    score={onRankedPage ? rankScore : null}
                    shown={shown}
                    hovered={hovered}
                    onHover={hoverClaim}
                    onZoom={(claim, region) => setZoom({ claim, region })}
                    onMeasure={setPageAspect}
                  />
                  <PageRail
                    candidates={alternates}
                    activeDoc={retrieved.doc_sha}
                    activePage={retrieved.page}
                    isRead={(sha, page) => results[keyOf(sha, page)] !== undefined}
                    pending={pending}
                    onSelect={(cand) =>
                      void run(asked, { doc: cand.doc_sha, page: cand.page })
                    }
                  />
                </>
              ) : (
                <div className="flex h-72 items-center justify-center rounded-2xl border border-dashed border-border text-sm text-faint">
                  The retrieved page appears here.
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {zoom && retrieved && zoomRef && (
        <ZoomOverlay
          imageUrl={imageFor(zoomRef)}
          claim={zoom.claim}
          region={zoom.region}
          docName={retrieved.doc_name}
          page={zoomRef.page}
          pageAspect={pageAspect}
          onClose={() => setZoom(null)}
        />
      )}
    </>
  );
}
