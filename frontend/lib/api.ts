import type { Region } from "./overlay";

// EventSource is GET-only, which would force the question into a query string
// and give up a request body for doc, page, k and threshold. Streaming a POST
// response is well supported and needs no extra machinery.

export const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

export type ClaimLabel =
  | "supported"
  | "partially_supported"
  | "insufficient_evidence"
  | "unsupported";

export type ClaimEvent = {
  index: number;
  text: string;
  label: ClaimLabel | null;
  confidence: number;
  reason: string | null;
  compound: boolean;
  withheld: boolean;
  regions: Region[];
  // Where the answer breaks into a new paragraph. Set by the reader, which is
  // the only thing that knows where the topic turns.
  starts_paragraph: boolean;
  // True only on the lead claim, and only when it was withheld. It says the
  // answer is already abstaining, while claims are still arriving and long
  // before `done`. Non-optional, like starts_paragraph: the wire always sends
  // it, and an `undefined` reading as falsy would mean "not abstaining", which
  // is the wrong direction to fail in.
  abstains_answer: boolean;
};

export type Candidate = {
  doc_sha: string;
  page: number;
  score: number;
  // Retrieval is corpus-wide and takes no document filter, so a candidate is
  // often in a different document than the one on screen.
  doc_name: string;
};

export type RetrievedEvent = {
  doc_sha: string;
  doc_name: string;
  page: number;
  score: number | null;
  candidates: Candidate[];
  // Sent before any model call. Currently only "this page is not embedded",
  // which otherwise surfaces as every claim coming back insufficient_evidence
  // and reads as a verdict about the evidence rather than a missing index.
  warning: string | null;
};

export type DoneEvent = { shown: number; withheld: number; abstained_overall: boolean };

export type Frame = { name: string; data: unknown };

type Handlers = {
  onRetrieved: (e: RetrievedEvent) => void;
  onClaims: (n: number) => void;
  onClaim: (e: ClaimEvent) => void;
  onDone: (e: DoneEvent) => void;
  onError: (message: string) => void;
};

/**
 * Split whatever complete SSE frames the buffer holds, and hand back the tail.
 *
 * The tail is the whole point. A chunk boundary falls wherever the network put
 * it, so a frame routinely arrives in two reads; parsing what is present would
 * drop the event and leave a fragment that corrupts every frame after it.
 */
export function parseFrames(buffer: string): { frames: Frame[]; rest: string } {
  const frames: Frame[] = [];
  let rest = buffer;

  // Frames are separated by a blank line. A partial frame stays in the
  // buffer until its terminator arrives; parsing early would drop events.
  let split: number;
  while ((split = rest.indexOf("\n\n")) !== -1) {
    const block = rest.slice(0, split);
    rest = rest.slice(split + 2);
    const [nameLine, dataLine] = block.split("\n");
    if (!nameLine || dataLine === undefined) continue;
    const name = nameLine.replace("event: ", "");
    const data = JSON.parse(dataLine.replace("data: ", ""));
    frames.push({ name, data });
  }

  return { frames, rest };
}

export async function ask(
  body: { question: string; doc?: string; page?: number },
  h: Handlers,
): Promise<void> {
  const res = await fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    h.onError(`${res.status}: ${(await res.text()) || res.statusText}`);
    return;
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const { frames, rest } = parseFrames(buffer);
    buffer = rest;
    for (const { name, data } of frames) {
      if (name === "retrieved") h.onRetrieved(data as RetrievedEvent);
      else if (name === "claims") h.onClaims((data as { n: number }).n);
      else if (name === "claim") h.onClaim(data as ClaimEvent);
      else if (name === "done") h.onDone(data as DoneEvent);
      else if (name === "error") h.onError((data as { message: string }).message);
    }
  }
}
