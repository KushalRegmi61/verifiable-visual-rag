import { describe, expect, it } from "vitest";
import { parseFrames } from "./api";

const frame = (name: string, data: unknown) =>
  `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;

describe("parseFrames", () => {
  it("returns both frames when one chunk carries two", () => {
    const chunk = frame("claims", { n: 2 }) + frame("claim", { index: 0, text: "a" });

    const { frames, rest } = parseFrames(chunk);

    expect(frames.map((f) => f.name)).toEqual(["claims", "claim"]);
    expect(frames[0].data).toEqual({ n: 2 });
    expect(frames[1].data).toEqual({ index: 0, text: "a" });
    expect(rest).toBe("");
  });

  // The bug this exists to catch: a chunk boundary falls wherever the network
  // put it, and emitting a fragment silently drops the event and corrupts every
  // frame after it. The tail must survive to the next read untouched.
  it("holds a frame back until its terminator arrives", () => {
    const whole = frame("claim", { index: 1, text: "split me" });
    const cut = whole.indexOf("text") + 2;

    const first = parseFrames(whole.slice(0, cut));
    expect(first.frames).toEqual([]);
    expect(first.rest).toBe(whole.slice(0, cut));

    const second = parseFrames(first.rest + whole.slice(cut));
    expect(second.frames).toHaveLength(1);
    expect(second.frames[0].name).toBe("claim");
    expect(second.frames[0].data).toEqual({ index: 1, text: "split me" });
    expect(second.rest).toBe("");
  });

  it("keeps a partial frame that trails a complete one", () => {
    const chunk = frame("claims", { n: 1 }) + "event: claim\ndata: {\"index\"";

    const { frames, rest } = parseFrames(chunk);

    expect(frames).toHaveLength(1);
    expect(rest).toBe('event: claim\ndata: {"index"');
  });

  // The server escapes newlines with json.dumps precisely because a raw newline
  // in a data: field would terminate it. A verifier reason spanning two lines is
  // ordinary, so the escape has to round-trip rather than split the frame.
  it("round-trips an escaped newline inside the payload", () => {
    const reason = "line one\nline two";
    const { frames, rest } = parseFrames(frame("claim", { index: 0, reason }));

    expect(rest).toBe("");
    expect(frames).toHaveLength(1);
    expect((frames[0].data as { reason: string }).reason).toBe(reason);
  });

  it("returns nothing for an empty buffer", () => {
    expect(parseFrames("")).toEqual({ frames: [], rest: "" });
  });
});
