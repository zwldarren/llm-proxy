import { describe, expect, it } from "vitest";
import { useRequestSequence } from "./useRequestSequence";

describe("useRequestSequence", () => {
  it("only treats the newest request as current", () => {
    const requests = useRequestSequence();
    const first = requests.begin();
    const second = requests.begin();

    expect(requests.isLatest(first)).toBe(false);
    expect(requests.isLatest(second)).toBe(true);
  });

  it("can discard an in-flight request without starting another", () => {
    const requests = useRequestSequence();
    const pending = requests.begin();

    requests.invalidate();

    expect(requests.isLatest(pending)).toBe(false);
  });
});
