import { describe, expect, it } from "vitest";
import { clockFromMinutes, isKnownTimezone, minutesOfClock, parseClockInput } from "./clock";

describe("parseClockInput", () => {
  it("reads a bare hour as the top of that hour", () => {
    expect(parseClockInput("9")).toBe("09:00");
    expect(parseClockInput("09")).toBe("09:00");
    expect(parseClockInput("23")).toBe("23:00");
  });

  it("reads keypad digits as hours and minutes", () => {
    expect(parseClockInput("930")).toBe("09:30");
    expect(parseClockInput("0930")).toBe("09:30");
    expect(parseClockInput("130")).toBe("01:30");
    expect(parseClockInput("0000")).toBe("00:00");
  });

  it("reads a colon, dot or full-width colon", () => {
    expect(parseClockInput("09:30")).toBe("09:30");
    expect(parseClockInput("9:30")).toBe("09:30");
    expect(parseClockInput("1:5")).toBe("01:05");
    expect(parseClockInput("9.30")).toBe("09:30");
    expect(parseClockInput("9：30")).toBe("09:30");
  });

  it("folds the end of a day onto midnight", () => {
    expect(parseClockInput("24")).toBe("00:00");
    expect(parseClockInput("24:00")).toBe("00:00");
  });

  it("rejects what is not a clock instead of wrapping it", () => {
    // Wrapping silently would reprice a band at a different hour than typed.
    expect(parseClockInput("")).toBeNull();
    expect(parseClockInput("  ")).toBeNull();
    expect(parseClockInput("25")).toBeNull();
    expect(parseClockInput("24:30")).toBeNull();
    expect(parseClockInput("09:60")).toBeNull();
    expect(parseClockInput("9:5:2")).toBeNull();
    expect(parseClockInput("abc")).toBeNull();
    expect(parseClockInput(":")).toBeNull();
  });
});

describe("clockFromMinutes", () => {
  it("formats and wraps minutes into a day", () => {
    expect(clockFromMinutes(0)).toBe("00:00");
    expect(clockFromMinutes(90)).toBe("01:30");
    expect(clockFromMinutes(1440)).toBe("00:00");
    expect(clockFromMinutes(1500)).toBe("01:00");
    expect(clockFromMinutes(-60)).toBe("23:00");
  });
});

describe("minutesOfClock", () => {
  it("reads normalized clocks only", () => {
    expect(minutesOfClock("01:30")).toBe(90);
    expect(minutesOfClock("00:00")).toBe(0);
    expect(minutesOfClock("1:30")).toBeNull();
    expect(minutesOfClock("24:00")).toBeNull();
  });
});

describe("isKnownTimezone", () => {
  it("accepts platform zones, aliases and UTC", () => {
    expect(isKnownTimezone("UTC")).toBe(true);
    expect(isKnownTimezone("Asia/Shanghai")).toBe(true);
    expect(isKnownTimezone("Etc/GMT+8")).toBe(true);
  });

  it("rejects blank and unknown names so the editor can warn first", () => {
    expect(isKnownTimezone("")).toBe(false);
    expect(isKnownTimezone("   ")).toBe(false);
    expect(isKnownTimezone("Asia/Atlantis")).toBe(false);
    expect(isKnownTimezone("UTC+8")).toBe(false);
  });
});
