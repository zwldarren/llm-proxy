/**
 * `HH:MM` clock plumbing for the 24-hour time fields.
 *
 * A native `<input type="time">` renders in the browser's locale, so an en-US
 * browser shows a 12-hour clock with an AM/PM selector and gives no way to turn
 * it off — wrong for a field whose stored value is a 24-hour wall clock. These
 * helpers keep the wire format as `HH:MM` and accept the ways an operator
 * actually types a time (`9`, `930`, `09:30`, `9:30`).
 */

/** Minutes since midnight for a normalized `HH:MM`, or null when unreadable. */
export function minutesOfClock(value: string): number | null {
  const match = /^([01]\d|2[0-3]):([0-5]\d)$/.exec(value);
  return match ? Number(match[1]) * 60 + Number(match[2]) : null;
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

/** Zero-padded `HH:MM` for minutes since midnight, wrapping past a day. */
export function clockFromMinutes(minutes: number): string {
  const wrapped = ((Math.round(minutes) % 1440) + 1440) % 1440;
  return `${pad(Math.floor(wrapped / 60))}:${pad(wrapped % 60)}`;
}

/**
 * A typed clock as normalized `HH:MM`, or null when it cannot be read.
 *
 * Bare digits are read the way a numeric keypad produces them: `9` is the hour,
 * `930` and `0930` are hours and minutes. `24:00` folds onto `00:00` so a day's
 * end is expressible either way; any other minute outside `00:00`–`23:59` is
 * rejected rather than silently wrapped.
 */
export function parseClockInput(input: string): string | null {
  const text = input.trim().replace(/\s+/g, "");
  if (!text) return null;
  if (/^[：:.]$/.test(text)) return null;

  let hours: number;
  let minutes: number;
  if (/^\d{1,2}$/.test(text)) {
    hours = Number(text);
    minutes = 0;
  } else if (/^\d{3,4}$/.test(text)) {
    hours = Number(text.slice(0, -2));
    minutes = Number(text.slice(-2));
  } else {
    const match = /^(\d{1,2})[：:.](\d{1,2})$/.exec(text);
    if (!match) return null;
    hours = Number(match[1]);
    minutes = Number(match[2]);
  }

  if (hours === 24 && minutes === 0) return "00:00";
  if (hours > 23 || minutes > 59) return null;
  return `${pad(hours)}:${pad(minutes)}`;
}

/** Validity of each probed zone name, so a keystroke costs one format at most. */
const timezoneValidity = new Map<string, boolean>();

/**
 * Whether `value` is an IANA zone name the platform can resolve.
 *
 * The pricing backend resolves the name with `ZoneInfo`, and a window whose
 * zone does not resolve is dropped from billing, so the editor has to say so
 * before a save rather than after. Asking the platform's own formatter keeps
 * aliases (`Asia/Shanghai`, `Etc/GMT+8`) valid and unknown names invalid.
 */
export function isKnownTimezone(value: string): boolean {
  const name = value.trim();
  if (!name) return false;
  const cached = timezoneValidity.get(name);
  if (cached !== undefined) return cached;
  let valid = true;
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: name });
  } catch {
    valid = false;
  }
  timezoneValidity.set(name, valid);
  return valid;
}
