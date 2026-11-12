/**
 * One response-order guard for overlapping asynchronous reads.
 *
 * Call `begin()` when a request starts and pass its token to `isLatest()` before
 * committing data, errors, or loading flags. A later request makes every earlier
 * token stale; `invalidate()` does the same without starting a request.
 */
export interface RequestSequence {
  begin: () => number;
  isLatest: (token: number) => boolean;
  invalidate: () => void;
}

export function useRequestSequence(): RequestSequence {
  let current = 0;

  return {
    begin: () => ++current,
    isLatest: (token) => token === current,
    invalidate: () => {
      current++;
    },
  };
}
