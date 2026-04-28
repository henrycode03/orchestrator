import { describe, expect, it } from 'vitest';

import { isNoisySessionLogMessage } from './sessionLogNoise';

describe('isNoisySessionLogMessage', () => {
  it('filters telemetry fragments in clean mode heuristics', () => {
    expect(isNoisySessionLogMessage('"replayInvalid": false,')).toBe(true);
    expect(isNoisySessionLogMessage('"livenessState": "working",')).toBe(true);
    expect(isNoisySessionLogMessage('"stopReason": "stop"')).toBe(true);
    expect(isNoisySessionLogMessage('"schemaChars": 1234,')).toBe(true);
  });

  it('keeps meaningful structured diagnostics visible', () => {
    expect(isNoisySessionLogMessage('"error": "model backend unavailable"')).toBe(
      false
    );
    expect(
      isNoisySessionLogMessage(
        '[ORCHESTRATION] Planning response received; parsing and validating plan'
      )
    ).toBe(false);
  });
});
