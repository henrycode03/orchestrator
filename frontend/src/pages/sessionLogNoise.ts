const NOISY_LOG_PATTERNS = [
  /^"propertiesCount":\s*\d+,?$/,
  /^"schemaChars":\s*\d+,?$/,
  /^"summaryChars":\s*\d+,?$/,
  /^"promptChars":\s*\d+,?$/,
  /^"blockChars":\s*\d+,?$/,
  /^"rawChars":\s*\d+,?$/,
  /^"injectedChars":\s*\d+,?$/,
  /^"truncated":\s*(true|false),?$/,
  /^"missing":\s*(true|false),?$/,
  /^"replayInvalid":\s*(true|false),?$/,
  /^"livenessState":\s*"[^"]+",?$/,
  /^"stopReason":\s*"[^"]+",?$/,
  /^"path":\s*".*",?$/,
  /^"name":\s*"[^"]+",?$/,
  /^"name":\s*"(healthcheck|memory_get|memory_search|session_status|update_plan|web_search|web_fetch|image|pdf|browser|BOOTSTRAP\.md|MEMORY\.md)".*$/,
  /^"entries":\s*\[$/,
  /^"skills":\s*{$/,
  /^[[\]{}],?$/,
];

const NOISY_SUBSTRINGS = [
  '"propertiesCount"',
  '"schemaChars"',
  '"summaryChars"',
  '"promptChars"',
  '"blockChars"',
  '"rawChars"',
  '"injectedChars"',
  '"replayInvalid"',
  '"livenessState"',
  '"stopReason"',
  '"bootstrapTotalMaxChars"',
  '"bootstrapTruncation"',
  '"systemPromptReport"',
  '"injectedWorkspaceFiles"',
];

export function isNoisySessionLogMessage(message?: string | null): boolean {
  const trimmed = (message || '').trim();
  if (!trimmed) {
    return true;
  }

  if (NOISY_SUBSTRINGS.some((token) => trimmed.includes(token))) {
    return true;
  }

  return NOISY_LOG_PATTERNS.some((pattern) => pattern.test(trimmed));
}
