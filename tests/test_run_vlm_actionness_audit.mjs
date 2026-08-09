import assert from 'node:assert/strict';
import http from 'node:http';
import test from 'node:test';

import {
  FRAME_OFFSETS_SEC,
  HIGH_PROMPT_PROTOCOL_VERSION,
  HIGH_PROMPT_ROLE,
  HIGH_PROMPT_SHA256,
  HIGH_PROMPT_TEXT,
  HttpStatusError,
  LOW_PROMPT_PROTOCOL_VERSION,
  LOW_PROMPT_ROLE,
  LOW_PROMPT_SHA256,
  LOW_PROMPT_TEXT,
  PROMPT_SHA256,
  RESPONSE_CONTRACT_VERSION,
  RESPONSE_TEXT_FORMAT,
  SEMANTIC_CACHE_SCHEMA_VERSION,
  RequestTimeoutError,
  absoluteCenterTime,
  absoluteFeatureIndex,
  argsOf,
  assertCompleteSummary,
  completionSummary,
  fetchCompat,
  parseGateResponse,
  promptProtocol,
  promptRoleForScore,
  requestFingerprint,
  requestWithRetry,
  semanticKey,
  validateManifestRows,
  validatePriorRows,
} from '../tools/run_vlm_actionness_audit.mjs';


function manifestRow() {
  return {
    candidate_id: 'p123',
    anchor_id: 'p123',
    video_id: 'video',
    absolute_feature_index: 123,
    absolute_center_time: 12.5,
    center_sec: 12.5,
  };
}


function clone(value) {
  return JSON.parse(JSON.stringify(value));
}


test('schema is semantic-only and absolute-index keyed', () => {
  assert.equal(
    SEMANTIC_CACHE_SCHEMA_VERSION,
    'offline-vlm-semantic-evidence-v1',
  );
  assert.deepEqual(
    FRAME_OFFSETS_SEC,
    [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
  );
  const row = manifestRow();
  assert.equal(absoluteFeatureIndex(row), 123);
  assert.equal(semanticKey(row), 'low_rescue|video|123');
  assert.equal(
    RESPONSE_CONTRACT_VERSION,
    'vlm-actionness-label-json-schema-v1',
  );
  assert.equal(RESPONSE_TEXT_FORMAT.type, 'json_schema');
  assert.equal(RESPONSE_TEXT_FORMAT.strict, true);
  assert.deepEqual(
    RESPONSE_TEXT_FORMAT.schema.properties.label.enum,
    ['TARGET_ACTION', 'OTHER_ACTIVITY', 'TRUE_BACKGROUND', 'UNCERTAIN'],
  );
});


test('frozen low prompt is byte-identical to the audited v3 protocol', () => {
  assert.equal(LOW_PROMPT_PROTOCOL_VERSION, 'candidate-blind-epic-semantic-gate-v3');
  assert.equal(
    LOW_PROMPT_SHA256,
    'fe53deb05bb2b13faa8ecfdaae0ddc73fb1eafe1909493aeedf646087402efd3',
  );
  assert.equal(PROMPT_SHA256, LOW_PROMPT_SHA256);
  assert.match(LOW_PROMPT_TEXT, /False rescue is more harmful/);
});


test('teacher score region selects the independent prompt role', () => {
  assert.equal(promptRoleForScore(0.2, 0.1, 0.7), LOW_PROMPT_ROLE);
  assert.equal(promptRoleForScore(0.699, 0.1, 0.7), LOW_PROMPT_ROLE);
  assert.equal(promptRoleForScore(0.7, 0.1, 0.7), HIGH_PROMPT_ROLE);
  assert.equal(promptRoleForScore(0.9, 0.1, 0.7), HIGH_PROMPT_ROLE);
  assert.equal(promptRoleForScore(0.1, 0.1, 0.7), null);
});


test('high veto prompt is independent and candidate-blind', () => {
  const high = promptProtocol({ ...manifestRow(), prompt_role: HIGH_PROMPT_ROLE });
  assert.equal(high.version, HIGH_PROMPT_PROTOCOL_VERSION);
  assert.equal(high.sha256, HIGH_PROMPT_SHA256);
  assert.equal(high.text, HIGH_PROMPT_TEXT);
  assert.notEqual(HIGH_PROMPT_SHA256, LOW_PROMPT_SHA256);
  for (const forbidden of [
    'teacher score', 'model confidence', 'target gt', 'ground truth',
    'fpn level', 'assignment validity', 'regression range', 'cohort',
  ]) {
    assert.equal(HIGH_PROMPT_TEXT.toLowerCase().includes(forbidden), false);
  }
  assert.match(HIGH_PROMPT_TEXT, /Object presence is not action evidence/);
  assert.match(HIGH_PROMPT_TEXT, /Hand-object contact alone is not/);
  assert.match(HIGH_PROMPT_TEXT, /prefer UNCERTAIN/);
});


test('routing metadata and post-hoc GT never enter the request prompt identity', () => {
  const base = { ...manifestRow(), prompt_role: HIGH_PROMPT_ROLE };
  const auditOnly = {
    ...base,
    sada_score: 0.99,
    alpha: 0.7,
    beta: 0.1,
    fpn_level: 5,
    assignment_valid: false,
    gt_is_foreground: false,
    target_gt: 'background',
    cohort: 'high_background',
  };
  assert.equal(
    requestFingerprint(base, 'model', 'responses'),
    requestFingerprint(auditOnly, 'model', 'responses'),
  );
  assert.equal(promptProtocol(base).text, promptProtocol(auditOnly).text);
});


test('valid target foreground is the only semantic pass', () => {
  const parsed = parseGateResponse(JSON.stringify({ label: 'TARGET_ACTION' }));
  assert.equal(parsed.semantic_pass, true);
  assert.equal(parsed.prediction, 'TARGET_ACTION');
  for (const label of ['OTHER_ACTIVITY', 'TRUE_BACKGROUND', 'UNCERTAIN']) {
    const ignored = parseGateResponse(JSON.stringify({ label }));
    assert.equal(ignored.semantic_pass, false);
    assert.equal(ignored.prediction, label);
  }
});


test('malformed and inconsistent responses are protocol failures', () => {
  assert.equal(parseGateResponse('not-json').parse_error, true);
  assert.equal(parseGateResponse('null').protocol_error, true);
  const inconsistent = parseGateResponse(JSON.stringify({
    label: 'TARGET_ACTION', extra: true,
  }));
  assert.equal(inconsistent.consistency_error, true);
  assert.equal(inconsistent.semantic_pass, false);
});


test('manifest enforces exact absolute index and timestamp', () => {
  const row = manifestRow();
  validateManifestRows([row]);
  assert.equal(absoluteCenterTime(row), 12.5);

  const badAlias = clone(row);
  badAlias.center_sec = 12.6;
  assert.throws(() => validateManifestRows([badAlias]), /center_sec differs/);

  const nonInteger = clone(row);
  nonInteger.absolute_feature_index = 123.5;
  assert.throws(() => validateManifestRows([nonInteger]), /absolute_feature_index/);

  const duplicate = clone(row);
  duplicate.candidate_id = 'duplicate';
  assert.throws(
    () => validateManifestRows([row, duplicate]),
    /duplicate exact semantic key/,
  );

  const distinctRole = clone(row);
  distinctRole.candidate_id = 'high-copy';
  distinctRole.prompt_role = HIGH_PROMPT_ROLE;
  assert.doesNotThrow(() => validateManifestRows([row, distinctRole]));
});


test('legacy prompt-mode argument is rejected', () => {
  assert.throws(
    () => argsOf(['node', 'runner', '--prompt-mode', 'anchor_rescue']),
    /unsupported CLI argument/,
  );
});


test('retry only handles timeout, 429, and server errors', async () => {
  let calls = 0;
  const retryThenSucceed = async () => {
    calls += 1;
    if (calls === 1) throw new HttpStatusError(429, 'quota');
    if (calls === 2) throw new RequestTimeoutError('timeout');
    return { text: '{}', usage: null };
  };
  const result = await requestWithRetry({}, 3, 0, retryThenSucceed);
  assert.equal(result.attempts, 3);

  calls = 0;
  await assert.rejects(
    requestWithRetry({}, 5, 0, async () => {
      calls += 1;
      throw new HttpStatusError(401, 'unauthorized');
    }),
    /unauthorized/,
  );
  assert.equal(calls, 1);
});


test('native fetch timeout aborts an outstanding request', async () => {
  const server = http.createServer((request, response) => {
    setTimeout(() => response.end('late'), 100);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  try {
    await assert.rejects(
      fetchCompat(`http://127.0.0.1:${port}`, { method: 'GET' }, 10),
      error => error instanceof RequestTimeoutError,
    );
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});


test('resume output cannot mix schema, prompt, model, or wire API', () => {
  const manifest = manifestRow();
  const model = 'qwen-test';
  const wireApi = 'chat';
  const prior = {
    candidate_id: manifest.candidate_id,
    semantic_cache_schema_version: SEMANTIC_CACHE_SCHEMA_VERSION,
    response_contract_version: RESPONSE_CONTRACT_VERSION,
    prompt_sha256: PROMPT_SHA256,
    model,
    wire_api: wireApi,
    request_fingerprint: requestFingerprint(manifest, model, wireApi),
  };
  assert.doesNotThrow(
    () => validatePriorRows([prior], [manifest], model, wireApi),
  );
  assert.throws(
    () => validatePriorRows(
      [{ ...prior, semantic_cache_schema_version: 'old-v3' }],
      [manifest], model, wireApi,
    ),
    /semantic cache schema/,
  );

  const highManifest = { ...manifest, prompt_role: HIGH_PROMPT_ROLE };
  assert.throws(
    () => validatePriorRows([prior], [highManifest], model, wireApi),
    /prompt_role/,
  );
});


test('completion summary reports semantic pass and ignore only', () => {
  const rows = [
    { candidate_id: 'a' },
    { candidate_id: 'b' },
    { candidate_id: 'c' },
  ];
  const latest = new Map([
    ['a', { status: 'ok', semantic_pass: true }],
    ['b', { status: 'ok', semantic_pass: false }],
    ['c', { status: 'ok', semantic_pass: false }],
  ]);
  const summary = completionSummary(rows, latest, {
    alreadyCompleted: 1,
    apiAttempted: 1,
    apiSuccess: 1,
    localUncertain: 1,
  });
  assert.deepEqual(summary, {
    manifest_total: 3,
    already_completed: 1,
    api_attempted: 1,
    api_success: 1,
    local_uncertain: 1,
    completed_total: 3,
    semantic_pass: 1,
    semantic_ignore: 2,
    error: 0,
  });
  assert.doesNotThrow(() => assertCompleteSummary(summary));
});
