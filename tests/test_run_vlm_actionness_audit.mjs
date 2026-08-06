import assert from 'node:assert/strict';
import http from 'node:http';
import test from 'node:test';

import {
  HttpStatusError,
  PROMPT_SHA256,
  ROUTING_SCHEMA_VERSION,
  RequestTimeoutError,
  absoluteCenterTime,
  argsOf,
  assertCompleteSummary,
  completionSummary,
  fetchCompat,
  parseGateResponse,
  requestFingerprint,
  requestWithRetry,
  validateManifestRows,
  validatePriorRows,
} from '../tools/run_vlm_actionness_audit.mjs';

function manifestRow() {
  return {
    candidate_id: 'exact',
    video_id: 'video',
    fpn_level: 2,
    anchor_index: 7,
    global_anchor_index: 123,
    absolute_center_time: 12.5,
    center_sec: 12.5,
    teacher_top1_class: 3,
    member_anchors: [{
      fpn_level: 2,
      anchor_index: 7,
      global_anchor_index: 123,
      absolute_center_time: 12.5,
      teacher_top1_class: 3,
    }],
  };
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

test('routing schema is locked to candidate-blind exact-anchor v1', () => {
  assert.equal(
    ROUTING_SCHEMA_VERSION,
    'candidate-blind-exact-anchor-routing-v1',
  );
});

test('valid target foreground is the only action rescue', () => {
  const parsed = parseGateResponse(JSON.stringify({
    center_interaction: 'YES',
    target_signature: 'YES',
    label: 'TARGET_FOREGROUND',
  }));
  assert.equal(parsed.action_rescue, true);
  assert.equal(parsed.prediction, 'TARGET_FOREGROUND');
  assert.equal(parsed.protocol_error, false);
  assert.equal(parsed.consistency_error, false);
});

test('valid non-confirm responses remain normalized ignores', () => {
  const cases = [
    ['YES', 'NO', 'OTHER_ACTIVITY'],
    ['NO', 'NO', 'TRUE_BACKGROUND'],
    ['UNCERTAIN', 'UNCERTAIN', 'UNCERTAIN'],
  ];
  for (const [center, signature, label] of cases) {
    const parsed = parseGateResponse(JSON.stringify({
      center_interaction: center,
      target_signature: signature,
      label,
    }));
    assert.equal(parsed.action_rescue, false);
    assert.equal(parsed.prediction, label);
    assert.equal(parsed.consistency_error, false);
  }
});

test('malformed and inconsistent responses are protocol failures', () => {
  assert.equal(parseGateResponse('not-json').parse_error, true);
  assert.equal(parseGateResponse('null').protocol_error, true);
  const inconsistent = parseGateResponse(JSON.stringify({
    center_interaction: 'YES',
    target_signature: 'NO',
    label: 'TARGET_FOREGROUND',
  }));
  assert.equal(inconsistent.consistency_error, true);
  assert.equal(inconsistent.action_rescue, false);
});

test('manifest enforces absolute time and one exact member', () => {
  const row = manifestRow();
  validateManifestRows([row]);
  assert.equal(absoluteCenterTime(row), 12.5);

  const badAlias = clone(row);
  badAlias.center_sec = 12.6;
  assert.throws(
    () => validateManifestRows([badAlias]),
    /center_sec differs/,
  );

  const broadcast = clone(row);
  broadcast.member_anchors.push(clone(row.member_anchors[0]));
  assert.throws(
    () => validateManifestRows([broadcast]),
    /exactly one member anchor/,
  );
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

test('resume output cannot mix Prompt, model, or wire API', () => {
  const manifest = manifestRow();
  const model = 'qwen-test';
  const wireApi = 'chat';
  const prior = {
    candidate_id: manifest.candidate_id,
    routing_schema_version: ROUTING_SCHEMA_VERSION,
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
      [{ ...prior, model: 'another-model' }],
      [manifest],
      model,
      wireApi,
    ),
    /model or wire_api differs/,
  );
});

test('completion summary distinguishes resumed, API, and local work', () => {
  const rows = [
    { candidate_id: 'a' },
    { candidate_id: 'b' },
    { candidate_id: 'c' },
  ];
  const latest = new Map([
    ['a', { status: 'ok', route: 'action' }],
    ['b', { status: 'ok', route: 'ignore' }],
    ['c', { status: 'ok', route: 'ignore' }],
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
    action: 1,
    ignore: 2,
    background: 0,
    error: 0,
  });
  assert.doesNotThrow(() => assertCompleteSummary(summary));
  assert.throws(
    () => assertCompleteSummary({ ...summary, background: 1 }),
    /background/,
  );
});
