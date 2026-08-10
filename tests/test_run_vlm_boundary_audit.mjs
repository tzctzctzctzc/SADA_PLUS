import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PREDICTION_SCHEMA_VERSION,
  parseDecision,
  promptFields,
  promptFor,
  requestFingerprint,
  requestWithRetry,
  validateManifestRows,
  validatePriorRows,
} from '../tools/run_vlm_boundary_audit.mjs';


function coarseRow() {
  return {
    query_schema_version: 'vlm-boundary-query-v1',
    stage: 'coarse',
    query_id: 'p:coarse',
    proposal_id: 'p',
    predicted_class_name: 'PUT',
    predicted_class_id: 1,
    coarse_segment_seconds: [2, 5],
    score: 0.999,
    fpn_level: 2,
    posthoc_gt: { kind: 'NO_ACTION' },
    frames: [
      { frame_id: 'F1', timestamp_sec: 1 },
      { frame_id: 'F2', timestamp_sec: 3 },
      { frame_id: 'F3', timestamp_sec: 6 },
    ],
  };
}


function fineRow() {
  return {
    ...coarseRow(),
    stage: 'fine',
    query_id: 'p:fine:start',
    boundary_side: 'start',
    frames: [
      { frame_id: 'F1', timestamp_sec: 1.8, option: 'C' },
      { frame_id: 'F2', timestamp_sec: 2.0, option: 'A' },
      { frame_id: 'F3', timestamp_sec: 2.2, option: 'B' },
    ],
  };
}


test('prompt whitelist excludes model metadata and post-hoc labels', () => {
  const row = coarseRow();
  const fields = promptFields(row);
  assert.equal('score' in fields, false);
  assert.equal('fpn_level' in fields, false);
  assert.equal('posthoc_gt' in fields, false);
  const prompt = promptFor(row);
  assert.equal(prompt.includes('0.999'), false);
  assert.equal(prompt.includes('NO_ACTION' + JSON.stringify(row.posthoc_gt)), false);
  assert.equal(
    requestFingerprint(row, 'model', 'responses'),
    requestFingerprint({ ...row, score: 0.1, fpn_level: 5, posthoc_gt: {} }, 'model', 'responses'),
  );
});


test('strict coarse and fine response contracts reject invalid geometry', () => {
  const coarse = coarseRow();
  const valid = parseDecision(JSON.stringify({
    status: 'VALID_ACTION', rough_start_frame: 'F1', rough_end_frame: 'F3', reason: 'visible',
  }), coarse);
  assert.equal(valid.status, 'VALID_ACTION');
  assert.throws(() => parseDecision(JSON.stringify({
    status: 'VALID_ACTION', rough_start_frame: 'F3', rough_end_frame: 'F1', reason: 'bad',
  }), coarse), /invalid/);
  assert.equal(parseDecision(JSON.stringify({
    status: 'VALID_ACTION', selected_option: 'A', reason: 'edge',
  }), fineRow()).selected_option, 'A');
  assert.throws(() => parseDecision(JSON.stringify({
    status: 'VALID_ACTION', selected_option: 'Z', reason: 'bad',
  }), fineRow()), /invalid/);
});


test('resume rejects a mismatched fingerprint', () => {
  const row = coarseRow();
  validateManifestRows([row]);
  const prior = {
    query_id: row.query_id,
    prediction_schema_version: PREDICTION_SCHEMA_VERSION,
    request_fingerprint: requestFingerprint(row, 'm1', 'chat'),
  };
  assert.doesNotThrow(() => validatePriorRows([prior], [row], 'm1', 'chat'));
  assert.throws(() => validatePriorRows([prior], [row], 'm2', 'chat'), /fingerprint/);
  assert.throws(
    () => validatePriorRows([prior], [row], 'm1', 'chat', 'high'),
    /fingerprint/,
  );
});


test('retry is limited to transient failures', async () => {
  let calls = 0;
  const transient = async () => {
    calls += 1;
    if (calls < 3) {
      const error = new Error('busy');
      error.status = 429;
      throw error;
    }
    return { text: '{}', usage: null };
  };
  const result = await requestWithRetry({}, 3, 0, transient);
  assert.equal(result.attempts, 3);
  calls = 0;
  await assert.rejects(requestWithRetry({}, 5, 0, async () => {
    calls += 1;
    const error = new Error('unauthorized');
    error.status = 401;
    throw error;
  }), /unauthorized/);
  assert.equal(calls, 1);
});
