import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PREDICTION_SCHEMA_VERSION,
  PROMPT_VERSION,
  parseDecision,
  promptFields,
  promptFor,
  requestFingerprint,
  requestWithRetry,
  resolvedBoundaryOutput,
  validateManifestRows,
  validatePriorRows,
} from '../tools/run_vlm_boundary_audit.mjs';


function coarseRow() {
  return {
    query_schema_version: 'vlm-boundary-query-v3',
    stage: 'coarse',
    task_mode: 'validity-and-boundary',
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
      { frame_id: 'F4', timestamp_sec: 8 },
    ],
  };
}


function fineRow() {
  return {
    ...coarseRow(),
    stage: 'fine',
    query_id: 'p:fine:start',
    boundary_side: 'start',
    rough_boundary_seconds: 2,
    frames: [
      { frame_id: 'F1', timestamp_sec: 1.7 },
      { frame_id: 'F2', timestamp_sec: 1.9 },
      { frame_id: 'F3', timestamp_sec: 2.1 },
      { frame_id: 'F4', timestamp_sec: 2.3 },
    ],
    candidates: [
      { option: 'C', before_frame_id: 'F1', after_frame_id: 'F2', timestamp_sec: 1.8 },
      { option: 'A', before_frame_id: 'F2', after_frame_id: 'F3', timestamp_sec: 2.0 },
      { option: 'B', before_frame_id: 'F3', after_frame_id: 'F4', timestamp_sec: 2.2 },
    ],
  };
}


function oneSideCoarseDecision() {
  return {
    action_status: 'VALID_ACTION',
    start_status: 'VISIBLE',
    rough_start_before_frame: 'F1',
    rough_start_after_frame: 'F2',
    end_status: 'NOT_VISIBLE',
    rough_end_before_frame: null,
    rough_end_after_frame: null,
    reason: 'start only',
  };
}


test('prompt whitelist excludes metadata and defines one transition protocol', () => {
  const row = coarseRow();
  const fields = promptFields(row);
  assert.equal('score' in fields, false);
  assert.equal('fpn_level' in fields, false);
  assert.equal('posthoc_gt' in fields, false);
  const prompt = promptFor(row);
  assert.equal(PROMPT_VERSION, 'vlm-boundary-sada-top1-boundary-only-v4');
  assert.match(prompt, /complete narrated action interval/);
  assert.match(prompt, /only part of its complete interval is\s+shown/);
  assert.match(prompt, /return VALID_ACTION/);
  assert.match(prompt, /full place\/leave event/);
  assert.match(prompt, /not in the action; adjacent after-frame is in the action/);
  assert.match(prompt, /before-frame is in the action; adjacent after-frame is not/);
  assert.equal(prompt.includes('0.999'), false);
  assert.equal(
    requestFingerprint(row, 'model', 'responses'),
    requestFingerprint({ ...row, score: 0.1, fpn_level: 5, posthoc_gt: {} }, 'model', 'responses'),
  );
});


test('boundary-only contract freezes SADA action and accepts only boundary fields', () => {
  const row = { ...coarseRow(), task_mode: 'boundary-only' };
  const decision = { ...oneSideCoarseDecision() };
  delete decision.action_status;
  assert.equal(parseDecision(JSON.stringify(decision), row).start_status, 'VISIBLE');
  assert.throws(() => parseDecision(JSON.stringify({
    ...decision, action_status: 'VALID_ACTION',
  }), row), /keys violate/);
  const prompt = promptFor(row);
  assert.match(prompt, /fixed inputs from high-confidence SADA/);
  assert.match(prompt, /only task is to locate its\s+complete START and END/);
  assert.equal(prompt.includes('WRONG_ACTION'), false);
  assert.equal(prompt.includes('NO_ACTION'), false);
});


test('coarse contract accepts one visible side and enforces adjacent pairs', () => {
  const coarse = coarseRow();
  assert.equal(
    parseDecision(JSON.stringify(oneSideCoarseDecision()), coarse).start_status,
    'VISIBLE',
  );
  assert.throws(() => parseDecision(JSON.stringify({
    ...oneSideCoarseDecision(), rough_start_after_frame: 'F3',
  }), coarse), /invalid/);
  assert.throws(() => parseDecision(JSON.stringify({
    ...oneSideCoarseDecision(), action_status: 'NO_ACTION',
  }), coarse), /cannot have a visible/);
  const normalized = parseDecision(JSON.stringify({
    ...oneSideCoarseDecision(), end_status: 'UNCERTAIN', rough_end_before_frame: 'F3',
  }), coarse);
  assert.equal(normalized.rough_end_before_frame, null);
  assert.equal(normalized.rough_end_after_frame, null);
});


test('fine contract supports candidate, KEEP, abstention, and rejection', () => {
  const row = fineRow();
  assert.equal(parseDecision(JSON.stringify({
    decision: 'SELECT_CANDIDATE', selected_option: 'A', reason: 'edge',
  }), row).selected_option, 'A');
  for (const decision of [
    'KEEP_COARSE', 'NO_VISIBLE_BOUNDARY', 'UNCERTAIN', 'WRONG_ACTION', 'NO_ACTION',
  ]) {
    assert.equal(parseDecision(JSON.stringify({
      decision, selected_option: null, reason: 'valid alternative',
    }), row).decision, decision);
  }
  assert.throws(() => parseDecision(JSON.stringify({
    decision: 'SELECT_CANDIDATE', selected_option: 'Z', reason: 'bad',
  }), row), /invalid/);
  assert.match(promptFor(row), /KEEP_COARSE/);
  assert.match(promptFor(row), /Current coarse start boundary: 2\.000s/);
  assert.match(promptFor(row), /only when one option/);
  assert.match(promptFor(row), /Do not move\s+the boundary inward/);
  assert.deepEqual(resolvedBoundaryOutput(row, {
    decision: 'SELECT_CANDIDATE', selected_option: 'A', reason: 'edge',
  }), { seconds: 2, valid: true, source: 'VLM' });
  assert.deepEqual(resolvedBoundaryOutput(row, {
    decision: 'KEEP_COARSE', selected_option: null, reason: 'keep',
  }), { seconds: 2, valid: true, source: 'KEEP' });
  assert.deepEqual(resolvedBoundaryOutput(row, {
    decision: 'UNCERTAIN', selected_option: null, reason: 'blur',
  }), { seconds: 2, valid: false, source: 'INVALID' });
});


test('boundary-only fine contract cannot reject or reclassify SADA action', () => {
  const row = { ...fineRow(), task_mode: 'boundary-only' };
  assert.equal(parseDecision(JSON.stringify({
    decision: 'KEEP_COARSE', selected_option: null, reason: 'best boundary',
  }), row).decision, 'KEEP_COARSE');
  for (const decision of ['WRONG_ACTION', 'NO_ACTION']) {
    assert.throws(() => parseDecision(JSON.stringify({
      decision, selected_option: null, reason: 'forbidden',
    }), row), /invalid/);
  }
  const prompt = promptFor(row);
  assert.match(prompt, /Do not verify, reject, or reclassify/);
  assert.equal(prompt.includes('WRONG_ACTION'), false);
  assert.equal(prompt.includes('NO_ACTION'), false);
});


test('manifest validation rejects non-adjacent fine candidate', () => {
  validateManifestRows([coarseRow(), fineRow()]);
  const bad = fineRow();
  bad.candidates[0] = { ...bad.candidates[0], after_frame_id: 'F3' };
  assert.throws(() => validateManifestRows([bad]), /transition candidate/);
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
