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


function frame(frameId, frameIndex) {
  return { frame_id: frameId, frame_index: frameIndex, timestamp_sec: frameIndex / 30 };
}


function coarseRow() {
  return {
    query_schema_version: 'vlm-boundary-query-v4',
    stage: 'coarse',
    task_mode: 'validity-and-boundary',
    query_id: 'p:coarse',
    proposal_id: 'p',
    predicted_class_name: 'PUT',
    predicted_class_id: 1,
    coarse_segment_seconds: [1, 3],
    fps: 30,
    score: 0.999,
    fpn_level: 2,
    posthoc_gt: { kind: 'NO_ACTION' },
    frames: [
      frame('F1', 0), frame('F2', 2), frame('F3', 30), frame('F4', 32),
      frame('F5', 60), frame('F6', 62), frame('F7', 90), frame('F8', 92),
    ],
    context_frame_ids: ['F1', 'F8'],
    instance_anchor_frame_ids: ['F3', 'F6'],
    start_candidates: [
      { option: 'B', before_frame_id: 'F1', after_frame_id: 'F2', timestamp_sec: 1 / 30 },
      { option: 'A', before_frame_id: 'F3', after_frame_id: 'F4', timestamp_sec: 31 / 30 },
    ],
    end_candidates: [
      { option: 'D', before_frame_id: 'F5', after_frame_id: 'F6', timestamp_sec: 61 / 30 },
      { option: 'C', before_frame_id: 'F7', after_frame_id: 'F8', timestamp_sec: 91 / 30 },
    ],
  };
}


function fineRow() {
  return {
    ...coarseRow(),
    stage: 'fine',
    query_id: 'p:fine:start',
    boundary_side: 'start',
    instance_description: 'right hand places the white bowl on the counter',
    sada_boundary_seconds: 1,
    rough_boundary_seconds: 31 / 30,
    frames: [frame('F1', 27), frame('F2', 29), frame('F3', 30), frame('F4', 32)],
    instance_anchor_frame_ids: ['F3'],
    candidates: [
      { option: 'B', before_frame_id: 'F1', after_frame_id: 'F2', timestamp_sec: 28 / 30 },
      { option: 'A', before_frame_id: 'F3', after_frame_id: 'F4', timestamp_sec: 31 / 30 },
    ],
  };
}


function coarseDecision() {
  return {
    action_status: 'VALID_ACTION',
    instance_description: 'right hand places the white bowl on the counter',
    start_decision: 'KEEP_SADA',
    selected_start_option: null,
    end_decision: 'SELECT_CANDIDATE',
    selected_end_option: 'C',
    reason: 'SADA start is already best; option C clearly brackets the same bowl',
  };
}


test('prompt whitelist excludes metadata and defines conservative residual protocol', () => {
  const row = coarseRow();
  const fields = promptFields(row);
  assert.equal('score' in fields, false);
  assert.equal('fpn_level' in fields, false);
  assert.equal('posthoc_gt' in fields, false);
  const prompt = promptFor(row);
  assert.equal(PROMPT_VERSION, 'vlm-boundary-sada-residual-instance-v5');
  assert.match(prompt, /complete narrated action instance/);
  assert.match(prompt, /instance_description/);
  assert.match(prompt, /KEEP_SADA is the safe default/);
  assert.match(prompt, /context only/);
  assert.match(prompt, /complete narrated place\/leave instance/);
  assert.equal(prompt.includes('0.999'), false);
  assert.equal(
    requestFingerprint(row, 'model', 'responses'),
    requestFingerprint({ ...row, score: 0.1, fpn_level: 5, posthoc_gt: {} }, 'model', 'responses'),
  );
});


test('boundary-only contract freezes SADA action and omits action status', () => {
  const row = { ...coarseRow(), task_mode: 'boundary-only' };
  const decision = { ...coarseDecision() };
  delete decision.action_status;
  assert.equal(parseDecision(JSON.stringify(decision), row).start_decision, 'KEEP_SADA');
  assert.throws(() => parseDecision(JSON.stringify({
    ...decision, action_status: 'VALID_ACTION',
  }), row), /keys violate/);
  const prompt = promptFor(row);
  assert.match(prompt, /frozen high-confidence SADA inputs/);
  assert.equal(prompt.includes('WRONG_ACTION'), false);
  assert.equal(prompt.includes('NO_ACTION'), false);
});


test('coarse contract validates side-specific options and legal final order', () => {
  const row = coarseRow();
  assert.equal(parseDecision(JSON.stringify(coarseDecision()), row).selected_end_option, 'C');
  assert.throws(() => parseDecision(JSON.stringify({
    ...coarseDecision(), selected_end_option: 'A',
  }), row), /invalid/);
  assert.throws(() => parseDecision(JSON.stringify({
    ...coarseDecision(), action_status: 'NO_ACTION',
  }), row), /cannot select/);
  assert.throws(() => parseDecision(JSON.stringify({
    ...coarseDecision(), end_decision: 'SELECT_CANDIDATE', selected_end_option: 'D',
  }), { ...row, coarse_segment_seconds: [2.5, 3] }), /must precede/);
});


test('fine contract supports SADA, coarse, candidate, and abstention', () => {
  const row = fineRow();
  assert.equal(parseDecision(JSON.stringify({
    decision: 'SELECT_CANDIDATE', selected_option: 'A', reason: 'edge',
  }), row).selected_option, 'A');
  for (const decision of [
    'KEEP_SADA', 'KEEP_COARSE', 'NO_VISIBLE_BOUNDARY', 'UNCERTAIN',
    'WRONG_ACTION', 'NO_ACTION',
  ]) {
    assert.equal(parseDecision(JSON.stringify({
      decision, selected_option: null, reason: 'valid alternative',
    }), row).decision, decision);
  }
  assert.match(promptFor(row), /Original SADA start boundary: 1\.000s/);
  assert.match(promptFor(row), /Frozen visual instance/);
  assert.deepEqual(resolvedBoundaryOutput(row, {
    decision: 'KEEP_SADA', selected_option: null, reason: 'keep',
  }), { seconds: 1, valid: true, source: 'SADA_KEEP' });
  assert.deepEqual(resolvedBoundaryOutput(row, {
    decision: 'KEEP_COARSE', selected_option: null, reason: 'keep',
  }), { seconds: 31 / 30, valid: true, source: 'COARSE_KEEP' });
});


test('boundary-only fine contract cannot reject or reclassify SADA action', () => {
  const row = { ...fineRow(), task_mode: 'boundary-only' };
  assert.equal(parseDecision(JSON.stringify({
    decision: 'KEEP_SADA', selected_option: null, reason: 'best boundary',
  }), row).decision, 'KEEP_SADA');
  for (const decision of ['WRONG_ACTION', 'NO_ACTION']) {
    assert.throws(() => parseDecision(JSON.stringify({
      decision, selected_option: null, reason: 'forbidden',
    }), row), /invalid/);
  }
});


test('manifest validation rejects a transition wider than three video frames', () => {
  validateManifestRows([coarseRow(), fineRow()]);
  const bad = fineRow();
  bad.frames[1] = frame('F2', 35);
  bad.frames.sort((a, b) => a.frame_index - b.frame_index);
  assert.throws(() => validateManifestRows([bad]), /timeline|transition candidate/);
});


test('coarse resolved output explicitly preserves SADA per side', () => {
  const row = coarseRow();
  assert.deepEqual(resolvedBoundaryOutput(row, coarseDecision()), {
    start: { seconds: 1, valid: true, source: 'SADA_KEEP' },
    end: { seconds: 91 / 30, valid: true, source: 'VLM' },
  });
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
