import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';


const QUERY_SCHEMA_VERSION = 'vlm-boundary-query-v5';
const PREDICTION_SCHEMA_VERSION = 'vlm-boundary-prediction-v5';
const PROMPT_VERSION = 'vlm-boundary-sada-residual-instance-v5';
const ACTION_STATUS_VALUES = new Set([
  'VALID_ACTION', 'NO_ACTION', 'WRONG_ACTION', 'UNCERTAIN',
]);
const SIDE_DECISION_VALUES = new Set([
  'SELECT_CANDIDATE', 'KEEP_SADA', 'NOT_VISIBLE', 'UNCERTAIN',
]);
const FINE_DECISION_VALUES = new Set([
  'SELECT_CANDIDATE', 'KEEP_SADA', 'KEEP_COARSE', 'NO_VISIBLE_BOUNDARY',
  'UNCERTAIN', 'WRONG_ACTION', 'NO_ACTION',
]);
const BOUNDARY_ONLY_FINE_DECISION_VALUES = new Set([
  'SELECT_CANDIDATE', 'KEEP_SADA', 'KEEP_COARSE', 'NO_VISIBLE_BOUNDARY', 'UNCERTAIN',
]);
const ACTION_GUIDANCE = Object.freeze({
  TAKE: 'the complete narrated retrieve/pick-up instance (collect, fetch, get, grab, pick, pull off, retrieve, take, or unload), distinguished from a preceding reach and later unrelated carry by the same hand-object interaction',
  PUT: 'the complete narrated place/leave instance (lay, place, position, put, replace, reposition, rest, return, set down, or store), distinguished from preceding transport and a neighboring placement by the same object and destination',
  WASH: 'the complete narrated cleaning instance (clean, lather, rinse, soap, sponge, wash, or wipe), including its continuous cleaning strokes but not a separate neighboring cleaning action',
  OPEN: 'the complete narrated opening instance (open, uncork, or unzip) for the same closure, not a nearby opening of another container or part',
  CLOSE: 'the complete narrated closing instance (close, shut, screw on, or twist on) for the same closure, not a nearby closing action',
  INSERT: 'the complete narrated insertion/loading instance (drop into, fit, insert, move in, place in, put in, load, or pack) for the same object and destination',
  'TURN-ON': 'the complete narrated activation instance (activate, ignite, light, start, switch on, turn on, or water on) for the same control',
  CUT: 'the complete narrated cutting instance (chop, cut, dice, halve, julienne, or slice) on the same object, separated from neighboring cutting instances',
  'TURN-OFF': 'the complete narrated deactivation instance (shut off, switch off, tap off, turn off, or water off) for the same control',
  POUR: 'the complete narrated pouring/transfer instance (drizzle, pour, shake into, sieve, or tip) for the same source and destination',
});

const FULL_ACTION_EXTENT = `Estimate the visible extent of the same complete narrated action instance,
not only its most salient contact or state-change instant. Do not automatically
include an earlier reach, generic preparation, transport, recovery, or a nearby
same-verb event. Include such motion only when the visual sequence clearly shows
that it is continuous evidence for this exact hand-object-location instance.`;


function sha256(value) {
  return crypto.createHash('sha256').update(String(value)).digest('hex');
}


function readJsonl(file, { allowMissing = false } = {}) {
  if (!fs.existsSync(file)) {
    if (allowMissing) return [];
    throw new Error(`JSONL file not found: ${file}`);
  }
  return fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean)
    .map((line, index) => {
      try { return JSON.parse(line); } catch (error) {
        throw new Error(`invalid JSONL at ${file}:${index + 1}: ${error.message}`);
      }
    });
}


function promptFields(row) {
  const fields = {
    stage: row.stage,
    task_mode: row.task_mode,
    predicted_class_name: row.predicted_class_name,
    coarse_segment_seconds: row.coarse_segment_seconds,
    frames: row.frames.map(frame => ({
      frame_id: frame.frame_id,
      frame_index: frame.frame_index,
      timestamp_sec: frame.timestamp_sec,
    })),
  };
  fields.instance_anchor_frame_ids = row.instance_anchor_frame_ids;
  if (row.stage === 'fine') {
    fields.boundary_side = row.boundary_side;
    fields.instance_description = row.instance_description;
    fields.sada_boundary_seconds = row.sada_boundary_seconds;
    fields.rough_boundary_seconds = row.rough_boundary_seconds;
    fields.candidates = row.candidates.map(candidate => ({
      option: candidate.option,
      before_frame_id: candidate.before_frame_id,
      after_frame_id: candidate.after_frame_id,
      timestamp_sec: candidate.timestamp_sec,
    }));
  } else {
    fields.context_frame_ids = row.context_frame_ids;
    for (const side of ['start', 'end']) {
      fields[`${side}_candidates`] = row[`${side}_candidates`].map(candidate => ({
        option: candidate.option,
        before_frame_id: candidate.before_frame_id,
        after_frame_id: candidate.after_frame_id,
        timestamp_sec: candidate.timestamp_sec,
      }));
    }
  }
  return fields;
}


function isBoundaryOnly(row) {
  return row.task_mode === 'boundary-only';
}


function normalizedClassName(value) {
  return String(value).trim().toUpperCase().replace(/_/g, '-');
}


function actionGuidance(row) {
  return ACTION_GUIDANCE[normalizedClassName(row.predicted_class_name)]
    || 'the named action must be visually supported by the temporal sequence';
}


function coarsePrompt(row) {
  const [left, right] = row.coarse_segment_seconds.map(Number);
  const timeline = row.frames.map(frame => (
    `${frame.frame_id}: ${Number(frame.timestamp_sec).toFixed(3)}s (video frame ${frame.frame_index})`
  )).join('\n');
  const candidates = side => row[`${side}_candidates`].map(candidate => (
    `option ${candidate.option}: ${candidate.before_frame_id} -> ${candidate.after_frame_id}, midpoint ${Number(candidate.timestamp_sec).toFixed(3)}s`
  )).join('\n');
  const validity = isBoundaryOnly(row)
    ? `The occurrence and class are frozen high-confidence SADA inputs. Do not
verify, reject, or reclassify the action.`
    : `First return action_status: VALID_ACTION, WRONG_ACTION, NO_ACTION, or
UNCERTAIN. Only VALID_ACTION may select a replacement boundary.`;
  return `You are conservatively auditing the temporal boundaries of one SADA
proposal in an egocentric video.

Frozen SADA action: ${row.predicted_class_name}
Action boundary meaning: ${actionGuidance(row)}
Original SADA interval: ${left.toFixed(3)}s to ${right.toFixed(3)}s

${validity}

${FULL_ACTION_EXTENT}

The images are chronologically ordered:
${timeline}

Broad context frames (context only; only listed transition pairs are choices):
${row.context_frame_ids.join(', ')}
Instance-anchor frames inside the original SADA proposal:
${row.instance_anchor_frame_ids.join(', ')}

First write instance_description: a short visual identity for this exact action
instance using the anchor frames, preferably hand/object/location. Track only
that same instance at both boundaries. Do not switch to a nearby occurrence of
the same verb. If the object is unclear, state that it is unclear.

Selectable START transition pairs:
${candidates('start')}

Selectable END transition pairs:
${candidates('end')}

Each option is a short pair of real video frames. Context frames between options
are not additional choices. Judge START and END independently against the exact
original SADA boundary:
- SELECT_CANDIDATE only when one option visibly brackets the same instance and is
  clearly better than SADA.
- KEEP_SADA when SADA is at least as good, evidence is tied, or improvement is
  not clear. Changing a boundary is never required; KEEP_SADA is the safe default.
- NOT_VISIBLE when the transition is outside all shown evidence.
- UNCERTAIN when competing interpretations remain.

START means before-frame is not in the action and after-frame is in it. END means
before-frame is in the action and after-frame is not. A selected_option is required
only for SELECT_CANDIDATE and must otherwise be null. The final start must precede
the final end. Return JSON only.`;
}


function finePrompt(row) {
  const side = row.boundary_side === 'start' ? 'start' : 'end';
  const timeline = row.frames.map(frame => (
    `${frame.frame_id}: ${Number(frame.timestamp_sec).toFixed(3)}s (video frame ${frame.frame_index})`
  )).join('\n');
  const candidates = row.candidates.map(candidate => (
    `option ${candidate.option}: ${candidate.before_frame_id} -> ${candidate.after_frame_id}, midpoint ${Number(candidate.timestamp_sec).toFixed(3)}s`
  )).join('\n');
  const fixedInput = isBoundaryOnly(row)
    ? `The high-confidence SADA action occurrence and class are frozen inputs.
Do not verify, reject, or reclassify the action.`
    : '';
  const finalChoices = isBoundaryOnly(row)
    ? `Return NO_VISIBLE_BOUNDARY when this side is outside the shown evidence,
or UNCERTAIN when the boundary evidence is ambiguous.`
    : `You may also return WRONG_ACTION or NO_ACTION when the coarse action
decision is contradicted.`;
  return `You are conservatively refining the ${side} boundary of one visible action.

Action: ${row.predicted_class_name}
Action boundary meaning: ${actionGuidance(row)}
Frozen visual instance: ${row.instance_description}
Original SADA ${side} boundary: ${Number(row.sada_boundary_seconds).toFixed(3)}s
Current coarse ${side} boundary: ${Number(row.rough_boundary_seconds).toFixed(3)}s

${fixedInput}

${FULL_ACTION_EXTENT}

Chronological frames:
${timeline}

Instance-anchor frames (identity labels; only listed transition pairs are choices):
${row.instance_anchor_frame_ids.join(', ')}

Option letters are deliberately shuffled. Candidate adjacent-frame transitions:
${candidates}

Use the same transition definition as the coarse stage:
${side === 'start' ? 'START: before-frame is not in the action; after-frame is in the action.' : 'END: before-frame is in the action; after-frame is not in the action.'}

Compare every choice to the original SADA boundary and keep tracking the frozen
visual instance. Return SELECT_CANDIDATE only when one option is clearly the best
and clearly improves SADA. Return KEEP_COARSE only when the current coarse boundary
clearly improves SADA and no fine option clearly improves the coarse boundary.
Return KEEP_SADA whenever SADA remains best or the comparison is tied/ambiguous.
Do not move inward to a salient core within the action. ${finalChoices}
Return JSON only.`;
}


function promptFor(row) {
  return row.stage === 'coarse' ? coarsePrompt(row) : finePrompt(row);
}


function responseSchema(row) {
  const reason = { reason: { type: 'string', maxLength: 240 } };
  if (row.stage === 'coarse') {
    const nullableOption = side => ({
      anyOf: [
        { type: 'string', enum: row[`${side}_candidates`].map(item => item.option) },
        { type: 'null' },
      ],
    });
    const boundaryProperties = {
      instance_description: { type: 'string', maxLength: 160 },
      start_decision: { type: 'string', enum: [...SIDE_DECISION_VALUES] },
      selected_start_option: nullableOption('start'),
      end_decision: { type: 'string', enum: [...SIDE_DECISION_VALUES] },
      selected_end_option: nullableOption('end'),
      ...reason,
    };
    const boundaryRequired = [
      'instance_description', 'start_decision', 'selected_start_option',
      'end_decision', 'selected_end_option', 'reason',
    ];
    return {
      type: 'object',
      properties: isBoundaryOnly(row)
        ? boundaryProperties
        : {
          action_status: { type: 'string', enum: [...ACTION_STATUS_VALUES] },
          ...boundaryProperties,
        },
      required: isBoundaryOnly(row)
        ? boundaryRequired
        : ['action_status', ...boundaryRequired],
      additionalProperties: false,
    };
  }
  const options = row.candidates.map(candidate => candidate.option);
  const fineDecisionValues = isBoundaryOnly(row)
    ? BOUNDARY_ONLY_FINE_DECISION_VALUES : FINE_DECISION_VALUES;
  return {
    type: 'object',
    properties: {
      decision: { type: 'string', enum: [...fineDecisionValues] },
      selected_option: {
        anyOf: [{ type: 'string', enum: options }, { type: 'null' }],
      },
      ...reason,
    },
    required: ['decision', 'selected_option', 'reason'],
    additionalProperties: false,
  };
}


function parseDecision(text, row) {
  let value;
  try { value = JSON.parse(String(text).trim()); } catch {
    throw new Error('response is not valid JSON');
  }
  const required = row.stage === 'coarse'
    ? [
      ...(isBoundaryOnly(row) ? [] : ['action_status']),
      'instance_description', 'start_decision', 'selected_start_option',
      'end_decision', 'selected_end_option', 'reason',
    ]
    : ['decision', 'selected_option', 'reason'];
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).sort().join('|') !== required.sort().join('|')) {
    throw new Error('response keys violate the contract');
  }
  if (typeof value.reason !== 'string') {
    throw new Error('response reason is invalid');
  }
  value.reason = value.reason.slice(0, 240);
  if (row.stage === 'coarse') {
    if (typeof value.instance_description !== 'string'
        || value.instance_description.length < 1
        || value.instance_description.length > 160) {
      throw new Error('instance description is invalid');
    }
    if ((!isBoundaryOnly(row) && !ACTION_STATUS_VALUES.has(value.action_status))
        || !SIDE_DECISION_VALUES.has(value.start_decision)
        || !SIDE_DECISION_VALUES.has(value.end_decision)) {
      throw new Error('coarse decision is invalid');
    }
    const boundaries = {
      start: Number(row.coarse_segment_seconds[0]),
      end: Number(row.coarse_segment_seconds[1]),
    };
    for (const side of ['start', 'end']) {
      const decision = value[`${side}_decision`];
      const option = value[`selected_${side}_option`];
      if (decision === 'SELECT_CANDIDATE') {
        const candidate = row[`${side}_candidates`].find(item => item.option === option);
        if (!candidate) throw new Error(`selected coarse ${side} option is invalid`);
        boundaries[side] = Number(candidate.timestamp_sec);
      } else if (option !== null) {
        throw new Error(`non-selection coarse ${side} response must use a null option`);
      }
    }
    if (!isBoundaryOnly(row) && value.action_status !== 'VALID_ACTION'
        && (value.start_decision === 'SELECT_CANDIDATE'
          || value.end_decision === 'SELECT_CANDIDATE')) {
      throw new Error('non-valid action cannot select a replacement boundary');
    }
    if (boundaries.start >= boundaries.end) {
      throw new Error('coarse start must precede coarse end');
    }
  } else {
    const options = new Set(row.candidates.map(candidate => candidate.option));
    const fineDecisionValues = isBoundaryOnly(row)
      ? BOUNDARY_ONLY_FINE_DECISION_VALUES : FINE_DECISION_VALUES;
    if (!fineDecisionValues.has(value.decision)) {
      throw new Error('fine decision is invalid');
    }
    if (value.decision === 'SELECT_CANDIDATE' && !options.has(value.selected_option)) {
      throw new Error('selected fine option is invalid');
    }
    if (value.decision !== 'SELECT_CANDIDATE' && value.selected_option !== null) {
      throw new Error('non-selection fine response must use a null option');
    }
  }
  return value;
}


function resolvedBoundaryOutput(row, decision) {
  const invalid = side => ({
    seconds: Number(row.coarse_segment_seconds[side === 'start' ? 0 : 1]),
    valid: false,
    source: 'INVALID',
  });
  if (row.stage === 'fine') {
    if (decision.decision === 'SELECT_CANDIDATE') {
      const candidate = row.candidates.find(item => item.option === decision.selected_option);
      return { seconds: Number(candidate.timestamp_sec), valid: true, source: 'VLM' };
    }
    if (decision.decision === 'KEEP_COARSE') {
      return { seconds: Number(row.rough_boundary_seconds), valid: true, source: 'COARSE_KEEP' };
    }
    if (decision.decision === 'KEEP_SADA') {
      return { seconds: Number(row.sada_boundary_seconds), valid: true, source: 'SADA_KEEP' };
    }
    return invalid(row.boundary_side);
  }
  const result = {};
  for (const side of ['start', 'end']) {
    const sideIndex = side === 'start' ? 0 : 1;
    const sideDecision = decision[`${side}_decision`];
    if (sideDecision === 'KEEP_SADA') {
      result[side] = {
        seconds: Number(row.coarse_segment_seconds[sideIndex]),
        valid: true,
        source: 'SADA_KEEP',
      };
      continue;
    }
    if (sideDecision !== 'SELECT_CANDIDATE') {
      result[side] = invalid(side);
      continue;
    }
    const candidate = row[`${side}_candidates`].find(
      item => item.option === decision[`selected_${side}_option`],
    );
    result[side] = {
      seconds: Number(candidate.timestamp_sec),
      valid: true,
      source: 'VLM',
    };
  }
  return result;
}


function validateManifestRows(rows) {
  const seen = new Set();
  for (const row of rows) {
    if (row.query_schema_version !== QUERY_SCHEMA_VERSION) {
      throw new Error(`invalid query schema for ${row.query_id}`);
    }
    if (!['coarse', 'fine'].includes(row.stage) || !row.query_id || seen.has(row.query_id)) {
      throw new Error(`invalid or duplicate query_id: ${row.query_id}`);
    }
    if (!['validity-and-boundary', 'boundary-only'].includes(row.task_mode)) {
      throw new Error(`invalid task mode for ${row.query_id}`);
    }
    seen.add(row.query_id);
    if (!String(row.predicted_class_name || '').trim()) {
      throw new Error(`missing predicted class for ${row.query_id}`);
    }
    if (!Array.isArray(row.coarse_segment_seconds)
        || row.coarse_segment_seconds.length !== 2
        || !(Number(row.coarse_segment_seconds[0]) < Number(row.coarse_segment_seconds[1]))) {
      throw new Error(`invalid coarse segment for ${row.query_id}`);
    }
    if (!Array.isArray(row.frames) || row.frames.length < 2) {
      throw new Error(`insufficient frames for ${row.query_id}`);
    }
    if (!Number.isFinite(Number(row.fps)) || Number(row.fps) <= 0) {
      throw new Error(`invalid fps for ${row.query_id}`);
    }
    let prior = -Infinity;
    let priorFrameIndex = -1;
    const frameIds = new Set();
    for (const frame of row.frames) {
      const timestamp = Number(frame.timestamp_sec);
      const frameIndex = Number(frame.frame_index);
      if (!frame.frame_id || frameIds.has(frame.frame_id)
          || !Number.isFinite(timestamp)
          || !Number.isInteger(frameIndex)
          || frameIndex <= priorFrameIndex
          || (prior !== -Infinity && timestamp <= prior)
          || Math.abs(timestamp - frameIndex / Number(row.fps)) > 1e-6) {
        throw new Error(`invalid frame timeline for ${row.query_id}`);
      }
      frameIds.add(frame.frame_id);
      prior = timestamp;
      priorFrameIndex = frameIndex;
    }
    const validateCandidates = (candidates, label) => {
      if (!Array.isArray(candidates) || candidates.length < 1) {
        throw new Error(`missing ${label} candidates for ${row.query_id}`);
      }
      const options = new Set();
      for (const candidate of candidates) {
        const before = row.frames.find(frame => frame.frame_id === candidate.before_frame_id);
        const after = row.frames.find(frame => frame.frame_id === candidate.after_frame_id);
        const gap = before && after ? after.frame_index - before.frame_index : NaN;
        const midpoint = before && after
          ? 0.5 * (Number(before.timestamp_sec) + Number(after.timestamp_sec)) : NaN;
        if (!candidate.option || options.has(candidate.option) || !before || !after
            || gap < 1 || gap > 3
            || !Number.isFinite(Number(candidate.timestamp_sec))
            || Math.abs(Number(candidate.timestamp_sec) - midpoint) > 1e-6) {
          throw new Error(`invalid ${label} transition candidate for ${row.query_id}`);
        }
        options.add(candidate.option);
      }
    };
    if (!Array.isArray(row.instance_anchor_frame_ids)
        || row.instance_anchor_frame_ids.length < 1
        || row.instance_anchor_frame_ids.some(id => !frameIds.has(id))) {
      throw new Error(`invalid instance anchors for ${row.query_id}`);
    }
    if (row.stage === 'fine') {
      if (!['start', 'end'].includes(row.boundary_side)
          || typeof row.instance_description !== 'string'
          || !Number.isFinite(Number(row.sada_boundary_seconds))) {
        throw new Error(`invalid boundary side or candidates for ${row.query_id}`);
      }
      validateCandidates(row.candidates, 'fine');
    } else {
      if (!Array.isArray(row.context_frame_ids)
          || row.context_frame_ids.length < 2
          || row.context_frame_ids.some(id => !frameIds.has(id))) {
        throw new Error(`invalid context frames for ${row.query_id}`);
      }
      validateCandidates(row.start_candidates, 'coarse start');
      validateCandidates(row.end_candidates, 'coarse end');
    }
  }
}


function requestFingerprint(row, model, wireApi, reasoningEffort = null) {
  return sha256(JSON.stringify({
    protocol: PROMPT_VERSION,
    query_id: row.query_id,
    prompt_fields: promptFields(row),
    prompt_sha256: sha256(promptFor(row)),
    model,
    wire_api: wireApi,
    reasoning_effort: reasoningEffort,
  }));
}


function validatePriorRows(
  priorRows, manifestRows, model, wireApi, reasoningEffort = null,
) {
  const manifest = new Map(manifestRows.map(row => [row.query_id, row]));
  for (const prior of priorRows) {
    const row = manifest.get(prior.query_id);
    if (!row) throw new Error(`prior output has unknown query ${prior.query_id}`);
    if (prior.prediction_schema_version !== PREDICTION_SCHEMA_VERSION) {
      throw new Error(`prediction schema mismatch for ${prior.query_id}`);
    }
    if (prior.request_fingerprint !== requestFingerprint(
      row, model, wireApi, reasoningEffort,
    )) {
      throw new Error(`request fingerprint mismatch for ${prior.query_id}`);
    }
  }
}


function findVideo(root, videoId) {
  for (const extension of ['.MP4', '.mp4']) {
    const candidate = path.join(root, videoId + extension);
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error(`video not found: ${videoId}`);
}


function prepareFrames(video, row, cacheRoot) {
  const frameIdentity = JSON.stringify({
    video: path.resolve(video),
    query_id: row.query_id,
    frames: row.frames.map(frame => [frame.frame_id, Number(frame.timestamp_sec)]),
  });
  const outputDir = path.join(cacheRoot, sha256(frameIdentity).slice(0, 20));
  fs.mkdirSync(outputDir, { recursive: true });
  return row.frames.map((frame, index) => {
    const file = path.join(outputDir, `frame_${index + 1}.jpg`);
    if (!fs.existsSync(file)) {
      const result = spawnSync('ffmpeg', [
        '-hide_banner', '-loglevel', 'error', '-ss', Number(frame.timestamp_sec).toFixed(3),
        '-i', video, '-frames:v', '1', '-vf', 'scale=640:-2', '-q:v', '3', '-y', file,
      ], { encoding: 'utf8' });
      if (result.status !== 0 || !fs.existsSync(file)) {
        throw new Error(`ffmpeg failed: ${String(result.stderr || '').trim()}`);
      }
    }
    return { ...frame, file };
  });
}


function multimodalContent(row, frames, wireApi) {
  const textType = wireApi === 'responses' ? 'input_text' : 'text';
  const imageType = wireApi === 'responses' ? 'input_image' : 'image_url';
  const content = [{ type: textType, text: promptFor(row) }];
  for (const frame of frames) {
    content.push({ type: textType, text: `${frame.frame_id} at ${Number(frame.timestamp_sec).toFixed(3)}s` });
    const url = `data:image/jpeg;base64,${fs.readFileSync(frame.file).toString('base64')}`;
    content.push(wireApi === 'responses'
      ? { type: imageType, image_url: url }
      : { type: imageType, image_url: { url } });
  }
  return content;
}


function responseText(response, wireApi) {
  if (wireApi === 'responses') {
    return (response.output || []).flatMap(item => item.content || [])
      .filter(item => item.type === 'output_text').map(item => item.text).join('').trim();
  }
  return String(response.choices?.[0]?.message?.content || '').trim();
}


function postJson(url, headers, payload, timeoutMs) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const transport = parsed.protocol === 'https:' ? https : http;
    const request = transport.request(parsed, {
      method: 'POST', headers: { ...headers, 'content-type': 'application/json' },
    }, response => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => { body += chunk; });
      response.on('end', () => {
        if (response.statusCode < 200 || response.statusCode >= 300) {
          const error = new Error(`HTTP ${response.statusCode}: ${body.slice(0, 500)}`);
          error.status = response.statusCode;
          reject(error);
          return;
        }
        try { resolve(JSON.parse(body)); } catch (error) { reject(error); }
      });
    });
    request.setTimeout(timeoutMs, () => request.destroy(new Error('request timeout')));
    request.on('error', reject);
    request.end(JSON.stringify(payload));
  });
}


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


async function requestWithRetry(options, maxRetries, retryBaseMs, implementation = requestModel) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      const result = await implementation(options);
      return { ...result, attempts: attempt + 1 };
    } catch (error) {
      const retryable = error?.status === 429
        || Number(error?.status) >= 500
        || /timeout|ECONNRESET|EAI_AGAIN/i.test(String(error?.message || error));
      if (!retryable || attempt >= maxRetries) throw error;
      await sleep(retryBaseMs * (2 ** attempt));
    }
  }
}


async function requestModel({
  baseUrl, apiKey, model, row, frames, wireApi, maxTokens, timeoutMs,
  reasoningEffort,
}) {
  const content = multimodalContent(row, frames, wireApi);
  const schema = responseSchema(row);
  const root = baseUrl.replace(/\/$/, '');
  if (wireApi === 'responses') {
    const payload = {
      model, input: [{ role: 'user', content }], max_output_tokens: maxTokens,
      text: { format: { type: 'json_schema', name: 'vlm_boundary', strict: true, schema } },
    };
    if (reasoningEffort) payload.reasoning = { effort: reasoningEffort };
    const response = await postJson(
      `${root}/responses`, { authorization: `Bearer ${apiKey}` }, payload, timeoutMs,
    );
    return {
      text: responseText(response, wireApi),
      usage: response.usage || null,
      server_metadata: {
        id: response.id || null,
        model: response.model || null,
        system_fingerprint: response.system_fingerprint || null,
        status: response.status || null,
      },
    };
  }
  const payload = {
    model, messages: [{ role: 'user', content }], max_tokens: maxTokens,
    response_format: { type: 'json_schema', json_schema: { name: 'vlm_boundary', strict: true, schema } },
  };
  if (reasoningEffort) payload.reasoning_effort = reasoningEffort;
  const response = await postJson(
    `${root}/chat/completions`, { authorization: `Bearer ${apiKey}` }, payload, timeoutMs,
  );
  return {
    text: responseText(response, wireApi),
    usage: response.usage || null,
    server_metadata: {
      id: response.id || null,
      model: response.model || null,
      system_fingerprint: response.system_fingerprint || null,
      status: null,
    },
  };
}


function argsOf(argv) {
  const allowed = new Set([
    'manifest', 'video-root', 'output', 'cache-dir', 'base-url', 'model',
    'wire-api', 'api-key-env', 'max-output-tokens', 'request-timeout-sec', 'limit',
    'max-retries', 'retry-base-ms',
    'reasoning-effort',
  ]);
  const result = {};
  for (let index = 2; index < argv.length; index += 2) {
    const raw = argv[index];
    const key = String(raw || '').replace(/^--/, '');
    if (!String(raw || '').startsWith('--') || !allowed.has(key) || index + 1 >= argv.length) {
      throw new Error(`invalid argument: ${raw}`);
    }
    result[key] = argv[index + 1];
  }
  return result;
}


async function main() {
  const args = argsOf(process.argv);
  for (const key of ['manifest', 'video-root', 'output', 'cache-dir', 'base-url', 'model', 'wire-api']) {
    if (!args[key]) throw new Error(`missing --${key}`);
  }
  if (!['responses', 'chat'].includes(args['wire-api'])) throw new Error('invalid --wire-api');
  const keyEnv = args['api-key-env'] || 'VLM_API_KEY';
  const apiKey = process.env[keyEnv];
  if (!apiKey) throw new Error(`environment variable ${keyEnv} is empty`);
  const maxTokens = Number(args['max-output-tokens'] || 256);
  const timeoutMs = Number(args['request-timeout-sec'] || 120) * 1000;
  const maxRetries = Number(args['max-retries'] || 5);
  const retryBaseMs = Number(args['retry-base-ms'] || 1000);
  const reasoningEffort = args['reasoning-effort'] || null;
  const limit = Number(args.limit || Infinity);
  if (!Number.isInteger(maxTokens) || maxTokens <= 0
      || !Number.isFinite(timeoutMs) || timeoutMs <= 0
      || !Number.isInteger(maxRetries) || maxRetries < 0
      || !Number.isFinite(retryBaseMs) || retryBaseMs < 0
      || (limit !== Infinity && (!Number.isInteger(limit) || limit <= 0))) {
    throw new Error('invalid numeric request option');
  }
  const rows = readJsonl(args.manifest);
  validateManifestRows(rows);
  const active = rows.slice(0, limit);
  const prior = readJsonl(args.output, { allowMissing: true });
  validatePriorRows(
    prior, rows, args.model, args['wire-api'], reasoningEffort,
  );
  const latest = new Map(prior.map(row => [row.query_id, row]));
  fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });

  for (const row of active) {
    if (latest.get(row.query_id)?.run_status === 'ok') continue;
    const fingerprint = requestFingerprint(
      row, args.model, args['wire-api'], reasoningEffort,
    );
    const output = {
      prediction_schema_version: PREDICTION_SCHEMA_VERSION,
      query_id: row.query_id,
      proposal_id: row.proposal_id,
      stage: row.stage,
      task_mode: row.task_mode,
      boundary_side: row.boundary_side || null,
      run_status: 'error',
      request_fingerprint: fingerprint,
      prompt_version: PROMPT_VERSION,
      prompt_sha256: sha256(promptFor(row)),
      query_sha256: row.query_sha256 || null,
      frame_timestamps_seconds: row.frames.map(frame => Number(frame.timestamp_sec)),
      model: args.model,
      wire_api: args['wire-api'],
      reasoning_effort: reasoningEffort,
    };
    try {
      const video = findVideo(args['video-root'], row.video_id);
      const frames = prepareFrames(video, row, args['cache-dir']);
      const response = await requestWithRetry({
        baseUrl: args['base-url'], apiKey, model: args.model, row, frames,
        wireApi: args['wire-api'], maxTokens, timeoutMs, reasoningEffort,
      }, maxRetries, retryBaseMs);
      output.decision = parseDecision(response.text, row);
      output.boundary_results = resolvedBoundaryOutput(row, output.decision);
      output.response_text = response.text;
      output.usage = response.usage;
      output.server_metadata = response.server_metadata || null;
      output.request_attempts = response.attempts;
      output.run_status = 'ok';
    } catch (error) {
      output.error = String(error?.message || error);
    }
    fs.appendFileSync(args.output, `${JSON.stringify(output)}\n`, 'utf8');
    latest.set(row.query_id, output);
    process.stdout.write(`${row.query_id} ${output.run_status}\n`);
    if (output.run_status !== 'ok') throw new Error(output.error);
  }
}


const isDirectRun = process.argv[1]
  && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isDirectRun) {
  main().catch(error => { console.error(error?.stack || String(error)); process.exitCode = 1; });
}


export {
  PREDICTION_SCHEMA_VERSION,
  PROMPT_VERSION,
  QUERY_SCHEMA_VERSION,
  argsOf,
  coarsePrompt,
  finePrompt,
  parseDecision,
  promptFields,
  promptFor,
  requestFingerprint,
  requestWithRetry,
  resolvedBoundaryOutput,
  responseSchema,
  validateManifestRows,
  validatePriorRows,
};
