import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';


const QUERY_SCHEMA_VERSION = 'vlm-boundary-query-v3';
const PREDICTION_SCHEMA_VERSION = 'vlm-boundary-prediction-v3';
const PROMPT_VERSION = 'vlm-boundary-sada-top1-boundary-only-v4';
const ACTION_STATUS_VALUES = new Set([
  'VALID_ACTION', 'NO_ACTION', 'WRONG_ACTION', 'UNCERTAIN',
]);
const SIDE_STATUS_VALUES = new Set(['VISIBLE', 'NOT_VISIBLE', 'UNCERTAIN']);
const FINE_DECISION_VALUES = new Set([
  'SELECT_CANDIDATE', 'KEEP_COARSE', 'NO_VISIBLE_BOUNDARY',
  'UNCERTAIN', 'WRONG_ACTION', 'NO_ACTION',
]);
const BOUNDARY_ONLY_FINE_DECISION_VALUES = new Set([
  'SELECT_CANDIDATE', 'KEEP_COARSE', 'NO_VISIBLE_BOUNDARY', 'UNCERTAIN',
]);
const ACTION_GUIDANCE = Object.freeze({
  TAKE: 'the full retrieve/pick-up event (collect, fetch, get, grab, pick, pull off, retrieve, take, or unload), from the first purposeful retrieval motion through the completed lift/removal into control; exclude later mere holding or carrying',
  PUT: 'the full place/leave event (lay, place, position, put, replace, reposition, rest, return, set down, or store), from the first purposeful destination-directed placement motion through release or completion at the destination; exclude unrelated transport',
  WASH: 'the full cleaning event (clean, lather, rinse, soap, sponge, wash, or wipe), from the first purposeful cleaning motion through the last cleaning stroke or rinse motion',
  OPEN: 'the full opening event (open, uncork, or unzip), from the first purposeful manipulation of the closure through completion of the opening motion',
  CLOSE: 'the full closing event (close, shut, screw on, or twist on), from the first purposeful closing manipulation through completion or securing',
  INSERT: 'the full insertion/loading event (drop into, fit, insert, move in, place in, put in, load, or pack), from the first purposeful motion into the destination through completion inside it',
  'TURN-ON': 'the full activation event (activate, ignite, light, start, switch on, turn on, or water on), from the first purposeful control manipulation through completion of activation',
  CUT: 'the full cutting event (chop, cut, dice, halve, julienne, or slice), from the first purposeful cutting stroke through the last cutting stroke in the event',
  'TURN-OFF': 'the full deactivation event (shut off, switch off, tap off, turn off, or water off), from the first purposeful control manipulation through completion of deactivation',
  POUR: 'the full pouring/transfer event (drizzle, pour, shake into, sieve, or tip), from the first purposeful transfer or tilting motion through the end of the transfer and recovery motion',
});

const FULL_ACTION_EXTENT = `Use an EPIC-KITCHENS-style complete narrated action interval, not a single
state-change instant or only the most salient core. START is the earliest
purposeful motion that belongs to the named verb event. END is after its last
purposeful motion, before later mere holding, unrelated transport, idle, or the
next action. Do not expand the interval to generic preparation that does not yet
belong to the named action.`;


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
      timestamp_sec: frame.timestamp_sec,
    })),
  };
  if (row.stage === 'fine') {
    fields.boundary_side = row.boundary_side;
    fields.rough_boundary_seconds = row.rough_boundary_seconds;
    fields.candidates = row.candidates.map(candidate => ({
      option: candidate.option,
      before_frame_id: candidate.before_frame_id,
      after_frame_id: candidate.after_frame_id,
      timestamp_sec: candidate.timestamp_sec,
    }));
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
    `${frame.frame_id}: ${Number(frame.timestamp_sec).toFixed(3)}s`
  )).join('\n');
  if (isBoundaryOnly(row)) {
    return `You are locating the temporal boundaries of one frozen SADA action
proposal in an egocentric video.

Frozen SADA top-1 action: ${row.predicted_class_name}
Action boundary meaning: ${actionGuidance(row)}
Frozen SADA temporal proposal: ${left.toFixed(3)}s to ${right.toFixed(3)}s

The action occurrence and class label are fixed inputs from high-confidence SADA.
Do not verify, reject, or reclassify the action. Your only task is to locate its
complete START and END boundaries from the chronological evidence.

${FULL_ACTION_EXTENT}

The images below are chronologically ordered:
${timeline}

Judge START and END independently:
- VISIBLE: the transition is bracketed by one adjacent frame pair.
- NOT_VISIBLE: this transition lies outside the shown timeline.
- UNCERTAIN: the shown evidence cannot locate this transition reliably.

Use exactly one transition definition throughout:
- START: before-frame is not in the action; adjacent after-frame is in the action.
- END: before-frame is in the action; adjacent after-frame is not in the action.
The boundary time is the midpoint of that adjacent pair. Either side may be
NOT_VISIBLE or UNCERTAIN. Return JSON only.`;
  }
  return `You are checking one temporal action proposal in an egocentric video.

Proposed action: ${row.predicted_class_name}
Action boundary meaning: ${actionGuidance(row)}
Proposed temporal region: ${left.toFixed(3)}s to ${right.toFixed(3)}s

${FULL_ACTION_EXTENT}

The images below are chronologically ordered:
${timeline}

First decide whether the proposed action is actually visible in this sequence.
- VALID_ACTION: the proposed action occurs.
- WRONG_ACTION: a meaningful action occurs, but it is not the proposed action.
- NO_ACTION: no meaningful action occurs in the proposed region.
- UNCERTAIN: visual evidence is genuinely insufficient.

If the named action is identifiable but only part of its complete interval is
shown, return VALID_ACTION and mark each missing side NOT_VISIBLE or UNCERTAIN.
Do not return NO_ACTION merely because a grasp, release, contact, or state-change
instant is outside the shown frames.

The surrounding context is only evidence. Do not confirm an action that occurs
entirely outside the proposed temporal region.

Then judge START and END independently:
- VISIBLE: the transition is bracketed by one adjacent frame pair.
- NOT_VISIBLE: the action is valid, but this transition lies outside the shown timeline.
- UNCERTAIN: the shown evidence cannot locate this transition reliably.

Use exactly one transition definition throughout:
- START: before-frame is not in the action; adjacent after-frame is in the action.
- END: before-frame is in the action; adjacent after-frame is not in the action.
The boundary time is the midpoint of that adjacent pair. A VALID_ACTION may have
only one visible side. For a non-valid action, no side may be VISIBLE. Return JSON only.`;
}


function finePrompt(row) {
  const side = row.boundary_side === 'start' ? 'start' : 'end';
  const timeline = row.frames.map(frame => (
    `${frame.frame_id}: ${Number(frame.timestamp_sec).toFixed(3)}s`
  )).join('\n');
  const candidates = row.candidates.map(candidate => (
    `option ${candidate.option}: ${candidate.before_frame_id} -> ${candidate.after_frame_id}, midpoint ${Number(candidate.timestamp_sec).toFixed(3)}s`
  )).join('\n');
  const fixedInput = isBoundaryOnly(row)
    ? `The high-confidence SADA action occurrence and class are frozen inputs.
Do not verify, reject, or reclassify the action.`
    : '';
  const finalChoices = isBoundaryOnly(row)
    ? `Return KEEP_COARSE when no candidate clearly improves it. Return
NO_VISIBLE_BOUNDARY when this side is outside the shown timeline, or UNCERTAIN
when the boundary evidence is ambiguous.`
    : `Return KEEP_COARSE when no candidate clearly improves it. Return
NO_VISIBLE_BOUNDARY when the action is visible but this side is outside the
shown timeline; UNCERTAIN when evidence is ambiguous; or WRONG_ACTION /
NO_ACTION when the coarse action decision is contradicted.`;
  return `You are refining the ${side} boundary of one visible action.

Action: ${row.predicted_class_name}
Action boundary meaning: ${actionGuidance(row)}
Current coarse ${side} boundary: ${Number(row.rough_boundary_seconds).toFixed(3)}s

${fixedInput}

${FULL_ACTION_EXTENT}

Chronological frames:
${timeline}

Option letters are deliberately shuffled. Candidate adjacent-frame transitions:
${candidates}

Use the same transition definition as the coarse stage:
${side === 'start' ? 'START: before-frame is not in the action; after-frame is in the action.' : 'END: before-frame is in the action; after-frame is not in the action.'}

Return SELECT_CANDIDATE only when one option visibly brackets the complete-action
transition and is clearly better than the current coarse boundary. Do not move
the boundary inward to a grasp, release, contact, state-change instant, or other
salient core within the action. ${finalChoices}
Return JSON only.`;
}


function promptFor(row) {
  return row.stage === 'coarse' ? coarsePrompt(row) : finePrompt(row);
}


function responseSchema(row) {
  const reason = { reason: { type: 'string', maxLength: 240 } };
  if (row.stage === 'coarse') {
    const frameIds = row.frames.map(frame => frame.frame_id);
    const nullableFrame = {
      anyOf: [{ type: 'string', enum: frameIds }, { type: 'null' }],
    };
    const boundaryProperties = {
      start_status: { type: 'string', enum: [...SIDE_STATUS_VALUES] },
      rough_start_before_frame: nullableFrame,
      rough_start_after_frame: nullableFrame,
      end_status: { type: 'string', enum: [...SIDE_STATUS_VALUES] },
      rough_end_before_frame: nullableFrame,
      rough_end_after_frame: nullableFrame,
      ...reason,
    };
    const boundaryRequired = [
      'start_status', 'rough_start_before_frame', 'rough_start_after_frame',
      'end_status', 'rough_end_before_frame', 'rough_end_after_frame', 'reason',
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
      'start_status', 'rough_start_before_frame',
      'rough_start_after_frame', 'end_status', 'rough_end_before_frame',
      'rough_end_after_frame', 'reason',
    ]
    : ['decision', 'selected_option', 'reason'];
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).sort().join('|') !== required.sort().join('|')) {
    throw new Error('response keys violate the contract');
  }
  if (typeof value.reason !== 'string' || value.reason.length > 240) {
    throw new Error('response reason is invalid');
  }
  if (row.stage === 'coarse') {
    const ids = row.frames.map(frame => frame.frame_id);
    if ((!isBoundaryOnly(row) && !ACTION_STATUS_VALUES.has(value.action_status))
        || !SIDE_STATUS_VALUES.has(value.start_status)
        || !SIDE_STATUS_VALUES.has(value.end_status)) {
      throw new Error('coarse status is invalid');
    }
    const midpoints = {};
    for (const side of ['start', 'end']) {
      const status = value[`${side}_status`];
      const before = value[`rough_${side}_before_frame`];
      const after = value[`rough_${side}_after_frame`];
      if (status === 'VISIBLE') {
        const beforeIndex = ids.indexOf(before);
        const afterIndex = ids.indexOf(after);
        if (beforeIndex < 0 || afterIndex !== beforeIndex + 1) {
          throw new Error(`coarse ${side} transition frames are invalid`);
        }
        midpoints[side] = 0.5 * (
          Number(row.frames[beforeIndex].timestamp_sec)
          + Number(row.frames[afterIndex].timestamp_sec)
        );
      } else if (before !== null || after !== null) {
        throw new Error(`non-visible coarse ${side} must use null frames`);
      }
    }
    if (!isBoundaryOnly(row) && value.action_status !== 'VALID_ACTION'
        && (value.start_status === 'VISIBLE' || value.end_status === 'VISIBLE')) {
      throw new Error('non-valid action cannot have a visible boundary');
    }
    if (midpoints.start !== undefined && midpoints.end !== undefined
        && midpoints.start >= midpoints.end) {
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
      return { seconds: Number(row.rough_boundary_seconds), valid: true, source: 'KEEP' };
    }
    return invalid(row.boundary_side);
  }
  const result = {};
  for (const side of ['start', 'end']) {
    if (decision[`${side}_status`] !== 'VISIBLE') {
      result[side] = invalid(side);
      continue;
    }
    const beforeId = decision[`rough_${side}_before_frame`];
    const afterId = decision[`rough_${side}_after_frame`];
    const before = row.frames.find(frame => frame.frame_id === beforeId);
    const after = row.frames.find(frame => frame.frame_id === afterId);
    result[side] = {
      seconds: 0.5 * (Number(before.timestamp_sec) + Number(after.timestamp_sec)),
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
    let prior = -Infinity;
    const frameIds = new Set();
    for (const frame of row.frames) {
      const timestamp = Number(frame.timestamp_sec);
      if (!frame.frame_id || frameIds.has(frame.frame_id)
          || !Number.isFinite(timestamp)
          || (prior !== -Infinity && timestamp <= prior)) {
        throw new Error(`invalid frame timeline for ${row.query_id}`);
      }
      frameIds.add(frame.frame_id);
      prior = timestamp;
    }
    if (row.stage === 'fine') {
      if (!['start', 'end'].includes(row.boundary_side)
          || !Array.isArray(row.candidates) || row.candidates.length < 1) {
        throw new Error(`invalid boundary side or candidates for ${row.query_id}`);
      }
      const options = new Set();
      for (const candidate of row.candidates) {
        const before = row.frames.findIndex(frame => frame.frame_id === candidate.before_frame_id);
        const after = row.frames.findIndex(frame => frame.frame_id === candidate.after_frame_id);
        const midpoint = before >= 0 && after >= 0
          ? 0.5 * (Number(row.frames[before].timestamp_sec) + Number(row.frames[after].timestamp_sec))
          : NaN;
        if (!candidate.option || options.has(candidate.option) || before < 0 || after !== before + 1
            || !Number.isFinite(Number(candidate.timestamp_sec))
            || Math.abs(Number(candidate.timestamp_sec) - midpoint) > 1e-6) {
          throw new Error(`invalid fine transition candidate for ${row.query_id}`);
        }
        options.add(candidate.option);
      }
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
    return { text: responseText(response, wireApi), usage: response.usage || null };
  }
  const payload = {
    model, messages: [{ role: 'user', content }], max_tokens: maxTokens,
    response_format: { type: 'json_schema', json_schema: { name: 'vlm_boundary', strict: true, schema } },
  };
  if (reasoningEffort) payload.reasoning_effort = reasoningEffort;
  const response = await postJson(
    `${root}/chat/completions`, { authorization: `Bearer ${apiKey}` }, payload, timeoutMs,
  );
  return { text: responseText(response, wireApi), usage: response.usage || null };
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
