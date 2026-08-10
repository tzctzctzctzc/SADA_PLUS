import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';


const QUERY_SCHEMA_VERSION = 'vlm-boundary-query-v1';
const PREDICTION_SCHEMA_VERSION = 'vlm-boundary-prediction-v1';
const PROMPT_VERSION = 'vlm-boundary-coarse-to-fine-v1';
const STATUS_VALUES = new Set([
  'VALID_ACTION', 'NO_ACTION', 'WRONG_ACTION', 'UNCERTAIN',
]);
const ACTION_GUIDANCE = Object.freeze({
  TAKE: 'gaining possession or stable control; merely holding, carrying, removing, or lifting is not enough',
  PUT: 'placing, positioning, returning, storing, or setting down an object; generic movement is not enough',
  WASH: 'cleaning, rinsing, wiping, lathering, sponging, or soaping an object',
  OPEN: 'changing a closure from closed to open or more accessible',
  CLOSE: 'changing a closure from open to closed or secured',
  INSERT: 'moving an object from outside to inside a container or opening',
  'TURN-ON': 'operating a control so a device, light, flame, or flow becomes active',
  CUT: 'slicing, dividing, or severing material with a cutting tool',
  'TURN-OFF': 'operating a control so an active device, light, flame, or flow stops',
  POUR: 'controlling a source container so contents visibly transfer to a destination',
});


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
    predicted_class_name: row.predicted_class_name,
    coarse_segment_seconds: row.coarse_segment_seconds,
    frames: row.frames.map(frame => ({
      frame_id: frame.frame_id,
      timestamp_sec: frame.timestamp_sec,
      ...(row.stage === 'fine' ? { option: frame.option } : {}),
    })),
  };
  if (row.stage === 'fine') fields.boundary_side = row.boundary_side;
  return fields;
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
  return `You are checking one temporal action proposal in an egocentric video.

Proposed action: ${row.predicted_class_name}
Action boundary meaning: ${actionGuidance(row)}
Proposed temporal region: ${left.toFixed(3)}s to ${right.toFixed(3)}s

The images below are chronologically ordered:
${timeline}

First decide whether the proposed action is actually visible in this sequence.
- VALID_ACTION: the proposed action occurs.
- WRONG_ACTION: a meaningful action occurs, but it is not the proposed action.
- NO_ACTION: no meaningful action occurs in the proposed region.
- UNCERTAIN: visual evidence is genuinely insufficient.

The surrounding context is only evidence. Do not confirm an action that occurs
entirely outside the proposed temporal region.

If and only if the status is VALID_ACTION, select the frame at or immediately
before the visible action start and the frame at or immediately after its end.
The start frame must precede the end frame. Return JSON only.`;
}


function finePrompt(row) {
  const side = row.boundary_side === 'start' ? 'start' : 'end';
  const candidates = row.frames.map(frame => (
    `${frame.frame_id}: option ${frame.option}, ${Number(frame.timestamp_sec).toFixed(3)}s`
  )).join('\n');
  return `You are refining the ${side} boundary of one visible action.

Action: ${row.predicted_class_name}
Action boundary meaning: ${actionGuidance(row)}
The images are chronologically ordered, while option letters are deliberately
shuffled. Candidate frames:
${candidates}

Choose the option whose image best marks the action ${side} boundary. For a
start boundary choose the first frame belonging to the action; for an end
boundary choose the last frame belonging to the action. If the action identity
is contradicted, absent, or genuinely ambiguous, use WRONG_ACTION, NO_ACTION,
or UNCERTAIN. Return JSON only.`;
}


function promptFor(row) {
  return row.stage === 'coarse' ? coarsePrompt(row) : finePrompt(row);
}


function responseSchema(row) {
  const common = {
    status: { type: 'string', enum: [...STATUS_VALUES] },
    reason: { type: 'string', maxLength: 240 },
  };
  if (row.stage === 'coarse') {
    const frameIds = row.frames.map(frame => frame.frame_id);
    return {
      type: 'object',
      properties: {
        ...common,
        rough_start_frame: {
          anyOf: [{ type: 'string', enum: frameIds }, { type: 'null' }],
        },
        rough_end_frame: {
          anyOf: [{ type: 'string', enum: frameIds }, { type: 'null' }],
        },
      },
      required: ['status', 'rough_start_frame', 'rough_end_frame', 'reason'],
      additionalProperties: false,
    };
  }
  const options = row.frames.map(frame => frame.option);
  return {
    type: 'object',
    properties: {
      ...common,
      selected_option: {
        anyOf: [{ type: 'string', enum: options }, { type: 'null' }],
      },
    },
    required: ['status', 'selected_option', 'reason'],
    additionalProperties: false,
  };
}


function parseDecision(text, row) {
  let value;
  try { value = JSON.parse(String(text).trim()); } catch {
    throw new Error('response is not valid JSON');
  }
  const required = row.stage === 'coarse'
    ? ['status', 'rough_start_frame', 'rough_end_frame', 'reason']
    : ['status', 'selected_option', 'reason'];
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).sort().join('|') !== required.sort().join('|')) {
    throw new Error('response keys violate the contract');
  }
  if (!STATUS_VALUES.has(value.status) || typeof value.reason !== 'string'
      || value.reason.length > 240) {
    throw new Error('response status or reason is invalid');
  }
  if (row.stage === 'coarse') {
    const ids = row.frames.map(frame => frame.frame_id);
    if (value.status === 'VALID_ACTION') {
      const start = ids.indexOf(value.rough_start_frame);
      const end = ids.indexOf(value.rough_end_frame);
      if (start < 0 || end < 0 || start >= end) {
        throw new Error('coarse boundary frames are invalid');
      }
    } else if (value.rough_start_frame !== null || value.rough_end_frame !== null) {
      throw new Error('non-valid coarse response must use null boundaries');
    }
  } else {
    const options = new Set(row.frames.map(frame => frame.option));
    if (value.status === 'VALID_ACTION' && !options.has(value.selected_option)) {
      throw new Error('selected fine option is invalid');
    }
    if (value.status !== 'VALID_ACTION' && value.selected_option !== null) {
      throw new Error('non-valid fine response must use a null option');
    }
  }
  return value;
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
    const options = new Set();
    for (const frame of row.frames) {
      const timestamp = Number(frame.timestamp_sec);
      if (!frame.frame_id || frameIds.has(frame.frame_id)
          || !Number.isFinite(timestamp)
          || (prior !== -Infinity && timestamp <= prior)) {
        throw new Error(`invalid frame timeline for ${row.query_id}`);
      }
      frameIds.add(frame.frame_id);
      prior = timestamp;
      if (row.stage === 'fine') {
        if (!frame.option || options.has(frame.option)) {
          throw new Error(`invalid fine options for ${row.query_id}`);
        }
        options.add(frame.option);
      }
    }
    if (row.stage === 'fine' && !['start', 'end'].includes(row.boundary_side)) {
      throw new Error(`invalid boundary side for ${row.query_id}`);
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
  responseSchema,
  validateManifestRows,
  validatePriorRows,
};
