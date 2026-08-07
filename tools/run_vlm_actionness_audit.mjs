import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

/**
 * Candidate-blind, exact-anchor foreground gate for SADA Stage 1.
 *
 * This runner intentionally has one task only:
 *   1. Read one exact low-confidence SADA anchor from the manifest.
 *   2. Extract nine dense frames centered on that exact anchor.
 *   3. Hide SADA's top-1 class from the VLM.
 *   4. Ask whether Frame 5 belongs to an event in the ten-action ontology.
 *   5. Combine the semantic result with a source-supervised structure probe.
 *
 * The VLM never changes the class. A rescued anchor keeps the frozen teacher's
 * top-1 class. Valid non-confirmed responses are routed to IGNORE. Malformed or
 * internally inconsistent responses are protocol errors and stop the run.
 */

const FRAME_OFFSETS_SEC = Object.freeze([
  -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8,
]);
const PROMPT_VERSION = 'candidate-blind-epic-semantic-gate-v3';
const ROUTING_SCHEMA_VERSION = 'semantic-structural-exact-anchor-routing-v3';

const ALLOWED_ARGS = new Set([
  'manifest',
  'video-root',
  'output',
  'cache-dir',
  'base-url',
  'model',
  'wire-api',
  'api-key-env',
  'reasoning-effort',
  'max-output-tokens',
  'request-timeout-sec',
  'max-retries',
  'retry-base-ms',
  'limit',
]);

const LABEL_VALUES = new Set([
  'TARGET_ACTION',
  'OTHER_ACTIVITY',
  'TRUE_BACKGROUND',
  'UNCERTAIN',
]);

function argsOf(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 2) {
    const key = String(argv[i] || '').replace(/^--/, '');
    if (
      !String(argv[i] || '').startsWith('--')
      || !key
      || i + 1 >= argv.length
    ) {
      throw new Error(`invalid CLI argument: ${argv[i]}`);
    }
    if (!ALLOWED_ARGS.has(key)) {
      throw new Error(`unsupported CLI argument: --${key}`);
    }
    out[key] = argv[i + 1];
  }
  return out;
}

function readJsonl(file, { allowMissing = false, allowEmpty = false } = {}) {
  if (!fs.existsSync(file)) {
    if (allowMissing) return [];
    throw new Error(`JSONL file not found: ${file}`);
  }

  const rows = fs.readFileSync(file, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(
          `invalid JSONL at ${file}:${index + 1}: ${error.message}`,
        );
      }
    });

  if (!allowEmpty && rows.length === 0) {
    throw new Error(`JSONL file is empty: ${file}`);
  }

  return rows;
}

function sha256(value) {
  return crypto
    .createHash('sha256')
    .update(String(value))
    .digest('hex');
}

function normalizeEnum(value) {
  return String(value ?? '')
    .trim()
    .toUpperCase()
    .replace(/[\s-]+/g, '_');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function absoluteCenterTime(row) {
  const absolute = Number(row.absolute_center_time);
  const alias = Number(row.center_sec);

  if (!Number.isFinite(absolute) || absolute < 0) {
    throw new Error(
      `invalid absolute_center_time for ${row.candidate_id}`,
    );
  }

  if (!Number.isFinite(alias) || Math.abs(alias - absolute) > 1e-6) {
    throw new Error(
      `center_sec differs from absolute_center_time for ${row.candidate_id}`,
    );
  }

  return absolute;
}

function findVideo(root, videoId) {
  for (const ext of ['.MP4', '.mp4']) {
    const candidate = path.join(root, `${videoId}${ext}`);
    if (fs.existsSync(candidate)) return candidate;
  }

  throw new Error(`video not found for video_id=${videoId}`);
}

const VIDEO_DURATIONS = new Map();

function videoDuration(video) {
  if (VIDEO_DURATIONS.has(video)) {
    return VIDEO_DURATIONS.get(video);
  }

  const result = spawnSync(
    'ffprobe',
    [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=nw=1:nk=1',
      video,
    ],
    { encoding: 'utf8' },
  );

  const duration = Number(String(result.stdout || '').trim());

  if (
    result.status !== 0
    || !Number.isFinite(duration)
    || duration <= 0
  ) {
    throw new Error(
      `ffprobe failed for ${video}: `
      + String(result.stderr || '').trim(),
    );
  }

  VIDEO_DURATIONS.set(video, duration);
  return duration;
}

function sameFrameManifest(existing, expected) {
  if (!existing || typeof existing !== 'object') return false;

  return (
    existing.video === expected.video
    && Number(existing.center_sec) === Number(expected.center_sec)
    && JSON.stringify(existing.timestamps_sec)
      === JSON.stringify(expected.timestamps_sec)
  );
}

function prepareFrames(video, centerSec, outputDir) {
  fs.mkdirSync(outputDir, { recursive: true });

  const center = Number(centerSec);

  if (!Number.isFinite(center) || center < 0) {
    throw new Error(`invalid center_sec: ${centerSec}`);
  }

  const duration = videoDuration(video);

  const timestamps = FRAME_OFFSETS_SEC.map(offset => Number(
    Math.max(
      0,
      Math.min(
        center + offset,
        Math.max(0, duration - 0.05),
      ),
    ).toFixed(3),
  ));

  const expectedManifest = {
    video: path.resolve(video),
    center_sec: center,
    offsets_sec: [...FRAME_OFFSETS_SEC],
    timestamps_sec: timestamps,
  };

  const frameManifestPath = path.join(
    outputDir,
    'frames_manifest.json',
  );

  let regenerate = true;

  if (fs.existsSync(frameManifestPath)) {
    try {
      const existing = JSON.parse(
        fs.readFileSync(frameManifestPath, 'utf8'),
      );

      regenerate = !sameFrameManifest(
        existing,
        expectedManifest,
      );
    } catch {
      regenerate = true;
    }
  }

  const frames = timestamps.map((timestamp, index) => ({
    index: index + 1,
    offset_sec: FRAME_OFFSETS_SEC[index],
    timestamp_sec: timestamp,
    file: path.join(outputDir, `frame_${index + 1}.jpg`),
  }));

  if (!regenerate) {
    regenerate = frames.some(
      frame => !fs.existsSync(frame.file),
    );
  }

  if (regenerate) {
    for (const frame of frames) {
      const result = spawnSync(
        'ffmpeg',
        [
          '-hide_banner',
          '-loglevel',
          'error',
          '-ss',
          frame.timestamp_sec.toFixed(3),
          '-i',
          video,
          '-frames:v',
          '1',
          '-vf',
          'scale=640:-2',
          '-q:v',
          '3',
          '-y',
          frame.file,
        ],
        { encoding: 'utf8' },
      );

      if (
        result.status !== 0
        || !fs.existsSync(frame.file)
      ) {
        throw new Error(
          `ffmpeg failed at `
          + `${frame.timestamp_sec.toFixed(3)}s: `
          + String(result.stderr || '').trim(),
        );
      }
    }

    fs.writeFileSync(
      frameManifestPath,
      `${JSON.stringify(expectedManifest, null, 2)}\n`,
      'utf8',
    );
  }

  const uniqueTimestamps = new Set(
    timestamps.map(value => value.toFixed(3)),
  ).size;

  const hasPreContext = timestamps.some(
    value => value < center - 0.05,
  );

  const hasPostContext = timestamps.some(
    value => value > center + 0.05,
  );

  const temporalContextValid = (
    uniqueTimestamps >= 7
    && hasPreContext
    && hasPostContext
  );

  return {
    frames,
    duration_sec: duration,
    unique_timestamp_count: uniqueTimestamps,
    has_pre_context: hasPreContext,
    has_post_context: hasPostContext,
    temporal_context_valid: temporalContextValid,
  };
}

function foregroundGatePrompt() {
  return `You are auditing the event-level semantic evidence around one exact
temporal anchor for temporal action localization in an egocentric kitchen video.

You are given nine chronologically ordered frames sampled every 0.2 seconds.
Frame 5 is the exact anchor. Frames 1-4 are preceding context and Frames 6-9
are following context.

Determine whether the temporal neighborhood of Frame 5 belongs to an event in
the ten-action target ontology below. This is only a semantic judgment. Do not
judge FPN level, anchor scale, center sampling, regression range, or alignment
eligibility; a separate source-supervised structural gate handles those.

No candidate label is provided. Do not assume the anchor is foreground merely
because it was selected for review.

Target ontology:
- TAKE: gaining possession or stable control of an object.
- PUT: placing an object at a destination and relinquishing stable control.
- WASH: cleaning using water, rubbing, rinsing, or another cleaning medium.
- OPEN: changing a closure from closed to open or more accessible.
- CLOSE: changing a closure from open to closed or secured.
- INSERT: moving an object from outside to inside a container or opening.
- TURN-ON: operating a control so a device, light, flame, or flow becomes active.
- CUT: visibly slicing, dividing, or severing material with a cutting tool.
- TURN-OFF: operating a control so an active device, light, flame, or flow stops.
- POUR: controlling a source container so contents visibly transfer to a destination.

Related non-target activities must not be broadened into a target action. These
include merely holding or moving an already controlled object, removing from a
constrained location, adjusting, setting without the target state transition,
emptying, scraping, filling, scrubbing, pulling, pressing, or wrapping.

Decision rules:
- False rescue is more harmful than leaving an ambiguous anchor ignored.
- Choose TARGET_ACTION only when Frame 5's temporal neighborhood clearly belongs
  to one continuous event in the target ontology.
- Frame 5 may be near the beginning, middle, or end of the event. The complete
  transition need not be visible, but the central neighborhood must belong to it.
- Use surrounding frames as evidence, but do not confirm an event that occurs
  only in the outer context and does not include Frame 5.
- Do not combine evidence from different objects or separate events.
- Visible hands, objects, tools, camera motion, and general motion are not enough.

Labels:
- TARGET_ACTION: Frame 5 belongs to one event in the ten-action target ontology.
- OTHER_ACTIVITY: a meaningful interaction occurs, but it is outside that ontology.
- TRUE_BACKGROUND: no meaningful object interaction occurs around Frame 5.
- UNCERTAIN: the semantic or temporal evidence is insufficient or contradictory.

Return exactly one JSON object with exactly one key and no other text:
{"label":"TARGET_ACTION|OTHER_ACTIVITY|TRUE_BACKGROUND|UNCERTAIN"}`;
}

const PROMPT_TEXT = foregroundGatePrompt();
const PROMPT_SHA256 = sha256(PROMPT_TEXT);

function frameLabel(frame) {
  const offset = Number(frame.offset_sec);

  if (frame.index === 5 && offset === 0) {
    return 'FRAME 5 - EXACT TEMPORAL ANCHOR (0.0 seconds)';
  }

  const sign = offset > 0 ? '+' : '';

  return (
    `FRAME ${frame.index} - `
    + `${sign}${offset.toFixed(1)} seconds relative to the anchor`
  );
}

function buildMultimodalContent(frames, apiStyle) {
  const textType = (
    apiStyle === 'responses'
      ? 'input_text'
      : 'text'
  );

  const imageType = (
    apiStyle === 'responses'
      ? 'input_image'
      : 'image_url'
  );

  const content = [
    {
      type: textType,
      text: PROMPT_TEXT,
    },
  ];

  for (const frame of frames) {
    content.push({
      type: textType,
      text: frameLabel(frame),
    });

    const image = fs
      .readFileSync(frame.file)
      .toString('base64');

    const url = `data:image/jpeg;base64,${image}`;

    if (apiStyle === 'responses') {
      content.push({
        type: imageType,
        image_url: url,
      });
    } else {
      content.push({
        type: imageType,
        image_url: { url },
      });
    }
  }

  return content;
}

function outputText(response) {
  return (response.output || [])
    .flatMap(item => item.content || [])
    .filter(item => item.type === 'output_text')
    .map(item => item.text)
    .join('')
    .trim();
}

class HttpStatusError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'HttpStatusError';
    this.status = Number(status);
  }
}

class RequestTimeoutError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RequestTimeoutError';
  }
}

class ProtocolError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ProtocolError';
  }
}

function fetchCompat(url, options, timeoutMs = 0) {
  if (typeof globalThis.fetch === 'function') {
    const controller = new AbortController();
    const timeout = timeoutMs > 0
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;

    return globalThis.fetch(
      url,
      {
        ...options,
        signal: controller.signal,
      },
    ).catch(error => {
      if (error?.name === 'AbortError') {
        throw new RequestTimeoutError(
          `request timed out after ${timeoutMs} ms`,
        );
      }
      throw error;
    }).finally(() => {
      if (timeout !== null) clearTimeout(timeout);
    });
  }

  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const transport = (
      parsed.protocol === 'https:'
        ? https
        : http
    );

    const request = transport.request(
      parsed,
      {
        method: options.method,
        headers: options.headers,
      },
      response => {
        const chunks = [];

        response.on(
          'data',
          chunk => chunks.push(chunk),
        );

        response.on('end', () => {
          const body = Buffer
            .concat(chunks)
            .toString('utf8');

          const status = Number(
            response.statusCode || 0,
          );

          resolve({
            ok: status >= 200 && status < 300,
            status,
            text: async () => body,
          });
        });
      },
    );

    request.on('error', reject);

    if (timeoutMs > 0) {
      request.setTimeout(timeoutMs, () => {
        request.destroy(
          new RequestTimeoutError(
            `request timed out after ${timeoutMs} ms`,
          ),
        );
      });
    }

    request.end(options.body);
  });
}

async function requestOnce({
  baseUrl,
  apiKey,
  model,
  frames,
  wireApi,
  reasoningEffort,
  maxOutputTokens,
  requestTimeoutMs,
}) {
  const apiStyle = (
    wireApi === 'chat'
      ? 'chat'
      : 'responses'
  );

  const content = buildMultimodalContent(
    frames,
    apiStyle,
  );

  let endpoint;
  let body;

  if (wireApi === 'chat') {
    endpoint = 'chat/completions';

    body = {
      model,
      messages: [
        {
          role: 'user',
          content,
        },
      ],
      max_tokens: maxOutputTokens,
      temperature: 0,
      enable_thinking: false,
      response_format: {
        type: 'json_object',
      },
    };
  } else {
    endpoint = 'responses';

    body = {
      model,
      input: [
        {
          role: 'user',
          content,
        },
      ],
      max_output_tokens: maxOutputTokens,
    };

    if (reasoningEffort) {
      body.reasoning = {
        effort: reasoningEffort,
      };
    }
  }

  const response = await fetchCompat(
    `${baseUrl.replace(/\/$/, '')}/${endpoint}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    },
    requestTimeoutMs,
  );

  const responseBody = await response.text();

  let parsed;

  try {
    parsed = JSON.parse(responseBody);
  } catch {
    parsed = {
      raw: responseBody,
    };
  }

  if (!response.ok) {
    throw new HttpStatusError(
      response.status,
      `HTTP ${response.status}: `
      + JSON.stringify(parsed).slice(0, 600),
    );
  }

  const text = (
    wireApi === 'chat'
      ? String(
        parsed.choices?.[0]?.message?.content || '',
      ).trim()
      : outputText(parsed)
  );

  if (!text) {
    throw new Error('empty model response');
  }

  return {
    text,
    usage: parsed.usage || null,
  };
}

async function requestWithRetry(
  options,
  maxRetries,
  retryBaseMs,
  requestFn = requestOnce,
) {
  let lastError;

  for (
    let attempt = 0;
    attempt <= maxRetries;
    attempt += 1
  ) {
    try {
      const response = await requestFn(options);

      return {
        ...response,
        attempts: attempt + 1,
      };
    } catch (error) {
      lastError = error;

      const retryable = (
        error instanceof RequestTimeoutError
        || error?.status === 408
        || error?.status === 429
        || error?.status >= 500
        || (
          !(error instanceof HttpStatusError)
          && error?.name !== 'ProtocolError'
        )
      );

      if (attempt >= maxRetries || !retryable) break;

      const delay = retryBaseMs * (2 ** attempt);
      await sleep(delay);
    }
  }

  throw lastError;
}

function parseGateResponse(text) {
  let parsed;

  try {
    parsed = JSON.parse(text);
  } catch {
    return {
      parse_error: true,
      protocol_error: true,
      model_label: null,
      prediction: 'UNCERTAIN',
      semantic_pass: false,
      consistency_error: true,
    };
  }

  const exactObject = (
    parsed !== null
    && !Array.isArray(parsed)
    && typeof parsed === 'object'
    && Object.keys(parsed).length === 1
    && Object.hasOwn(parsed, 'label')
  );
  const modelLabel = exactObject ? normalizeEnum(parsed.label) : null;
  const valid = exactObject && LABEL_VALUES.has(modelLabel);

  return {
    parse_error: false,
    protocol_error: !valid,
    model_label: valid ? modelLabel : null,
    prediction: valid ? modelLabel : 'UNCERTAIN',
    semantic_pass: valid && modelLabel === 'TARGET_ACTION',
    consistency_error: !valid,
  };
}

function teacherTop1Class(row) {
  const top1 = Number(row.teacher_top1_class);

  if (!Number.isInteger(top1) || top1 < 0 || top1 >= 10) {
    throw new Error(
      `invalid teacher_top1_class for ${row.candidate_id}`,
    );
  }

  return top1;
}

const SOURCE_STRUCTURE_FIELDS = Object.freeze([
  'structure_probe_schema_version',
  'structure_probe_sha256',
  'structure_score',
  'structure_threshold',
  'structural_pass',
  'structure_feature_values',
]);

function sourceSupervisedStructure(row) {
  for (const field of SOURCE_STRUCTURE_FIELDS) {
    if (!(field in row)) {
      throw new Error(
        `manifest row ${row.candidate_id} is missing ${field}`,
      );
    }
  }
  if (
    row.structure_probe_schema_version
      !== 'source-supervised-anchor-validity-v1'
    || !/^[0-9a-f]{64}$/.test(row.structure_probe_sha256)
    || !Array.isArray(row.structure_feature_values)
    || row.structure_feature_values.length !== 13
    || row.structure_feature_values.some(value => !Number.isFinite(Number(value)))
  ) {
    throw new Error(`invalid source-supervised structure provenance for ${row.candidate_id}`);
  }
  const score = Number(row.structure_score);
  const threshold = Number(row.structure_threshold);
  if (
    !Number.isFinite(score)
    || !Number.isFinite(threshold)
    || score < 0
    || score > 1
    || threshold < 0
    || threshold > 1
  ) {
    throw new Error(`invalid source-supervised structure score for ${row.candidate_id}`);
  }
  const pass = score >= threshold;
  if (row.structural_pass !== pass) {
    throw new Error(`source-supervised structure decision disagrees for ${row.candidate_id}`);
  }
  return { score, threshold, pass };
}

function exactAnchorKey(row) {
  const top1 = teacherTop1Class(row);
  const center = absoluteCenterTime(row);

  if (
    row.fpn_level == null
    || row.global_anchor_index == null
    || row.teacher_top1_class == null
  ) {
    throw new Error(
      `manifest row ${row.candidate_id} must include `
      + 'fpn_level, global_anchor_index, and '
      + 'teacher_top1_class/sada_top1_class',
    );
  }

  return [
    row.video_id,
    Number(row.fpn_level),
    Number(row.global_anchor_index),
    center.toFixed(6),
    top1,
  ].join('|');
}

function validateManifestRows(rows) {
  const candidateIds = new Set();
  const anchorKeys = new Set();

  for (const row of rows) {
    if (
      !row.candidate_id
      || !row.video_id
      || row.center_sec == null
      || row.absolute_center_time == null
      || row.anchor_index == null
      || row.teacher_top1_class == null
    ) {
      throw new Error(
        'each manifest row requires '
        + 'candidate_id, video_id, anchor_index, '
        + 'center_sec, absolute_center_time, and teacher_top1_class',
      );
    }

    if (candidateIds.has(row.candidate_id)) {
      throw new Error(
        `duplicate candidate_id in manifest: `
        + `${row.candidate_id}`,
      );
    }

    candidateIds.add(row.candidate_id);

    const key = exactAnchorKey(row);
    const members = row.member_anchors;

    if (!Array.isArray(members) || members.length !== 1) {
      throw new Error(
        `manifest row ${row.candidate_id} must contain exactly one member anchor`,
      );
    }

    const member = members[0];
    sourceSupervisedStructure(row);
    sourceSupervisedStructure({ ...member, candidate_id: row.candidate_id });
    const memberMatches = (
      Number(member.fpn_level) === Number(row.fpn_level)
      && Number(member.anchor_index) === Number(row.anchor_index)
      && Number(member.global_anchor_index)
        === Number(row.global_anchor_index)
      && Math.abs(
        Number(member.absolute_center_time)
        - absoluteCenterTime(row)
      ) <= 1e-6
      && Number(member.teacher_top1_class)
        === teacherTop1Class(row)
      && SOURCE_STRUCTURE_FIELDS.every(
        field => member[field] === row[field],
      )
    );

    if (!memberMatches) {
      throw new Error(
        `member anchor differs from exact query fields for ${row.candidate_id}`,
      );
    }

    if (anchorKeys.has(key)) {
      throw new Error(
        `duplicate exact-anchor five-tuple `
        + `in manifest: ${key}`,
      );
    }

    anchorKeys.add(key);
  }
}

function requestFingerprint(row, model, wireApi) {
  return sha256(
    JSON.stringify({
      candidate_id: row.candidate_id,
      anchor_id: row.anchor_id ?? null,
      video_id: row.video_id,
      absolute_center_time: absoluteCenterTime(row),
      fpn_level: row.fpn_level ?? null,
      global_anchor_index:
        row.global_anchor_index ?? null,
      teacher_top1_class:
        teacherTop1Class(row),
      source_supervised_structure: Object.fromEntries(
        SOURCE_STRUCTURE_FIELDS.map(field => [field, row[field]]),
      ),
      frame_offsets_sec: FRAME_OFFSETS_SEC,
      prompt_sha256: PROMPT_SHA256,
      routing_schema_version: ROUTING_SCHEMA_VERSION,
      model,
      wire_api: wireApi,
    }),
  );
}

function localUncertainResult(
  row,
  frameInfo,
  fingerprint,
  model,
  wireApi,
  reason,
) {
  return {
    candidate_id: row.candidate_id,
    anchor_id: row.anchor_id ?? row.candidate_id,
    request_fingerprint: fingerprint,
    status: 'ok',
    prediction: 'UNCERTAIN',
    route: 'ignore',
    semantic_pass: false,
    structural_pass: sourceSupervisedStructure(row).pass,
    ablation_routes: {
      semantic_only: 'ignore',
      structure_only: sourceSupervisedStructure(row).pass ? 'action' : 'ignore',
      semantic_and_structure: 'ignore',
    },
    action_rescue: false,
    pseudo_label: null,
    model_reported_label: null,
    parse_error: false,
    protocol_error: false,
    consistency_error: false,
    local_gate_reason: reason,
    response_text: null,
    usage: null,
    model,
    wire_api: wireApi,
    prompt_version: PROMPT_VERSION,
    prompt_sha256: PROMPT_SHA256,
    routing_schema_version: ROUTING_SCHEMA_VERSION,
    frame_offsets_sec: [...FRAME_OFFSETS_SEC],
    frame_timestamps_sec:
      frameInfo.frames.map(frame => frame.timestamp_sec),
    temporal_context_valid:
      frameInfo.temporal_context_valid,
    unique_timestamp_count:
      frameInfo.unique_timestamp_count,
    video_id: row.video_id,
    anchor_index: Number(row.anchor_index),
    absolute_center_time: absoluteCenterTime(row),
    center_sec: absoluteCenterTime(row),
    fpn_level: row.fpn_level ?? null,
    global_anchor_index:
      row.global_anchor_index ?? null,
    teacher_top1_class:
      teacherTop1Class(row),
    teacher_top1_score:
      row.teacher_top1_score
      ?? row.sada_top1_score
      ?? null,
    structure_probe_schema_version: row.structure_probe_schema_version,
    structure_probe_sha256: row.structure_probe_sha256,
    structure_score: Number(row.structure_score),
    structure_threshold: Number(row.structure_threshold),
    structure_feature_values: [...row.structure_feature_values],
  };
}

function latestRowsByCandidate(rows) {
  const latest = new Map();
  for (const row of rows) {
    latest.set(row.candidate_id, row);
  }
  return latest;
}

function validatePriorRows(priorRows, manifestRows, model, wireApi) {
  const manifestById = new Map(
    manifestRows.map(row => [row.candidate_id, row]),
  );

  for (const prior of priorRows) {
    const manifest = manifestById.get(prior.candidate_id);
    if (!manifest) {
      throw new Error(
        `output contains candidate not present in manifest: ${prior.candidate_id}`,
      );
    }
    if (prior.routing_schema_version !== ROUTING_SCHEMA_VERSION) {
      throw new Error('output routing_schema_version does not match this runner');
    }
    if (prior.prompt_sha256 !== PROMPT_SHA256) {
      throw new Error('output prompt_sha256 does not match this runner');
    }
    if (prior.model !== model || prior.wire_api !== wireApi) {
      throw new Error('output model or wire_api differs from this run');
    }

    const expectedFingerprint = requestFingerprint(
      manifest,
      model,
      wireApi,
    );
    if (prior.request_fingerprint !== expectedFingerprint) {
      throw new Error(
        `output request fingerprint differs for ${prior.candidate_id}`,
      );
    }
  }
}

function completionSummary(
  activeRows,
  latestRows,
  {
    alreadyCompleted = 0,
    apiAttempted = 0,
    apiSuccess = 0,
    localUncertain = 0,
    localStructuralIgnore = 0,
  } = {},
) {
  const current = activeRows
    .map(row => latestRows.get(row.candidate_id))
    .filter(Boolean);
  const completed = current.filter(row => row.status === 'ok');
  const action = completed.filter(row => row.route === 'action').length;
  const ignore = completed.filter(row => row.route === 'ignore').length;
  const background = completed.filter(row => row.route === 'background').length;
  const error = current.filter(row => row.status === 'error').length;

  return {
    manifest_total: activeRows.length,
    already_completed: alreadyCompleted,
    api_attempted: apiAttempted,
    api_success: apiSuccess,
    local_uncertain: localUncertain,
    local_structural_ignore: localStructuralIgnore,
    completed_total: completed.length,
    action,
    ignore,
    background,
    error,
  };
}

function assertCompleteSummary(summary) {
  if (summary.completed_total !== summary.manifest_total) {
    throw new Error('completed_total differs from manifest_total');
  }
  if (summary.action + summary.ignore !== summary.manifest_total) {
    throw new Error('action + ignore differs from manifest_total');
  }
  if (summary.error !== 0) {
    throw new Error('run contains error rows');
  }
  if (summary.background !== 0) {
    throw new Error('candidate-blind rescue must never route background');
  }
}

function printSummary(summary) {
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

async function main() {
  const args = argsOf(process.argv);

  for (
    const required of [
      'manifest',
      'video-root',
      'output',
      'cache-dir',
      'base-url',
      'model',
      'wire-api',
    ]
  ) {
    if (!args[required]) {
      throw new Error(`missing --${required}`);
    }
  }

  const baseUrl = args['base-url'];
  const model = args.model;
  const wireApi = args['wire-api'];

  if (!['responses', 'chat'].includes(wireApi)) {
    throw new Error(
      '--wire-api must be responses or chat',
    );
  }

  const keyEnv = (
    args['api-key-env']
    || 'VLM_API_KEY'
  );

  const apiKey = process.env[keyEnv];

  if (!apiKey) {
    throw new Error(
      `environment variable ${keyEnv} is empty`,
    );
  }

  const reasoningEffort = (
    args['reasoning-effort']
    || process.env.VLM_REASONING_EFFORT
    || null
  );

  const maxOutputTokens = Number(
    args['max-output-tokens']
    || 96,
  );

  const requestTimeoutMs = Number(
    args['request-timeout-sec']
    || process.env.VLM_REQUEST_TIMEOUT_SEC
    || 120,
  ) * 1000;

  const maxRetries = Number(
    args['max-retries']
    || 5,
  );

  const retryBaseMs = Number(
    args['retry-base-ms']
    || 1000,
  );

  const limit = Number(
    args.limit
    || Infinity,
  );

  const rows = readJsonl(args.manifest);
  validateManifestRows(rows);

  if (
    limit !== Infinity
    && (!Number.isInteger(limit) || limit <= 0)
  ) {
    throw new Error('--limit must be a positive integer');
  }

  const activeRows = rows.slice(0, limit);

  const priorRows = readJsonl(
    args.output,
    { allowMissing: true, allowEmpty: true },
  );
  validatePriorRows(priorRows, rows, model, wireApi);

  const latestRows = latestRowsByCandidate(priorRows);
  const alreadyCompleted = activeRows.filter(
    row => latestRows.get(row.candidate_id)?.status === 'ok',
  ).length;

  const counters = {
    alreadyCompleted,
    apiAttempted: 0,
    apiSuccess: 0,
    localUncertain: 0,
    localStructuralIgnore: 0,
  };

  const numericOptions = {
    maxOutputTokens,
    requestTimeoutMs,
    maxRetries,
    retryBaseMs,
  };
  if (
    !Number.isInteger(maxOutputTokens)
    || maxOutputTokens <= 0
    || !Number.isFinite(requestTimeoutMs)
    || requestTimeoutMs <= 0
    || !Number.isInteger(maxRetries)
    || maxRetries < 0
    || !Number.isFinite(retryBaseMs)
    || retryBaseMs < 0
  ) {
    throw new Error(
      `invalid numeric request options: ${JSON.stringify(numericOptions)}`,
    );
  }

  const appendResult = result => {
    fs.appendFileSync(
      args.output,
      `${JSON.stringify(result)}\n`,
      'utf8',
    );
    latestRows.set(result.candidate_id, result);
  };

  const summary = () => completionSummary(
    activeRows,
    latestRows,
    counters,
  );

  fs.mkdirSync(
    path.dirname(path.resolve(args.output)),
    { recursive: true },
  );

  let processed = 0;

  for (const row of activeRows) {
    if (latestRows.get(row.candidate_id)?.status === 'ok') {
      continue;
    }

    const fingerprint = requestFingerprint(
      row,
      model,
      wireApi,
    );

    processed += 1;
    const center = absoluteCenterTime(row);

    const result = {
      candidate_id: row.candidate_id,
      anchor_id:
        row.anchor_id ?? row.candidate_id,
      request_fingerprint: fingerprint,
      status: 'error',
      model,
      wire_api: wireApi,
      prompt_version: PROMPT_VERSION,
      prompt_sha256: PROMPT_SHA256,
      routing_schema_version: ROUTING_SCHEMA_VERSION,
      video_id: row.video_id,
      anchor_index: Number(row.anchor_index),
      absolute_center_time: center,
      center_sec: center,
      fpn_level: row.fpn_level ?? null,
      global_anchor_index:
        row.global_anchor_index ?? null,
      teacher_top1_class:
        teacherTop1Class(row),
      teacher_top1_score:
        row.teacher_top1_score
        ?? row.sada_top1_score
        ?? null,
      structure_probe_schema_version: row.structure_probe_schema_version,
      structure_probe_sha256: row.structure_probe_sha256,
      structure_score: Number(row.structure_score),
      structure_threshold: Number(row.structure_threshold),
      structural_pass: sourceSupervisedStructure(row).pass,
      structure_feature_values: [...row.structure_feature_values],
    };

    try {
      const video = findVideo(
        args['video-root'],
        row.video_id,
      );

      const frameInfo = prepareFrames(
        video,
        center,
        path.join(
          args['cache-dir'],
          row.frame_cache_id ?? row.candidate_id,
        ),
      );

      if (!frameInfo.temporal_context_valid) {
        const local = localUncertainResult(
          row,
          frameInfo,
          fingerprint,
          model,
          wireApi,
          'insufficient_pre_or_post_context',
        );

        appendResult(local);
        counters.localUncertain += 1;

        process.stdout.write(
          `${processed}/${activeRows.length - alreadyCompleted} `
          + `${row.candidate_id} ok UNCERTAIN local\n`,
        );

        continue;
      }

      counters.apiAttempted += 1;
      const response = await requestWithRetry(
        {
          baseUrl,
          apiKey,
          model,
          frames: frameInfo.frames,
          wireApi,
          reasoningEffort,
          maxOutputTokens,
          requestTimeoutMs,
        },
        maxRetries,
        retryBaseMs,
      );

      const parsed = parseGateResponse(
        response.text,
      );

      if (
        parsed.parse_error
        || parsed.protocol_error
        || parsed.consistency_error
      ) {
        throw new ProtocolError(
          `invalid routing response for ${row.candidate_id}: ${response.text}`,
        );
      }

      counters.apiSuccess += 1;

      const top1 = teacherTop1Class(row);

      const structure = sourceSupervisedStructure(row);
      const semanticPass = parsed.semantic_pass;
      const jointPass = semanticPass && structure.pass;
      Object.assign(result, {
        status: 'ok',
        prediction: parsed.prediction,
        semantic_pass: semanticPass,
        structural_pass: structure.pass,
        ablation_routes: {
          semantic_only: semanticPass ? 'action' : 'ignore',
          structure_only: structure.pass ? 'action' : 'ignore',
          semantic_and_structure: jointPass ? 'action' : 'ignore',
        },
        route: jointPass ? 'action' : 'ignore',
        action_rescue: jointPass,
        pseudo_label: (
          jointPass
            ? top1
            : null
        ),
        model_reported_label:
          parsed.model_label,
        parse_error:
          parsed.parse_error,
        protocol_error:
          parsed.protocol_error,
        consistency_error:
          parsed.consistency_error,
        local_gate_reason: null,
        response_text:
          response.text,
        usage:
          response.usage,
        request_attempts:
          response.attempts,
        reasoning_effort:
          reasoningEffort,
        max_output_tokens:
          maxOutputTokens,
        request_timeout_sec:
          requestTimeoutMs / 1000,
        frame_offsets_sec:
          [...FRAME_OFFSETS_SEC],
        frame_timestamps_sec:
          frameInfo.frames.map(
            frame => frame.timestamp_sec,
          ),
        temporal_context_valid:
          frameInfo.temporal_context_valid,
        unique_timestamp_count:
          frameInfo.unique_timestamp_count,
        gate_rule:
          'candidate_blind_epic_semantics'
          + '_and_source_supervised_structure_and_exact_anchor',
      });
    } catch (error) {
      result.error = String(
        error?.message || error,
      );
      result.error_type = error?.name || 'Error';
    }

    appendResult(result);

    process.stdout.write(
      `${processed}/${activeRows.length - alreadyCompleted} `
      + `${row.candidate_id} `
      + `${result.status} `
      + `${result.prediction || ''}\n`,
    );

    if (result.status !== 'ok') {
      printSummary(summary());
      throw new Error(
        `failed at ${row.candidate_id}: ${result.error}`,
      );
    }
  }

  const finalSummary = summary();
  printSummary(finalSummary);
  assertCompleteSummary(finalSummary);
  return finalSummary;
}

const isDirectRun = (
  process.argv[1]
  && path.resolve(process.argv[1])
    === path.resolve(fileURLToPath(import.meta.url))
);

if (isDirectRun) {
  main().catch(error => {
    console.error(
      error?.stack
      || error?.message
      || String(error),
    );

    process.exitCode = 1;
  });
}

export {
  FRAME_OFFSETS_SEC,
  PROMPT_SHA256,
  PROMPT_VERSION,
  ROUTING_SCHEMA_VERSION,
  HttpStatusError,
  ProtocolError,
  RequestTimeoutError,
  absoluteCenterTime,
  argsOf,
  assertCompleteSummary,
  completionSummary,
  exactAnchorKey,
  fetchCompat,
  parseGateResponse,
  requestFingerprint,
  requestWithRetry,
  validateManifestRows,
  validatePriorRows,
};
