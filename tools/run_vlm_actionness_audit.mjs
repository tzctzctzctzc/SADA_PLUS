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
 *   4. Ask whether a target-defining state change passes through Frame 5.
 *   5. Rescue only when visual state evidence and teacher structure both pass.
 *
 * The VLM never changes the class. A rescued anchor keeps the frozen teacher's
 * top-1 class. Valid non-confirmed responses are routed to IGNORE. Malformed or
 * internally inconsistent responses are protocol errors and stop the run.
 */

const FRAME_OFFSETS_SEC = Object.freeze([
  -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8,
]);
const PROMPT_VERSION = 'candidate-blind-state-transition-gate-v2';
const ROUTING_SCHEMA_VERSION = 'candidate-blind-exact-anchor-routing-v2';

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

const STATE_VALUES = new Set([
  'SUPPORTS_TARGET_EVENT',
  'CONTRADICTS_TARGET_EVENT',
  'UNCERTAIN',
]);
const CHANGE_VALUES = new Set([
  'TARGET_DEFINING_CHANGE',
  'OTHER_CHANGE',
  'NO_MEANINGFUL_CHANGE',
  'UNCERTAIN',
]);
const LABEL_VALUES = new Set([
  'TARGET_FOREGROUND',
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
  return `You are auditing one exact temporal anchor for temporal action localization
in an egocentric kitchen video.

You are given nine chronologically ordered frames sampled every 0.2 seconds.
Frame 5 is the exact anchor being audited. Frames 1-4 are preceding context
and Frames 6-9 are following context.

Determine whether one continuous target action event passes through Frame 5.
Use the dense sequence to compare the relevant object state before, during,
and after the anchor. Do not classify the whole clip from general activity.

No candidate action label is provided. Do not assume that Frame 5 is an action
anchor merely because it was selected for review.

Target action event signatures:

TAKE:
The person gains possession or stable control of an object. An object that is
already being held, carried, moved, adjusted, lifted, or removed does not by
itself establish TAKE.

PUT:
The person places an object at a destination and relinquishes stable control,
leaving the object supported at or remaining in that destination.

WASH:
The person performs a cleaning event using water, rubbing, rinsing, or another
cleaning medium. Merely being near a sink, moving a wet object, filling,
pouring, squeezing, or performing one unrelated manipulation is insufficient.

OPEN:
A door, lid, drawer, container, or closure changes from closed to open or
becomes more accessible.

CLOSE:
A door, lid, drawer, container, or closure changes from open to closed or
becomes less accessible or secured.

INSERT:
An object crosses from outside to inside a container, slot, opening, or
enclosing object.

TURN-ON:
A control operation produces an observable transition from inactive to active
in a device, appliance, light, flame, or flow.

CUT:
A blade or cutting tool visibly slices, divides, or severs material. Merely
holding a knife, positioning material, pressing, peeling, or scraping is
insufficient.

TURN-OFF:
A control operation produces an observable transition from active to inactive
in a device, appliance, light, flame, or flow.

POUR:
A source container is controlled so that visible contents flow or are clearly
transferred into or onto a destination.

Related but non-target activities must not be broadened into a target action.
Examples include moving an already controlled object, removing an object from
a container or constrained location, adjusting a position or setting without
a target state transition, emptying, scraping, filling, scrubbing, pulling,
or wrapping.

Before deciding, internally inspect three stages:

1. BEFORE STATE — Frames 1-4:
   Identify the preceding possession, location, containment, closure, device,
   cleaning, cutting, or content state for the same hand and object.

2. CENTER CHANGE — Frames 4-6, especially Frame 5:
   Identify the visible change or continuing target-defining manipulation that
   passes through the exact anchor. An event only in Frames 1-3 or 7-9 does not
   pass through the anchor.

3. AFTER STATE — Frames 6-9:
   Check the resulting state or continuation of that same event on the same
   object. Do not combine separate events or different objects.

Do not output this internal analysis.

Important decision rules:

- A false target-foreground rescue is more harmful than failing to rescue an
  ambiguous true action.

- Choose TARGET_FOREGROUND only when all three stages support one continuous
  target event: the before state is compatible, a target-defining change or
  manipulation passes through Frame 5, and the after state confirms the result
  or continuation.

- For transition actions such as TAKE, PUT, OPEN, CLOSE, INSERT, TURN-ON,
  TURN-OFF, and POUR, compare the before and after object states. Do not infer
  an unseen transition only from one static state.

- For ongoing actions such as WASH and CUT, the target-defining manipulation
  must be visibly active through Frames 4-6 and supported by adjacent frames.

- Do not use an action occurring only in the outer context if it does not pass
  through Frames 4-6.

- Do not combine evidence from different objects or separate events.

- Visible hands, relevant objects, tools, controls, camera motion, and general
  motion are not sufficient by themselves.

Choose exactly one semantic label:

TARGET_FOREGROUND:
The before state, center change, and after state consistently show that Frame 5
belongs to one continuous event from the ten-action target set.

OTHER_ACTIVITY:
A meaningful non-target interaction passes through Frame 5, but the event is
better described as a non-target activity.

TRUE_BACKGROUND:
No meaningful object interaction passes through Frame 5.

UNCERTAIN:
The before state, center change, after state, object identity, or temporal
relation to Frame 5 is insufficient or contradictory.

Return exactly one JSON object with exactly these keys and no other text:

"before_state": one of "SUPPORTS_TARGET_EVENT",
"CONTRADICTS_TARGET_EVENT", or "UNCERTAIN".
"center_change": one of "TARGET_DEFINING_CHANGE", "OTHER_CHANGE",
"NO_MEANINGFUL_CHANGE", or "UNCERTAIN".
"after_state": one of "SUPPORTS_TARGET_EVENT",
"CONTRADICTS_TARGET_EVENT", or "UNCERTAIN".
"label": one of "TARGET_FOREGROUND", "OTHER_ACTIVITY",
"TRUE_BACKGROUND", or "UNCERTAIN".

The four fields must be logically consistent. TARGET_FOREGROUND requires
SUPPORTS_TARGET_EVENT, TARGET_DEFINING_CHANGE, and SUPPORTS_TARGET_EVENT in
that order.

Use this exact label mapping after filling the three evidence fields:
- If the three fields are SUPPORTS_TARGET_EVENT, TARGET_DEFINING_CHANGE,
  SUPPORTS_TARGET_EVENT, return TARGET_FOREGROUND.
- Otherwise, if center_change is OTHER_CHANGE, return OTHER_ACTIVITY.
- Otherwise, if center_change is NO_MEANINGFUL_CHANGE, return TRUE_BACKGROUND.
- In every remaining case, return UNCERTAIN.

Do not return explanations, markdown, probabilities, action names, or
additional keys.`;
}

const PROMPT_TEXT = foregroundGatePrompt();
const PROMPT_SHA256 = sha256(PROMPT_TEXT);

function frameLabel(frame) {
  const offset = Number(frame.offset_sec);

  if (frame.index === 5 && offset === 0) {
    return 'FRAME 5 - EXACT TEMPORAL ANCHOR (0.0 seconds)';
  }

  if (offset === 0) {
    return 'FRAME 3 — EXACT TEMPORAL ANCHOR (0.0 seconds)';
  }

  const sign = offset > 0 ? '+' : '';

  return (
    `FRAME ${frame.index} — `
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
      before_state: 'UNCERTAIN',
      center_change: 'UNCERTAIN',
      after_state: 'UNCERTAIN',
      model_label: null,
      prediction: 'UNCERTAIN',
      action_rescue: false,
      consistency_error: true,
    };
  }

  if (
    parsed === null
    || Array.isArray(parsed)
    || typeof parsed !== 'object'
  ) {
    return {
      parse_error: false,
      protocol_error: true,
      before_state: 'UNCERTAIN',
      center_change: 'UNCERTAIN',
      after_state: 'UNCERTAIN',
      model_label: null,
      prediction: 'UNCERTAIN',
      action_rescue: false,
      consistency_error: true,
    };
  }

  const beforeState = normalizeEnum(
    parsed.before_state,
  );

  const centerChange = normalizeEnum(
    parsed.center_change,
  );

  const afterState = normalizeEnum(
    parsed.after_state,
  );

  const modelLabel = normalizeEnum(
    parsed.label,
  );

  const keys = Object.keys(parsed).sort();

  const exactKeys = (
    JSON.stringify(keys)
    === JSON.stringify([
      'after_state',
      'before_state',
      'center_change',
      'label',
    ])
  );

  const fieldsValid = (
    STATE_VALUES.has(beforeState)
    && CHANGE_VALUES.has(centerChange)
    && STATE_VALUES.has(afterState)
    && LABEL_VALUES.has(modelLabel)
  );

  const actionRescue = (
    fieldsValid
    && exactKeys
    && beforeState === 'SUPPORTS_TARGET_EVENT'
    && centerChange === 'TARGET_DEFINING_CHANGE'
    && afterState === 'SUPPORTS_TARGET_EVENT'
    && modelLabel === 'TARGET_FOREGROUND'
  );

  let expectedLabel = 'UNCERTAIN';

  if (
    beforeState === 'SUPPORTS_TARGET_EVENT'
    && centerChange === 'TARGET_DEFINING_CHANGE'
    && afterState === 'SUPPORTS_TARGET_EVENT'
  ) {
    expectedLabel = 'TARGET_FOREGROUND';
  } else if (
    centerChange === 'OTHER_CHANGE'
  ) {
    expectedLabel = 'OTHER_ACTIVITY';
  } else if (
    centerChange === 'NO_MEANINGFUL_CHANGE'
  ) {
    expectedLabel = 'TRUE_BACKGROUND';
  }

  const consistencyError = (
    !fieldsValid
    || !exactKeys
    || modelLabel !== expectedLabel
  );

  const safePrediction = (
    actionRescue
      ? 'TARGET_FOREGROUND'
      : (
        !consistencyError
        && LABEL_VALUES.has(modelLabel)
          ? modelLabel
          : 'UNCERTAIN'
      )
  );

  return {
    parse_error: false,
    protocol_error: !fieldsValid || !exactKeys,
    before_state: (
      STATE_VALUES.has(beforeState)
        ? beforeState
        : 'UNCERTAIN'
    ),
    center_change: (
      CHANGE_VALUES.has(centerChange)
        ? centerChange
        : 'UNCERTAIN'
    ),
    after_state: (
      STATE_VALUES.has(afterState)
        ? afterState
        : 'UNCERTAIN'
    ),
    model_label: (
      LABEL_VALUES.has(modelLabel)
        ? modelLabel
        : null
    ),
    prediction: safePrediction,
    action_rescue: actionRescue,
    consistency_error: consistencyError,
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

const STRUCTURE_FIELDS = Object.freeze([
  'teacher_regression_left',
  'teacher_regression_right',
  'point_stride',
  'point_regression_range_min',
  'point_regression_range_max',
  'teacher_predicted_max_distance',
  'teacher_predicted_center_distance',
  'center_sample_radius',
  'teacher_range_eligible',
  'teacher_center_eligible',
  'teacher_duration_eligible',
  'teacher_structural_eligible',
]);

function teacherStructure(row) {
  for (const field of STRUCTURE_FIELDS) {
    if (!(field in row)) {
      throw new Error(
        `manifest row ${row.candidate_id} is missing ${field}`,
      );
    }
  }

  const left = Number(row.teacher_regression_left);
  const right = Number(row.teacher_regression_right);
  const stride = Number(row.point_stride);
  const rangeMin = Number(row.point_regression_range_min);
  const rangeMax = Number(row.point_regression_range_max);
  const radius = Number(row.center_sample_radius);
  const numbers = [left, right, stride, rangeMin, rangeMax, radius];

  if (
    numbers.some(value => !Number.isFinite(value))
    || left < 0
    || right < 0
    || stride <= 0
    || rangeMin < 0
    || rangeMax < rangeMin
    || radius <= 0
  ) {
    throw new Error(
      `invalid teacher structure for ${row.candidate_id}`,
    );
  }

  const maxDistance = Math.max(left, right) * stride;
  const centerDistance = Math.abs(right - left) / 2;
  const rangeEligible = (
    maxDistance >= rangeMin
    && maxDistance <= rangeMax
  );
  const centerEligible = centerDistance <= radius;
  const durationEligible = left + right > 0;
  const eligible = (
    rangeEligible
    && centerEligible
    && durationEligible
  );

  const close = (actual, expected) => (
    Math.abs(Number(actual) - expected) <= 1e-6
  );
  const booleansMatch = (
    row.teacher_range_eligible === rangeEligible
    && row.teacher_center_eligible === centerEligible
    && row.teacher_duration_eligible === durationEligible
    && row.teacher_structural_eligible === eligible
  );

  if (
    !close(row.teacher_predicted_max_distance, maxDistance)
    || !close(row.teacher_predicted_center_distance, centerDistance)
    || !booleansMatch
  ) {
    throw new Error(
      `teacher structure fields disagree for ${row.candidate_id}`,
    );
  }

  return {
    eligible,
    range_eligible: rangeEligible,
    center_eligible: centerEligible,
    duration_eligible: durationEligible,
  };
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
    teacherStructure(row);
    teacherStructure({ ...member, candidate_id: row.candidate_id });
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
      && STRUCTURE_FIELDS.every(
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
      teacher_structure: Object.fromEntries(
        STRUCTURE_FIELDS.map(field => [field, row[field]]),
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
    action_rescue: false,
    pseudo_label: null,
    before_state: 'UNCERTAIN',
    center_change: 'UNCERTAIN',
    after_state: 'UNCERTAIN',
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
    teacher_structural_eligible:
      teacherStructure(row).eligible,
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
      teacher_structural_eligible:
        teacherStructure(row).eligible,
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

      if (!teacherStructure(row).eligible) {
        const local = localUncertainResult(
          row,
          frameInfo,
          fingerprint,
          model,
          wireApi,
          'teacher_fpn_structural_ineligible',
        );

        appendResult(local);
        counters.localStructuralIgnore += 1;

        process.stdout.write(
          `${processed}/${activeRows.length - alreadyCompleted} `
          + `${row.candidate_id} ok UNCERTAIN structural\n`,
        );

        continue;
      }

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

      Object.assign(result, {
        status: 'ok',
        prediction: parsed.prediction,
        route: (
          parsed.action_rescue
            ? 'action'
            : 'ignore'
        ),
        action_rescue:
          parsed.action_rescue,
        pseudo_label: (
          parsed.action_rescue
            ? top1
            : null
        ),
        before_state:
          parsed.before_state,
        center_change:
          parsed.center_change,
        after_state:
          parsed.after_state,
        teacher_structural_eligible: true,
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
          'candidate_blind_state_transition'
          + '_and_teacher_structure_and_exact_anchor',
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
