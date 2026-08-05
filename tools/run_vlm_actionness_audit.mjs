import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function argsOf(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 2) out[argv[i].replace(/^--/, '')] = argv[i + 1];
  return out;
}

function readJsonl(file) {
  return fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
}

function findVideo(root, videoId) {
  for (const ext of ['.MP4', '.mp4']) {
    const candidate = path.join(root, videoId + ext);
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error(`video not found: ${videoId}`);
}

function prepareFrames(video, center, outputDir) {
  fs.mkdirSync(outputDir, { recursive: true });
  const offsets = [-1.0, -0.5, 0.0, 0.5, 1.0];
  return offsets.map((offset, index) => {
    const file = path.join(outputDir, `frame_${index}.jpg`);
    if (!fs.existsSync(file)) {
      const timestamp = Math.max(0, Number(center) + offset).toFixed(3);
      const result = spawnSync('ffmpeg', [
        '-hide_banner', '-loglevel', 'error', '-ss', timestamp, '-i', video,
        '-frames:v', '1', '-vf', 'scale=640:-2', '-q:v', '3', '-y', file,
      ], { encoding: 'utf8' });
      if (result.status !== 0 || !fs.existsSync(file)) {
        throw new Error(`ffmpeg failed at ${timestamp}s: ${result.stderr}`);
      }
    }
    return file;
  });
}

function loadOntology(csvPath) {
  if (!csvPath) return null;
  const rows = fs.readFileSync(csvPath, 'utf8').split(/\r?\n/).slice(1).filter(Boolean);
  return rows.map(line => {
    const match = line.match(/^(\d+),([^,]+),/);
    if (!match) throw new Error(`cannot parse ontology row: ${line.slice(0, 80)}`);
    return { id: Number(match[1]), verb: match[2] };
  }).sort((a, b) => a.id - b.id);
}

function normalizeVerb(value) {
  return String(value || '').trim().toLowerCase().replace(/[\s_]+/g, '-');
}

function prompt(ontology, promptMode) {
  const target = ontology
    ? ontology.filter(item => item.id < 10).map(item => item.verb)
    : ['take', 'put', 'wash', 'open', 'close', 'insert', 'turn on', 'cut', 'turn off', 'pour'];
  const contrastiveKeys = new Set([
    'mix', 'move', 'remove', 'throw', 'dry', 'shake', 'scoop', 'adjust',
    'squeeze', 'lift', 'pull', 'press', 'fill', 'scrape', 'wrap', 'set',
    'add', 'hold', 'wear', 'transition',
  ]);
  const nonTarget = ontology
    ? ontology.filter(item => item.id >= 10 && (
      promptMode === 'full_ontology' || contrastiveKeys.has(item.verb)
    )).map(item => item.verb)
    : [];
  const taxonomyRule = ontology ? `
The dataset uses a strict verb taxonomy. First identify the single best verb at the center moment.
Target verb labels: ${target.join(', ')}.
${promptMode === 'full_ontology' ? 'All non-target verb labels' : 'Easily confused non-target verb labels'} that MUST be treated as background even when they are meaningful actions: ${nonTarget.join(', ')}.
Near-neighbor labels are intentionally distinct. Do not map a non-target verb to a broad target synonym merely because both involve hand-object interaction.` : `
Target verb labels: ${target.join(', ')}.`;
  return `You are a temporal action localization verifier.
You are given five chronologically ordered frames sampled from a short egocentric kitchen video clip. Frame 3 is the current temporal anchor; frames 1-2 are before it and frames 4-5 are after it.

Decide whether the CENTER MOMENT (frame 3), not merely another part of the clip, lies inside one of the target verb labels.${taxonomyRule}

Do not count camera motion, visible hands without meaningful interaction, idle/preparation states, or a target action appearing only before or after the center moment.

Return JSON only with label, best verb, and target_probability from 0 to 100.
target_probability means the probability that the CENTER MOMENT belongs to one of the exact target verb labels, not merely the probability that some action is happening.
Examples:
{"label":"A","verb":"target verb","target_probability":90}
{"label":"B","verb":"best non-target verb or background","target_probability":10}
{"label":"U","verb":"uncertain","target_probability":50}`;
}

function outputText(response) {
  return (response.output || []).flatMap(item => item.content || [])
    .filter(item => item.type === 'output_text').map(item => item.text).join('').trim();
}

async function requestOne(baseUrl, apiKey, model, frames, wireApi, ontology, promptMode) {
  const content = [{ type: 'input_text', text: prompt(ontology, promptMode) }];
  for (const file of frames) {
    const image = fs.readFileSync(file).toString('base64');
    content.push({ type: 'input_image', image_url: `data:image/jpeg;base64,${image}` });
  }
  let endpoint = 'responses';
  let body = { model, input: [{ role: 'user', content }], max_output_tokens: 32 };
  if (wireApi === 'chat') {
    endpoint = 'chat/completions';
    body = {
      model,
      messages: [{
        role: 'user',
        content: content.map(item => item.type === 'input_text'
          ? { type: 'text', text: item.text }
          : { type: 'image_url', image_url: { url: item.image_url } }),
      }],
      max_tokens: 32,
      temperature: 0,
      enable_thinking: false,
      response_format: { type: 'json_object' },
    };
  }
  const response = await fetch(`${baseUrl.replace(/\/$/, '')}/${endpoint}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const responseBody = await response.text();
  let parsed;
  try { parsed = JSON.parse(responseBody); } catch { parsed = { raw: responseBody }; }
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${JSON.stringify(parsed).slice(0, 400)}`);
  const text = wireApi === 'chat'
    ? String(parsed.choices?.[0]?.message?.content || '').trim()
    : outputText(parsed);
  return { text, usage: parsed.usage || null };
}

async function main() {
  const args = argsOf(process.argv);
  for (const required of ['manifest', 'video-root', 'output', 'cache-dir']) {
    if (!args[required]) throw new Error(`missing --${required}`);
  }
  const baseUrl = args['base-url'] || process.env.VLM_BASE_URL || 'https://api.asxs.top/v1';
  const model = args.model || process.env.VLM_MODEL || 'gpt-5.4';
  const wireApi = args['wire-api'] || process.env.VLM_WIRE_API || 'responses';
  if (!['responses', 'chat'].includes(wireApi)) throw new Error('--wire-api must be responses or chat');
  const keyEnv = args['api-key-env'] || 'CCH_API_KEY';
  const apiKey = process.env[keyEnv];
  const ontology = loadOntology(args['ontology-csv']);
  const promptMode = args['prompt-mode'] || (ontology ? 'full_ontology' : 'target_only');
  if (!['target_only', 'contrastive', 'contrastive_score', 'full_ontology'].includes(promptMode)) {
    throw new Error('--prompt-mode must be target_only, contrastive, contrastive_score, or full_ontology');
  }
  if (promptMode !== 'target_only' && !ontology) {
    throw new Error('--ontology-csv is required for taxonomy-aware prompt modes');
  }
  if (!apiKey) throw new Error(`environment variable ${keyEnv} is empty`);
  const limit = Number(args.limit || Infinity);
  const rows = readJsonl(args.manifest);
  const done = fs.existsSync(args.output)
    ? new Set(readJsonl(args.output).filter(row => row.status === 'ok').map(row => row.candidate_id))
    : new Set();
  fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });

  let attempted = 0;
  for (const row of rows) {
    if (done.has(row.candidate_id) || attempted >= limit) continue;
    attempted += 1;
    const result = { candidate_id: row.candidate_id, status: 'error' };
    try {
      const video = findVideo(args['video-root'], row.video_id);
      const frames = prepareFrames(video, row.center_sec, path.join(args['cache-dir'], row.candidate_id));
      const response = await requestOne(
        baseUrl, apiKey, model, frames, wireApi, ontology, promptMode,
      );
      let label = null;
      let reportedVerb = null;
      let targetProbability = null;
      try {
        const parsed = JSON.parse(response.text);
        label = parsed.label;
        reportedVerb = parsed.verb || null;
        targetProbability = Number.isFinite(Number(parsed.target_probability))
          ? Number(parsed.target_probability) : null;
      } catch {}
      const match = String(label || '').match(/^([ABU])$/i);
      let prediction = match ? match[1].toUpperCase() : 'U';
      if (ontology && reportedVerb) {
        const targetVerbs = new Set(
          ontology.filter(item => item.id < 10).map(item => normalizeVerb(item.verb)),
        );
        prediction = normalizeVerb(reportedVerb) === 'uncertain'
          ? 'U'
          : (targetVerbs.has(normalizeVerb(reportedVerb)) ? 'A' : 'B');
      }
      result.status = 'ok';
      result.prediction = prediction;
      result.model_reported_label = match ? match[1].toUpperCase() : null;
      result.parse_error = !match;
      result.response_text = response.text;
      result.reported_verb = reportedVerb;
      result.target_probability = targetProbability;
      result.usage = response.usage;
      result.model = model;
      result.wire_api = wireApi;
      result.prompt_mode = promptMode;
      result.gate_rule = ontology ? 'deterministic_from_reported_verb' : 'model_label';
    } catch (error) {
      result.error = String(error.message || error);
    }
    fs.appendFileSync(args.output, JSON.stringify(result) + '\n', 'utf8');
    process.stdout.write(`${attempted}/${Math.min(rows.length, limit)} ${row.candidate_id} ${result.status} ${result.prediction || ''}\n`);
    if (result.status !== 'ok') throw new Error(result.error);
  }
}

main().catch(error => { console.error(error.message || error); process.exitCode = 1; });
