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

function prompt() {
  return `You are a temporal action localization verifier.
You are given five chronologically ordered frames sampled from a short egocentric kitchen video clip. Frame 3 is the current temporal anchor; frames 1-2 are before it and frames 4-5 are after it.

Decide whether the CENTER MOMENT (frame 3), not merely another part of the clip, lies inside one of these target actions:
take, put, wash, open, close, insert, turn on, cut, turn off, pour.

Do not count camera motion, visible hands without meaningful interaction, idle/preparation states, or a target action appearing only before or after the center moment.

Output exactly one letter:
A = center moment is inside a target action
B = center moment is background
U = genuinely uncertain`;
}

function outputText(response) {
  return (response.output || []).flatMap(item => item.content || [])
    .filter(item => item.type === 'output_text').map(item => item.text).join('').trim();
}

async function requestOne(baseUrl, apiKey, model, frames) {
  const content = [{ type: 'input_text', text: prompt() }];
  for (const file of frames) {
    const image = fs.readFileSync(file).toString('base64');
    content.push({ type: 'input_image', image_url: `data:image/jpeg;base64,${image}` });
  }
  const response = await fetch(`${baseUrl.replace(/\/$/, '')}/responses`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, input: [{ role: 'user', content }], max_output_tokens: 32 }),
  });
  const body = await response.text();
  let parsed;
  try { parsed = JSON.parse(body); } catch { parsed = { raw: body }; }
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${JSON.stringify(parsed).slice(0, 400)}`);
  return { text: outputText(parsed), usage: parsed.usage || null };
}

async function main() {
  const args = argsOf(process.argv);
  for (const required of ['manifest', 'video-root', 'output', 'cache-dir']) {
    if (!args[required]) throw new Error(`missing --${required}`);
  }
  const baseUrl = args['base-url'] || process.env.VLM_BASE_URL || 'https://api.asxs.top/v1';
  const model = args.model || process.env.VLM_MODEL || 'gpt-5.4';
  const keyEnv = args['api-key-env'] || 'CCH_API_KEY';
  const apiKey = process.env[keyEnv];
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
      const response = await requestOne(baseUrl, apiKey, model, frames);
      const match = response.text.match(/^\s*([ABU])(?:\b|\s|$)/i);
      result.status = 'ok';
      result.prediction = match ? match[1].toUpperCase() : 'U';
      result.parse_error = !match;
      result.response_text = response.text;
      result.usage = response.usage;
      result.model = model;
    } catch (error) {
      result.error = String(error.message || error);
    }
    fs.appendFileSync(args.output, JSON.stringify(result) + '\n', 'utf8');
    process.stdout.write(`${attempted}/${Math.min(rows.length, limit)} ${row.candidate_id} ${result.status} ${result.prediction || ''}\n`);
    if (result.status !== 'ok') throw new Error(result.error);
  }
}

main().catch(error => { console.error(error.message || error); process.exitCode = 1; });
