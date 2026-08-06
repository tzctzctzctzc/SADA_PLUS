import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
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

const VIDEO_DURATIONS = new Map();

function videoDuration(video) {
  if (VIDEO_DURATIONS.has(video)) return VIDEO_DURATIONS.get(video);
  const result = spawnSync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=nw=1:nk=1', video,
  ], { encoding: 'utf8' });
  const duration = Number(String(result.stdout || '').trim());
  if (result.status !== 0 || !Number.isFinite(duration) || duration <= 0) {
    throw new Error(`ffprobe failed for ${video}: ${result.stderr}`);
  }
  VIDEO_DURATIONS.set(video, duration);
  return duration;
}

function prepareFrames(video, center, outputDir) {
  fs.mkdirSync(outputDir, { recursive: true });
  const offsets = [-1.0, -0.5, 0.0, 0.5, 1.0];
  const duration = videoDuration(video);
  return offsets.map((offset, index) => {
    const file = path.join(outputDir, `frame_${index}.jpg`);
    if (!fs.existsSync(file)) {
      const timestamp = Math.max(
        0, Math.min(Number(center) + offset, duration - 0.05),
      ).toFixed(3);
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

const CANDIDATE_SPECS = new Map([
  [0, {
    name: 'take',
    definition: 'acquires possession or stable control of an object by grasping, retrieving, or picking it up',
  }],
  [1, {
    name: 'put',
    definition: 'places an object at a destination and relinquishes stable control of it',
    evidence: 'The person holds or carries the object before or near the anchor, moves it to a destination, and places or releases it so that it remains there.',
    negatives: 'Do not confirm if the object remains held, is only moved or adjusted, or merely contacts a surface without a clear placement-and-release event.',
  }],
  [2, {
    name: 'wash',
    definition: 'deliberately cleans an object or the hands using water, rubbing, or another cleaning medium',
    evidence: 'The central neighborhood shows sustained cleaning interaction, such as rinsing under water, rubbing, scrubbing, or applying and removing a cleaning medium.',
    negatives: 'Being near a sink, holding an object under an inactive tap, filling, pouring, wiping once, or merely moving a wet object is not sufficient.',
  }],
  [3, {
    name: 'open',
    definition: 'changes a door, lid, drawer, container, or closure from a closed state to an open or accessible state',
    evidence: 'The person manipulates a closure and the opening or accessibility increases before, around, or after the anchor.',
    negatives: 'Do not confirm for reaching into something already open, holding an already removed lid, unrelated pulling, or touching a closure without an opening state change.',
  }],
  [4, {
    name: 'close',
    definition: 'changes a door, lid, drawer, container, or closure from an open state to a closed or secured state',
    evidence: 'The person moves or secures a closure and the opening or accessibility decreases before, around, or after the anchor.',
    negatives: 'Do not confirm for touching or adjusting something already closed, putting an object nearby, or moving a door or lid without evidence that it becomes closed.',
  }],
  [5, {
    name: 'insert',
    definition: 'moves an object from outside into the interior of a container, slot, opening, or enclosing object',
    evidence: 'The object approaches an opening, crosses its boundary, and becomes at least partly contained inside it around the central neighborhood.',
    negatives: 'Putting an object on a surface, holding it near an opening, briefly dipping it, or removing it from a container is not insert.',
  }],
  [6, {
    name: 'turn-on',
    definition: 'operates a control so that a device, appliance, light, flame, or flow changes from inactive to active',
    evidence: 'The person presses, rotates, toggles, or otherwise operates a control and there is visual evidence that the device or output activates.',
    negatives: 'Do not confirm for touching or adjusting a control without activation, using a device that was already on, or an activation occurring only outside the central neighborhood.',
  }],
  [7, {
    name: 'cut',
    definition: 'uses a blade or cutting tool to slice, divide, or sever material',
    evidence: 'The central neighborhood shows purposeful blade contact and cutting motion that slices or separates the material.',
    negatives: 'Holding a knife, positioning food, scraping, peeling, pressing, or moving an already cut piece is not sufficient.',
  }],
  [8, {
    name: 'turn-off',
    definition: 'operates a control so that a device, appliance, light, flame, or flow changes from active to inactive',
    evidence: 'The person presses, rotates, toggles, or otherwise operates a control and there is visual evidence that the device or output deactivates.',
    negatives: 'Do not confirm for touching or adjusting a control without deactivation, interacting with a device already off, or a deactivation occurring only outside the central neighborhood.',
  }],
  [9, {
    name: 'pour',
    definition: 'tilts or controls a source container so that its contents flow into or onto a destination',
    evidence: 'The source is oriented for transfer and visible liquid, granules, or other contents flow or are clearly transferred around the central neighborhood.',
    negatives: 'Moving or tilting a container without evidence of content transfer, filling from a tap, stirring, or merely holding containers near each other is not sufficient.',
  }],
  [11, {
    name: 'move',
    definition: 'changes the location or position of an object without requiring a new possession or release event',
    evidence: 'The object is deliberately translated or repositioned around the central neighborhood. Stable possession may continue throughout.',
    negatives: 'Do not confirm merely for acquiring, releasing, opening, or closing an object when no distinct relocation event is visible.',
  }],
  [12, {
    name: 'remove',
    definition: 'takes an object out of, off, or away from a container, attachment, or prior location',
    evidence: 'The object crosses from an inside, attached, or resting state to an outside, detached, or removed state.',
    negatives: 'Generic picking up without evidence of removal from something, continued holding, or unrelated movement is not sufficient.',
  }],
  [17, {
    name: 'adjust',
    definition: 'changes the position, setting, or configuration of an object or control without requiring activation or deactivation',
    evidence: 'The central neighborhood shows a deliberate fine change to position, orientation, fit, or setting.',
    negatives: 'Do not confirm when the defining event is clearly turning a device on or off, opening, closing, inserting, or removing.',
  }],
  [20, {
    name: 'empty',
    definition: 'removes or discharges contents from a container so that it becomes substantially less full',
    evidence: 'Contents visibly leave the source container by tipping, dumping, draining, or another discharge action.',
    negatives: 'Pouring a small controlled amount while retaining contents, filling a destination, or tilting without visible discharge is not sufficient.',
  }],
  [25, {
    name: 'scrape',
    definition: 'drags a tool or object edge across a surface to detach, collect, clean, or move material',
    evidence: 'The central neighborhood shows sustained edge-to-surface contact with a dragging or scraping motion.',
    negatives: 'Cutting through material, merely holding a blade, wiping softly, or moving already detached pieces is not sufficient.',
  }],
  [26, {
    name: 'fill',
    definition: 'adds material or contents into a container so that its amount or occupancy increases',
    evidence: 'Material enters a destination container and visibly increases its contents around the central neighborhood.',
    negatives: 'Inserting one solid object, pouring without a clear destination increase, or handling an already full container is not sufficient.',
  }],
  [29, {
    name: 'scrub',
    definition: 'cleans a surface or object by repeated forceful rubbing with a hand, pad, brush, or cleaning tool',
    evidence: 'Repeated back-and-forth rubbing contact for cleaning is visible around the central neighborhood.',
    negatives: 'Rinsing under water, squeezing a sponge, wiping once, or merely being near a sink is not sufficient.',
  }],
  [31, {
    name: 'pull',
    definition: 'applies force to draw an object or movable part toward the person or along its constrained path',
    evidence: 'The object or part translates in the pulling direction while contact and pulling force are maintained.',
    negatives: 'Opening without visible pulling, lifting vertically, taking possession, or touching a handle without translation is not sufficient.',
  }],
  [36, {
    name: 'wrap',
    definition: 'covers, encloses, seals, or secures an object with material or a flexible closure',
    evidence: 'Covering material or a seal is deliberately placed around or over the object and becomes secured.',
    negatives: 'Closing a rigid door or lid, merely placing material nearby, or holding an already wrapped object is not sufficient.',
  }],
]);

// Parallel descriptions used only by the listwise experiment. Keeping this
// separate preserves the exact prompts used by the earlier independent audit.
const LISTWISE_SPECS = new Map([
  [0, {
    name: 'take',
    definition: 'acquires possession or stable control of an object, typically by grasping, retrieving, or picking it up',
    evidence: 'The person approaches, reaches for, contacts, grasps, or retrieves the object; control is acquired or consolidated around the central neighborhood; and the object may subsequently be held or carried. The complete acquisition transition does not need to be fully visible.',
    negatives: 'Do not confirm merely because the object is already held, carried, lifted, pulled, adjusted, or moved. Brief contact without evidence that the central neighborhood belongs to an acquisition event is not sufficient.',
    distinction: '"Take" is primarily defined by gaining possession or stable control. Unlike "move", control does not simply continue; unlike "remove", extraction from a container, attachment, or constrained location is not the defining event.',
  }],
  [1, {
    name: 'put',
    definition: 'places an object at a destination and relinquishes stable control of it',
    evidence: 'The person holds or carries the object before or near the anchor, moves it toward a destination, and places or releases it so that the object becomes supported by or remains at the destination.',
    negatives: 'Do not confirm if the object remains held, is only moved or adjusted, merely contacts a surface, or passes near a destination without a placement-and-release event.',
    distinction: '"Put" requires relinquishing control and establishing a new resting or supported state. Unlike "move", continued possession without release is insufficient.',
  }],
  [2, {
    name: 'wash',
    definition: 'cleans an object or the hands using water, rubbing, rinsing, or another cleaning medium',
    evidence: 'The central neighborhood shows a sustained cleaning process, such as rinsing under water, rubbing, applying a cleaning medium, or removing residue from the hands or an object.',
    negatives: 'Being near a sink, holding an object under an inactive tap, filling, pouring, squeezing a sponge, wiping once, or merely moving a wet object is not sufficient.',
    distinction: '"Wash" describes the broader cleaning event, often involving water or rinsing. If repeated forceful rubbing is clearly the defining interaction, "scrub" is more specific.',
  }],
  [3, {
    name: 'open',
    definition: 'changes a door, lid, drawer, container, or closure from a closed state to an open or accessible state',
    evidence: 'The person manipulates a closure and the opening, separation, or accessibility increases before, around, or after the central neighborhood.',
    negatives: 'Do not confirm for reaching into something already open, holding an already removed lid, generic pulling, touching a closure without a state change, or moving an open part without increasing accessibility.',
    distinction: '"Open" is defined by a closed-to-open state change. A pulling motion is only the means; if no opening or accessibility change occurs, "pull" is more accurate.',
  }],
  [4, {
    name: 'close',
    definition: 'changes a door, lid, drawer, container, or closure from an open state to a closed or secured state',
    evidence: 'The person moves, presses, aligns, or secures a closure and the opening or accessibility decreases before, around, or after the anchor.',
    negatives: 'Do not confirm for touching or adjusting something already closed, placing an object nearby, moving a door or lid without evidence of closure, or handling a separate covering material.',
    distinction: '"Close" requires an open-to-closed state change of a rigid or defined closure. Using flexible material to cover or enclose an object is better described as "wrap".',
  }],
  [5, {
    name: 'insert',
    definition: 'moves an object from outside into the interior of a container, slot, opening, or enclosing object',
    evidence: 'The object approaches an opening, crosses its boundary, and becomes at least partly contained inside it around the central neighborhood.',
    negatives: 'Putting an object on a surface, holding it near an opening, briefly dipping it, placing it beside a container, or removing it from an interior is not insert.',
    distinction: '"Insert" is defined by crossing an outside-to-inside containment boundary. "Put" may place an object at a destination without entering an enclosing space.',
  }],
  [6, {
    name: 'turn-on',
    definition: 'operates a control so that a device, appliance, light, flame, or flow changes from inactive to active',
    evidence: 'The person presses, rotates, toggles, or otherwise operates a control, with visual evidence that the device, output, flame, light, or flow activates.',
    negatives: 'Do not confirm for touching or adjusting a control without activation, using a device that was already active, or activation occurring only outside the central neighborhood.',
    distinction: '"Turn-on" requires an inactive-to-active state change. A control manipulation without activation is better described as "adjust".',
  }],
  [7, {
    name: 'cut',
    definition: 'uses a blade or cutting tool to slice, divide, or sever material',
    evidence: 'The central neighborhood shows purposeful blade contact and cutting motion, with evidence that the material is sliced, divided, or structurally separated.',
    negatives: 'Holding a knife, positioning food, pressing without separation, peeling, scraping across a surface, or moving an already cut piece is not sufficient.',
    distinction: '"Cut" requires slicing or severing material. "Scrape" uses edge-to-surface dragging without cutting through the material.',
  }],
  [8, {
    name: 'turn-off',
    definition: 'operates a control so that a device, appliance, light, flame, or flow changes from active to inactive',
    evidence: 'The person presses, rotates, toggles, or otherwise operates a control, with visual evidence that the device, output, flame, light, or flow deactivates.',
    negatives: 'Do not confirm for touching or adjusting a control without deactivation, interacting with a device already inactive, or deactivation occurring only outside the central neighborhood.',
    distinction: '"Turn-off" requires an active-to-inactive state change. A setting change without deactivation is better described as "adjust".',
  }],
  [9, {
    name: 'pour',
    definition: 'tilts or controls a source container so that its contents flow into or onto a destination',
    evidence: 'The source container is oriented for transfer and liquid, granules, or other contents visibly flow or are clearly transferred around the anchor.',
    negatives: 'Moving or tilting a container without content transfer, filling directly from a tap, stirring, handling containers near each other, or merely changing the destination amount without evidence of source-controlled flow is insufficient.',
    distinction: '"Pour" is defined from the source-transfer perspective. Unlike "empty", the source need not become substantially depleted; unlike "fill", the destination state is not the sole defining event.',
  }],
  [11, {
    name: 'move',
    definition: 'changes the location or position of an object while stable possession or support may continue',
    evidence: 'The object is deliberately translated or repositioned around the central neighborhood without a defining acquisition, release, containment, or state change event.',
    negatives: 'Do not confirm merely for acquiring possession, releasing an object at a destination, opening, closing, inserting, or removing when relocation is only a consequence of another defining action.',
    distinction: '"Move" is primarily relocation. Unlike "take", possession need not be newly acquired; unlike "put", possession need not be relinquished.',
  }],
  [12, {
    name: 'remove',
    definition: 'extracts an object from an inside, attached, inserted, covering, or constrained state into an outside or detached state',
    evidence: 'The object crosses from inside to outside, attached to detached, inserted to withdrawn, or constrained to released around the central neighborhood.',
    negatives: 'Generic picking up from an unconstrained resting position, continued holding, simple relocation, or movement without evidence of extraction or detachment is not sufficient.',
    distinction: '"Remove" is defined by extraction or detachment from a prior containing or constrained state. "Take" is defined by gaining possession even when no such state transition exists.',
  }],
  [17, {
    name: 'adjust',
    definition: 'changes the position, setting, orientation, fit, or configuration of an object or control without a defining activation, deactivation, opening, or closing event',
    evidence: 'The central neighborhood shows a deliberate fine modification to position, orientation, alignment, fit, or setting.',
    negatives: 'Do not confirm when the defining event is clearly turning a device on or off, opening, closing, inserting, removing, taking, or moving an object to a new destination.',
    distinction: '"Adjust" changes degree, alignment, configuration, or setting while preserving the broader state. It does not require the binary state transition of turn-on, turn-off, open, or close.',
  }],
  [20, {
    name: 'empty',
    definition: 'discharges contents from a container so that the source becomes substantially less full or nearly empty',
    evidence: 'Contents visibly leave the source through tipping, dumping, draining, shaking, or another discharge process, with evidence of substantial depletion.',
    negatives: 'Pouring a small controlled amount while retaining substantial contents, tilting without visible discharge, filling a destination, or handling an empty container is not sufficient.',
    distinction: '"Empty" is defined by substantial depletion of the source. "Pour" describes controlled transfer and does not require the source to become nearly empty.',
  }],
  [25, {
    name: 'scrape',
    definition: 'drags a tool or object edge across a surface to detach, collect, clean, spread, or move material',
    evidence: 'The central neighborhood shows sustained edge-to-surface contact with a dragging or scraping motion along the surface.',
    negatives: 'Cutting through material, merely holding a blade, wiping softly, pressing without dragging, or moving already detached pieces is not sufficient.',
    distinction: '"Scrape" is defined by dragging an edge across a surface. "Cut" is defined by slicing or severing through material.',
  }],
  [26, {
    name: 'fill',
    definition: 'adds material or contents into a destination container so that its amount, occupancy, or fullness increases',
    evidence: 'Material enters a destination container and produces a visible or strongly supported increase in its contents around the central neighborhood.',
    negatives: 'Inserting one solid object, handling an already full container, tilting without transfer, or transferring material without evidence that a destination is being filled is not sufficient.',
    distinction: '"Fill" is defined by increasing the destination contents. "Pour" is defined by source-controlled flow, and "insert" is defined by one object crossing into an enclosing space.',
  }],
  [29, {
    name: 'scrub',
    definition: 'cleans a surface or object through repeated forceful rubbing with a hand, pad, brush, sponge, or cleaning tool',
    evidence: 'Repeated back-and-forth or circular rubbing contact for cleaning is visible around the central neighborhood.',
    negatives: 'Rinsing under water without repeated rubbing, squeezing a sponge, wiping once, merely being near a sink, or handling a cleaning tool without sustained rubbing is not sufficient.',
    distinction: '"Scrub" is the specific repeated forceful rubbing action. "Wash" is the broader cleaning event and may instead be dominated by rinsing or water-based cleaning.',
  }],
  [31, {
    name: 'pull',
    definition: 'applies force to draw an object or movable part toward the person or along its constrained path',
    evidence: 'The object or part translates in the pulling direction while contact and pulling force are maintained around the central neighborhood.',
    negatives: 'Touching a handle without translation, lifting vertically, acquiring possession, or opening something without visible pulling evidence is not sufficient.',
    distinction: '"Pull" is defined by the direction and maintained drawing force. If the defining result is a closed-to-open state change, "open" is more accurate.',
  }],
  [36, {
    name: 'wrap',
    definition: 'covers, encloses, seals, or secures an object using flexible material or a flexible closure',
    evidence: 'Flexible covering material or a seal is deliberately placed around or over the object and becomes folded, enclosed, sealed, or secured.',
    negatives: 'Closing a rigid door or lid, placing material nearby, covering only momentarily, or holding an already wrapped object is not sufficient.',
    distinction: '"Wrap" requires flexible material enclosing or securing an object. "Close" uses a defined rigid closure or movable part to reduce accessibility.',
  }],
]);

function listwiseCandidateBlock(letter, option) {
  const spec = LISTWISE_SPECS.get(Number(option?.class_id));
  if (!spec) throw new Error(`unsupported listwise candidate: ${option?.class_id}`);
  return `Candidate ${letter}

Action name:
"${spec.name}"

Candidate definition:
"${spec.name}" means that the person deliberately ${spec.definition}.

Expected temporal evidence:
${spec.evidence}

Important negative evidence:
${spec.negatives}

Key distinction:
${spec.distinction}`;
}

function listwisePrompt(optionMapping) {
  if (!optionMapping || !['A', 'B', 'C'].every(key => optionMapping[key])) {
    throw new Error('listwise option_mapping must contain A, B, and C');
  }
  const candidateText = ['A', 'B', 'C']
    .map(letter => listwiseCandidateBlock(letter, optionMapping[letter]))
    .join('\n\n\n');
  return `You are comparing candidate pseudo-labels for temporal action localization
in an egocentric kitchen video.

You are given five chronologically ordered frames sampled around a temporal
anchor. Frame 3 corresponds to the temporal anchor. Frames 1-2 provide
preceding context, and Frames 4-5 provide following context.

Your task is not to identify the exact instant when an action starts or ends.
Determine which single option, if any, best and most specifically describes
the temporal neighborhood of Frame 3. Do not assume that Frame 3 belongs to
an action merely because candidate actions are provided.

The candidate actions are presented in an arbitrary order. Their order does
not indicate confidence, priority, likelihood, or whether they were proposed
by another model. Evaluate all candidates using exactly the same standard.

${candidateText}


Temporal localization principles:

- Evaluate the temporal neighborhood of Frame 3, not merely whether an action
  appears somewhere in the five frames.
- The complete state transition of an action does not need to be fully visible
  within the five frames. Frame 3 may lie near the beginning, middle, or end
  of the corresponding action segment.
- However, the frames must contain temporally consistent evidence that the
  central neighborhood belongs to the selected action event.
- Evidence appearing only in the outer context frames must not determine the
  answer if that event does not extend into the neighborhood of Frame 3.
- The evidence should concern the same object, control, device, material, or
  interaction and should form one continuous event around Frame 3.
- Do not combine unrelated evidence from different objects or separate moments.
- Visible hands, tools, controls, relevant objects, or general motion are not
  sufficient by themselves.

Before deciding, internally perform the following comparison:

1. Frames 1-2: Determine the preceding object state, hand-object relationship,
   possession state, location, closure state, device state, or content state.
2. Frame 3 and its immediate neighborhood: Determine the interaction and event
   phase occurring at the anchor.
3. Frames 4-5: Determine whether the same interaction continues, completes,
   changes into another event, or produces a relevant new state.
4. Compare Candidates A, B, and C directly: Determine which candidate explains
   the complete before-anchor-after evidence most accurately and specifically.
5. Check counter-evidence: Reject a candidate if confirming it requires
   assuming an unobserved state transition, ignoring contradictory frames,
   combining unrelated events, or relying only on broad semantic similarity.

Do not output this internal analysis.

Choose exactly one label:

A: The temporal neighborhood of Frame 3 is best and most specifically
explained by Candidate A, with better support than Candidates B and C.

B: The temporal neighborhood of Frame 3 is best and most specifically
explained by Candidate B, with better support than Candidates A and C.

C: The temporal neighborhood of Frame 3 is best and most specifically
explained by Candidate C, with better support than Candidates A and B.

OTHER_ACTIVITY: A meaningful interaction clearly occurs around Frame 3, but
none of Candidates A, B, or C accurately describes the central event.

TRUE_BACKGROUND: No deliberate or meaningful object interaction occurs around
Frame 3. Camera motion, walking, waiting, observation, visible hands, object
visibility, and idle preparation alone count as background.

UNCERTAIN: The visual or temporal evidence is insufficient to distinguish the
options reliably, or two or more candidates remain similarly plausible.

Important decision rules:

- Choose a candidate only when it is the single best explanation of the
  central temporal neighborhood.
- Semantic compatibility is not sufficient. A candidate must explain the
  event more accurately and specifically than the other candidates.
- Do not choose a candidate merely because one or two frames can be made
  consistent with its definition.
- Shared objects, hand motions, tools, or intermediate states do not prove that
  two semantically related actions are equivalent.
- Prefer the action whose defining state change or interaction is directly
  supported by the temporal evidence.
- If Candidates A, B, and C describe related actions, distinguish them using
  their Key distinction fields.
- If two or more candidates remain similarly plausible after comparison,
  choose UNCERTAIN rather than selecting one arbitrarily.
- Choose OTHER_ACTIVITY only when a meaningful interaction is clearly visible
  but all three candidates are contradicted or less accurate.
- Choose TRUE_BACKGROUND only when meaningful interaction is absent.
- The use of A, B, or C does not indicate any preferred ordering.

Return exactly one JSON object with no explanation, markdown, or additional
text. The value of "label" must be exactly one of: "A", "B", "C",
"OTHER_ACTIVITY", "TRUE_BACKGROUND", "UNCERTAIN".

{"label":"A"}`;
}

function genericCandidatePrompt(candidateLabelId) {
  const spec = CANDIDATE_SPECS.get(Number(candidateLabelId));
  if (!spec) throw new Error(`unsupported candidate label id: ${candidateLabelId}`);
  return `You are verifying one candidate pseudo-label for temporal action localization
in an egocentric kitchen video.

You are given five chronologically ordered frames sampled around a temporal
anchor. Frame 3 corresponds to the anchor. Frames 1-2 provide preceding
context, and Frames 4-5 provide following context.

Your task is not to identify the exact instant when an action starts or ends.
Determine whether the temporal neighborhood of Frame 3 belongs to the action
segment represented by the SADA candidate.

SADA candidate action: "${spec.name}"

Candidate definition:
"${spec.name}" means that the person deliberately ${spec.definition}.

Expected temporal evidence:
${spec.evidence}
No single cue is sufficient by itself. The frames should provide temporally
consistent evidence that the central neighborhood belongs to this event. The
complete state transition does not need to be fully visible within five frames.

Important negative evidence:
${spec.negatives}
Do not confirm merely because hands, tools, controls, or relevant objects are
visible. Do not confirm an action that occurs only in the outer context frames
and does not extend into the temporal neighborhood of Frame 3.

Choose exactly one label:

CONFIRM_TARGET: The temporal neighborhood of Frame 3 is clearly part of the
candidate "${spec.name}" event.

OTHER_ACTIVITY: A meaningful interaction clearly occurs around Frame 3, but
the overall event is better described by another action rather than
"${spec.name}".

TRUE_BACKGROUND: No deliberate object interaction or target action occurs
around Frame 3. Camera motion, walking, waiting, visible hands, object
visibility, observation, and idle preparation alone count as background.

UNCERTAIN: The frames suggest that an interaction may be occurring, but
occlusion, sparse temporal evidence, ambiguous state, or insufficient context
makes it impossible to determine the correct case reliably.

Decision order:
1. Choose CONFIRM_TARGET only when the central neighborhood is clearly part of
   the candidate event.
2. Otherwise choose OTHER_ACTIVITY when another meaningful interaction is
   clearly identifiable.
3. Choose TRUE_BACKGROUND only when no meaningful interaction occurs around
   the anchor.
4. Choose UNCERTAIN for all remaining ambiguous cases.

Return exactly one JSON object with no explanation, markdown, or additional
text, for example: {"label":"CONFIRM_TARGET"}`;
}

function prompt(ontology, promptMode, candidateLabelId, optionMapping) {
  if (promptMode === 'candidate_listwise') {
    return listwisePrompt(optionMapping);
  }
  if (promptMode === 'candidate_verify' && Number(candidateLabelId) !== 0) {
    return genericCandidatePrompt(candidateLabelId);
  }
  if (promptMode === 'take_verify' || promptMode === 'candidate_verify') {
    return `You are verifying one candidate pseudo-label for temporal action localization
in an egocentric kitchen video.

You are given five chronologically ordered frames sampled around a temporal
anchor. Frame 3 corresponds to the anchor. Frames 1-2 provide preceding
context, and Frames 4-5 provide following context.

Your task is not to identify the exact instant when an action starts or ends.
Determine whether the temporal neighborhood of Frame 3 belongs to the action
segment represented by the SADA candidate.

SADA candidate action: "take"

Candidate definition:
"Take" means that the person deliberately acquires possession or stable
control of an object, typically by grasping, retrieving, or picking it up.

Typical temporal evidence may include approaching, reaching for, contacting,
grasping, or retrieving the object before or near the anchor; acquiring or
consolidating control around the anchor; and holding or carrying the object
after or near the anchor. None of these cues is sufficient by itself. They
should form temporally consistent evidence that the central neighborhood
belongs to an acquisition event.

The complete transition from not possessing the object to possessing it does
not need to be fully visible within the five frames. Frame 3 may lie near the
beginning, middle, or end of the annotated "take" segment. However, the
available evidence should indicate that the central temporal neighborhood is
part of a possession-acquisition event, rather than merely showing an object
that was already being held and moved for an unrelated purpose.

Choose exactly one label:

CONFIRM_TARGET: The temporal neighborhood of Frame 3 is clearly part of the
possession-acquisition event described by "take".

OTHER_ACTIVITY: A meaningful interaction clearly occurs around Frame 3, but
the overall event is better described by another action rather than "take".

TRUE_BACKGROUND: No deliberate object interaction occurs around Frame 3.
Camera motion, walking, waiting, visible hands, object visibility, observation,
and idle preparation alone count as background.

UNCERTAIN: The frames suggest that an interaction may be occurring, but
occlusion, sparse temporal evidence, ambiguous object state, or insufficient
context makes it impossible to determine reliably whether the central
neighborhood belongs to "take", another activity, or background.

Important:
- Evaluate whether Frame 3 lies inside the candidate action event, not whether
  the exact action transition happens precisely at Frame 3.
- Use Frames 1-2 and Frames 4-5 as temporal context, but do not confirm an
  action that occurs only in the outer context frames and does not extend into
  the temporal neighborhood of Frame 3.
- Do not confirm merely because hands or relevant objects are visible.
- If the object remains under stable control throughout the sequence and
  there is no evidence that the central neighborhood belongs to an acquisition
  event, do not confirm "take".
- Brief contact without evidence of acquiring stable control is not sufficient.
- Actions such as lift, pull, remove, or adjust should not automatically be
  treated as "take". Confirm only when they are part of gaining possession or
  stable control and the overall event is best described as "take".
- If a meaningful interaction is clearly visible but it does not match "take",
  choose OTHER_ACTIVITY.
- If a meaningful interaction may be occurring but its identity or temporal
  relation to Frame 3 is unclear, choose UNCERTAIN rather than TRUE_BACKGROUND.

Decision order:
1. Choose CONFIRM_TARGET when the central neighborhood is clearly part of the
   candidate "take" event.
2. Otherwise choose OTHER_ACTIVITY when another meaningful interaction is
   clearly identifiable.
3. Choose TRUE_BACKGROUND only when no meaningful interaction occurs around
   the anchor.
4. Choose UNCERTAIN for all remaining ambiguous cases.

Return exactly one JSON object with no explanation, markdown, or additional
text, for example: {"label":"CONFIRM_TARGET"}`;
  }
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

function fetchCompat(url, options) {
  if (typeof globalThis.fetch === 'function') return globalThis.fetch(url, options);
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const transport = parsed.protocol === 'https:' ? https : http;
    const request = transport.request(parsed, {
      method: options.method,
      headers: options.headers,
    }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf8');
        const status = Number(response.statusCode || 0);
        resolve({
          ok: status >= 200 && status < 300,
          status,
          text: async () => body,
        });
      });
    });
    request.on('error', reject);
    request.end(options.body);
  });
}

async function requestOne(baseUrl, apiKey, model, frames, wireApi, ontology, promptMode, candidateLabelId, optionMapping) {
  const content = [{
    type: 'input_text',
    text: prompt(ontology, promptMode, candidateLabelId, optionMapping),
  }];
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
  const response = await fetchCompat(`${baseUrl.replace(/\/$/, '')}/${endpoint}`, {
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
  if (!['target_only', 'contrastive', 'contrastive_score', 'full_ontology', 'take_verify', 'candidate_verify', 'candidate_listwise'].includes(promptMode)) {
    throw new Error('--prompt-mode is not supported');
  }
  if (!['target_only', 'take_verify', 'candidate_verify', 'candidate_listwise'].includes(promptMode) && !ontology) {
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
      const frames = prepareFrames(
        video,
        row.center_sec,
        path.join(args['cache-dir'], row.frame_cache_id ?? row.candidate_id),
      );
      const candidateLabelId = row.candidate_label_id ?? row.sada_top1_class;
      if (promptMode === 'candidate_verify' && candidateLabelId == null) {
        throw new Error(`candidate class missing for ${row.candidate_id}`);
      }
      if (promptMode === 'candidate_listwise' && !row.option_mapping) {
        throw new Error(`option_mapping missing for ${row.candidate_id}`);
      }
      const response = await requestOne(
        baseUrl, apiKey, model, frames, wireApi, ontology, promptMode,
        candidateLabelId, row.option_mapping,
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
      const verifierLabels = new Set([
        'CONFIRM_TARGET', 'OTHER_ACTIVITY', 'TRUE_BACKGROUND', 'UNCERTAIN',
      ]);
      const listwiseLabels = new Set([
        'A', 'B', 'C', 'OTHER_ACTIVITY', 'TRUE_BACKGROUND', 'UNCERTAIN',
      ]);
      const normalizedLabel = String(label || '').trim().toUpperCase();
      const match = normalizedLabel.match(/^([ABU])$/i);
      const verifierMode = ['take_verify', 'candidate_verify'].includes(promptMode);
      const listwiseMode = promptMode === 'candidate_listwise';
      let prediction = listwiseMode
        ? (listwiseLabels.has(normalizedLabel) ? normalizedLabel : 'UNCERTAIN')
        : (verifierMode
          ? (verifierLabels.has(normalizedLabel) ? normalizedLabel : 'UNCERTAIN')
          : (match ? match[1].toUpperCase() : 'U'));
      if (!verifierMode && !listwiseMode && ontology && reportedVerb) {
        const targetVerbs = new Set(
          ontology.filter(item => item.id < 10).map(item => normalizeVerb(item.verb)),
        );
        prediction = normalizeVerb(reportedVerb) === 'uncertain'
          ? 'U'
          : (targetVerbs.has(normalizeVerb(reportedVerb)) ? 'A' : 'B');
      }
      result.status = 'ok';
      result.prediction = prediction;
      result.model_reported_label = listwiseMode
        ? (listwiseLabels.has(normalizedLabel) ? normalizedLabel : null)
        : (verifierMode
          ? (verifierLabels.has(normalizedLabel) ? normalizedLabel : null)
          : (match ? match[1].toUpperCase() : null));
      result.parse_error = listwiseMode
        ? !listwiseLabels.has(normalizedLabel)
        : (verifierMode ? !verifierLabels.has(normalizedLabel) : !match);
      result.response_text = response.text;
      result.reported_verb = reportedVerb;
      result.target_probability = targetProbability;
      result.usage = response.usage;
      result.model = model;
      result.wire_api = wireApi;
      result.prompt_mode = promptMode;
      result.anchor_id = row.anchor_id ?? row.candidate_id;
      result.candidate_role = row.candidate_role ?? null;
      result.option_mapping = listwiseMode ? row.option_mapping : null;
      result.candidate_label_id = verifierMode ? Number(candidateLabelId ?? 0) : null;
      result.candidate_name = verifierMode
        ? CANDIDATE_SPECS.get(Number(candidateLabelId ?? 0))?.name || null
        : null;
      result.gate_rule = listwiseMode
        ? 'listwise_candidate_competition'
        : (verifierMode
          ? 'candidate_semantic_verifier'
          : (ontology ? 'deterministic_from_reported_verb' : 'model_label'));
    } catch (error) {
      result.error = String(error.message || error);
    }
    fs.appendFileSync(args.output, JSON.stringify(result) + '\n', 'utf8');
    process.stdout.write(`${attempted}/${Math.min(rows.length, limit)} ${row.candidate_id} ${result.status} ${result.prediction || ''}\n`);
    if (result.status !== 'ok') throw new Error(result.error);
  }
}

main().catch(error => { console.error(error.message || error); process.exitCode = 1; });
